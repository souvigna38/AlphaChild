/*
 * DeepSeek-V2 tiny forward + loss + head-only training step.
 * Mirrors llmc/deepseek_v2.py and llm.c train loop structure.
 */
#include "model.h"

#include "../rmsnorm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_random(float *p, size_t n, unsigned int *seed) {
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        p[i] = 0.02f * ((float)((*seed >> 16) % 1000) / 500.0f - 1.0f);
    }
}

static void linear_bt(
    float *logits_bt,
    const float *h,
    const float *W,
    int V,
    int C) {
    for (int v = 0; v < V; v++) {
        float s = 0.0f;
        const float *row = W + (size_t)v * (size_t)C;
        for (int c = 0; c < C; c++) {
            s += row[c] * h[c];
        }
        logits_bt[v] = s;
    }
}

size_t dsv2_model_param_bytes(const Dsv2ModelConfig *cfg) {
    const int V = cfg->vocab_size;
    const int C = cfg->n_embd;
    const int L = cfg->n_layer;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate;
    const int S = cfg->n_shared_experts;
    size_t n = (size_t)V * (size_t)C;
    n += (size_t)L * (size_t)C * 2;
    n += (size_t)L * ((size_t)C * (size_t)C + (size_t)cfg->kv_lora_rank * (size_t)C + (size_t)C * (size_t)cfg->kv_lora_rank * 2 + (size_t)C * (size_t)C);
    n += (size_t)L * ((size_t)E * (size_t)C + (size_t)E * (size_t)I * (size_t)C * 3 + (size_t)S * (size_t)I * (size_t)C * 3);
    n += (size_t)C;
    return n * sizeof(float);
}

size_t dsv2_model_activations_bytes(const Dsv2ModelConfig *cfg, int B, int T) {
    const int C = cfg->n_embd;
    const int V = cfg->vocab_size;
    const int L = cfg->n_layer;
    size_t n = (size_t)B * (size_t)T * (size_t)C * 2;
    n += (size_t)B * (size_t)T * (size_t)V;
    Dsv2BlockConfig bc = {.attn = {.n_embd = C, .n_head = cfg->n_head, .kv_lora_rank = cfg->kv_lora_rank, .block_size = cfg->block_size},
                          .moe = {.n_embd = C,
                                  .n_routed_experts = cfg->n_routed_experts,
                                  .n_shared_experts = cfg->n_shared_experts,
                                  .num_experts_per_tok = cfg->num_experts_per_tok,
                                  .moe_intermediate = cfg->moe_intermediate},
                          .rms_eps = cfg->rms_eps};
    n += (size_t)L * (dsv2_block_scratch_bytes(&bc, T) / sizeof(float));
    return n * sizeof(float);
}

void dsv2_model_init(Dsv2Model *m, const Dsv2ModelConfig *cfg, unsigned int seed) {
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    size_t bytes = dsv2_model_param_bytes(cfg);
    m->memory = (float *)malloc(bytes);
    float *p = m->memory;
    const int V = cfg->vocab_size;
    const int C = cfg->n_embd;
    const int L = cfg->n_layer;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate;
    const int S = cfg->n_shared_experts;
    const int r = cfg->kv_lora_rank;

    m->wte = p;
    p += (size_t)V * (size_t)C;
    m->rms1_w = p;
    p += (size_t)L * (size_t)C;
    m->rms2_w = p;
    p += (size_t)L * (size_t)C;
    m->wq = p;
    p += (size_t)L * (size_t)C * (size_t)C;
    m->w_dkv = p;
    p += (size_t)L * (size_t)r * (size_t)C;
    m->w_uk = p;
    p += (size_t)L * (size_t)C * (size_t)r;
    m->w_uv = p;
    p += (size_t)L * (size_t)C * (size_t)r;
    m->wo = p;
    p += (size_t)L * (size_t)C * (size_t)C;
    m->gate = p;
    p += (size_t)L * (size_t)E * (size_t)C;
    m->expert_w1 = p;
    p += (size_t)L * (size_t)E * (size_t)I * (size_t)C;
    m->expert_w2 = p;
    p += (size_t)L * (size_t)E * (size_t)C * (size_t)I;
    m->expert_w3 = p;
    p += (size_t)L * (size_t)E * (size_t)I * (size_t)C;
    m->shared_w1 = p;
    p += (size_t)L * (size_t)S * (size_t)I * (size_t)C;
    m->shared_w2 = p;
    p += (size_t)L * (size_t)S * (size_t)C * (size_t)I;
    m->shared_w3 = p;
    p += (size_t)L * (size_t)S * (size_t)I * (size_t)C;
    m->rms_f = p;

    fill_random(m->memory, bytes / sizeof(float), &seed);
    for (int l = 0; l < L; l++) {
        for (int c = 0; c < C; c++) {
            m->rms1_w[l * C + c] = 1.0f;
            m->rms2_w[l * C + c] = 1.0f;
        }
    }
    for (int c = 0; c < C; c++) {
        m->rms_f[c] = 1.0f;
    }
}

void dsv2_model_free(Dsv2Model *m) {
    free(m->memory);
    memset(m, 0, sizeof(*m));
}

void dsv2_model_forward(
    float *logits,
    const int *idx,
    const Dsv2Model *m,
    int B,
    int T,
    float *activations) {
    const Dsv2ModelConfig *cfg = &m->cfg;
    const int C = cfg->n_embd;
    const int V = cfg->vocab_size;
    const int L = cfg->n_layer;
    const int r = cfg->kv_lora_rank;

    float *x = activations;
    float *x2 = x + (size_t)B * (size_t)T * (size_t)C;
    size_t act_off = (size_t)B * (size_t)T * (size_t)C * 2;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int tok = idx[b * T + t];
            memcpy(x + ((size_t)b * (size_t)T + (size_t)t) * (size_t)C, m->wte + (size_t)tok * (size_t)C, (size_t)C * sizeof(float));
        }
    }

    Dsv2BlockConfig bc = {
        .attn = {.n_embd = C, .n_head = cfg->n_head, .kv_lora_rank = cfg->kv_lora_rank, .block_size = cfg->block_size},
        .moe = {.n_embd = C,
                .n_routed_experts = cfg->n_routed_experts,
                .n_shared_experts = cfg->n_shared_experts,
                .num_experts_per_tok = cfg->num_experts_per_tok,
                .moe_intermediate = cfg->moe_intermediate},
        .rms_eps = cfg->rms_eps};

    for (int l = 0; l < L; l++) {
        float *scr = activations + act_off;
        act_off += dsv2_block_scratch_bytes(&bc, T) / sizeof(float);
        for (int b = 0; b < B; b++) {
            float *xb = x + (size_t)b * (size_t)T * (size_t)C;
            float *x2b = x2 + (size_t)b * (size_t)T * (size_t)C;
            dsv2_block_forward(
                x2b,
                xb,
                &bc,
                T,
                m->rms1_w + (size_t)l * (size_t)C,
                m->rms2_w + (size_t)l * (size_t)C,
                m->wq + (size_t)l * (size_t)C * (size_t)C,
                m->w_dkv + (size_t)l * (size_t)r * (size_t)C,
                m->w_uk + (size_t)l * (size_t)C * (size_t)r,
                m->w_uv + (size_t)l * (size_t)C * (size_t)r,
                m->wo + (size_t)l * (size_t)C * (size_t)C,
                m->gate + (size_t)l * (size_t)cfg->n_routed_experts * (size_t)C,
                m->expert_w1 + (size_t)l * (size_t)cfg->n_routed_experts * (size_t)cfg->moe_intermediate * (size_t)C,
                m->expert_w2 + (size_t)l * (size_t)cfg->n_routed_experts * (size_t)C * (size_t)cfg->moe_intermediate,
                m->expert_w3 + (size_t)l * (size_t)cfg->n_routed_experts * (size_t)cfg->moe_intermediate * (size_t)C,
                m->shared_w1 + (size_t)l * (size_t)cfg->n_shared_experts * (size_t)cfg->moe_intermediate * (size_t)C,
                m->shared_w2 + (size_t)l * (size_t)cfg->n_shared_experts * (size_t)C * (size_t)cfg->moe_intermediate,
                m->shared_w3 + (size_t)l * (size_t)cfg->n_shared_experts * (size_t)cfg->moe_intermediate * (size_t)C,
                scr);
            memcpy(xb, x2b, (size_t)T * (size_t)C * sizeof(float));
        }
    }

    for (int b = 0; b < B; b++) {
        float *xb = x + (size_t)b * (size_t)T * (size_t)C;
        float *x2b = x2 + (size_t)b * (size_t)T * (size_t)C;
        ds4_rmsnorm_forward(x2b, xb, m->rms_f, T, C, cfg->rms_eps);
        for (int t = 0; t < T; t++) {
            linear_bt(logits + ((size_t)b * (size_t)T + (size_t)t) * (size_t)V, x2b + (size_t)t * (size_t)C, m->wte, V, C);
        }
    }
}

float dsv2_cross_entropy_loss(const float *logits, const int *targets, int B, int T, int V, int C) {
    (void)C;
    float loss = 0.0f;
    int n = 0;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *lg = logits + ((size_t)b * (size_t)T + (size_t)t) * (size_t)V;
            int y = targets[b * T + t];
            float maxv = lg[0];
            for (int v = 1; v < V; v++) {
                if (lg[v] > maxv) {
                    maxv = lg[v];
                }
            }
            float sum = 0.0f;
            for (int v = 0; v < V; v++) {
                sum += expf(lg[v] - maxv);
            }
            float logp = lg[y] - maxv - logf(sum);
            loss -= logp;
            n++;
        }
    }
    return loss / (float)n;
}

int dsv2_sample_greedy_last(const float *logits, int B, int T, int V) {
    const float *lg = logits + ((size_t)(B - 1) * (size_t)T + (size_t)(T - 1)) * (size_t)V;
    int best = 0;
    float bv = lg[0];
    for (int v = 1; v < V; v++) {
        if (lg[v] > bv) {
            bv = lg[v];
            best = v;
        }
    }
    return best;
}

void dsv2_train_head_step(
    Dsv2Model *m,
    const int *idx,
    const int *targets,
    int B,
    int T,
    float lr,
    float *activations,
    float *logits) {
    const int V = m->cfg.vocab_size;
    const int C = m->cfg.n_embd;

    dsv2_model_forward(logits, idx, m, B, T, activations);
    const float *hf_base = activations + (size_t)B * (size_t)T * (size_t)C;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *lg = logits + ((size_t)b * (size_t)T + (size_t)t) * (size_t)V;
            int y = targets[b * T + t];
            float maxv = lg[0];
            for (int v = 1; v < V; v++) {
                if (lg[v] > maxv) {
                    maxv = lg[v];
                }
            }
            float sum = 0.0f;
            for (int v = 0; v < V; v++) {
                sum += expf(lg[v] - maxv);
            }
            float inv = 1.0f / sum;
            const float *hf = hf_base + ((size_t)b * (size_t)T + (size_t)t) * (size_t)C;
            for (int v = 0; v < V; v++) {
                float p = expf(lg[v] - maxv) * inv;
                float dlogit = (p - (v == y ? 1.0f : 0.0f)) / (float)(B * T);
                float *wrow = m->wte + (size_t)v * (size_t)C;
                for (int c = 0; c < C; c++) {
                    wrow[c] -= lr * dlogit * hf[c];
                }
            }
        }
    }
}

int dsv2_model_load_checkpoint(Dsv2Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != DSV2_CKPT_MAGIC) {
        fclose(f);
        return -1;
    }
    Dsv2ModelConfig cfg;
    if (fread(&cfg, sizeof(cfg), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    if (m->memory) {
        dsv2_model_free(m);
    }
    dsv2_model_init(m, &cfg, 42u);
    size_t n = dsv2_model_param_bytes(&cfg) / sizeof(float);
    if (fread(m->memory, sizeof(float), n, f) != n) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}
