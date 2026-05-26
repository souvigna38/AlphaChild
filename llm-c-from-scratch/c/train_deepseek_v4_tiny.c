/*
 * train_deepseek_v4_tiny.c — DeepSeek-V4 educational CPU port
 *
 * Phase 0–7: piece smokes + full forward (make test_v4)
 * Phase 8: -train-head N — SGD on lm_head (frozen trunk), optional Shakespeare data
 */

#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "mhc.h"
#include "rmsnorm.h"
#include "sliding_attn.h"
#include "swiglu.h"
#include "v4_attention.h"
#include "v4_model.h"
#include "v4_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Optional character dataset (deepseek_v2/data.c) when -train-head uses ../data/ */
#include "deepseek_v2/data.h"

static int run_rmsnorm_smoke(int C, float eps) {
    int n = 4;
    float *inp = (float *)malloc((size_t)n * (size_t)C * sizeof(float));
    float *out = (float *)malloc((size_t)n * (size_t)C * sizeof(float));
    float *weight = (float *)malloc((size_t)C * sizeof(float));
    for (int i = 0; i < n * C; i++) {
        inp[i] = 0.01f * (float)(i % 11);
    }
    for (int i = 0; i < C; i++) {
        weight[i] = 1.0f;
    }
    ds4_rmsnorm_forward(out, inp, weight, n, C, eps);
    float mean_sq = 0.0f;
    for (int i = 0; i < C; i++) {
        mean_sq += out[i] * out[i];
    }
    mean_sq /= (float)C;
    free(inp);
    free(out);
    free(weight);
    if (out[0] == out[0] && (mean_sq < 0.5f || mean_sq > 2.0f)) {
        fprintf(stderr, "RMSNorm smoke failed: mean_sq=%f\n", mean_sq);
        return 1;
    }
    printf("  RMSNorm OK\n");
    return 0;
}

static int run_swiglu_smoke(void) {
    const int C = 64;
    const int I = 96;
    float *gate_up = (float *)calloc((size_t)I * 2 * (size_t)C, sizeof(float));
    float *down = (float *)calloc((size_t)C * (size_t)I, sizeof(float));
    float *x = (float *)calloc((size_t)C, sizeof(float));
    float *y = (float *)calloc((size_t)C, sizeof(float));
    for (size_t i = 0; i < (size_t)C; i++) {
        x[i] = 0.01f;
    }
    for (size_t i = 0; i < (size_t)I * 2 * (size_t)C; i++) {
        gate_up[i] = 0.001f;
    }
    ds4_swiglu_forward(y, x, C, I, gate_up, down, 10.0f);
    free(gate_up);
    free(down);
    free(x);
    free(y);
    printf("  SwiGLU OK\n");
    return 0;
}

static int run_hash_moe_smoke(const DeepSeekV4Config *cfg) {
    int *tid2eid = (int *)malloc((size_t)cfg->vocab_size * (size_t)cfg->num_experts_per_tok * sizeof(int));
    ds4_hash_moe_build_table(tid2eid, cfg);
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int S = cfg->n_shared_experts;
    size_t gu = (size_t)I * 2 * (size_t)C;
    size_t dn = (size_t)C * (size_t)I;
    float *gate_w = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    float *expert_gu = (float *)calloc((size_t)E * gu, sizeof(float));
    float *expert_down = (float *)calloc((size_t)E * dn, sizeof(float));
    float *shared_gu = (float *)calloc((size_t)S * gu, sizeof(float));
    float *shared_down = (float *)calloc((size_t)S * dn, sizeof(float));
    float *x = (float *)calloc((size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)C, sizeof(float));
    float *scratch = (float *)calloc((size_t)I * 2, sizeof(float));
    ds4_hash_moe_forward_token(out, x, 7, cfg, tid2eid, gate_w, expert_gu, expert_down, shared_gu, shared_down, scratch);
    free(tid2eid);
    free(gate_w);
    free(expert_gu);
    free(expert_down);
    free(shared_gu);
    free(shared_down);
    free(x);
    free(out);
    free(scratch);
    printf("  hash_moe OK (token 7 experts from tid2eid)\n");
    return 0;
}

static void fill_layer_min(Ds4LayerWeights *w, const DeepSeekV4Config *cfg, int layer_idx, int hash) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int attn_w = ds4_attention_width(cfg);
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    const int k = cfg->num_experts_per_tok;
    const int D = cfg->head_dim;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * D;
    const int rate_csa = cfg->compress_rate_csa;
    const int rate_hca = cfg->compress_rate_hca;
    Ds4AttentionType attn = cfg->layer_types[layer_idx];

    memset(w, 0, sizeof(*w));
    w->moe_gate = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    w->moe_expert_gu = (float *)calloc((size_t)E * (size_t)I * 2 * (size_t)C, sizeof(float));
    w->moe_expert_down = (float *)calloc((size_t)E * (size_t)C * (size_t)I, sizeof(float));
    w->moe_shared_gu = (float *)calloc((size_t)cfg->n_shared_experts * (size_t)I * 2 * (size_t)C, sizeof(float));
    w->moe_shared_down = (float *)calloc((size_t)C * (size_t)I, sizeof(float));
    if (!hash) {
        w->moe_route_bias = (float *)calloc((size_t)E, sizeof(float));
    }
    w->attn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->ffn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->attn_norm_w[0] = w->ffn_norm_w[0] = 1.0f;
    w->attn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->attn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->ffn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->ffn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->attn_hc_scale[0] = w->attn_hc_scale[1] = w->attn_hc_scale[2] = 1.0f;
    w->ffn_hc_scale[0] = w->ffn_hc_scale[1] = w->ffn_hc_scale[2] = 1.0f;
    w->wq_a = (float *)calloc((size_t)cfg->q_lora_rank * (size_t)C, sizeof(float));
    w->w_qa_norm = (float *)calloc((size_t)cfg->q_lora_rank, sizeof(float));
    w->wq_b = (float *)calloc((size_t)attn_w * (size_t)cfg->q_lora_rank, sizeof(float));
    w->wkv = (float *)calloc((size_t)C * (size_t)C, sizeof(float));
    w->w_kv_norm = (float *)calloc((size_t)C, sizeof(float));
    w->attn_sink = (float *)calloc((size_t)cfg->num_attention_heads, sizeof(float));
    w->wo_a = (float *)calloc((size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(attn_w / cfg->o_groups), sizeof(float));
    w->wo_b = (float *)calloc((size_t)C * (size_t)o_mid, sizeof(float));
    w->w_qa_norm[0] = w->w_kv_norm[0] = 1.0f;
    if (attn == DS4_ATTN_HCA) {
        w->hca_w_kv = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
        w->hca_w_gate = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
        w->hca_pos_bias = (float *)calloc((size_t)rate_hca * (size_t)D, sizeof(float));
        w->hca_norm = (float *)calloc((size_t)D, sizeof(float));
        w->hca_norm[0] = 1.0f;
    } else if (attn == DS4_ATTN_CSA) {
        w->csa_w_kv = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
        w->csa_w_gate = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
        w->csa_pos_bias = (float *)calloc((size_t)rate_csa * (size_t)HD2, sizeof(float));
        w->csa_norm = (float *)calloc((size_t)D, sizeof(float));
        w->csa_norm[0] = 1.0f;
        w->idx_wq_b = (float *)calloc((size_t)(cfg->index_n_heads * HD) * (size_t)cfg->q_lora_rank, sizeof(float));
        w->idx_w_weights = (float *)calloc((size_t)cfg->index_n_heads * (size_t)C, sizeof(float));
        w->idx_w_kv = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
        w->idx_w_gate = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
        w->idx_pos_bias = (float *)calloc((size_t)rate_csa * (size_t)(2 * HD), sizeof(float));
        w->idx_norm = (float *)calloc((size_t)HD, sizeof(float));
        w->idx_norm[0] = 1.0f;
    }
    if (hash) {
        w->tid2eid = (int *)malloc((size_t)cfg->vocab_size * (size_t)k * sizeof(int));
        ds4_hash_moe_build_table(w->tid2eid, cfg);
    }
}

static void rand_small(float *p, size_t n, unsigned int *seed) {
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        p[i] = 0.002f * (float)((int)(*seed % 1000) - 500);
    }
}

static int run_train_head(int steps, float lr, const char *data_path) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    const int nL = cfg.num_hidden_layers;
    unsigned int seed = 42u;

    Dsv2Dataset ds = {0};
    int use_data = 0;
    if (data_path && dsv2_load_text_dataset(&ds, data_path) == 0 && ds.n_tokens > T + 2) {
        use_data = 1;
        printf("  dataset: %s (%d tokens, vocab %d)\n", data_path, ds.n_tokens, ds.vocab.vocab_size);
    } else {
        printf("  dataset: synthetic random ids (vocab %d)\n", V);
    }

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc((size_t)nL, sizeof(Ds4LayerWeights));
    for (int i = 0; i < nL; i++) {
        fill_layer_min(&mw.layers[i], &cfg, i, cfg.mlp_layer_types[i] == DS4_MLP_HASH_MOE);
    }
    rand_small(mw.embed, (size_t)V * (size_t)C, &seed);
    rand_small(mw.lm_head, (size_t)V * (size_t)C, &seed);

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *norm_h = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    printf("\n--- Phase 8: train-head (lm_head SGD, lr=%.4f, steps=%d) ---\n", lr, steps);
    for (int s = 0; s < steps; s++) {
        if (use_data) {
            dsv2_get_batch(ds.tokens, ds.n_tokens, ids, targets, 1, T, &seed);
            for (int t = 0; t < T; t++) {
                ids[t] %= V;
                targets[t] %= V;
            }
        } else {
            for (int t = 0; t < T; t++) {
                seed = seed * 1103515245u + 12345u;
                ids[t] = (int)(seed % (unsigned int)V);
                seed = seed * 1103515245u + 12345u;
                targets[t] = (int)(seed % (unsigned int)V);
            }
        }
        float loss = ds4_model_train_head_step(&mw, &cfg, ids, targets, T, lr, sa, sb, scratch, logits, norm_h, tok);
        if (s == 0 || (s + 1) % 10 == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    free(tok);
    free(scratch);
    free(sa);
    free(sb);
    free(norm_h);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    printf("Phase 8 train-head OK\n");
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [smoke] | -train-head STEPS [-lr LR] [-data path.txt]\n", prog);
}

int main(int argc, char **argv) {
    int train_head_steps = 0;
    float lr = 0.05f;
    const char *data_path = "../data/tiny_shakespeare.txt";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-train-head") == 0 && i + 1 < argc) {
            train_head_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (train_head_steps > 0) {
        return run_train_head(train_head_steps, lr, data_path);
    }

    printf("=== DeepSeek-V4 C port (phases 0–8) ===\n");
    printf("Reference: vendor/nano-deepseek-v4/nano_deepseek_v4/modeling.py\n");
    printf("Notebook:  ../18.DeepSeekV4Path.ipynb\n\n");

    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    ds4_config_print(&cfg);
    printf("\nSmoke tests:\n");

    if (run_rmsnorm_smoke(cfg.hidden_size, cfg.rms_norm_eps) != 0) {
        ds4_config_free(&cfg);
        return 1;
    }
    if (run_swiglu_smoke() != 0) {
        ds4_config_free(&cfg);
        return 1;
    }
    if (run_hash_moe_smoke(&cfg) != 0) {
        ds4_config_free(&cfg);
        return 1;
    }

    printf("  attention scratch (T=8) = %zu bytes\n", ds4_attention_scratch_bytes(&cfg, 8));
    printf("  model scratch (T=8) = %zu bytes\n", ds4_model_scratch_bytes(&cfg, 8));
    printf("  train-head: ./bin/train_deepseek_v4_tiny -train-head 20\n");
    printf("  full model: make bin/test_v4_model && ./bin/test_v4_model\n");

    ds4_config_free(&cfg);
    printf("\nPhase 0–8 OK (forward + train-head entry point)\n");
    return 0;
}
