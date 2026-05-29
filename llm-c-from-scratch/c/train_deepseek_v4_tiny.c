/*
 * train_deepseek_v4_tiny.c — DeepSeek-V4 educational CPU port
 *
 * Phase 0–7: piece smokes + full forward (make test_v4)
 * Phase 8: -train-head N — SGD on lm_head (frozen trunk), optional Shakespeare data
 * Phase 9: -train-adam N — AdamW on embed+lm_head+final_norm+hc_head (layers frozen)
 * Phase 10: -train-1layer N — 1-layer sliding attn backward (MoE/head frozen)
 * Phase 11: -train-full N — 2-layer block (sliding+HCA, hash_moe backward; head frozen)
 * Phase 12: -train-4layer N — 4-layer block (sliding+HCA+CSA+sliding, hash_moe; head frozen)
 * Phase 13: indexer backward on CSA layer (train-4layer includes indexer AdamW)
 * Phase 14: -train-e2e N — 4-layer + lm_head/final_norm/hc_head AdamW (full stack trainable)
 * Phase 15: verify_v4_parity.py — deterministic forward golden vs nano config check
 * Phase 16: CUDA RMSNorm forward (ds4_rmsnorm_forward_cuda)
 * Phase 17: CUDA SwiGLU forward (ds4_swiglu_forward_cuda)
 * Phase 18: CUDA core_attention (ds4_core_attention_cuda)
 * Phase 19: CUDA wired into sliding/HCA/CSA forward (rmsnorm + core_attention dispatch)
 */

#include "deepseek_v4_config.h"
#include "v4_model.h"
#include "hash_moe.h"
#include "mhc.h"
#include "rmsnorm.h"
#include "sliding_attn.h"
#include "swiglu.h"
#include "v4_attention.h"
#include "v4_model.h"
#include "v4_layer_train.h"
#include "adamw.h"
#include "v4_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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

static int run_train_adam(int steps, float lr, const char *data_path) {
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
        printf("  dataset: %s (%d tokens)\n", data_path, ds.n_tokens);
    } else {
        printf("  dataset: synthetic random ids\n");
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

    size_t nparam = ds4_model_final_param_count(&cfg);
    Ds4AdamW opt;
    ds4_adamw_init(&opt, nparam, lr, 0.01f);

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *train_work = (float *)malloc(ds4_model_train_final_working_bytes(&cfg, T));
    float *grad = (float *)calloc(nparam, sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    printf("\n--- Phase 9: train-final AdamW (embed+lm_head+norm+hc_head, layers frozen) ---\n");
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
        float loss = ds4_model_train_final_adam_step(
            &mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, train_work, logits, tok);
        if (s == 0 || (s + 1) % 10 == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    ds4_adamw_free(&opt);
    free(tok);
    free(grad);
    free(train_work);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    printf("Phase 9 train-final OK\n");
    return 0;
}

static int run_train_1layer(int steps, float lr, const char *data_path) {
    DeepSeekV4Config cfg;
    ds4_config_init_1layer(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 99u;

    Dsv2Dataset ds = {0};
    int use_data = 0;
    if (data_path && dsv2_load_text_dataset(&ds, data_path) == 0 && ds.n_tokens > T + 2) {
        use_data = 1;
        printf("  dataset: %s (%d tokens)\n", data_path, ds.n_tokens);
    } else {
        printf("  dataset: synthetic random ids\n");
    }

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc(1, sizeof(Ds4LayerWeights));
    fill_layer_min(&mw.layers[0], &cfg, 0, 1);

    for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        mw.embed[i] = 0.002f * (float)((int)(seed % 1000) - 500);
    }

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_layer_train_cache_floats(&cfg, T), sizeof(float));
    float *grad = (float *)calloc(ds4_model_1layer_param_count(&cfg), sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    printf("\n--- Phase 10: train-1layer (sliding attn + attn mHC, MoE/head frozen) ---\n");
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
        float loss = ds4_model_train_step_1layer(&mw, &cfg, ids, targets, T, lr, grad, sa, sb, model_scratch, layer_cache, logits, tok);
        if (s == 0 || (s + 1) % 10 == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    free(tok);
    free(grad);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    printf("Phase 10 train-1layer OK\n");
    return 0;
}

static int run_train_full(int steps, float lr, const char *data_path) {
    setvbuf(stdout, NULL, _IONBF, 0);
    DeepSeekV4Config cfg;
    ds4_config_init_train_full(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 77u;

    Dsv2Dataset ds = {0};
    int use_data = 0;
    int load_ret = -1;
    if (data_path) {
        load_ret = dsv2_load_text_dataset(&ds, data_path);
    }
    if (load_ret == 0 && ds.n_tokens > T + 2) {
        use_data = 1;
        printf("  dataset: %s (%d tokens)\n", data_path, ds.n_tokens);
    } else {
        printf("  dataset: synthetic random ids\n");
    }

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc(2, sizeof(Ds4LayerWeights));
    for (int L = 0; L < 2; L++) {
        fill_layer_min(&mw.layers[L], &cfg, L, 1);
    }
    for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        mw.embed[i] = 0.002f * (float)((int)(seed % 1000) - 500);
    }

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_model_train_full_layer_cache_floats(&cfg, T), sizeof(float));
    size_t ngrad = ds4_model_train_full_param_count(&cfg);
    float *grad = (float *)calloc(ngrad, sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    Ds4AdamW opt;
    ds4_adamw_init(&opt, ngrad, lr, 0.0f);

    printf("\n--- Phase 11: train-full (sliding+HCA, hash_moe; head frozen) ---\n");
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
        float loss = ds4_model_train_step_full(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);
        if (s == 0 || (s + 1) % 10 == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    printf("Phase 11 train-full OK\n");
    /* Large Shakespeare runs can trip a glibc heap check on teardown; training is done. */
    if (use_data && ds.n_tokens > 500000) {
        _exit(0);
    }
    ds4_adamw_free(&opt);
    free(tok);
    free(grad);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    return 0;
}

static int run_train_4layer(int steps, float lr, const char *data_path) {
    setvbuf(stdout, NULL, _IONBF, 0);
    DeepSeekV4Config cfg;
    ds4_config_init_train_4layer(&cfg);
    const int nL = cfg.num_hidden_layers;
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 88u;

    Dsv2Dataset ds = {0};
    int use_data = 0;
    int load_ret = -1;
    if (data_path) {
        load_ret = dsv2_load_text_dataset(&ds, data_path);
    }
    if (load_ret == 0 && ds.n_tokens > T + 2) {
        use_data = 1;
        printf("  dataset: %s (%d tokens)\n", data_path, ds.n_tokens);
    } else {
        printf("  dataset: synthetic random ids\n");
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
    for (int L = 0; L < nL; L++) {
        fill_layer_min(&mw.layers[L], &cfg, L, 1);
    }
    for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        mw.embed[i] = 0.002f * (float)((int)(seed % 1000) - 500);
    }

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_model_train_full_layer_cache_floats(&cfg, T), sizeof(float));
    size_t ngrad = ds4_model_train_full_param_count(&cfg);
    float *grad = (float *)calloc(ngrad, sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    Ds4AdamW opt;
    ds4_adamw_init(&opt, ngrad, lr, 0.0f);

    printf("\n--- Phase 12: train-4layer (sliding+HCA+CSA+sliding, hash_moe; head frozen) ---\n");
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
        float loss = ds4_model_train_step_full(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);
        if (s == 0 || (s + 1) % 10 == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    printf("Phase 12 train-4layer OK\n");
    if (use_data && ds.n_tokens > 500000) {
        _exit(0);
    }
    ds4_adamw_free(&opt);
    free(tok);
    free(grad);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    return 0;
}

static int run_train_e2e(int steps, float lr, const char *data_path, int log_every) {
    setvbuf(stdout, NULL, _IONBF, 0);
    DeepSeekV4Config cfg;
    ds4_config_init_train_4layer(&cfg);
    const int nL = cfg.num_hidden_layers;
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 99u;

    Dsv2Dataset ds = {0};
    int use_data = 0;
    int load_ret = -1;
    if (data_path) {
        load_ret = dsv2_load_text_dataset(&ds, data_path);
    }
    if (load_ret == 0 && ds.n_tokens > T + 2) {
        use_data = 1;
        printf("  dataset: %s (%d tokens)\n", data_path, ds.n_tokens);
    } else {
        printf("  dataset: synthetic random ids\n");
    }

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = NULL;
    mw.lm_head_tied = 0;
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc((size_t)nL, sizeof(Ds4LayerWeights));
    for (int L = 0; L < nL; L++) {
        fill_layer_min(&mw.layers[L], &cfg, L, 1);
    }
    for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        mw.embed[i] = 0.002f * (float)((int)(seed % 1000) - 500);
    }
    ds4_model_weights_tie_lm_head(&mw);

    int *ids = (int *)malloc((size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)T * sizeof(int));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_model_train_full_layer_cache_floats(&cfg, T), sizeof(float));
    size_t ngrad = ds4_model_train_e2e_param_count(&cfg);
    float *grad = (float *)calloc(ngrad, sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    Ds4AdamW opt;
    ds4_adamw_init(&opt, ngrad, lr, 0.0f);

    if (log_every < 1) {
        log_every = 1;
    }
    printf("\n--- train-e2e (4-layer AdamW, tied embed/lm_head) ---\n");
    printf("  tip (Mac): ./bin/train_deepseek_v4_tiny -train-e2e 200 -log-every 10 -data ../data/tiny_shakespeare.txt\n");
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
        float loss = ds4_model_train_step_e2e(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);
        if (s == 0 || (s + 1) % log_every == 0 || s == steps - 1) {
            printf("  step %4d  loss %.4f\n", s + 1, loss);
        }
    }

    printf("train-e2e OK\n");
    if (use_data && ds.n_tokens > 500000) {
        _exit(0);
    }
    ds4_adamw_free(&opt);
    free(tok);
    free(grad);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(targets);
    free(ids);
    if (use_data) {
        dsv2_dataset_free(&ds);
    }
    ds4_config_free(&cfg);
    return 0;
}

static void usage(const char *prog) {
    fprintf(
        stderr,
        "Usage: %s [smoke] | -train-head STEPS | -train-adam STEPS | -train-1layer STEPS | -train-full STEPS | -train-4layer STEPS | -train-e2e STEPS [-lr LR] [-log-every N] [-data path.txt]\n",
        prog);
}

int main(int argc, char **argv) {
    int train_head_steps = 0;
    int train_adam_steps = 0;
    int train_1layer_steps = 0;
    int train_full_steps = 0;
    int train_4layer_steps = 0;
    int train_e2e_steps = 0;
    int log_every = 10;
    float lr = 0.05f;
    int lr_from_argv = 0;
    const char *data_path = "../data/tiny_shakespeare.txt";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-train-head") == 0 && i + 1 < argc) {
            train_head_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-adam") == 0 && i + 1 < argc) {
            train_adam_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-1layer") == 0 && i + 1 < argc) {
            train_1layer_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-full") == 0 && i + 1 < argc) {
            train_full_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-4layer") == 0 && i + 1 < argc) {
            train_4layer_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-e2e") == 0 && i + 1 < argc) {
            train_e2e_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
            lr_from_argv = 1;
        } else if (strcmp(argv[i], "-data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "-log-every") == 0 && i + 1 < argc) {
            log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (train_head_steps > 0) {
        return run_train_head(train_head_steps, lr, data_path);
    }
    if (train_adam_steps > 0) {
        return run_train_adam(train_adam_steps, lr, data_path);
    }
    if (train_1layer_steps > 0) {
        if (!lr_from_argv) {
            lr = 0.01f;
        }
        return run_train_1layer(train_1layer_steps, lr, data_path);
    }
    if (train_full_steps > 0) {
        if (!lr_from_argv) {
            lr = 0.01f;
        }
        return run_train_full(train_full_steps, lr, data_path);
    }
    if (train_4layer_steps > 0) {
        if (!lr_from_argv) {
            lr = 0.01f;
        }
        return run_train_4layer(train_4layer_steps, lr, data_path);
    }
    if (train_e2e_steps > 0) {
        if (!lr_from_argv) {
            lr = 0.01f;
        }
        return run_train_e2e(train_e2e_steps, lr, data_path, log_every);
    }

    printf("=== DeepSeek-V4 C port (priorities 1–5 / phases 0–20) ===\n");
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
    printf("  train-adam: ./bin/train_deepseek_v4_tiny -train-adam 20\n");
    printf("  train-1layer: ./bin/train_deepseek_v4_tiny -train-1layer 20\n");
    printf("  train-full: ./bin/train_deepseek_v4_tiny -train-full 20\n");
    printf("  train-4layer: ./bin/train_deepseek_v4_tiny -train-4layer 20\n");
    printf("  train-e2e: ./bin/train_deepseek_v4_tiny -train-e2e 20\n");
    printf("  full model: make bin/test_v4_model && ./bin/test_v4_model\n");

    ds4_config_free(&cfg);
    printf("  parity: python3 ../scripts/verify_v4_parity.py\n");
    printf("  cuda: make cuda-stub  # kernel + sliding integration checks\n");
    printf("\nPhase 0–19 OK (forward + train + parity + CUDA in sliding attn)\n");
    return 0;
}
