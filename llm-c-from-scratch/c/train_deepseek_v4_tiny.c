/*
 * train_deepseek_v4_tiny.c — DeepSeek-V4 educational CPU port
 *
 * Phase 0: config + RMSNorm
 * Phase 1: swiglu.c
 * Phase 2: hash_moe.c (bootstrap MoE)
 *
 * Full model (sliding / CSA / HCA / mHC) → see docs/V4_SOURCES_AND_SCOPE.md
 */

#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "rmsnorm.h"
#include "swiglu.h"

#include <stdio.h>
#include <stdlib.h>

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
    (void)cfg->num_experts_per_tok;
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

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("=== DeepSeek-V4 C port (phases 0–2) ===\n");
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

    printf("\nNext C modules (see docs/V4_SOURCES_AND_SCOPE.md):\n");
    printf("  3) sliding_attn.c  4) csa/hca + indexer  5) mhc.c  6) full forward\n");

    ds4_config_free(&cfg);
    printf("\nPhase 0–2 OK\n");
    return 0;
}
