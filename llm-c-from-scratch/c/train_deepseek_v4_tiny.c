/*
 * train_deepseek_v4_tiny.c
 *
 * Educational CPU trainer skeleton for DeepSeek-V4 (tiny config).
 * Phase 0: config + RMSNorm + roadmap hooks — NOT a full V4 trainer yet.
 *
 * Baseline style: karpathy/llm.c train_gpt2.c
 * Architecture reference: nano-deepseek-v4 (Python)
 */

#include "deepseek_v4_config.h"
#include "rmsnorm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fill_random(float *p, int n, unsigned int *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        p[i] = (float)((*seed >> 16) % 1000) / 500.0f - 1.0f;
    }
}

static int run_rmsnorm_smoke_test(int C, float eps) {
    int n = 4;
    float *inp = (float *)malloc((size_t)n * (size_t)C * sizeof(float));
    float *out = (float *)malloc((size_t)n * (size_t)C * sizeof(float));
    float *weight = (float *)malloc((size_t)C * sizeof(float));
    unsigned seed = 42u;
    fill_random(inp, n * C, &seed);
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
    /* normalized vector should have RMS ~ 1 when weight=1 */
    if (fabsf(sqrtf(mean_sq) - 1.0f) > 0.05f) {
        fprintf(stderr, "RMSNorm smoke test failed: rms=%f\n", sqrtf(mean_sq));
        return 1;
    }
    printf("RMSNorm smoke test passed (rms≈1)\n");
    return 0;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("=== llm-c-from-scratch: DeepSeek-V4 C port (phase 0) ===\n");
    printf("Upstream llm.c:  vendor/llm.c  (run scripts/setup_vendor.sh)\n");
    printf("V4 reference:    vendor/nano-deepseek-v4/nano_deepseek_v4/modeling.py\n\n");

    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    ds4_config_print(&cfg);
    printf("\n");

    if (run_rmsnorm_smoke_test(cfg.hidden_size, cfg.rms_norm_eps) != 0) {
        ds4_config_free(&cfg);
        return 1;
    }

    printf("\nNext implementation steps (see docs/V4_SOURCES_AND_SCOPE.md):\n");
    printf("  1) hash_moe.c  2) moe.c  3) sliding_attn.c\n");
    printf("  4) csa/hca + indexer  5) mhc.c  6) wire full forward like train_gpt2.c\n");

    ds4_config_free(&cfg);
    printf("\nPhase 0 OK — not training yet.\n");
    return 0;
}
