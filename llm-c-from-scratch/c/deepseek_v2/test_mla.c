/*
 * Smoke test for MLA — prints cache sizes and runs a tiny forward pass.
 * No trained weights: random-ish fixed pattern so the binary always runs.
 */
#include "mla.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_pattern(float *w, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) {
        w[i] = scale * (float)((i % 17) - 8) * 0.01f;
    }
}

int main(void) {
    Dsv2MlaConfig cfg = {.n_embd = 128, .n_head = 4, .kv_lora_rank = 32, .block_size = 32};
    const int T = 8;
    const int C = cfg.n_embd;
    const int r = cfg.kv_lora_rank;

    printf("=== DeepSeek-V2 MLA (educational C) ===\n");
    printf("C=%d NH=%d hs=%d kv_lora_rank=%d T=%d\n", C, cfg.n_head, C / cfg.n_head, r, T);
    printf("MLA KV bytes/token:  %zu\n", dsv2_mla_kv_cache_bytes_per_token(&cfg));
    printf("MHA KV bytes/token: %zu\n", dsv2_mha_kv_cache_bytes_per_token(&cfg));
    printf(
        "ratio MHA/MLA (one layer): %.2f\n",
        (double)dsv2_mha_kv_cache_bytes_per_token(&cfg) / (double)dsv2_mla_kv_cache_bytes_per_token(&cfg));

    float *x = (float *)malloc((size_t)T * C * sizeof(float));
    float *out = (float *)malloc((size_t)T * C * sizeof(float));
    float *c_kv = (float *)malloc(dsv2_mla_c_kv_bytes(&cfg, T));
    float *scratch = (float *)malloc(dsv2_mla_scratch_bytes(&cfg, T));

    float *wq = (float *)malloc((size_t)C * C * sizeof(float));
    float *w_dkv = (float *)malloc((size_t)r * C * sizeof(float));
    float *w_uk = (float *)malloc((size_t)C * r * sizeof(float));
    float *w_uv = (float *)malloc((size_t)C * r * sizeof(float));
    float *wo = (float *)malloc((size_t)C * C * sizeof(float));

    fill_pattern(x, T, C, 1.0f);
    fill_pattern(wq, C, C, 0.5f);
    fill_pattern(w_dkv, r, C, 0.5f);
    fill_pattern(w_uk, C, r, 0.5f);
    fill_pattern(w_uv, C, r, 0.5f);
    fill_pattern(wo, C, C, 0.5f);

    dsv2_mla_forward(out, c_kv, x, &cfg, T, wq, w_dkv, w_uk, w_uv, wo, scratch);

    printf("c_kv[0..3] = %.4f %.4f %.4f %.4f\n", c_kv[0], c_kv[1], c_kv[2], c_kv[3]);
    printf("out[0] first 4 dims = %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
    printf("OK — compare to notebook 12 PyTorch shapes.\n");

    free(x);
    free(out);
    free(c_kv);
    free(scratch);
    free(wq);
    free(w_dkv);
    free(w_uk);
    free(w_uv);
    free(wo);
    return 0;
}
