#include "block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill(float *w, size_t n, float s) {
    for (size_t i = 0; i < n; i++) {
        w[i] = s * (float)((int)(i % 11) - 5) * 0.01f;
    }
}

int main(void) {
    Dsv2BlockConfig cfg = {
        .attn = {.n_embd = 128, .n_head = 4, .kv_lora_rank = 32, .block_size = 16},
        .moe = {.n_embd = 128, .n_routed_experts = 8, .n_shared_experts = 1, .num_experts_per_tok = 2, .moe_intermediate = 192},
        .rms_eps = 1e-6f,
    };
    const int C = 128;
    const int T = 4;
    const int n = T * C;

    printf("=== DeepSeek-V2 block (MLA + MoE + RMSNorm) ===\n");

    float *scratch = (float *)malloc(dsv2_block_scratch_bytes(&cfg, T));
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    fill(x, (size_t)n, 1.0f);

    size_t rms_n = (size_t)C;
    float *rms1 = (float *)malloc(rms_n * sizeof(float));
    float *rms2 = (float *)malloc(rms_n * sizeof(float));
    for (size_t i = 0; i < rms_n; i++) {
        rms1[i] = rms2[i] = 1.0f;
    }

    /* Minimal weight blobs — same pattern as test_mla / test_moe */
    float *wq = (float *)calloc((size_t)C * C, sizeof(float));
    float *w_dkv = (float *)calloc((size_t)32 * C, sizeof(float));
    float *w_uk = (float *)calloc((size_t)C * 32, sizeof(float));
    float *w_uv = (float *)calloc((size_t)C * 32, sizeof(float));
    float *wo = (float *)calloc((size_t)C * C, sizeof(float));
    fill(wq, (size_t)C * C, 0.01f);
    fill(w_dkv, (size_t)32 * C, 0.01f);
    fill(w_uk, (size_t)C * 32, 0.01f);
    fill(w_uv, (size_t)C * 32, 0.01f);
    fill(wo, (size_t)C * C, 0.01f);

    const int E = 8, I = 192;
    float *gate = (float *)calloc((size_t)E * C, sizeof(float));
    float *w1 = (float *)calloc((size_t)E * I * C, sizeof(float));
    float *w2 = (float *)calloc((size_t)E * C * I, sizeof(float));
    float *w3 = (float *)calloc((size_t)E * I * C, sizeof(float));
    float *sw1 = (float *)calloc((size_t)I * C, sizeof(float));
    float *sw2 = (float *)calloc((size_t)C * I, sizeof(float));
    float *sw3 = (float *)calloc((size_t)I * C, sizeof(float));
    fill(gate, (size_t)E * C, 0.01f);
    fill(w1, (size_t)E * I * C, 0.01f);
    fill(w2, (size_t)E * C * I, 0.01f);
    fill(w3, (size_t)E * I * C, 0.01f);
    fill(sw1, (size_t)I * C, 0.01f);
    fill(sw2, (size_t)C * I, 0.01f);
    fill(sw3, (size_t)I * C, 0.01f);

    dsv2_block_forward(
        out,
        x,
        &cfg,
        T,
        rms1,
        rms2,
        wq,
        w_dkv,
        w_uk,
        w_uv,
        wo,
        gate,
        w1,
        w2,
        w3,
        sw1,
        sw2,
        sw3,
        scratch);

    printf("block out[0..3] = %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
    printf("OK — compare to notebook 14 forward.\n");

    free(scratch);
    free(x);
    free(out);
    free(rms1);
    free(rms2);
    free(wq);
    free(w_dkv);
    free(w_uk);
    free(w_uv);
    free(wo);
    free(gate);
    free(w1);
    free(w2);
    free(w3);
    free(sw1);
    free(sw2);
    free(sw3);
    return 0;
}
