/*
 * Smoke test for MoE — one token + short sequence.
 * Compare printed expert indices to notebook 13.
 */
#include "moe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill(float *w, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        w[i] = scale * (float)((int)(i % 13) - 6) * 0.01f;
    }
}

int main(void) {
    Dsv2MoeConfig cfg = {
        .n_embd = 128,
        .n_routed_experts = 8,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .moe_intermediate = 192,
    };
    const int C = cfg.n_embd;
    const int E = cfg.n_routed_experts;
    const int I = cfg.moe_intermediate;
    const int k = cfg.num_experts_per_tok;
    const int T = 4;

    printf("=== DeepSeek-V2 MoE (educational C) ===\n");
    printf("C=%d E=%d topk=%d I=%d shared=%d T=%d\n", C, E, k, I, cfg.n_shared_experts, T);

    size_t gate_n = (size_t)E * (size_t)C;
    size_t w1_n = (size_t)E * (size_t)I * (size_t)C;
    size_t w2_n = (size_t)E * (size_t)C * (size_t)I;
    size_t w3_n = w1_n;
    size_t sh1 = (size_t)cfg.n_shared_experts * (size_t)I * (size_t)C;

    float *gate = (float *)malloc(gate_n * sizeof(float));
    float *w1 = (float *)malloc(w1_n * sizeof(float));
    float *w2 = (float *)malloc(w2_n * sizeof(float));
    float *w3 = (float *)malloc(w3_n * sizeof(float));
    float *sw1 = (float *)malloc(sh1 * sizeof(float));
    float *sw2 = (float *)malloc((size_t)cfg.n_shared_experts * C * I * sizeof(float));
    float *sw3 = (float *)malloc(sh1 * sizeof(float));

    fill(gate, gate_n, 0.3f);
    fill(w1, w1_n, 0.2f);
    fill(w2, w2_n, 0.2f);
    fill(w3, w3_n, 0.2f);
    fill(sw1, sh1, 0.2f);
    fill(sw2, (size_t)cfg.n_shared_experts * C * I, 0.2f);
    fill(sw3, sh1, 0.2f);

    float *x = (float *)malloc((size_t)T * C * sizeof(float));
    float *out = (float *)malloc((size_t)T * C * sizeof(float));
    float *scratch = (float *)malloc(dsv2_moe_scratch_bytes(&cfg, T));
    int *topi = (int *)malloc((size_t)T * k * sizeof(int));
    float *topw = (float *)malloc((size_t)T * k * sizeof(float));

    fill(x, (size_t)T * C, 1.0f);

    dsv2_moe_forward(out, x, &cfg, T, gate, w1, w2, w3, sw1, sw2, sw3, topi, topw, scratch);

    printf("token 0 routed experts:");
    for (int j = 0; j < k; j++) {
        printf(" %d(w=%.3f)", topi[j], topw[j]);
    }
    printf("\nout[0..3] = %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
    printf("OK — match notebook 13 routing printout.\n");

    free(gate);
    free(w1);
    free(w2);
    free(w3);
    free(sw1);
    free(sw2);
    free(sw3);
    free(x);
    free(out);
    free(scratch);
    free(topi);
    free(topw);
    return 0;
}
