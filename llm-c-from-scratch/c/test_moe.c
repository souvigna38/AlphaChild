#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "moe.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = 4;
    const int C = cfg.hidden_size;
    const int E = cfg.n_routed_experts;
    const int I = cfg.moe_intermediate_size;
    const int S = cfg.n_shared_experts;
    const int k = cfg.num_experts_per_tok;
    size_t gu = (size_t)I * 2 * (size_t)C;
    size_t dn = (size_t)C * (size_t)I;

    int ids[8] = {1, 7, 3, 9};
    int *tid2eid = (int *)malloc((size_t)cfg.vocab_size * (size_t)k * sizeof(int));
    ds4_hash_moe_build_table(tid2eid, &cfg);

    float *gate = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    float *bias = (float *)calloc((size_t)E, sizeof(float));
    float *egu = (float *)calloc((size_t)E * gu, sizeof(float));
    float *ed = (float *)calloc((size_t)E * dn, sizeof(float));
    float *sgu = (float *)calloc((size_t)S * gu, sizeof(float));
    float *sd = (float *)calloc((size_t)S * dn, sizeof(float));
    float *x = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *y_hash = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *y_moe = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *scratch = (float *)calloc(ds4_moe_scratch_bytes(&cfg) / sizeof(float), sizeof(float));

    for (size_t i = 0; i < (size_t)E * gu; i++) {
        egu[i] = 0.0005f;
    }
    for (size_t i = 0; i < (size_t)E * dn; i++) {
        ed[i] = 0.0005f;
    }

    ds4_moe_forward(
        y_hash, x, ids, T, DS4_MLP_HASH_MOE, &cfg, tid2eid, gate, NULL, egu, ed, sgu, sd, scratch);
    ds4_moe_forward(
        y_moe, x, ids, T, DS4_MLP_MOE, &cfg, tid2eid, gate, bias, egu, ed, sgu, sd, scratch);
    printf("hash_moe out[0]=%.6f routed_moe out[0]=%.6f\n", y_hash[0], y_moe[0]);

    free(tid2eid);
    free(gate);
    free(bias);
    free(egu);
    free(ed);
    free(sgu);
    free(sd);
    free(x);
    free(y_hash);
    free(y_moe);
    free(scratch);
    ds4_config_free(&cfg);
    printf("OK — hash_moe and routed MoE forward\n");
    return 0;
}
