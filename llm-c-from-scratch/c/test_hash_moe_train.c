#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "hash_moe_train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int C = cfg.hidden_size;
    const int E = cfg.n_routed_experts;
    const int I = cfg.moe_intermediate_size;
    const int S = cfg.n_shared_experts;
    const size_t gu = (size_t)I * 2 * (size_t)C;
    const size_t dn = (size_t)C * (size_t)I;

    int *tid2eid = (int *)malloc((size_t)cfg.vocab_size * (size_t)cfg.num_experts_per_tok * sizeof(int));
    ds4_hash_moe_build_table(tid2eid, &cfg);

    float *gate = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    float *egu = (float *)calloc((size_t)E * gu, sizeof(float));
    float *ed = (float *)calloc((size_t)E * dn, sizeof(float));
    float *sgu = (float *)calloc((size_t)S * gu, sizeof(float));
    float *sd = (float *)calloc((size_t)S * dn, sizeof(float));
    for (size_t i = 0; i < (size_t)E * gu; i++) {
        egu[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)E * dn; i++) {
        ed[i] = 0.001f;
    }

    float *x = (float *)calloc((size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)C, sizeof(float));
    float *cache = (float *)calloc(ds4_hash_moe_train_cache_floats(&cfg), sizeof(float));
    for (int c = 0; c < C; c++) {
        x[c] = 0.01f;
    }

    ds4_hash_moe_forward_train(out, x, 7, &cfg, tid2eid, gate, egu, ed, sgu, sd, cache);

    float *dx = (float *)calloc((size_t)C, sizeof(float));
    float *dgu = (float *)calloc((size_t)E * gu, sizeof(float));
    float *ddn = (float *)calloc((size_t)E * dn, sizeof(float));
    float dout[64];
    for (int c = 0; c < C; c++) {
        dout[c] = 0.01f;
    }
    ds4_hash_moe_backward(dx, gate, dgu, ddn, sgu, sd, dout, x, 7, &cfg, tid2eid, gate, egu, ed, sgu, sd, cache, cfg.swiglu_limit);

    printf("hash_moe_train out[0]=%.6f dx[0]=%.6f\n", out[0], dx[0]);

    free(dx);
    free(dgu);
    free(ddn);
    free(cache);
    free(x);
    free(out);
    free(gate);
    free(egu);
    free(ed);
    free(sgu);
    free(sd);
    free(tid2eid);
    ds4_config_free(&cfg);
    printf("OK — hash_moe train forward/backward\n");
    return 0;
}
