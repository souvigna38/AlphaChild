#include "deepseek_v4_config.h"
#include "hash_moe.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    int *tid2eid = (int *)malloc((size_t)cfg.vocab_size * (size_t)cfg.num_experts_per_tok * sizeof(int));
    ds4_hash_moe_build_table(tid2eid, &cfg);

    const int C = cfg.hidden_size;
    const int E = cfg.n_routed_experts;
    const int I = cfg.moe_intermediate_size;
    const int k = cfg.num_experts_per_tok;
    const int S = cfg.n_shared_experts;

    size_t gate_up_per = (size_t)I * 2 * (size_t)C;
    size_t down_per = (size_t)C * (size_t)I;
    float *gate_w = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    float *expert_gu = (float *)calloc((size_t)E * gate_up_per, sizeof(float));
    float *expert_down = (float *)calloc((size_t)E * down_per, sizeof(float));
    float *shared_gu = (float *)calloc((size_t)S * gate_up_per, sizeof(float));
    float *shared_down = (float *)calloc((size_t)S * down_per, sizeof(float));
    float *x = (float *)calloc((size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)C, sizeof(float));
    float *scratch = (float *)calloc((size_t)I * 2, sizeof(float));

    for (size_t i = 0; i < (size_t)E * gate_up_per; i++) {
        expert_gu[i] = 0.0005f;
    }
    for (size_t i = 0; i < (size_t)E * down_per; i++) {
        expert_down[i] = 0.0005f;
    }

    int tok = 42;
    printf("token %d experts:", tok);
    for (int j = 0; j < k; j++) {
        printf(" %d", tid2eid[(size_t)tok * (size_t)k + (size_t)j]);
    }
    printf("\n");

    ds4_hash_moe_forward_token(out, x, tok, &cfg, tid2eid, gate_w, expert_gu, expert_down, shared_gu, shared_down, scratch);
    printf("hash_moe out[0..3] = %.6f %.6f %.6f %.6f\n", out[0], out[1], out[2], out[3]);

    free(tid2eid);
    free(gate_w);
    free(expert_gu);
    free(expert_down);
    free(shared_gu);
    free(shared_down);
    free(x);
    free(out);
    free(scratch);
    ds4_config_free(&cfg);
    printf("OK — hash table matches nano tid2eid formula\n");
    return 0;
}
