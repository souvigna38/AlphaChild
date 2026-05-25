#include "deepseek_v4_config.h"
#include "sliding_attn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int NH = cfg.num_attention_heads;
    const int D = cfg.head_dim;
    const int r = cfg.q_lora_rank;
    const int attn_w = ds4_attention_width(&cfg);
    const int o_mid = cfg.o_groups * cfg.o_lora_rank;

    size_t scratch_n = ds4_sliding_attn_scratch_bytes(&cfg, T);
    float *scratch = (float *)malloc(scratch_n);
    float *x = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *wq_a = (float *)calloc((size_t)r * (size_t)C, sizeof(float));
    float *wqa_n = (float *)calloc((size_t)r, sizeof(float));
    float *wq_b = (float *)calloc((size_t)attn_w * (size_t)r, sizeof(float));
    float *wkv = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
    float *wkvn = (float *)calloc((size_t)D, sizeof(float));
    float *sink = (float *)calloc((size_t)NH, sizeof(float));
    float *wo_a = (float *)calloc((size_t)cfg.o_groups * (size_t)cfg.o_lora_rank * (size_t)(attn_w / cfg.o_groups), sizeof(float));
    float *wo_b = (float *)calloc((size_t)C * (size_t)o_mid, sizeof(float));

    for (int i = 0; i < r; i++) {
        wqa_n[i] = 1.0f;
    }
    for (int i = 0; i < D; i++) {
        wkvn[i] = 1.0f;
    }
    for (size_t i = 0; i < (size_t)r * (size_t)C; i++) {
        wq_a[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)attn_w * (size_t)r; i++) {
        wq_b[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)D * (size_t)C; i++) {
        wkv[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)C * (size_t)o_mid; i++) {
        wo_b[i] = 0.001f;
    }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            x[(size_t)t * (size_t)C + (size_t)c] = 0.01f * (float)((t + c) % 5);
        }
    }

    ds4_sliding_attn_forward(out, x, T, &cfg, wq_a, wqa_n, wq_b, wkv, wkvn, sink, wo_a, wo_b, scratch);
    printf("sliding_attn out[0..3] = %.6f %.6f %.6f %.6f\n", out[0], out[1], out[2], out[3]);

    free(scratch);
    free(x);
    free(out);
    free(wq_a);
    free(wqa_n);
    free(wq_b);
    free(wkv);
    free(wkvn);
    free(sink);
    free(wo_a);
    free(wo_b);
    ds4_config_free(&cfg);
    printf("OK — sliding window + attention sink (notebook 18 / nano)\n");
    return 0;
}
