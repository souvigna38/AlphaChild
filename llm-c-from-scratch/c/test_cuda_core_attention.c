#include "deepseek_v4_config.h"
#include "ds4_cuda.h"
#include "sliding_attn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);

    const int NH = cfg.num_attention_heads;
    const int D = cfg.head_dim;
    const int T = 8;
    const int window = cfg.sliding_window;

    size_t bytes_q = (size_t)NH * (size_t)T * (size_t)D * sizeof(float);
    size_t bytes_k = bytes_q;
    size_t bytes_ctx = bytes_q;
    size_t bytes_mask = (size_t)T * (size_t)T * sizeof(int);

    float *q = (float *)malloc(bytes_q);
    float *keys = (float *)malloc(bytes_k);
    float *ctx_cpu = (float *)malloc(bytes_ctx);
    float *ctx_cuda = (float *)malloc(bytes_ctx);
    float *sink = (float *)malloc((size_t)NH * sizeof(float));
    int *mask = (int *)malloc(bytes_mask);
    if (!q || !keys || !ctx_cpu || !ctx_cuda || !sink || !mask) {
        return 1;
    }

    unsigned seed = 19u;
    for (size_t i = 0; i < bytes_q / sizeof(float); i++) {
        seed = seed * 1103515245u + 12345u;
        q[i] = 0.01f * (float)((int)(seed % 1000) - 500);
        keys[i] = 0.01f * (float)((int)((seed >> 8) % 1000) - 500);
    }
    for (int h = 0; h < NH; h++) {
        sink[h] = 0.1f * (float)(h + 1);
    }
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - window + 1);
            mask[(size_t)tq * (size_t)T + (size_t)tk] = causal && in_window;
        }
    }

    ds4_core_attention(ctx_cpu, q, keys, NH, T, T, D, sink, mask);
    ds4_core_attention_cuda(ctx_cuda, q, keys, NH, T, T, D, sink, mask);

    float max_err = 0.0f;
    for (size_t i = 0; i < bytes_ctx / sizeof(float); i++) {
        float e = fabsf(ctx_cpu[i] - ctx_cuda[i]);
        if (e > max_err) {
            max_err = e;
        }
    }

    if (ds4_cuda_available()) {
        printf("cuda devices=%d max_abs_err=%.6g\n", ds4_cuda_device_count(), max_err);
        if (max_err > 1e-4f) {
            fprintf(stderr, "CUDA core_attention differs from CPU reference\n");
            ds4_config_free(&cfg);
            free(mask);
            free(sink);
            free(ctx_cuda);
            free(ctx_cpu);
            free(keys);
            free(q);
            return 1;
        }
        printf("OK — CUDA core_attention matches CPU (NH=%d T=%d D=%d)\n", NH, T, D);
    } else {
        printf("OK — CUDA not available; core_attention_cuda uses CPU fallback (max_err=%.6g)\n", max_err);
    }

    ds4_config_free(&cfg);
    free(mask);
    free(sink);
    free(ctx_cuda);
    free(ctx_cpu);
    free(keys);
    free(q);
    return 0;
}
