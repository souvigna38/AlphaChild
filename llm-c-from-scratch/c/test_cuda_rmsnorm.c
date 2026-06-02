#include "ds4_cuda.h"
#include "rmsnorm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int n = 8;
    const int C = 64;
    const float eps = 1e-6f;
    size_t bytes = (size_t)n * (size_t)C * sizeof(float);

    float *inp = (float *)malloc(bytes);
    float *weight = (float *)malloc((size_t)C * sizeof(float));
    float *out_cpu = (float *)malloc(bytes);
    float *out_cuda = (float *)malloc(bytes);
    if (!inp || !weight || !out_cpu || !out_cuda) {
        return 1;
    }

    unsigned seed = 7u;
    for (size_t i = 0; i < (size_t)n * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        inp[i] = 0.01f * (float)((int)(seed % 1000) - 500);
    }
    for (int c = 0; c < C; c++) {
        weight[c] = 1.0f;
    }

    ds4_rmsnorm_forward(out_cpu, inp, weight, n, C, eps);
    ds4_rmsnorm_forward_cuda(out_cuda, inp, weight, n, C, eps);

    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)n * (size_t)C; i++) {
        float e = fabsf(out_cpu[i] - out_cuda[i]);
        if (e > max_err) {
            max_err = e;
        }
    }

    if (ds4_cuda_available()) {
        printf("cuda devices=%d max_abs_err=%.6g\n", ds4_cuda_device_count(), max_err);
        if (max_err > 1e-4f) {
            fprintf(stderr, "CUDA RMSNorm differs from CPU reference\n");
            free(out_cuda);
            free(out_cpu);
            free(weight);
            free(inp);
            return 1;
        }
        printf("OK — CUDA RMSNorm matches CPU (n=%d C=%d)\n", n, C);
    } else {
        printf("OK — CUDA not available; rmsnorm_forward_cuda uses CPU fallback (max_err=%.6g)\n", max_err);
    }

    free(out_cuda);
    free(out_cpu);
    free(weight);
    free(inp);
    return 0;
}
