#include "ds4_cuda.h"
#include "swiglu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int hidden = 64;
    const int intermediate = 96;
    const float swiglu_limit = 10.0f;

    size_t bytes_gu = (size_t)intermediate * 2u * (size_t)hidden * sizeof(float);
    size_t bytes_down = (size_t)hidden * (size_t)intermediate * sizeof(float);
    size_t bytes_x = (size_t)hidden * sizeof(float);

    float *gate_up = (float *)malloc(bytes_gu);
    float *down = (float *)malloc(bytes_down);
    float *x = (float *)malloc(bytes_x);
    float *out_cpu = (float *)malloc(bytes_x);
    float *out_cuda = (float *)malloc(bytes_x);
    if (!gate_up || !down || !x || !out_cpu || !out_cuda) {
        return 1;
    }

    unsigned seed = 11u;
    for (size_t i = 0; i < bytes_gu / sizeof(float); i++) {
        seed = seed * 1103515245u + 12345u;
        gate_up[i] = 0.001f * (float)((int)(seed % 200) - 100);
    }
    for (size_t i = 0; i < bytes_down / sizeof(float); i++) {
        seed = seed * 1103515245u + 12345u;
        down[i] = 0.001f * (float)((int)(seed % 200) - 100);
    }
    for (int i = 0; i < hidden; i++) {
        x[i] = 0.01f * (float)(i % 7);
    }

    ds4_swiglu_forward(out_cpu, x, hidden, intermediate, gate_up, down, swiglu_limit);
    ds4_swiglu_forward_cuda(out_cuda, x, hidden, intermediate, gate_up, down, swiglu_limit);

    float max_err = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float e = fabsf(out_cpu[i] - out_cuda[i]);
        if (e > max_err) {
            max_err = e;
        }
    }

    if (ds4_cuda_available()) {
        printf("cuda devices=%d max_abs_err=%.6g\n", ds4_cuda_device_count(), max_err);
        if (max_err > 1e-4f) {
            fprintf(stderr, "CUDA SwiGLU differs from CPU reference\n");
            free(out_cuda);
            free(out_cpu);
            free(x);
            free(down);
            free(gate_up);
            return 1;
        }
        printf("OK — CUDA SwiGLU matches CPU (hidden=%d I=%d)\n", hidden, intermediate);
    } else {
        printf("OK — CUDA not available; swiglu_forward_cuda uses CPU fallback (max_err=%.6g)\n", max_err);
    }

    free(out_cuda);
    free(out_cpu);
    free(x);
    free(down);
    free(gate_up);
    return 0;
}
