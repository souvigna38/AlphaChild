#include "swiglu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int C = 64;
    const int I = 96;
    float *gate_up = (float *)calloc((size_t)I * 2 * (size_t)C, sizeof(float));
    float *down = (float *)calloc((size_t)C * (size_t)I, sizeof(float));
    float *x = (float *)calloc((size_t)C, sizeof(float));
    float *y = (float *)calloc((size_t)C, sizeof(float));
    for (int i = 0; i < C; i++) {
        x[i] = 0.01f * (float)(i % 7);
    }
    for (size_t i = 0; i < (size_t)I * 2 * (size_t)C; i++) {
        gate_up[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)C * (size_t)I; i++) {
        down[i] = 0.001f;
    }
    ds4_swiglu_forward(y, x, C, I, gate_up, down, 10.0f);
    printf("SwiGLU out[0..3] = %.4f %.4f %.4f %.4f\n", y[0], y[1], y[2], y[3]);
    free(gate_up);
    free(down);
    free(x);
    free(y);
    printf("OK — compare notebook 18 / nano SwiGLUExpert\n");
    return 0;
}
