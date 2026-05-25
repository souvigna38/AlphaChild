#include "rmsnorm.h"

#include <math.h>

void ds4_rmsnorm_forward(float *out, float *inp, const float *weight, int n, int C, float eps) {
    for (int i = 0; i < n; i++) {
        const float *x = inp + i * C;
        float *y = out + i * C;
        float sum_sq = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_sq += x[c] * x[c];
        }
        float scale = 1.0f / sqrtf(sum_sq / (float)C + eps);
        for (int c = 0; c < C; c++) {
            y[c] = x[c] * scale * weight[c];
        }
    }
}
