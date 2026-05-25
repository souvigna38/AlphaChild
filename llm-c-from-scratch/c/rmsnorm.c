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

void ds4_rmsnorm_backward(
    float *dinp,
    float *dweight,
    const float *dout,
    const float *inp,
    const float *out,
    const float *weight,
    int n,
    int C,
    float eps) {
    (void)out;
    for (int i = 0; i < n; i++) {
        const float *x = inp + i * C;
        const float *dy = dout + i * C;
        float *dx = dinp + i * C;
        float sum_sq = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_sq += x[c] * x[c];
        }
        float mean_sq = sum_sq / (float)C + eps;
        float inv_rms = 1.0f / sqrtf(mean_sq);
        float dot = 0.0f;
        for (int c = 0; c < C; c++) {
            dot += dy[c] * weight[c] * x[c];
        }
        for (int c = 0; c < C; c++) {
            dweight[c] += dy[c] * x[c] * inv_rms;
            dx[c] += inv_rms * dy[c] * weight[c] - inv_rms * inv_rms * inv_rms * x[c] * dot / (float)C;
        }
    }
}
