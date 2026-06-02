#include "ops.h"

#include <math.h>
#include <stddef.h>

void dsv2_linear_forward(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
}

void dsv2_linear_backward(
    float *dx,
    float *dW,
    const float *dy,
    const float *x,
    int out_dim,
    int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float g = dy[o];
        float *row = dW + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            dx[i] += row[i] * g;
            row[i] += g * x[i];
        }
    }
}

void dsv2_silu_forward(const float *x, float *y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

void dsv2_silu_backward(float *dx, const float *x, const float *dy, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float s = 1.0f / (1.0f + expf(-v));
        float silu = v * s;
        float dsilu = s + v * s * (1.0f - s);
        dx[i] += dy[i] * dsilu;
        (void)silu;
    }
}

void dsv2_softmax_forward(const float *logits, float *probs, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > maxv) {
            maxv = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] *= inv;
    }
}

void dsv2_softmax_backward(float *dlogits, const float *probs, const float *dprobs, int n) {
    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += probs[i] * dprobs[i];
    }
    for (int i = 0; i < n; i++) {
        dlogits[i] += probs[i] * (dprobs[i] - dot);
    }
}
