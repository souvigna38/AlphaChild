#include "v4_ops.h"

#include <math.h>
#include <stddef.h>

void ds4_linear(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
}

float ds4_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

float ds4_sqrt_softplus(float x) {
    if (x > 20.0f) {
        return sqrtf(x);
    }
    return sqrtf(logf(1.0f + expf(x)));
}

void ds4_topk_select(const float *scores, int n, int k, int *out_idx, float *out_w) {
    int used[32];
    if (n > 32) {
        n = 32;
    }
    if (k > n) {
        k = n;
    }
    for (int i = 0; i < n; i++) {
        used[i] = 0;
    }
    for (int t = 0; t < k; t++) {
        int best = -1;
        float bestv = -1e30f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && scores[i] > bestv) {
                bestv = scores[i];
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        out_idx[t] = best;
        out_w[t] = scores[best];
        used[best] = 1;
    }
}

void ds4_softmax(float *out, const float *in, int n) {
    float maxv = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > maxv) {
            maxv = in[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - maxv);
        sum += out[i];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] *= inv;
    }
}
