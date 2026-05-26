#include "v4_ops.h"

#include <math.h>
#include <stddef.h>

void ds4_linear_backward(
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

float ds4_softmax_cross_entropy_backward(float *dlogits, const float *logits, int target, int vocab) {
    float maxv = logits[0];
    for (int v = 1; v < vocab; v++) {
        if (logits[v] > maxv) {
            maxv = logits[v];
        }
    }
    float sum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        dlogits[v] = expf(logits[v] - maxv);
        sum += dlogits[v];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    float loss = 0.0f;
    for (int v = 0; v < vocab; v++) {
        float p = dlogits[v] * inv;
        dlogits[v] = p;
        if (v == target) {
            loss = -logf(p > 1e-30f ? p : 1e-30f);
        }
    }
    for (int v = 0; v < vocab; v++) {
        dlogits[v] = (dlogits[v] - (v == target ? 1.0f : 0.0f));
    }
    return loss;
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
