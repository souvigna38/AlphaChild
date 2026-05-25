/*
 * SwiGLU expert — matches nano_deepseek_v4 SwiGLUExpert (clamp + SiLU gate).
 * gate_up_w: (2*intermediate, hidden) row-major — first I rows = gate, next I = up.
 * down_w: (hidden, intermediate) row-major.
 */
#include "swiglu.h"

#include <math.h>
#include <stddef.h>

static void linear(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

void ds4_swiglu_forward(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit) {
    float gate[256];
    float up[256];
    float h[256];
    if (intermediate > 256 || hidden > 256) {
        return;
    }

    linear(gate_up_w, x, gate, intermediate, hidden);
    linear(gate_up_w + (size_t)intermediate * (size_t)hidden, x, up, intermediate, hidden);

    for (int i = 0; i < intermediate; i++) {
        float g = clampf(gate[i], -1e30f, swiglu_limit);
        float u = clampf(up[i], -swiglu_limit, swiglu_limit);
        float silu = g / (1.0f + expf(-g));
        h[i] = silu * u;
    }
    linear(down_w, h, out, hidden, intermediate);
}
