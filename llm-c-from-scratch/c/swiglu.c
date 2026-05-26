/*
 * SwiGLU expert — matches nano_deepseek_v4 SwiGLUExpert (clamp + SiLU gate).
 * gate_up_w: (2*intermediate, hidden) row-major — first I rows = gate, next I = up.
 * down_w: (hidden, intermediate) row-major.
 */
#include "swiglu.h"

#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

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

static void swiglu_core(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit,
    float *gate_up_act,
    float *h_act) {
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
        if (gate_up_act != NULL) {
            gate_up_act[i] = g;
            gate_up_act[intermediate + i] = u;
        }
        float silu = g / (1.0f + expf(-g));
        h[i] = silu * u;
        if (h_act != NULL) {
            h_act[i] = h[i];
        }
    }
    linear(down_w, h, out, hidden, intermediate);
}

void ds4_swiglu_forward(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit) {
    swiglu_core(out, x, hidden, intermediate, gate_up_w, down_w, swiglu_limit, NULL, NULL);
}

void ds4_swiglu_forward_save(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit,
    float *gate_up_act,
    float *h_act) {
    swiglu_core(out, x, hidden, intermediate, gate_up_w, down_w, swiglu_limit, gate_up_act, h_act);
}

void ds4_swiglu_backward(
    float *dx,
    float *d_gate_up_w,
    float *d_down_w,
    const float *dout,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    const float *gate_up_act,
    const float *h_act,
    float swiglu_limit) {
    float dh[256];
    float dg_raw[256];
    float du_raw[256];
    if (intermediate > 256 || hidden > 256) {
        return;
    }
    (void)gate_up_w;
    (void)down_w;
    (void)swiglu_limit;
    memset(dh, 0, (size_t)intermediate * sizeof(float));
    ds4_linear_backward(dx, d_down_w, dout, h_act, hidden, intermediate);
    for (int i = 0; i < intermediate; i++) {
        float g = gate_up_act[i];
        float u = gate_up_act[intermediate + i];
        float silu = g / (1.0f + expf(-g));
        float dh_i = h_act[i];
        du_raw[i] = dh_i * silu;
        float dsilu = dh_i * u;
        float s = 1.0f / (1.0f + expf(-g));
        dg_raw[i] = dsilu * (s + g * s * (1.0f - s));
    }
    float dx_gu[256];
    memset(dx_gu, 0, (size_t)hidden * sizeof(float));
    ds4_linear_backward(dx_gu, d_gate_up_w, dg_raw, x, intermediate, hidden);
    ds4_linear_backward(
        dx,
        d_gate_up_w + (size_t)intermediate * (size_t)hidden,
        du_raw,
        x,
        intermediate,
        hidden);
    for (int i = 0; i < hidden; i++) {
        dx[i] += dx_gu[i];
    }
}
