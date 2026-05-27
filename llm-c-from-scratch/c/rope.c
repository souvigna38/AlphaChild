#include "rope.h"

#include <math.h>
#include <stddef.h>

void ds4_rope_cos_sin_buffer(float *cos, float *sin, int T, int rope_dim, float theta) {
    int half = rope_dim / 2;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < half; i++) {
            float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
            float freq = (float)t * inv_freq;
            cos[(size_t)t * (size_t)half + (size_t)i] = cosf(freq);
            sin[(size_t)t * (size_t)half + (size_t)i] = sinf(freq);
        }
    }
}

void ds4_apply_partial_rope_vec(float *hd, int head_dim, int rope_dim, const float *cos_t, const float *sin_t) {
    if (rope_dim <= 0) {
        return;
    }
    int nope = head_dim - rope_dim;
    int half = rope_dim / 2;
    float tmp[64];
    if (rope_dim > 64) {
        return;
    }
    for (int i = 0; i < rope_dim; i++) {
        tmp[i] = hd[nope + i];
    }
    for (int i = 0; i < half; i++) {
        float even = tmp[2 * i];
        float odd = tmp[2 * i + 1];
        float c = cos_t[i];
        float s = sin_t[i];
        hd[nope + 2 * i] = even * c - odd * s;
        hd[nope + 2 * i + 1] = odd * c + even * s;
    }
}

void ds4_apply_partial_rope_vec_t(float *hd, int head_dim, int rope_dim, const float *cos_row, const float *sin_row) {
    ds4_apply_partial_rope_vec(hd, head_dim, rope_dim, cos_row, sin_row);
}

void ds4_apply_partial_rope_backward_vec_t(float *dhd, int head_dim, int rope_dim, const float *cos_row, const float *sin_row) {
    if (rope_dim <= 0) {
        return;
    }
    int nope = head_dim - rope_dim;
    int half = rope_dim / 2;
    float deven[32];
    float dodd[32];
    if (rope_dim > 64) {
        return;
    }
    for (int i = 0; i < half; i++) {
        float c = cos_row[i];
        float s = sin_row[i];
        float o0 = dhd[nope + 2 * i];
        float o1 = dhd[nope + 2 * i + 1];
        deven[i] = c * o0 + s * o1;
        dodd[i] = -s * o0 + c * o1;
    }
    for (int i = 0; i < half; i++) {
        dhd[nope + 2 * i] = deven[i];
        dhd[nope + 2 * i + 1] = dodd[i];
    }
}

void ds4_apply_partial_rope_backward_vec_t_neg_sin(
    float *dhd,
    int head_dim,
    int rope_dim,
    const float *cos_row,
    const float *sin_row) {
    if (rope_dim <= 0) {
        return;
    }
    int nope = head_dim - rope_dim;
    int half = rope_dim / 2;
    float deven[32];
    float dodd[32];
    if (rope_dim > 64) {
        return;
    }
    for (int i = 0; i < half; i++) {
        float c = cos_row[i];
        float s = -sin_row[i];
        float o0 = dhd[nope + 2 * i];
        float o1 = dhd[nope + 2 * i + 1];
        deven[i] = c * o0 + s * o1;
        dodd[i] = -s * o0 + c * o1;
    }
    for (int i = 0; i < half; i++) {
        dhd[nope + 2 * i] = deven[i];
        dhd[nope + 2 * i + 1] = dodd[i];
    }
}

void ds4_apply_partial_rope_vec_t_neg_sin(float *hd, int head_dim, int rope_dim, const float *cos_row, const float *sin_row) {
    if (rope_dim <= 0) {
        return;
    }
    int nope = head_dim - rope_dim;
    int half = rope_dim / 2;
    float tmp[64];
    if (rope_dim > 64) {
        return;
    }
    for (int i = 0; i < rope_dim; i++) {
        tmp[i] = hd[nope + i];
    }
    for (int i = 0; i < half; i++) {
        float even = tmp[2 * i];
        float odd = tmp[2 * i + 1];
        float c = cos_row[i];
        float s = -sin_row[i];
        hd[nope + 2 * i] = even * c - odd * s;
        hd[nope + 2 * i + 1] = odd * c + even * s;
    }
}
