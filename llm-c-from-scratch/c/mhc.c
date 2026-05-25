#include "mhc.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void unweighted_rmsnorm(float *out, const float *x, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

void ds4_hyper_connection_forward(
    float *post,
    float *comb,
    float *collapsed,
    const float *streams,
    const DeepSeekV4Config *cfg,
    const float *fn_weight,
    const float *fn_base,
    const float *scale) {
    const int hc = cfg->hc_mult;
    const int C = cfg->hidden_size;
    const int mix = (2 + hc) * hc;
    const float eps = cfg->hc_eps;
    float flat[512];
    float mixv[128];
    float pre[16];
    if ((size_t)hc * (size_t)C > 512 || mix > 128 || hc > 16) {
        return;
    }

    for (int i = 0; i < hc; i++) {
        memcpy(flat + (size_t)i * (size_t)C, streams + (size_t)i * (size_t)C, (size_t)C * sizeof(float));
    }
    unweighted_rmsnorm(flat, flat, hc * C, cfg->rms_norm_eps);

    for (int m = 0; m < mix; m++) {
        float s = fn_base[m];
        const float *row = fn_weight + (size_t)m * (size_t)(hc * C);
        for (int i = 0; i < hc * C; i++) {
            s += row[i] * flat[i];
        }
        mixv[m] = s;
    }

    for (int i = 0; i < hc; i++) {
        pre[i] = sigmoidf(mixv[i] * scale[0] + fn_base[i]) + eps;
        post[i] = sigmoidf(mixv[hc + i] * scale[1] + fn_base[hc + i]) + eps;
    }

    int off = 2 * hc;
    for (int i = 0; i < hc; i++) {
        for (int j = 0; j < hc; j++) {
            comb[(size_t)i * (size_t)hc + (size_t)j] = sigmoidf(mixv[off + i * hc + j] * scale[2] + fn_base[off + i * hc + j]) + eps;
        }
    }
    for (int it = 0; it < cfg->hc_sinkhorn_iters; it++) {
        for (int i = 0; i < hc; i++) {
            float rs = 0.0f;
            for (int j = 0; j < hc; j++) {
                rs += comb[(size_t)i * (size_t)hc + (size_t)j];
            }
            rs += eps;
            for (int j = 0; j < hc; j++) {
                comb[(size_t)i * (size_t)hc + (size_t)j] /= rs;
            }
        }
        for (int j = 0; j < hc; j++) {
            float cs = 0.0f;
            for (int i = 0; i < hc; i++) {
                cs += comb[(size_t)i * (size_t)hc + (size_t)j];
            }
            cs += eps;
            for (int i = 0; i < hc; i++) {
                comb[(size_t)i * (size_t)hc + (size_t)j] /= cs;
            }
        }
    }

    for (int c = 0; c < C; c++) {
        collapsed[c] = 0.0f;
        for (int i = 0; i < hc; i++) {
            collapsed[c] += pre[i] * streams[(size_t)i * (size_t)C + (size_t)c];
        }
    }
}

void ds4_hc_stream_update(
    float *streams_out,
    const float *streams_in,
    const float *post,
    const float *comb,
    const float *branch,
    int hc_mult,
    int hidden) {
    for (int i = 0; i < hc_mult; i++) {
        for (int c = 0; c < hidden; c++) {
            streams_out[(size_t)i * (size_t)hidden + (size_t)c] = post[i] * branch[c];
        }
    }
    for (int i = 0; i < hc_mult; i++) {
        for (int j = 0; j < hc_mult; j++) {
            float w = comb[(size_t)i * (size_t)hc_mult + (size_t)j];
            for (int c = 0; c < hidden; c++) {
                streams_out[(size_t)i * (size_t)hidden + (size_t)c] +=
                    w * streams_in[(size_t)j * (size_t)hidden + (size_t)c];
            }
        }
    }
}

void ds4_hyper_head_forward(
    float *out,
    const float *streams,
    const DeepSeekV4Config *cfg,
    const float *fn_weight,
    const float *fn_base,
    const float *scale) {
    const int hc = cfg->hc_mult;
    const int C = cfg->hidden_size;
    float flat[512];
    float mixv[16];
    if ((size_t)hc * (size_t)C > 512 || hc > 16) {
        return;
    }
    for (int i = 0; i < hc; i++) {
        memcpy(flat + (size_t)i * (size_t)C, streams + (size_t)i * (size_t)C, (size_t)C * sizeof(float));
    }
    unweighted_rmsnorm(flat, flat, hc * C, cfg->rms_norm_eps);
    for (int i = 0; i < hc; i++) {
        float s = fn_base[i];
        const float *row = fn_weight + (size_t)i * (size_t)(hc * C);
        for (int j = 0; j < hc * C; j++) {
            s += row[j] * flat[j];
        }
        mixv[i] = sigmoidf(s * scale[0] + fn_base[i]) + cfg->hc_eps;
    }
    for (int c = 0; c < C; c++) {
        out[c] = 0.0f;
        for (int i = 0; i < hc; i++) {
            out[c] += mixv[i] * streams[(size_t)i * (size_t)C + (size_t)c];
        }
    }
}
