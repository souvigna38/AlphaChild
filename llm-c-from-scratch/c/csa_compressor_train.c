#include "csa_compressor_train.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_COMP 64

size_t ds4_csa_compress_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_csa;
    const int HD2 = 2 * D;
    const int n_win = (T / rate);
    if (n_win <= 0) {
        return 0;
    }
    size_t per_win = (size_t)rate * (size_t)C + (size_t)rate * (size_t)HD2 * 2 + (size_t)(2 * rate) * (size_t)D * 2 + (size_t)(2 * rate) + (size_t)D;
    return (size_t)n_win * per_win;
}

int ds4_csa_compress_forward_train(
    float *comp_kv,
    int *end_positions,
    const float *hidden,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache,
    int *out_n) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_csa;
    const int HD2 = 2 * D;
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;

    if (n_win <= 0 || n_win > DS4_MAX_COMP) {
        *out_n = 0;
        return 0;
    }

    size_t per_win = (size_t)rate * (size_t)C + (size_t)rate * (size_t)HD2 * 2 + (size_t)(2 * rate) * (size_t)D * 2 + (size_t)(2 * rate) + (size_t)D;

    for (int w = 0; w < n_win; w++) {
        float *wc = cache + (size_t)w * per_win;
        float *h_save = wc;
        float *kv = h_save + (size_t)rate * (size_t)C;
        float *gate = kv + (size_t)rate * (size_t)HD2;
        float *slot = gate + (size_t)rate * (size_t)HD2;
        float *slotg = slot + (size_t)(2 * rate) * (size_t)D;
        float *weights = slotg + (size_t)(2 * rate) * (size_t)D;
        float *pre_norm = weights + (size_t)(2 * rate);
        int base = w * rate;

        for (int ri = 0; ri < rate; ri++) {
            const float *h = hidden + (size_t)(base + ri) * (size_t)C;
            memcpy(h_save + (size_t)ri * (size_t)C, h, (size_t)C * sizeof(float));
            ds4_linear(w_kv, h, kv + (size_t)ri * (size_t)HD2, HD2, C);
            ds4_linear(w_gate, h, gate + (size_t)ri * (size_t)HD2, HD2, C);
            for (int d = 0; d < HD2; d++) {
                gate[(size_t)ri * (size_t)HD2 + (size_t)d] += pos_bias[(size_t)ri * (size_t)HD2 + (size_t)d];
            }
        }
        for (int s = 0; s < 2 * rate; s++) {
            for (int d = 0; d < D; d++) {
                slot[(size_t)s * (size_t)D + (size_t)d] = 0.0f;
                slotg[(size_t)s * (size_t)D + (size_t)d] = -1e30f;
            }
        }
        for (int ri = 0; ri < rate; ri++) {
            for (int d = 0; d < D; d++) {
                slot[(size_t)(rate + ri) * (size_t)D + (size_t)d] = kv[(size_t)ri * (size_t)HD2 + (size_t)D + (size_t)d];
                slotg[(size_t)(rate + ri) * (size_t)D + (size_t)d] = gate[(size_t)ri * (size_t)HD2 + (size_t)D + (size_t)d];
            }
        }
        if (w > 0) {
            for (int ri = 0; ri < rate; ri++) {
                for (int d = 0; d < D; d++) {
                    slot[(size_t)ri * (size_t)D + (size_t)d] = kv[(size_t)ri * (size_t)HD2 + (size_t)d];
                    slotg[(size_t)ri * (size_t)D + (size_t)d] = gate[(size_t)ri * (size_t)HD2 + (size_t)d];
                }
            }
        }
        float maxg = slotg[0];
        for (int i = 1; i < 2 * rate * D; i++) {
            if (slotg[i] > maxg) {
                maxg = slotg[i];
            }
        }
        float sum = 0.0f;
        for (int s = 0; s < 2 * rate; s++) {
            float sg = 0.0f;
            for (int d = 0; d < D; d++) {
                sg += expf(slotg[(size_t)s * (size_t)D + (size_t)d] - maxg);
            }
            weights[s] = sg;
            sum += sg;
        }
        float *out = comp_kv + (size_t)w * (size_t)D;
        for (int d = 0; d < D; d++) {
            out[d] = 0.0f;
        }
        for (int s = 0; s < 2 * rate; s++) {
            float ws = (sum > 0.0f) ? (weights[s] / sum) : (1.0f / (float)(2 * rate));
            for (int d = 0; d < D; d++) {
                out[d] += ws * slot[(size_t)s * (size_t)D + (size_t)d];
            }
        }
        memcpy(pre_norm, out, (size_t)D * sizeof(float));
        ds4_rmsnorm_forward(out, out, norm_weight, 1, D, cfg->rms_norm_eps);
        float cos[8];
        float sin[8];
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_dim, cfg->compress_rope_theta);
        ds4_apply_partial_rope_vec(out, D, rope_dim, cos, sin);
        end_positions[w] = base + rate - 1;
    }
    *out_n = n_win;
    return n_win;
}

void ds4_csa_compress_backward(
    float *dhidden,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_comp_kv,
    int T,
    int n_comp,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *norm_weight,
    float *cache) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_csa;
    const int HD2 = 2 * D;
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const float eps = cfg->rms_norm_eps;
    (void)T;

    if (n_comp <= 0) {
        return;
    }

    size_t per_win = (size_t)rate * (size_t)C + (size_t)rate * (size_t)HD2 * 2 + (size_t)(2 * rate) * (size_t)D * 2 + (size_t)(2 * rate) + (size_t)D;

    for (int w = 0; w < n_comp; w++) {
        float *wc = cache + (size_t)w * per_win;
        const float *h_save = wc;
        const float *kv = h_save + (size_t)rate * (size_t)C;
        const float *gate = kv + (size_t)rate * (size_t)HD2;
        const float *slot = gate + (size_t)rate * (size_t)HD2;
        const float *slotg = slot + (size_t)(2 * rate) * (size_t)D;
        const float *weights = slotg + (size_t)(2 * rate) * (size_t)D;
        const float *pre_norm = weights + (size_t)(2 * rate);
        const float *dout = d_comp_kv + (size_t)w * (size_t)D;

        float d_pre[32];
        float cos[8];
        float sin[8];
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_dim, cfg->compress_rope_theta);
        memcpy(d_pre, dout, (size_t)D * sizeof(float));
        ds4_apply_partial_rope_backward_vec_t(d_pre, D, rope_dim, cos, sin);
        float d_slot_agg[32];
        ds4_rmsnorm_backward(d_slot_agg, d_norm, d_pre, pre_norm, pre_norm, norm_weight, 1, D, eps);

        float d_slot[128];
        float d_weights[64];
        memset(d_slot, 0, (size_t)(2 * rate) * (size_t)D * sizeof(float));
        memset(d_weights, 0, (size_t)(2 * rate) * sizeof(float));
        float sumw = 0.0f;
        for (int s = 0; s < 2 * rate; s++) {
            sumw += weights[s];
        }
        for (int s = 0; s < 2 * rate; s++) {
            float ws = (sumw > 0.0f) ? (weights[s] / sumw) : (1.0f / (float)(2 * rate));
            for (int d = 0; d < D; d++) {
                d_slot[(size_t)s * (size_t)D + (size_t)d] += ws * d_slot_agg[d];
                d_weights[s] += d_slot_agg[d] * slot[(size_t)s * (size_t)D + (size_t)d];
            }
        }
        float d_slotg[128];
        memset(d_slotg, 0, (size_t)(2 * rate) * (size_t)D * sizeof(float));
        float maxg = slotg[0];
        for (int i = 1; i < 2 * rate * D; i++) {
            if (slotg[i] > maxg) {
                maxg = slotg[i];
            }
        }
        for (int s = 0; s < 2 * rate; s++) {
            float d_ws = 0.0f;
            for (int d = 0; d < D; d++) {
                d_ws += d_weights[s] * slot[(size_t)s * (size_t)D + (size_t)d];
            }
            if (sumw > 0.0f) {
                for (int j = 0; j < 2 * rate; j++) {
                    float pj = weights[j] / sumw;
                    d_weights[j] -= d_ws * pj;
                }
                d_weights[s] += d_ws / sumw;
            }
            float d_sg = d_weights[s];
            for (int d = 0; d < D; d++) {
                float e = expf(slotg[(size_t)s * (size_t)D + (size_t)d] - maxg);
                d_slotg[(size_t)s * (size_t)D + (size_t)d] += d_sg * e;
            }
        }
        float dsum = 0.0f;
        for (int i = 0; i < 2 * rate * D; i++) {
            dsum += d_slotg[i] * expf(slotg[i] - maxg);
        }
        for (int i = 0; i < 2 * rate * D; i++) {
            float e = expf(slotg[i] - maxg);
            d_slotg[i] = d_slotg[i] * e - e * dsum;
        }

        float d_kv[256];
        float d_gate[256];
        memset(d_kv, 0, (size_t)rate * (size_t)HD2 * sizeof(float));
        memset(d_gate, 0, (size_t)rate * (size_t)HD2 * sizeof(float));
        for (int ri = 0; ri < rate; ri++) {
            for (int d = 0; d < D; d++) {
                d_kv[(size_t)ri * (size_t)HD2 + (size_t)D + (size_t)d] += d_slot[(size_t)(rate + ri) * (size_t)D + (size_t)d];
                d_gate[(size_t)ri * (size_t)HD2 + (size_t)D + (size_t)d] += d_slotg[(size_t)(rate + ri) * (size_t)D + (size_t)d];
                if (w > 0) {
                    d_kv[(size_t)ri * (size_t)HD2 + (size_t)d] += d_slot[(size_t)ri * (size_t)D + (size_t)d];
                    d_gate[(size_t)ri * (size_t)HD2 + (size_t)d] += d_slotg[(size_t)ri * (size_t)D + (size_t)d];
                }
            }
        }
        for (int ri = 0; ri < rate; ri++) {
            for (int d = 0; d < HD2; d++) {
                d_pos_bias[(size_t)ri * (size_t)HD2 + (size_t)d] += d_gate[(size_t)ri * (size_t)HD2 + (size_t)d];
            }
            float dx_t[64];
            float dg_t[64];
            ds4_linear_backward(dx_t, d_w_kv, d_kv + (size_t)ri * (size_t)HD2, h_save + (size_t)ri * (size_t)C, w_kv, HD2, C);
            ds4_linear_backward(dg_t, d_w_gate, d_gate + (size_t)ri * (size_t)HD2, h_save + (size_t)ri * (size_t)C, w_gate, HD2, C);
            int base = w * rate + ri;
            for (int c = 0; c < C; c++) {
                dhidden[(size_t)base * (size_t)C + (size_t)c] += dx_t[c] + dg_t[c];
            }
        }
    }
}
