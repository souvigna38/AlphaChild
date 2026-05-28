#include "hca_compressor_train.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_COMP 64

size_t ds4_hca_compress_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_hca;
    const int n_win = (T / rate);
    if (n_win <= 0) {
        return 0;
    }
    size_t per_win = (size_t)rate * (size_t)C * 2 + (size_t)rate * (size_t)D * 2 + (size_t)rate + (size_t)D * 3;
    return (size_t)n_win * per_win;
}

int ds4_hca_compress_forward_train(
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
    const int rate = cfg->compress_rate_hca;
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;

    if (n_win <= 0 || n_win > DS4_MAX_COMP) {
        *out_n = 0;
        return 0;
    }

    size_t per_win = (size_t)rate * (size_t)C * 2 + (size_t)rate * (size_t)D * 2 + (size_t)rate + (size_t)D * 3;
    float *p = cache;

    for (int w = 0; w < n_win; w++) {
        float *win_cache = p + (size_t)w * per_win;
        float *h_save = win_cache;
        float *kv = h_save + (size_t)rate * (size_t)C;
        float *gate = kv + (size_t)rate * (size_t)D;
        float *weights = gate + (size_t)rate * (size_t)D;
        float *pre_norm = weights + (size_t)rate;
        int base = w * rate;

        for (int r = 0; r < rate; r++) {
            const float *h = hidden + (size_t)(base + r) * (size_t)C;
            memcpy(h_save + (size_t)r * (size_t)C, h, (size_t)C * sizeof(float));
            ds4_linear(w_kv, h, kv + (size_t)r * (size_t)D, D, C);
            ds4_linear(w_gate, h, gate + (size_t)r * (size_t)D, D, C);
            for (int d = 0; d < D; d++) {
                gate[(size_t)r * (size_t)D + (size_t)d] += pos_bias[(size_t)r * (size_t)D + (size_t)d];
            }
        }
        float maxg = gate[0];
        for (int i = 1; i < rate * D; i++) {
            if (gate[i] > maxg) {
                maxg = gate[i];
            }
        }
        float sum = 0.0f;
        for (int r = 0; r < rate; r++) {
            float sg = 0.0f;
            for (int d = 0; d < D; d++) {
                sg += expf(gate[(size_t)r * (size_t)D + (size_t)d] - maxg);
            }
            weights[r] = sg;
            sum += sg;
        }
        float *out = comp_kv + (size_t)w * (size_t)D;
        for (int d = 0; d < D; d++) {
            out[d] = 0.0f;
        }
        for (int r = 0; r < rate; r++) {
            float wr = (sum > 0.0f) ? (weights[r] / sum) : (1.0f / (float)rate);
            for (int d = 0; d < D; d++) {
                out[d] += wr * kv[(size_t)r * (size_t)D + (size_t)d];
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

void ds4_hca_compress_backward(
    float *dhidden,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_comp_kv,
    const float *hidden,
    int T,
    int n_comp,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache) {
    (void)pos_bias;
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_hca;
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const float eps = cfg->rms_norm_eps;
    (void)hidden;
    (void)T;

    if (n_comp <= 0) {
        return;
    }

    size_t per_win = (size_t)rate * (size_t)C * 2 + (size_t)rate * (size_t)D * 2 + (size_t)rate + (size_t)D * 3;

    for (int w = 0; w < n_comp; w++) {
        float *win_cache = cache + (size_t)w * per_win;
        const float *h_save = win_cache;
        float *kv = win_cache + (size_t)rate * (size_t)C;
        float *gate = kv + (size_t)rate * (size_t)D;
        const float *weights = gate + (size_t)rate * (size_t)D;
        const float *pre_norm = weights + (size_t)rate;
        const float *dout = d_comp_kv + (size_t)w * (size_t)D;

        float d_pre[32];
        float cos[8];
        float sin[8];
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_dim, cfg->compress_rope_theta);
        memcpy(d_pre, dout, (size_t)D * sizeof(float));
        ds4_apply_partial_rope_backward_vec_t(d_pre, D, rope_dim, cos, sin);
        float d_agg[32];
        ds4_rmsnorm_backward(d_agg, d_norm, d_pre, pre_norm, pre_norm, norm_weight, 1, D, eps);

        float d_kv[128];
        float d_weights[32];
        memset(d_kv, 0, (size_t)rate * (size_t)D * sizeof(float));
        float sum = 0.0f;
        for (int r = 0; r < rate; r++) {
            sum += weights[r];
        }
        for (int r = 0; r < rate; r++) {
            float wr = (sum > 0.0f) ? (weights[r] / sum) : (1.0f / (float)rate);
            for (int d = 0; d < D; d++) {
                d_kv[(size_t)r * (size_t)D + (size_t)d] += wr * d_agg[d];
                d_weights[r] += d_agg[d] * kv[(size_t)r * (size_t)D + (size_t)d];
            }
        }
        float d_gate[128];
        memset(d_gate, 0, (size_t)rate * (size_t)D * sizeof(float));
        float maxg = gate[0];
        for (int i = 1; i < rate * D; i++) {
            if (gate[i] > maxg) {
                maxg = gate[i];
            }
        }
        for (int r = 0; r < rate; r++) {
            float d_p = 0.0f;
            for (int d = 0; d < D; d++) {
                d_p += d_weights[r] * kv[(size_t)r * (size_t)D + (size_t)d];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < rate; j++) {
                    float pj = weights[j] / sum;
                    d_weights[j] -= d_p * pj;
                }
                d_weights[r] += d_p / sum;
            }
            float d_sg = d_weights[r];
            for (int d = 0; d < D; d++) {
                float g = gate[(size_t)r * (size_t)D + (size_t)d];
                float e = expf(g - maxg);
                d_gate[(size_t)r * (size_t)D + (size_t)d] += d_sg * e;
            }
        }
        float dsum = 0.0f;
        for (int i = 0; i < rate * D; i++) {
            dsum += d_gate[i] * expf(gate[i] - maxg);
        }
        for (int i = 0; i < rate * D; i++) {
            float e = expf(gate[i] - maxg);
            d_gate[i] = d_gate[i] * e - e * dsum;
        }
        for (int r = 0; r < rate; r++) {
            for (int d = 0; d < D; d++) {
                d_pos_bias[(size_t)r * (size_t)D + (size_t)d] += d_gate[(size_t)r * (size_t)D + (size_t)d];
            }
            float dx_t[64];
            float dg_t[64];
            ds4_linear_backward(dx_t, d_w_kv, d_kv + (size_t)r * (size_t)D, h_save + (size_t)r * (size_t)C, w_kv, D, C);
            ds4_linear_backward(dg_t, d_w_gate, d_gate + (size_t)r * (size_t)D, h_save + (size_t)r * (size_t)C, w_gate, D, C);
            int base = w * rate + r;
            for (int c = 0; c < C; c++) {
                dhidden[(size_t)base * (size_t)C + (size_t)c] += dx_t[c] + dg_t[c];
            }
        }
    }
}
