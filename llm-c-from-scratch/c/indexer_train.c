#include "indexer_train.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_COMP 64
#define DS4_MAX_T 128

static int indexer_rope_use(const DeepSeekV4Config *cfg) {
    int rope_dim = (cfg->index_head_dim < ds4_qk_rope_head_dim(cfg)) ? cfg->index_head_dim : ds4_qk_rope_head_dim(cfg);
    if (rope_dim % 2 != 0) {
        rope_dim -= 1;
    }
    return rope_dim;
}

static size_t indexer_per_win_floats(const DeepSeekV4Config *cfg) {
    const int rate = cfg->compress_rate_csa;
    const int C = cfg->hidden_size;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * HD;
    return (size_t)rate * (size_t)C + (size_t)rate * (size_t)HD2 * 2 + (size_t)rate + (size_t)HD;
}

size_t ds4_csa_indexer_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    const int rate = cfg->compress_rate_csa;
    const int NH = cfg->index_n_heads;
    const int HD = cfg->index_head_dim;
    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;
    if (n_win <= 0) {
        return 0;
    }
    size_t n = (size_t)n_win * indexer_per_win_floats(cfg);
    n += (size_t)DS4_MAX_COMP * (size_t)HD;
    n += (size_t)DS4_MAX_COMP;
    n += (size_t)T * ((size_t)NH * (size_t)HD + (size_t)NH + (size_t)DS4_MAX_COMP);
    return n;
}

static void indexer_compress_forward_win(
    float *idx_comp_k,
    int *idx_end_k,
    const float *hidden,
    int w,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *win_cache) {
    const int C = cfg->hidden_size;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * HD;
    const int rate = cfg->compress_rate_csa;
    const int rope_use = indexer_rope_use(cfg);

    float *h_save = win_cache;
    float *kv = h_save + (size_t)rate * (size_t)C;
    float *gate = kv + (size_t)rate * (size_t)HD2;
    float *weights = gate + (size_t)rate * (size_t)HD2;
    float *pre_norm = weights + (size_t)rate;
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
    float maxg = gate[0];
    for (int i = 1; i < rate * HD2; i++) {
        if (gate[i] > maxg) {
            maxg = gate[i];
        }
    }
    float sum = 0.0f;
    for (int ri = 0; ri < rate; ri++) {
        float sg = 0.0f;
        for (int d = 0; d < HD2; d++) {
            sg += expf(gate[(size_t)ri * (size_t)HD2 + (size_t)d] - maxg);
        }
        weights[ri] = sg;
        sum += sg;
    }
    for (int d = 0; d < HD; d++) {
        idx_comp_k[d] = 0.0f;
    }
    for (int ri = 0; ri < rate; ri++) {
        float wr = (sum > 0.0f) ? (weights[ri] / sum) : (1.0f / (float)rate);
        for (int d = 0; d < HD; d++) {
            idx_comp_k[d] += wr * kv[(size_t)ri * (size_t)HD2 + (size_t)HD + (size_t)d];
        }
    }
    memcpy(pre_norm, idx_comp_k, (size_t)HD * sizeof(float));
    ds4_rmsnorm_forward(idx_comp_k, idx_comp_k, norm_weight, 1, HD, cfg->rms_norm_eps);
    float cos[DS4_MAX_COMP];
    float sin[DS4_MAX_COMP];
    ds4_rope_cos_sin_buffer(cos, sin, 1, rope_use, cfg->compress_rope_theta);
    ds4_apply_partial_rope_vec(idx_comp_k, HD, rope_use, cos, sin);
    *idx_end_k = base + rate - 1;
}

static void indexer_compress_backward_win(
    float *dhidden,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_idx_comp,
    int w,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *norm_weight,
    float *win_cache) {
    const int C = cfg->hidden_size;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * HD;
    const int rate = cfg->compress_rate_csa;
    const int rope_use = indexer_rope_use(cfg);
    const float eps = cfg->rms_norm_eps;
    int base = w * rate;

    float *h_save = win_cache;
    float *kv = h_save + (size_t)rate * (size_t)C;
    float *gate = kv + (size_t)rate * (size_t)HD2;
    float *weights = gate + (size_t)rate * (size_t)HD2;
    const float *pre_norm = weights + (size_t)rate;

    float d_pre[32];
    float cos[8];
    float sin[8];
    ds4_rope_cos_sin_buffer(cos, sin, 1, rope_use, cfg->compress_rope_theta);
    memcpy(d_pre, d_idx_comp, (size_t)HD * sizeof(float));
    ds4_apply_partial_rope_backward_vec_t(d_pre, HD, rope_use, cos, sin);
    float d_agg[32];
    ds4_rmsnorm_backward(d_agg, d_norm, d_pre, pre_norm, pre_norm, norm_weight, 1, HD, eps);

    float d_kv[256];
    float d_weights[32];
    memset(d_kv, 0, (size_t)rate * (size_t)HD2 * sizeof(float));
    memset(d_weights, 0, (size_t)rate * sizeof(float));
    float sum = 0.0f;
    for (int ri = 0; ri < rate; ri++) {
        sum += weights[ri];
    }
    for (int ri = 0; ri < rate; ri++) {
        float wr = (sum > 0.0f) ? (weights[ri] / sum) : (1.0f / (float)rate);
        for (int d = 0; d < HD; d++) {
            d_kv[(size_t)ri * (size_t)HD2 + (size_t)HD + (size_t)d] += wr * d_agg[d];
            d_weights[ri] += d_agg[d] * kv[(size_t)ri * (size_t)HD2 + (size_t)HD + (size_t)d];
        }
    }
    float d_gate[256];
    memset(d_gate, 0, (size_t)rate * (size_t)HD2 * sizeof(float));
    float maxg = gate[0];
    for (int i = 1; i < rate * HD2; i++) {
        if (gate[i] > maxg) {
            maxg = gate[i];
        }
    }
    for (int ri = 0; ri < rate; ri++) {
        float d_p = 0.0f;
        for (int d = 0; d < HD; d++) {
            d_p += d_weights[ri] * kv[(size_t)ri * (size_t)HD2 + (size_t)HD + (size_t)d];
        }
        if (sum > 0.0f) {
            for (int j = 0; j < rate; j++) {
                float pj = weights[j] / sum;
                d_weights[j] -= d_p * pj;
            }
            d_weights[ri] += d_p / sum;
        }
        float d_sg = d_weights[ri];
        for (int d = 0; d < HD2; d++) {
            float e = expf(gate[(size_t)ri * (size_t)HD2 + (size_t)d] - maxg);
            d_gate[(size_t)ri * (size_t)HD2 + (size_t)d] += d_sg * e;
        }
    }
    float dsum = 0.0f;
    for (int i = 0; i < rate * HD2; i++) {
        dsum += d_gate[i] * expf(gate[i] - maxg);
    }
    for (int i = 0; i < rate * HD2; i++) {
        float e = expf(gate[i] - maxg);
        d_gate[i] = d_gate[i] * e - e * dsum;
    }
    for (int ri = 0; ri < rate; ri++) {
        for (int d = 0; d < HD2; d++) {
            d_pos_bias[(size_t)ri * (size_t)HD2 + (size_t)d] += d_gate[(size_t)ri * (size_t)HD2 + (size_t)d];
        }
        float dx_t[64];
        float dg_t[64];
        ds4_linear_backward(dx_t, d_w_kv, d_kv + (size_t)ri * (size_t)HD2, h_save + (size_t)ri * (size_t)C, w_kv, HD2, C);
        ds4_linear_backward(dg_t, d_w_gate, d_gate + (size_t)ri * (size_t)HD2, h_save + (size_t)ri * (size_t)C, w_gate, HD2, C);
        for (int c = 0; c < C; c++) {
            dhidden[(size_t)(base + ri) * (size_t)C + (size_t)c] += dx_t[c] + dg_t[c];
        }
    }
}

int ds4_csa_indexer_forward_train(
    int *sparse_mask,
    const float *hidden,
    const float *q_residual,
    int T,
    int n_comp,
    const int *comp_end_positions,
    const float *compressed,
    const DeepSeekV4Config *cfg,
    const float *wq_b,
    const float *w_weights,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache,
    int *out_n_idx) {
    const int C = cfg->hidden_size;
    const int NH = cfg->index_n_heads;
    const int HD = cfg->index_head_dim;
    const int topk = cfg->index_topk;
    const int r = cfg->q_lora_rank;
    const int rate = cfg->compress_rate_csa;
    const int rope_use = indexer_rope_use(cfg);
    const int rope_half = rope_use / 2;

    (void)n_comp;
    (void)comp_end_positions;
    (void)compressed;

    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;
    if (n_win <= 0 || n_win > DS4_MAX_COMP) {
        *out_n_idx = 0;
        return 0;
    }

    size_t per_win = indexer_per_win_floats(cfg);
    float *idx_comp_all = cache + (size_t)n_win * per_win;
    float *idx_end_f = idx_comp_all + (size_t)DS4_MAX_COMP * (size_t)HD;
    float *t_cache = idx_end_f + (size_t)DS4_MAX_COMP;

    int n_idx = 0;
    for (int w = 0; w < n_win; w++) {
        int end_i = 0;
        indexer_compress_forward_win(
            idx_comp_all + (size_t)w * (size_t)HD,
            &end_i,
            hidden,
            w,
            cfg,
            w_kv,
            w_gate,
            pos_bias,
            norm_weight,
            cache + (size_t)w * per_win);
        idx_end_f[w] = (float)end_i;
        n_idx++;
    }
    *out_n_idx = n_idx;

    float cos_q[DS4_MAX_T * 32];
    float sin_q[DS4_MAX_T * 32];
    ds4_rope_cos_sin_buffer(cos_q, sin_q, T, rope_use, cfg->compress_rope_theta);

    for (int t = 0; t < T; t++) {
        float *qflat = t_cache + ((size_t)t * ((size_t)NH * (size_t)HD + (size_t)NH + (size_t)DS4_MAX_COMP));
        float *wf = qflat + (size_t)NH * (size_t)HD;
        float *scores = wf + (size_t)NH;

        const float *qr = q_residual + (size_t)t * (size_t)r;
        ds4_linear(wq_b, qr, qflat, NH * HD, r);
        for (int h = 0; h < NH; h++) {
            ds4_apply_partial_rope_vec_t(
                qflat + (size_t)h * (size_t)HD,
                HD,
                rope_use,
                cos_q + (size_t)t * (size_t)rope_half,
                sin_q + (size_t)t * (size_t)rope_half);
        }
        ds4_linear(w_weights, hidden + (size_t)t * (size_t)C, wf, NH, C);
        for (int h = 0; h < NH; h++) {
            wf[h] *= 1.0f / sqrtf((float)NH);
        }
        for (int k = 0; k < n_idx; k++) {
            float s = 0.0f;
            const float *ck = idx_comp_all + (size_t)k * (size_t)HD;
            int end_k = (int)idx_end_f[k];
            for (int h = 0; h < NH; h++) {
                float dot = ds4_dot(qflat + (size_t)h * (size_t)HD, ck, HD);
                float relu_dot = 0.0f;
                if (dot > 0.0f) {
                    relu_dot = dot * (1.0f / sqrtf((float)HD));
                }
                s += relu_dot * wf[h];
            }
            scores[k] = s;
            if (end_k > t) {
                scores[k] = -1e30f;
            }
        }
        int ktop = (topk < n_idx) ? topk : n_idx;
        for (int k = 0; k < n_comp; k++) {
            sparse_mask[(size_t)t * (size_t)n_comp + (size_t)k] = 0;
        }
        for (int pick = 0; pick < ktop; pick++) {
            int best = -1;
            float bestv = -1e30f;
            for (int k = 0; k < n_idx; k++) {
                float v = scores[k];
                if (v > bestv) {
                    bestv = v;
                    best = k;
                }
            }
            if (best >= 0 && bestv > -1e20f) {
                if (best < n_comp) {
                    sparse_mask[(size_t)t * (size_t)n_comp + (size_t)best] = 1;
                }
                scores[best] = -1e30f;
            }
        }
    }
    return n_idx;
}

void ds4_csa_indexer_backward(
    float *dhidden,
    float *d_q_residual,
    float *d_wq_b,
    float *d_w_weights,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_scores,
    const float *hidden,
    const float *q_residual,
    int T,
    int n_comp,
    const DeepSeekV4Config *cfg,
    const float *wq_b,
    const float *w_weights,
    const float *w_kv,
    const float *w_gate,
    const float *norm_weight,
    float *cache) {
    const int C = cfg->hidden_size;
    const int NH = cfg->index_n_heads;
    const int HD = cfg->index_head_dim;
    const int r = cfg->q_lora_rank;
    const int rate = cfg->compress_rate_csa;
    const int rope_use = indexer_rope_use(cfg);
    const int rope_half = rope_use / 2;
    const float inv_sqrt_hd = 1.0f / sqrtf((float)HD);
    const float inv_sqrt_nh = 1.0f / sqrtf((float)NH);

    (void)n_comp;
    (void)hidden;

    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;
    if (n_win <= 0) {
        return;
    }

    size_t per_win = indexer_per_win_floats(cfg);
    float *idx_comp_all = cache + (size_t)n_win * per_win;
    float *idx_end_f = idx_comp_all + (size_t)DS4_MAX_COMP * (size_t)HD;
    float *t_cache = idx_end_f + (size_t)DS4_MAX_COMP;

    float d_idx_comp[DS4_MAX_COMP * 32];
    memset(d_idx_comp, 0, (size_t)n_win * (size_t)HD * sizeof(float));

    float cos_q[DS4_MAX_T * 32];
    float sin_q[DS4_MAX_T * 32];
    ds4_rope_cos_sin_buffer(cos_q, sin_q, T, rope_use, cfg->compress_rope_theta);

    for (int t = 0; t < T; t++) {
        const float *qflat = t_cache + ((size_t)t * ((size_t)NH * (size_t)HD + (size_t)NH + (size_t)DS4_MAX_COMP));
        const float *wf = qflat + (size_t)NH * (size_t)HD;
        float dqflat[64];
        float dwf[16];
        memset(dqflat, 0, (size_t)NH * (size_t)HD * sizeof(float));
        memset(dwf, 0, (size_t)NH * sizeof(float));

        for (int k = 0; k < n_win; k++) {
            float ds = d_scores[(size_t)t * (size_t)DS4_MAX_COMP + (size_t)k];
            if (ds == 0.0f) {
                continue;
            }
            int end_k = (int)idx_end_f[k];
            if (end_k > t) {
                continue;
            }
            const float *ck = idx_comp_all + (size_t)k * (size_t)HD;
            for (int h = 0; h < NH; h++) {
                float dot = ds4_dot(qflat + (size_t)h * (size_t)HD, ck, HD);
                float relu_dot = 0.0f;
                if (dot > 0.0f) {
                    relu_dot = dot * inv_sqrt_hd;
                }
                float d_relu_dot = wf[h] * ds;
                if (dot > 0.0f) {
                    float d_dot = d_relu_dot * inv_sqrt_hd;
                    for (int d = 0; d < HD; d++) {
                        dqflat[(size_t)h * (size_t)HD + (size_t)d] += d_dot * ck[d];
                        d_idx_comp[(size_t)k * (size_t)HD + (size_t)d] += d_dot * qflat[(size_t)h * (size_t)HD + (size_t)d];
                    }
                }
                dwf[h] += relu_dot * ds;
            }
        }

        float d_wf_raw[16];
        for (int h = 0; h < NH; h++) {
            d_wf_raw[h] = dwf[h] * inv_sqrt_nh;
        }
        ds4_linear_backward(dhidden + (size_t)t * (size_t)C, d_w_weights, d_wf_raw, hidden + (size_t)t * (size_t)C, w_weights, NH, C);

        float dqr[32];
        memset(dqr, 0, (size_t)r * sizeof(float));
        for (int h = 0; h < NH; h++) {
            ds4_apply_partial_rope_backward_vec_t(
                dqflat + (size_t)h * (size_t)HD,
                HD,
                rope_use,
                cos_q + (size_t)t * (size_t)rope_half,
                sin_q + (size_t)t * (size_t)rope_half);
        }
        ds4_linear_backward(dqr, d_wq_b, dqflat, q_residual + (size_t)t * (size_t)r, wq_b, NH * HD, r);
        for (int i = 0; i < r; i++) {
            d_q_residual[(size_t)t * (size_t)r + (size_t)i] += dqr[i];
        }
    }

    for (int w = 0; w < n_win; w++) {
        indexer_compress_backward_win(
            dhidden,
            d_w_kv,
            d_w_gate,
            d_pos_bias,
            d_norm,
            d_idx_comp + (size_t)w * (size_t)HD,
            w,
            cfg,
            w_kv,
            w_gate,
            norm_weight,
            cache + (size_t)w * per_win);
    }
}
