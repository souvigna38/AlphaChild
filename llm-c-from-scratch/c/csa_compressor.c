#include "csa_compressor.h"

#include "indexer.h"
#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>

#define DS4_MAX_COMP 64

int ds4_csa_compress_forward(
    float *comp_kv,
    int *end_positions,
    int *sparse_mask,
    const float *hidden,
    const float *q_residual,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    const float *idx_wq_b,
    const float *idx_w_weights,
    const float *idx_w_kv,
    const float *idx_w_gate,
    const float *idx_pos_bias,
    const float *idx_norm_weight,
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

    float kv[512];
    float gate[512];
    float slot[512];
    float slotg[512];
    float weights[128];
    float cos[DS4_MAX_COMP];
    float sin[DS4_MAX_COMP];

    for (int w = 0; w < n_win; w++) {
        int base = w * rate;
        for (int ri = 0; ri < rate; ri++) {
            const float *h = hidden + (size_t)(base + ri) * (size_t)C;
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
        ds4_rmsnorm_forward(out, out, norm_weight, 1, D, cfg->rms_norm_eps);
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_dim, cfg->compress_rope_theta);
        ds4_apply_partial_rope_vec(out, D, rope_dim, cos, sin);
        end_positions[w] = base + rate - 1;
    }
    *out_n = n_win;

    ds4_csa_indexer_forward(
        sparse_mask,
        hidden,
        q_residual,
        T,
        n_win,
        end_positions,
        comp_kv,
        cfg,
        idx_wq_b,
        idx_w_weights,
        idx_w_kv,
        idx_w_gate,
        idx_pos_bias,
        idx_norm_weight);
    return n_win;
}
