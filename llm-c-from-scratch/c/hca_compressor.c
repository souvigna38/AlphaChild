#include "hca_compressor.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>

#define DS4_MAX_COMP 64

int ds4_hca_compress_forward(
    float *comp_kv,
    int *end_positions,
    const float *hidden,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    int *out_n) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_hca;
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const int rope_half = rope_dim / 2;
    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;

    if (n_win <= 0 || n_win > DS4_MAX_COMP) {
        *out_n = 0;
        return 0;
    }

    float cos[DS4_MAX_COMP];
    float sin[DS4_MAX_COMP];
    float kv[256];
    float gate[256];
    float weights[128];

    for (int w = 0; w < n_win; w++) {
        int base = w * rate;
        for (int r = 0; r < rate; r++) {
            const float *h = hidden + (size_t)(base + r) * (size_t)C;
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
        ds4_rmsnorm_forward(out, out, norm_weight, 1, D, cfg->rms_norm_eps);
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_dim, cfg->compress_rope_theta);
        ds4_apply_partial_rope_vec(out, D, rope_dim, cos, sin);
        end_positions[w] = base + rate - 1;
    }
    *out_n = n_win;
    (void)rope_half;
    return n_win;
}
