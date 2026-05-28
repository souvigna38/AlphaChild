#include "indexer.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>

#define DS4_MAX_COMP 64

int ds4_csa_indexer_forward(
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
    const float *norm_weight) {
    const int C = cfg->hidden_size;
    const int D = cfg->head_dim;
    const int rate = cfg->compress_rate_csa;
    const int NH = cfg->index_n_heads;
    const int HD = cfg->index_head_dim;
    const int topk = cfg->index_topk;
    const int r = cfg->q_lora_rank;
    const int rope_dim = (cfg->index_head_dim < ds4_qk_rope_head_dim(cfg)) ? cfg->index_head_dim : ds4_qk_rope_head_dim(cfg);
    if (rope_dim % 2 != 0) {
        /* match nano: shrink by 1 if odd */
    }
    int rope_use = rope_dim;
    if (rope_use % 2 != 0) {
        rope_use -= 1;
    }
    const int rope_half = rope_use / 2;

    float idx_comp[DS4_MAX_COMP * 32];
    int idx_end[DS4_MAX_COMP];
    int n_idx = 0;
    if (n_comp <= 0) {
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < n_comp; k++) {
                sparse_mask[(size_t)t * (size_t)n_comp + (size_t)k] = 0;
            }
        }
        return 0;
    }

    /* Indexer-side compression (nano CSAIndexer._compress), cache=NULL. */
    const int usable = (T / rate) * rate;
    const int n_win = usable / rate;
    float kv[512];
    float gate[512];
    float weights[128];
    float cos[DS4_MAX_COMP];
    float sin[DS4_MAX_COMP];

    for (int w = 0; w < n_win && w < DS4_MAX_COMP; w++) {
        int base = w * rate;
        for (int ri = 0; ri < rate; ri++) {
            const float *h = hidden + (size_t)(base + ri) * (size_t)C;
            ds4_linear(w_kv, h, kv + (size_t)ri * 2 * (size_t)HD, 2 * HD, C);
            ds4_linear(w_gate, h, gate + (size_t)ri * 2 * (size_t)HD, 2 * HD, C);
            for (int d = 0; d < 2 * HD; d++) {
                gate[(size_t)ri * 2 * (size_t)HD + (size_t)d] += pos_bias[(size_t)ri * 2 * (size_t)HD + (size_t)d];
            }
        }
        float maxg = gate[0];
        for (int i = 1; i < rate * 2 * HD; i++) {
            if (gate[i] > maxg) {
                maxg = gate[i];
            }
        }
        float sum = 0.0f;
        for (int ri = 0; ri < rate; ri++) {
            float sg = 0.0f;
            for (int d = 0; d < 2 * HD; d++) {
                sg += expf(gate[(size_t)ri * 2 * (size_t)HD + (size_t)d] - maxg);
            }
            weights[ri] = sg;
            sum += sg;
        }
        float *out = idx_comp + (size_t)n_idx * (size_t)HD;
        for (int d = 0; d < HD; d++) {
            out[d] = 0.0f;
        }
        for (int ri = 0; ri < rate; ri++) {
            float wr = (sum > 0.0f) ? (weights[ri] / sum) : (1.0f / (float)rate);
            for (int d = 0; d < HD; d++) {
                out[d] += wr * kv[(size_t)ri * 2 * (size_t)HD + (size_t)HD + (size_t)d];
            }
        }
        ds4_rmsnorm_forward(out, out, norm_weight, 1, HD, cfg->rms_norm_eps);
        ds4_rope_cos_sin_buffer(cos, sin, 1, rope_use, cfg->compress_rope_theta);
        ds4_apply_partial_rope_vec(out, HD, rope_use, cos, sin);
        idx_end[n_idx] = base + rate - 1;
        n_idx++;
    }

    if (n_idx == 0) {
        return 0;
    }

    float cos_q[DS4_MAX_COMP];
    float sin_q[DS4_MAX_COMP];
    ds4_rope_cos_sin_buffer(cos_q, sin_q, T, rope_use, cfg->compress_rope_theta);

    float qflat[256];
    float scores[DS4_MAX_COMP * DS4_MAX_COMP];

    for (int t = 0; t < T; t++) {
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
        float wf[16];
        ds4_linear(w_weights, hidden + (size_t)t * (size_t)C, wf, NH, C);
        for (int h = 0; h < NH; h++) {
            wf[h] *= 1.0f / sqrtf((float)NH);
        }
        for (int k = 0; k < n_idx; k++) {
            float s = 0.0f;
            const float *ck = idx_comp + (size_t)k * (size_t)HD;
            for (int h = 0; h < NH; h++) {
                float dot = ds4_dot(qflat + (size_t)h * (size_t)HD, ck, HD);
                if (dot > 0.0f) {
                    dot *= 1.0f / sqrtf((float)HD);
                } else {
                    dot = 0.0f;
                }
                s += dot * wf[h];
            }
            scores[(size_t)t * (size_t)n_idx + (size_t)k] = s;
            if (idx_end[k] > t) {
                scores[(size_t)t * (size_t)n_idx + (size_t)k] = -1e30f;
            }
        }
        int ktop = (topk < n_idx) ? topk : n_idx;
        for (int k = 0; k < n_idx; k++) {
            sparse_mask[(size_t)t * (size_t)n_comp + (size_t)k] = 0;
        }
        for (int pick = 0; pick < ktop; pick++) {
            int best = -1;
            float bestv = -1e30f;
            for (int k = 0; k < n_idx; k++) {
                float v = scores[(size_t)t * (size_t)n_idx + (size_t)k];
                if (v > bestv) {
                    bestv = v;
                    best = k;
                }
            }
            if (best >= 0 && bestv > -1e20f) {
                sparse_mask[(size_t)t * (size_t)n_comp + (size_t)best] = 1;
                scores[(size_t)t * (size_t)n_idx + (size_t)best] = -1e30f;
            }
        }
    }

    (void)compressed;
    (void)comp_end_positions;
    (void)D;
    return n_idx;
}
