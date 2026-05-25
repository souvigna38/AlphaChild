/*
 * Sliding-window attention + attention sink (DeepSeek-V4 bootstrap layers).
 * Reference: nano_deepseek_v4/modeling.py DeepSeekV4Attention
 */
#include "sliding_attn.h"

#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_T 128

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

static void grouped_linear(
    float *y,
    const float *x,
    int groups,
    int in_per_group,
    int out_per_group,
    const float *W) {
    for (int g = 0; g < groups; g++) {
        const float *xg = x + (size_t)g * (size_t)in_per_group;
        for (int o = 0; o < out_per_group; o++) {
            const float *row = W + ((size_t)g * (size_t)out_per_group + (size_t)o) * (size_t)in_per_group;
            float s = 0.0f;
            for (int i = 0; i < in_per_group; i++) {
                s += row[i] * xg[i];
            }
            y[(size_t)g * (size_t)out_per_group + (size_t)o] = s;
        }
    }
}

void ds4_core_attention(
    float *context,
    const float *q,
    const float *keys,
    int NH,
    int Tq,
    int Tk,
    int head_dim,
    const float *attn_sink,
    const int *mask) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores[DS4_MAX_T + 1];
    float probs[DS4_MAX_T + 1];

    for (int h = 0; h < NH; h++) {
        for (int tq = 0; tq < Tq; tq++) {
            const float *qh = q + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            float *ch = context + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            int n_scores = 0;
            for (int tk = 0; tk < Tk; tk++) {
                if (!mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
                    scores[n_scores] = -1e30f;
                } else {
                    const float *kh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    scores[n_scores] = ds4_dot(qh, kh, head_dim) * scale;
                }
                n_scores++;
            }
            scores[n_scores] = attn_sink[h];
            n_scores++;
            ds4_softmax(probs, scores, n_scores);
            memset(ch, 0, (size_t)head_dim * sizeof(float));
            int idx = 0;
            for (int tk = 0; tk < Tk; tk++) {
                if (mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
                    const float *vh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    float p = probs[idx];
                    for (int d = 0; d < head_dim; d++) {
                        ch[d] += p * vh[d];
                    }
                }
                idx++;
            }
        }
    }
}

size_t ds4_sliding_attn_scratch_bytes(const DeepSeekV4Config *cfg, int T) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int attn_w = ds4_attention_width(cfg);
    int o_out = cfg->o_groups * cfg->o_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    size_t n = 0;
    n += (size_t)T * (size_t)r;
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)T * (size_t)D;
    n += (size_t)T * (size_t)rope_half * 2;
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)T * (size_t)attn_w;
    n += (size_t)T * (size_t)o_out;
    n += (size_t)attn_w;
    return n * sizeof(float);
}

void ds4_sliding_attn_forward(
    float *out,
    const float *x,
    int T,
    const DeepSeekV4Config *cfg,
    const float *wq_a,
    const float *w_qa_norm,
    const float *wq_b,
    const float *wkv,
    const float *w_kv_norm,
    const float *attn_sink,
    const float *wo_a,
    const float *wo_b,
    float *scratch) {
    const int C = cfg->hidden_size;
    const int NH = cfg->num_attention_heads;
    const int D = cfg->head_dim;
    const int r = cfg->q_lora_rank;
    const int attn_w = ds4_attention_width(cfg);
    const int rope_dim = ds4_qk_rope_head_dim(cfg);
    const int rope_half = rope_dim / 2;
    const int groups = cfg->o_groups;
    const int in_pg = attn_w / groups;
    const int out_pg = cfg->o_lora_rank;
    const int o_mid_dim = groups * out_pg;
    const float eps = cfg->rms_norm_eps;
    if (T > DS4_MAX_T || rope_half <= 0) {
        return;
    }

    float *q_mid = scratch;
    float *q_heads = q_mid + (size_t)T * (size_t)r;
    float *kv = q_heads + (size_t)NH * (size_t)T * (size_t)D;
    float *cos_buf = kv + (size_t)T * (size_t)D;
    float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    float *context = sin_buf + (size_t)T * (size_t)rope_half;
    float *keys = context + (size_t)NH * (size_t)T * (size_t)D;
    float *ctx_flat = keys + (size_t)NH * (size_t)T * (size_t)D;
    float *o_mid = ctx_flat + (size_t)T * (size_t)attn_w;
    float *tmp = o_mid + (size_t)T * (size_t)o_mid_dim;

    ds4_rope_cos_sin_buffer(cos_buf, sin_buf, T, rope_dim, cfg->rope_theta);

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
        ds4_linear(wq_a, xt, tmp, r, C);
        ds4_rmsnorm_forward(qm, tmp, w_qa_norm, 1, r, eps);
        ds4_linear(wq_b, qm, tmp, attn_w, r);
        for (int h = 0; h < NH; h++) {
            float *qh = q_heads + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            memcpy(qh, tmp + (size_t)h * (size_t)D, (size_t)D * sizeof(float));
            unweighted_rmsnorm(qh, qh, D, eps);
            ds4_apply_partial_rope_vec_t(
                qh, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
        }
        float *kvt = kv + (size_t)t * (size_t)D;
        ds4_linear(wkv, xt, tmp, D, C);
        ds4_rmsnorm_forward(kvt, tmp, w_kv_norm, 1, D, eps);
        ds4_apply_partial_rope_vec_t(
            kvt, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
    }

    int mask[DS4_MAX_T * DS4_MAX_T];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)T + (size_t)tk] = causal && in_window;
        }
    }

    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            memcpy(
                keys + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D,
                kv + (size_t)t * (size_t)D,
                (size_t)D * sizeof(float));
        }
    }

    ds4_core_attention(context, q_heads, keys, NH, T, T, D, attn_sink, mask);

    for (int t = 0; t < T; t++) {
        float *row = ctx_flat + (size_t)t * (size_t)attn_w;
        for (int h = 0; h < NH; h++) {
            float *ch = context + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            memcpy(row + (size_t)h * (size_t)D, ch, (size_t)D * sizeof(float));
            ds4_apply_partial_rope_vec_t_neg_sin(
                row + (size_t)h * (size_t)D,
                D,
                rope_dim,
                cos_buf + (size_t)t * (size_t)rope_half,
                sin_buf + (size_t)t * (size_t)rope_half);
        }
        grouped_linear(o_mid + (size_t)t * (size_t)o_mid_dim, row, groups, in_pg, out_pg, wo_a);
        ds4_linear(wo_b, o_mid + (size_t)t * (size_t)o_mid_dim, out + (size_t)t * (size_t)C, C, o_mid_dim);
    }
}
