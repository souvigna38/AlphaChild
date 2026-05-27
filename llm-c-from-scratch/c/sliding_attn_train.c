#include "sliding_attn_train.h"

#include "rmsnorm.h"
#include "rope.h"
#include "sliding_attn.h"
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

static void unweighted_rmsnorm_backward(float *dx, const float *dy, const float *x, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / (float)n + eps;
    float inv_rms = 1.0f / sqrtf(mean_sq);
    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += dy[i] * x[i];
    }
    for (int i = 0; i < n; i++) {
        dx[i] += inv_rms * dy[i] - inv_rms * inv_rms * inv_rms * x[i] * dot / (float)n;
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

static void grouped_linear_backward(
    float *dx,
    float *dW,
    const float *dy,
    const float *x,
    const float *W,
    int groups,
    int in_per_group,
    int out_per_group) {
    for (int g = 0; g < groups; g++) {
        const float *xg = x + (size_t)g * (size_t)in_per_group;
        float *dxg = dx + (size_t)g * (size_t)in_per_group;
        for (int o = 0; o < out_per_group; o++) {
            float gdy = dy[(size_t)g * (size_t)out_per_group + (size_t)o];
            const float *wrow = W + ((size_t)g * (size_t)out_per_group + (size_t)o) * (size_t)in_per_group;
            float *drow = dW + ((size_t)g * (size_t)out_per_group + (size_t)o) * (size_t)in_per_group;
            for (int i = 0; i < in_per_group; i++) {
                dxg[i] += wrow[i] * gdy;
                drow[i] += gdy * xg[i];
            }
        }
    }
}

static void core_attention_save_probs(
    float *context,
    const float *q,
    const float *keys,
    int NH,
    int Tq,
    int Tk,
    int head_dim,
    const float *attn_sink,
    const int *mask,
    float *probs_out) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores[DS4_MAX_T + 1];
    float probs[DS4_MAX_T + 1];

    for (int h = 0; h < NH; h++) {
        for (int tq = 0; tq < Tq; tq++) {
            const float *qh = q + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            float *ch = context + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            float *pout = probs_out + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)(Tk + 1);
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
            memcpy(pout, probs, (size_t)n_scores * sizeof(float));
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

static void core_attention_backward(
    float *dq,
    float *dkeys,
    float *d_sink,
    const float *dcontext,
    const float *q,
    const float *keys,
    const float *probs,
    const int *mask,
    int NH,
    int Tq,
    int Tk,
    int head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    float dprobs[DS4_MAX_T + 1];
    float dscores[DS4_MAX_T + 1];

    for (int h = 0; h < NH; h++) {
        for (int tq = 0; tq < Tq; tq++) {
            const float *qh = q + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            float *dqh = dq + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            const float *dch = dcontext + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
            const float *pout = probs + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)(Tk + 1);
            int n_scores = Tk + 1;
            for (int i = 0; i < n_scores; i++) {
                dprobs[i] = 0.0f;
            }
            int idx = 0;
            for (int tk = 0; tk < Tk; tk++) {
                if (mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
                    const float *vh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    float *dkh = dkeys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        dprobs[idx] += dch[d] * vh[d];
                        dkh[d] += pout[idx] * dch[d];
                    }
                }
                idx++;
            }
            ds4_softmax_backward(dscores, pout, dprobs, n_scores);
            d_sink[h] += dscores[n_scores - 1];
            idx = 0;
            for (int tk = 0; tk < Tk; tk++) {
                if (mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
                    const float *kh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    float *dkh = dkeys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
                    float g = dscores[idx] * scale;
                    for (int d = 0; d < head_dim; d++) {
                        dqh[d] += g * kh[d];
                        dkh[d] += g * qh[d];
                    }
                }
                idx++;
            }
        }
    }
}

size_t ds4_sliding_attn_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int attn_w = ds4_attention_width(cfg);
    int o_mid = cfg->o_groups * cfg->o_lora_rank;
    size_t n = 0;
    n += (size_t)NH * (size_t)T * (size_t)(T + 1);
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)T * (size_t)r;
    n += (size_t)T * (size_t)D;
    n += (size_t)NH * (size_t)T * (size_t)D;
    n += (size_t)T * (size_t)rope_half * 2;
    n += (size_t)T * (size_t)cfg->hidden_size;
    n += (size_t)T * (size_t)o_mid;
    n += (size_t)T * (size_t)attn_w;
    return n;
}

static void cache_offsets(
    const DeepSeekV4Config *cfg,
    int T,
    size_t *probs,
    size_t *q_heads,
    size_t *q_pre,
    size_t *q_mid,
    size_t *kv,
    size_t *keys,
    size_t *cos,
    size_t *x_in,
    size_t *o_mid,
    size_t *ctx_flat) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int o_mid_dim = cfg->o_groups * cfg->o_lora_rank;
    size_t off = 0;
    *probs = off;
    off += (size_t)NH * (size_t)T * (size_t)(T + 1);
    *q_heads = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_pre = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_mid = off;
    off += (size_t)T * (size_t)r;
    *kv = off;
    off += (size_t)T * (size_t)D;
    *keys = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *cos = off;
    off += (size_t)T * (size_t)rope_half * 2;
    *x_in = off;
    off += (size_t)T * (size_t)cfg->hidden_size;
    *o_mid = off;
    off += (size_t)T * (size_t)o_mid_dim;
    *ctx_flat = off;
}

void ds4_sliding_attn_forward_train(
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
    float *scratch,
    float *cache) {
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

    size_t c_probs, c_q, c_qpre, c_qmid, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    cache_offsets(cfg, T, &c_probs, &c_q, &c_qpre, &c_qmid, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    float *probs = cache + c_probs;
    float *q_heads = cache + c_q;
    float *q_pre = cache + c_qpre;
    float *q_mid = cache + c_qmid;
    float *kv = cache + c_kv;
    float *keys = cache + c_keys;
    float *cos_buf = cache + c_cos;
    float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    float *x_save = cache + c_xin;
    float *o_mid_save = cache + c_omid;
    float *ctx_flat_save = cache + c_ctx;

    memcpy(x_save, x, (size_t)T * (size_t)C * sizeof(float));

    float *q_mid_scr = scratch;
    float *q_heads_scr = q_mid_scr + (size_t)T * (size_t)r;
    float *kv_scr = q_heads_scr + (size_t)NH * (size_t)T * (size_t)D;
    float *cos_scr = kv_scr + (size_t)T * (size_t)D;
    float *sin_scr = cos_scr + (size_t)T * (size_t)rope_half;
    float *context = sin_scr + (size_t)T * (size_t)rope_half;
    float *keys_scr = context + (size_t)NH * (size_t)T * (size_t)D;
    float *ctx_flat = keys_scr + (size_t)NH * (size_t)T * (size_t)D;
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
            float *qh = q_heads_scr + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            float *qp = q_pre + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            memcpy(qh, tmp + (size_t)h * (size_t)D, (size_t)D * sizeof(float));
            unweighted_rmsnorm(qh, qh, D, eps);
            memcpy(qp, qh, (size_t)D * sizeof(float));
            ds4_apply_partial_rope_vec_t(qh, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
            memcpy(q_heads + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D, qh, (size_t)D * sizeof(float));
        }
        float *kvt = kv + (size_t)t * (size_t)D;
        ds4_linear(wkv, xt, tmp, D, C);
        ds4_rmsnorm_forward(kvt, tmp, w_kv_norm, 1, D, eps);
        ds4_apply_partial_rope_vec_t(kvt, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
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

    memcpy(q_heads_scr, q_heads, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    core_attention_save_probs(context, q_heads_scr, keys, NH, T, T, D, attn_sink, mask, probs);

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
        memcpy(ctx_flat_save + (size_t)t * (size_t)attn_w, row, (size_t)attn_w * sizeof(float));
        grouped_linear(o_mid + (size_t)t * (size_t)o_mid_dim, row, groups, in_pg, out_pg, wo_a);
        memcpy(o_mid_save + (size_t)t * (size_t)o_mid_dim, o_mid + (size_t)t * (size_t)o_mid_dim, (size_t)o_mid_dim * sizeof(float));
        ds4_linear(wo_b, o_mid + (size_t)t * (size_t)o_mid_dim, out + (size_t)t * (size_t)C, C, o_mid_dim);
    }
}

void ds4_sliding_attn_backward(
    float *dx,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    const float *dout,
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
    float *scratch,
    float *cache) {
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

    size_t c_probs, c_q, c_qpre, c_qmid, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    cache_offsets(cfg, T, &c_probs, &c_q, &c_qpre, &c_qmid, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    const float *probs = cache + c_probs;
    const float *q_heads = cache + c_q;
    const float *q_pre = cache + c_qpre;
    const float *q_mid = cache + c_qmid;
    const float *kv = cache + c_kv;
    const float *keys = cache + c_keys;
    const float *cos_buf = cache + c_cos;
    const float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    const float *o_mid_save = cache + c_omid;
    const float *ctx_flat_save = cache + c_ctx;

    float *dq = scratch;
    float *dkeys = dq + (size_t)NH * (size_t)T * (size_t)D;
    float *dcontext = dkeys + (size_t)NH * (size_t)T * (size_t)D;
    float *dctx_flat = dcontext + (size_t)NH * (size_t)T * (size_t)D;
    float *do_mid = dctx_flat + (size_t)T * (size_t)attn_w;

    memset(dx, 0, (size_t)T * (size_t)C * sizeof(float));
    memset(dq, 0, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    memset(dkeys, 0, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));

    for (int t = 0; t < T; t++) {
        ds4_linear_backward(
            do_mid + (size_t)t * (size_t)o_mid_dim,
            d_wo_b,
            dout + (size_t)t * (size_t)C,
            o_mid_save + (size_t)t * (size_t)o_mid_dim,
            wo_b,
            C,
            o_mid_dim);
        memset(dctx_flat + (size_t)t * (size_t)attn_w, 0, (size_t)attn_w * sizeof(float));
        grouped_linear_backward(
            dctx_flat + (size_t)t * (size_t)attn_w,
            d_wo_a,
            do_mid + (size_t)t * (size_t)o_mid_dim,
            ctx_flat_save + (size_t)t * (size_t)attn_w,
            wo_a,
            groups,
            in_pg,
            out_pg);
    }

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < NH; h++) {
            float *drow = dctx_flat + (size_t)t * (size_t)attn_w + (size_t)h * (size_t)D;
            ds4_apply_partial_rope_backward_vec_t_neg_sin(
                drow, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
            float *dch = dcontext + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            memcpy(dch, drow, (size_t)D * sizeof(float));
        }
    }

    int mask[DS4_MAX_T * DS4_MAX_T];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)T + (size_t)tk] = causal && in_window;
        }
    }

    core_attention_backward(dq, dkeys, d_attn_sink, dcontext, q_heads, keys, probs, mask, NH, T, T, D);

    for (int t = 0; t < T; t++) {
        float dkv_t[32];
        memset(dkv_t, 0, (size_t)D * sizeof(float));
        for (int h = 0; h < NH; h++) {
            for (int d = 0; d < D; d++) {
                dkv_t[d] += dkeys[((size_t)h * (size_t)T + (size_t)t) * (size_t)D + (size_t)d];
            }
        }
        const float *kvt = kv + (size_t)t * (size_t)D;
        ds4_apply_partial_rope_backward_vec_t(dkv_t, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
        ds4_rmsnorm_backward(dx + (size_t)t * (size_t)C, d_w_kv_norm, dkv_t, x + (size_t)t * (size_t)C, kvt, w_kv_norm, 1, D, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wkv, dkv_t, x + (size_t)t * (size_t)C, wkv, D, C);

        float dq_flat[64];
        memset(dq_flat, 0, (size_t)attn_w * sizeof(float));
        for (int h = 0; h < NH; h++) {
            float *dqh = dq + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            ds4_apply_partial_rope_backward_vec_t(dqh, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
            unweighted_rmsnorm_backward(
                dqh,
                dqh,
                q_pre + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D,
                D,
                eps);
            memcpy(dq_flat + (size_t)h * (size_t)D, dqh, (size_t)D * sizeof(float));
        }
        float dqm[32];
        memset(dqm, 0, (size_t)r * sizeof(float));
        ds4_linear_backward(dqm, d_wq_b, dq_flat, q_mid + (size_t)t * (size_t)r, wq_b, attn_w, r);
        float dtmp[64];
        memset(dtmp, 0, (size_t)r * sizeof(float));
        ds4_rmsnorm_backward(dtmp, d_w_qa_norm, dqm, x + (size_t)t * (size_t)C, q_mid + (size_t)t * (size_t)r, w_qa_norm, 1, r, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wq_a, dtmp, x + (size_t)t * (size_t)C, wq_a, r, C);
    }
    (void)q_heads;
    (void)wq_a;
    (void)wq_b;
    (void)wkv;
    (void)attn_sink;
    (void)wo_a;
    (void)wo_b;
}
