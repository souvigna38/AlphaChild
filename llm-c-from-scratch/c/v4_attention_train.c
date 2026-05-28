#include "v4_attention_train.h"

#include "csa_compressor_train.h"
#include "hca_compressor_train.h"
#include "indexer.h"
#include "rmsnorm.h"
#include "rope.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_T 128
#define DS4_MAX_COMP 64
#define DS4_MAX_SCORES (DS4_MAX_T + DS4_MAX_COMP + 1)

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
    float scores[DS4_MAX_SCORES];
    float probs[DS4_MAX_SCORES];

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
    float dprobs[DS4_MAX_SCORES];
    float dscores[DS4_MAX_SCORES];

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

static size_t attn_cache_base(const DeepSeekV4Config *cfg, int T) {
    return ds4_hca_compress_train_cache_floats(cfg, T);
}

static size_t attn_cache_tail(const DeepSeekV4Config *cfg, int T) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int attn_w = ds4_attention_width(cfg);
    int o_mid = cfg->o_groups * cfg->o_lora_rank;
    int Tk = T + DS4_MAX_COMP;
    size_t n = 1;
    n += (size_t)DS4_MAX_COMP * (size_t)D + (size_t)DS4_MAX_COMP;
    n += (size_t)NH * (size_t)T * (size_t)(Tk + 1);
    n += (size_t)NH * (size_t)T * (size_t)D * 2;
    n += (size_t)T * (size_t)r + (size_t)T * (size_t)D;
    n += (size_t)NH * (size_t)Tk * (size_t)D;
    n += (size_t)T * (size_t)rope_half * 2;
    n += (size_t)T * (size_t)cfg->hidden_size;
    n += (size_t)T * (size_t)o_mid + (size_t)T * (size_t)attn_w;
    return n;
}

size_t ds4_hca_attention_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    return attn_cache_base(cfg, T) + attn_cache_tail(cfg, T);
}

static void tail_offsets(
    const DeepSeekV4Config *cfg,
    int T,
    size_t base,
    size_t *n_comp,
    size_t *comp_kv,
    size_t *comp_end,
    size_t *probs,
    size_t *q_heads,
    size_t *q_pre,
    size_t *q_mid,
    size_t *kv,
    size_t *keys,
    size_t *cos,
    size_t *xin,
    size_t *omid,
    size_t *ctx) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int o_mid = cfg->o_groups * cfg->o_lora_rank;
    int Tk = T + DS4_MAX_COMP;
    size_t off = base;
    *n_comp = off++;
    *comp_kv = off;
    off += (size_t)DS4_MAX_COMP * (size_t)D;
    *comp_end = off;
    off += (size_t)DS4_MAX_COMP;
    *probs = off;
    off += (size_t)NH * (size_t)T * (size_t)(Tk + 1);
    *q_heads = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_pre = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_mid = off;
    off += (size_t)T * (size_t)r;
    *kv = off;
    off += (size_t)T * (size_t)D;
    *keys = off;
    off += (size_t)NH * (size_t)Tk * (size_t)D;
    *cos = off;
    off += (size_t)T * (size_t)rope_half * 2;
    *xin = off;
    off += (size_t)T * (size_t)cfg->hidden_size;
    *omid = off;
    off += (size_t)T * (size_t)o_mid;
    *ctx = off;
}

void ds4_hca_attention_forward_train(
    float *out,
    const float *hidden,
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
    const float *hca_w_kv,
    const float *hca_w_gate,
    const float *hca_pos_bias,
    const float *hca_norm,
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

    size_t c_nc, c_ckv, c_cend, c_probs, c_q, c_qp, c_qm, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    tail_offsets(cfg, T, attn_cache_base(cfg, T), &c_nc, &c_ckv, &c_cend, &c_probs, &c_q, &c_qp, &c_qm, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    float *hca_cache = cache;
    float *tail = cache + attn_cache_base(cfg, T);
    int *n_comp_i = (int *)(tail + c_nc);
    float *comp_kv = tail + c_ckv;
    int *comp_end = (int *)(tail + c_cend);
    float *probs = tail + c_probs;
    float *q_heads = tail + c_q;
    float *q_pre = tail + c_qp;
    float *q_mid = tail + c_qm;
    float *kv = tail + c_kv;
    float *keys = tail + c_keys;
    float *cos_buf = tail + c_cos;
    float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    float *x_save = tail + c_xin;
    float *o_mid_save = tail + c_omid;
    float *ctx_flat_save = tail + c_ctx;

    memcpy(x_save, hidden, (size_t)T * (size_t)C * sizeof(float));

    int n_comp = 0;
    ds4_hca_compress_forward_train(
        comp_kv, comp_end, hidden, T, cfg, hca_w_kv, hca_w_gate, hca_pos_bias, hca_norm, hca_cache, &n_comp);
    *n_comp_i = n_comp;

    float *q_mid_scr = scratch;
    float *q_heads_scr = q_mid_scr + (size_t)T * (size_t)r;
    float *context = q_heads_scr + (size_t)NH * (size_t)T * (size_t)D;
    float *ctx_flat = context + (size_t)NH * (size_t)T * (size_t)D;
    float *o_mid = ctx_flat + (size_t)T * (size_t)attn_w;
    float *tmp = o_mid + (size_t)T * (size_t)o_mid_dim;

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
        ds4_linear(wq_a, xt, tmp, r, C);
        ds4_rmsnorm_forward(qm, tmp, w_qa_norm, 1, r, eps);
    }

    ds4_rope_cos_sin_buffer(cos_buf, sin_buf, T, rope_dim, cfg->rope_theta);

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
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

    const int Tk = T + n_comp;
    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            memcpy(keys + ((size_t)h * (size_t)Tk + (size_t)t) * (size_t)D, kv + (size_t)t * (size_t)D, (size_t)D * sizeof(float));
        }
        for (int k = 0; k < n_comp; k++) {
            memcpy(
                keys + ((size_t)h * (size_t)Tk + (size_t)T + (size_t)k) * (size_t)D,
                comp_kv + (size_t)k * (size_t)D,
                (size_t)D * sizeof(float));
        }
    }

    int mask[DS4_MAX_T * (DS4_MAX_T + DS4_MAX_COMP)];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)Tk + (size_t)tk] = causal && in_window;
        }
        for (int k = 0; k < n_comp; k++) {
            mask[(size_t)tq * (size_t)Tk + (size_t)T + (size_t)k] = (comp_end[k] <= tq);
        }
    }

    memcpy(q_heads_scr, q_heads, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    core_attention_save_probs(context, q_heads_scr, keys, NH, T, Tk, D, attn_sink, mask, probs);

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

void ds4_hca_attention_backward(
    float *dx,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    float *d_hca_w_kv,
    float *d_hca_w_gate,
    float *d_hca_pos_bias,
    float *d_hca_norm,
    const float *dout,
    const float *hidden,
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
    const float *hca_w_kv,
    const float *hca_w_gate,
    const float *hca_pos_bias,
    const float *hca_norm,
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

    size_t c_nc, c_ckv, c_cend, c_probs, c_q, c_qp, c_qm, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    tail_offsets(cfg, T, attn_cache_base(cfg, T), &c_nc, &c_ckv, &c_cend, &c_probs, &c_q, &c_qp, &c_qm, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    float *hca_cache = cache;
    float *tail = cache + attn_cache_base(cfg, T);
    int n_comp = *(int *)(tail + c_nc);
    int *comp_end = (int *)(tail + c_cend);
    const float *probs = tail + c_probs;
    const float *q_heads = tail + c_q;
    const float *q_pre = tail + c_qp;
    const float *q_mid = tail + c_qm;
    const float *kv = tail + c_kv;
    const float *keys = tail + c_keys;
    const float *cos_buf = tail + c_cos;
    const float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    const float *o_mid_save = tail + c_omid;
    const float *ctx_flat_save = tail + c_ctx;

    const int Tk = T + n_comp;

    float *dq = scratch;
    float *dkeys = dq + (size_t)NH * (size_t)T * (size_t)D;
    float *dcontext = dkeys + (size_t)NH * (size_t)Tk * (size_t)D;
    float *dctx_flat = dcontext + (size_t)NH * (size_t)T * (size_t)D;
    float *do_mid = dctx_flat + (size_t)T * (size_t)attn_w;

    memset(dx, 0, (size_t)T * (size_t)C * sizeof(float));
    memset(dq, 0, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    memset(dkeys, 0, (size_t)NH * (size_t)Tk * (size_t)D * sizeof(float));

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

    int mask[DS4_MAX_T * (DS4_MAX_T + DS4_MAX_COMP)];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)Tk + (size_t)tk] = causal && in_window;
        }
        for (int k = 0; k < n_comp; k++) {
            mask[(size_t)tq * (size_t)Tk + (size_t)T + (size_t)k] = (comp_end[k] <= tq);
        }
    }

    core_attention_backward(dq, dkeys, d_attn_sink, dcontext, q_heads, keys, probs, mask, NH, T, Tk, D);

    float d_comp[DS4_MAX_COMP * 32];
    memset(d_comp, 0, (size_t)n_comp * (size_t)D * sizeof(float));
    for (int h = 0; h < NH; h++) {
        for (int k = 0; k < n_comp; k++) {
            float *dk = dkeys + ((size_t)h * (size_t)Tk + (size_t)T + (size_t)k) * (size_t)D;
            for (int d = 0; d < D; d++) {
                d_comp[(size_t)k * (size_t)D + (size_t)d] += dk[d];
            }
        }
    }
    ds4_hca_compress_backward(
        dx,
        d_hca_w_kv,
        d_hca_w_gate,
        d_hca_pos_bias,
        d_hca_norm,
        d_comp,
        hidden,
        T,
        n_comp,
        cfg,
        hca_w_kv,
        hca_w_gate,
        hca_pos_bias,
        hca_norm,
        hca_cache);

    for (int t = 0; t < T; t++) {
        float dkv_t[32];
        memset(dkv_t, 0, (size_t)D * sizeof(float));
        for (int h = 0; h < NH; h++) {
            for (int d = 0; d < D; d++) {
                dkv_t[d] += dkeys[((size_t)h * (size_t)Tk + (size_t)t) * (size_t)D + (size_t)d];
            }
        }
        const float *kvt = kv + (size_t)t * (size_t)D;
        ds4_apply_partial_rope_backward_vec_t(dkv_t, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
        ds4_rmsnorm_backward(dx + (size_t)t * (size_t)C, d_w_kv_norm, dkv_t, hidden + (size_t)t * (size_t)C, kvt, w_kv_norm, 1, D, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wkv, dkv_t, hidden + (size_t)t * (size_t)C, wkv, D, C);

        float dq_flat[64];
        memset(dq_flat, 0, (size_t)attn_w * sizeof(float));
        for (int h = 0; h < NH; h++) {
            float *dqh = dq + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            ds4_apply_partial_rope_backward_vec_t(dqh, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
            unweighted_rmsnorm_backward(dqh, dqh, q_pre + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D, D, eps);
            memcpy(dq_flat + (size_t)h * (size_t)D, dqh, (size_t)D * sizeof(float));
        }
        float dqm[32];
        memset(dqm, 0, (size_t)r * sizeof(float));
        ds4_linear_backward(dqm, d_wq_b, dq_flat, q_mid + (size_t)t * (size_t)r, wq_b, attn_w, r);
        float dtmp[64];
        memset(dtmp, 0, (size_t)r * sizeof(float));
        ds4_rmsnorm_backward(dtmp, d_w_qa_norm, dqm, hidden + (size_t)t * (size_t)C, q_mid + (size_t)t * (size_t)r, w_qa_norm, 1, r, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wq_a, dtmp, hidden + (size_t)t * (size_t)C, wq_a, r, C);
    }
    (void)attn_sink;
}

static size_t csa_attn_cache_base(const DeepSeekV4Config *cfg, int T) {
    return ds4_csa_compress_train_cache_floats(cfg, T);
}

static size_t csa_attn_cache_tail(const DeepSeekV4Config *cfg, int T) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int o_mid = cfg->o_groups * cfg->o_lora_rank;
    int Tk = T + DS4_MAX_COMP;
    size_t n = 1 + (size_t)DS4_MAX_COMP * (size_t)D + (size_t)DS4_MAX_COMP;
    n += (size_t)T * (size_t)DS4_MAX_COMP;
    n += (size_t)NH * (size_t)T * (size_t)(Tk + 1);
    n += (size_t)NH * (size_t)T * (size_t)D * 2;
    n += (size_t)T * (size_t)r + (size_t)T * (size_t)D;
    n += (size_t)NH * (size_t)Tk * (size_t)D;
    n += (size_t)T * (size_t)rope_half * 2;
    n += (size_t)T * (size_t)cfg->hidden_size;
    n += (size_t)T * (size_t)o_mid + (size_t)T * (size_t)ds4_attention_width(cfg);
    return n;
}

size_t ds4_csa_attention_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    return csa_attn_cache_base(cfg, T) + csa_attn_cache_tail(cfg, T);
}

static void csa_tail_offsets(
    const DeepSeekV4Config *cfg,
    int T,
    size_t base,
    size_t *n_comp,
    size_t *comp_kv,
    size_t *comp_end,
    size_t *sparse,
    size_t *probs,
    size_t *q_heads,
    size_t *q_pre,
    size_t *q_mid,
    size_t *kv,
    size_t *keys,
    size_t *cos,
    size_t *xin,
    size_t *omid,
    size_t *ctx) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    int r = cfg->q_lora_rank;
    int rope_half = ds4_qk_rope_head_dim(cfg) / 2;
    int o_mid = cfg->o_groups * cfg->o_lora_rank;
    int Tk = T + DS4_MAX_COMP;
    size_t off = base;
    *n_comp = off++;
    *comp_kv = off;
    off += (size_t)DS4_MAX_COMP * (size_t)D;
    *comp_end = off;
    off += (size_t)DS4_MAX_COMP;
    *sparse = off;
    off += (size_t)T * (size_t)DS4_MAX_COMP;
    *probs = off;
    off += (size_t)NH * (size_t)T * (size_t)(Tk + 1);
    *q_heads = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_pre = off;
    off += (size_t)NH * (size_t)T * (size_t)D;
    *q_mid = off;
    off += (size_t)T * (size_t)r;
    *kv = off;
    off += (size_t)T * (size_t)D;
    *keys = off;
    off += (size_t)NH * (size_t)Tk * (size_t)D;
    *cos = off;
    off += (size_t)T * (size_t)rope_half * 2;
    *xin = off;
    off += (size_t)T * (size_t)cfg->hidden_size;
    *omid = off;
    off += (size_t)T * (size_t)o_mid;
    *ctx = off;
    (void)off;
}

void ds4_csa_attention_forward_train(
    float *out,
    const float *hidden,
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
    const float *csa_w_kv,
    const float *csa_w_gate,
    const float *csa_pos_bias,
    const float *csa_norm,
    const float *idx_wq_b,
    const float *idx_w_weights,
    const float *idx_w_kv,
    const float *idx_w_gate,
    const float *idx_pos_bias,
    const float *idx_norm,
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

    size_t c_nc, c_ckv, c_cend, c_sparse, c_probs, c_q, c_qp, c_qm, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    csa_tail_offsets(cfg, T, csa_attn_cache_base(cfg, T), &c_nc, &c_ckv, &c_cend, &c_sparse, &c_probs, &c_q, &c_qp, &c_qm, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    float *csa_cache = cache;
    float *tail = cache + csa_attn_cache_base(cfg, T);
    int *n_comp_i = (int *)(tail + c_nc);
    float *comp_kv = tail + c_ckv;
    int *comp_end = (int *)(tail + c_cend);
    int *sparse_mask = (int *)(tail + c_sparse);
    float *probs = tail + c_probs;
    float *q_heads = tail + c_q;
    float *q_pre = tail + c_qp;
    float *q_mid = tail + c_qm;
    float *kv = tail + c_kv;
    float *keys = tail + c_keys;
    float *cos_buf = tail + c_cos;
    float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    float *x_save = tail + c_xin;
    float *o_mid_save = tail + c_omid;
    float *ctx_flat_save = tail + c_ctx;

    memcpy(x_save, hidden, (size_t)T * (size_t)C * sizeof(float));

    float *q_mid_scr = scratch;
    float *q_heads_scr = q_mid_scr + (size_t)T * (size_t)r;
    float *context = q_heads_scr + (size_t)NH * (size_t)T * (size_t)D;
    float *ctx_flat = context + (size_t)NH * (size_t)T * (size_t)D;
    float *o_mid = ctx_flat + (size_t)T * (size_t)attn_w;
    float *tmp = o_mid + (size_t)T * (size_t)o_mid_dim;

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
        ds4_linear(wq_a, xt, tmp, r, C);
        ds4_rmsnorm_forward(qm, tmp, w_qa_norm, 1, r, eps);
    }

    int n_comp = 0;
    ds4_csa_compress_forward_train(
        comp_kv, comp_end, hidden, T, cfg, csa_w_kv, csa_w_gate, csa_pos_bias, csa_norm, csa_cache, &n_comp);
    *n_comp_i = n_comp;

    ds4_csa_indexer_forward(
        sparse_mask,
        hidden,
        q_mid,
        T,
        n_comp,
        comp_end,
        comp_kv,
        cfg,
        idx_wq_b,
        idx_w_weights,
        idx_w_kv,
        idx_w_gate,
        idx_pos_bias,
        idx_norm);

    ds4_rope_cos_sin_buffer(cos_buf, sin_buf, T, rope_dim, cfg->rope_theta);

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
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

    const int Tk = T + n_comp;
    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            memcpy(keys + ((size_t)h * (size_t)Tk + (size_t)t) * (size_t)D, kv + (size_t)t * (size_t)D, (size_t)D * sizeof(float));
        }
        for (int k = 0; k < n_comp; k++) {
            memcpy(
                keys + ((size_t)h * (size_t)Tk + (size_t)T + (size_t)k) * (size_t)D,
                comp_kv + (size_t)k * (size_t)D,
                (size_t)D * sizeof(float));
        }
    }

    int mask[DS4_MAX_T * (DS4_MAX_T + DS4_MAX_COMP)];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)Tk + (size_t)tk] = causal && in_window;
        }
        for (int k = 0; k < n_comp; k++) {
            int ok = (comp_end[k] <= tq) && sparse_mask[(size_t)tq * (size_t)n_comp + (size_t)k];
            mask[(size_t)tq * (size_t)Tk + (size_t)T + (size_t)k] = ok;
        }
    }

    memcpy(q_heads_scr, q_heads, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    core_attention_save_probs(context, q_heads_scr, keys, NH, T, Tk, D, attn_sink, mask, probs);

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

void ds4_csa_attention_backward(
    float *dx,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    float *d_csa_w_kv,
    float *d_csa_w_gate,
    float *d_csa_pos_bias,
    float *d_csa_norm,
    const float *dout,
    const float *hidden,
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
    const float *csa_w_kv,
    const float *csa_w_gate,
    const float *csa_pos_bias,
    const float *csa_norm,
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

    size_t c_nc, c_ckv, c_cend, c_sparse, c_probs, c_q, c_qp, c_qm, c_kv, c_keys, c_cos, c_xin, c_omid, c_ctx;
    csa_tail_offsets(cfg, T, csa_attn_cache_base(cfg, T), &c_nc, &c_ckv, &c_cend, &c_sparse, &c_probs, &c_q, &c_qp, &c_qm, &c_kv, &c_keys, &c_cos, &c_xin, &c_omid, &c_ctx);

    float *csa_cache = cache;
    float *tail = cache + csa_attn_cache_base(cfg, T);
    int n_comp = *(int *)(tail + c_nc);
    int *comp_end = (int *)(tail + c_cend);
    int *sparse_mask = (int *)(tail + c_sparse);
    const float *probs = tail + c_probs;
    const float *q_heads = tail + c_q;
    const float *q_pre = tail + c_qp;
    const float *q_mid = tail + c_qm;
    const float *kv = tail + c_kv;
    const float *keys = tail + c_keys;
    const float *cos_buf = tail + c_cos;
    const float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    const float *o_mid_save = tail + c_omid;
    const float *ctx_flat_save = tail + c_ctx;

    const int Tk = T + n_comp;

    float *dq = scratch;
    float *dkeys = dq + (size_t)NH * (size_t)T * (size_t)D;
    float *dcontext = dkeys + (size_t)NH * (size_t)Tk * (size_t)D;
    float *dctx_flat = dcontext + (size_t)NH * (size_t)T * (size_t)D;
    float *do_mid = dctx_flat + (size_t)T * (size_t)attn_w;

    memset(dx, 0, (size_t)T * (size_t)C * sizeof(float));
    memset(dq, 0, (size_t)NH * (size_t)T * (size_t)D * sizeof(float));
    memset(dkeys, 0, (size_t)NH * (size_t)Tk * (size_t)D * sizeof(float));

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

    int mask[DS4_MAX_T * (DS4_MAX_T + DS4_MAX_COMP)];
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            int causal = (tk <= tq);
            int in_window = (tk >= tq - cfg->sliding_window + 1);
            mask[(size_t)tq * (size_t)Tk + (size_t)tk] = causal && in_window;
        }
        for (int k = 0; k < n_comp; k++) {
            int ok = (comp_end[k] <= tq) && sparse_mask[(size_t)tq * (size_t)n_comp + (size_t)k];
            mask[(size_t)tq * (size_t)Tk + (size_t)T + (size_t)k] = ok;
        }
    }

    core_attention_backward(dq, dkeys, d_attn_sink, dcontext, q_heads, keys, probs, mask, NH, T, Tk, D);

    float d_comp[DS4_MAX_COMP * 32];
    memset(d_comp, 0, (size_t)n_comp * (size_t)D * sizeof(float));
    for (int h = 0; h < NH; h++) {
        for (int k = 0; k < n_comp; k++) {
            float *dk = dkeys + ((size_t)h * (size_t)Tk + (size_t)T + (size_t)k) * (size_t)D;
            for (int d = 0; d < D; d++) {
                d_comp[(size_t)k * (size_t)D + (size_t)d] += dk[d];
            }
        }
    }
    ds4_csa_compress_backward(
        dx,
        d_csa_w_kv,
        d_csa_w_gate,
        d_csa_pos_bias,
        d_csa_norm,
        d_comp,
        T,
        n_comp,
        cfg,
        csa_w_kv,
        csa_w_gate,
        csa_norm,
        csa_cache);

    for (int t = 0; t < T; t++) {
        float dkv_t[32];
        memset(dkv_t, 0, (size_t)D * sizeof(float));
        for (int h = 0; h < NH; h++) {
            for (int d = 0; d < D; d++) {
                dkv_t[d] += dkeys[((size_t)h * (size_t)Tk + (size_t)t) * (size_t)D + (size_t)d];
            }
        }
        const float *kvt = kv + (size_t)t * (size_t)D;
        ds4_apply_partial_rope_backward_vec_t(dkv_t, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
        ds4_rmsnorm_backward(dx + (size_t)t * (size_t)C, d_w_kv_norm, dkv_t, hidden + (size_t)t * (size_t)C, kvt, w_kv_norm, 1, D, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wkv, dkv_t, hidden + (size_t)t * (size_t)C, wkv, D, C);

        float dq_flat[64];
        memset(dq_flat, 0, (size_t)attn_w * sizeof(float));
        for (int h = 0; h < NH; h++) {
            float *dqh = dq + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D;
            ds4_apply_partial_rope_backward_vec_t(dqh, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
            unweighted_rmsnorm_backward(dqh, dqh, q_pre + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D, D, eps);
            memcpy(dq_flat + (size_t)h * (size_t)D, dqh, (size_t)D * sizeof(float));
        }
        float dqm[32];
        memset(dqm, 0, (size_t)r * sizeof(float));
        ds4_linear_backward(dqm, d_wq_b, dq_flat, q_mid + (size_t)t * (size_t)r, wq_b, attn_w, r);
        float dtmp[64];
        memset(dtmp, 0, (size_t)r * sizeof(float));
        ds4_rmsnorm_backward(dtmp, d_w_qa_norm, dqm, hidden + (size_t)t * (size_t)C, q_mid + (size_t)t * (size_t)r, w_qa_norm, 1, r, eps);
        ds4_linear_backward(dx + (size_t)t * (size_t)C, d_wq_a, dtmp, hidden + (size_t)t * (size_t)C, wq_a, r, C);
    }
    (void)csa_pos_bias;
    (void)attn_sink;
    (void)wq_a;
}
