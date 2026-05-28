#include "v4_attention.h"

#include "csa_compressor.h"
#include "ds4_cuda.h"
#include "hca_compressor.h"
#include "rmsnorm.h"
#include "rope.h"
#include "sliding_attn.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_T 128
#define DS4_MAX_COMP 64

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

size_t ds4_attention_scratch_bytes(const DeepSeekV4Config *cfg, int T) {
    int NH = cfg->num_attention_heads;
    int D = cfg->head_dim;
    size_t n = ds4_sliding_attn_scratch_bytes(cfg, T) / sizeof(float);
    /* sliding scratch keys are NH*T*D; hybrid attention needs NH*(T+DS4_MAX_COMP)*D */
    n += (size_t)NH * (size_t)DS4_MAX_COMP * (size_t)D;
    return n * sizeof(float);
}

void ds4_attention_forward(
    float *out,
    const float *hidden,
    int T,
    Ds4AttentionType attn_type,
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
    float *scratch) {
    if (attn_type == DS4_ATTN_SLIDING) {
        ds4_sliding_attn_forward(
            out, hidden, T, cfg, wq_a, w_qa_norm, wq_b, wkv, w_kv_norm, attn_sink, wo_a, wo_b, scratch);
        return;
    }

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

    float comp_kv[DS4_MAX_COMP * 32];
    int comp_end[DS4_MAX_COMP];
    int sparse_mask[DS4_MAX_T * DS4_MAX_COMP];
    int n_comp = 0;

    float *q_mid = scratch;
    float *q_heads = q_mid + (size_t)T * (size_t)r;
    float *kv = q_heads + (size_t)NH * (size_t)T * (size_t)D;
    float *cos_buf = kv + (size_t)T * (size_t)D;
    float *sin_buf = cos_buf + (size_t)T * (size_t)rope_half;
    const int Tk_cap = T + DS4_MAX_COMP;
    float *context = sin_buf + (size_t)T * (size_t)rope_half;
    float *keys = context + (size_t)NH * (size_t)T * (size_t)D;
    float *ctx_flat = keys + (size_t)NH * (size_t)Tk_cap * (size_t)D;
    float *o_mid = ctx_flat + (size_t)T * (size_t)attn_w;
    float *tmp = o_mid + (size_t)T * (size_t)o_mid_dim;

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
        ds4_linear(wq_a, xt, tmp, r, C);
        ds4_rmsnorm_forward_cuda(qm, tmp, w_qa_norm, 1, r, eps);
    }

    if (attn_type == DS4_ATTN_HCA) {
        ds4_hca_compress_forward(comp_kv, comp_end, hidden, T, cfg, hca_w_kv, hca_w_gate, hca_pos_bias, hca_norm, &n_comp);
    } else if (attn_type == DS4_ATTN_CSA) {
        ds4_csa_compress_forward(
            comp_kv,
            comp_end,
            sparse_mask,
            hidden,
            q_mid,
            T,
            cfg,
            csa_w_kv,
            csa_w_gate,
            csa_pos_bias,
            csa_norm,
            idx_wq_b,
            idx_w_weights,
            idx_w_kv,
            idx_w_gate,
            idx_pos_bias,
            idx_norm,
            &n_comp);
    }

    ds4_rope_cos_sin_buffer(cos_buf, sin_buf, T, rope_dim, cfg->rope_theta);

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *qm = q_mid + (size_t)t * (size_t)r;
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
        ds4_rmsnorm_forward_cuda(kvt, tmp, w_kv_norm, 1, D, eps);
        ds4_apply_partial_rope_vec_t(
            kvt, D, rope_dim, cos_buf + (size_t)t * (size_t)rope_half, sin_buf + (size_t)t * (size_t)rope_half);
    }

    const int Tk = T + n_comp;
    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            memcpy(
                keys + ((size_t)h * (size_t)Tk + (size_t)t) * (size_t)D,
                kv + (size_t)t * (size_t)D,
                (size_t)D * sizeof(float));
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
            int ok = (comp_end[k] <= tq);
            if (attn_type == DS4_ATTN_CSA) {
                ok = ok && sparse_mask[(size_t)tq * (size_t)n_comp + (size_t)k];
            }
            mask[(size_t)tq * (size_t)Tk + (size_t)T + (size_t)k] = ok;
        }
    }

    ds4_core_attention_cuda(context, q_heads, keys, NH, T, Tk, D, attn_sink, mask);

    for (int t = 0; t < T; t++) {
        float *row = ctx_flat + (size_t)t * (size_t)attn_w;
        for (int h = 0; h < NH; h++) {
            memcpy(row + (size_t)h * (size_t)D, context + ((size_t)h * (size_t)T + (size_t)t) * (size_t)D, (size_t)D * sizeof(float));
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
