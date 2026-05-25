/*
 * MLA forward — heavily commented for learning.
 *
 * Reading guide:
 *   1) Skim attention_forward() in vendor/llm.c/train_gpt2.c (GPT-2 MHA).
 *   2) Read this file — same causal softmax pattern, but K/V come from latent c_kv.
 *   3) Run: make test_mla && ./bin/test_mla
 *   4) Compare shapes printed here to notebook 12 (PyTorch).
 */
#include "mla.h"

#include "ops.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Small helpers (no external BLAS — every line visible)                       */
/* -------------------------------------------------------------------------- */

/*
 * y = W @ x
 * W: (out_dim, in_dim) row-major — PyTorch Linear weight shape
 * x: (in_dim)
 * y: (out_dim)
 */
static void linear(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const float *row = W + o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += row[i] * x[i];
        }
        y[o] = sum;
    }
}

/* Dot product of two length-d vectors */
static float dot(const float *a, const float *b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) {
        s += a[i] * b[i];
    }
    return s;
}

/* -------------------------------------------------------------------------- */
/* Public size helpers                                                         */
/* -------------------------------------------------------------------------- */

size_t dsv2_mla_c_kv_bytes(const Dsv2MlaConfig *cfg, int T) {
    return (size_t)T * (size_t)cfg->kv_lora_rank * sizeof(float);
}

size_t dsv2_mla_scratch_bytes(const Dsv2MlaConfig *cfg, int T) {
    return dsv2_mla_train_scratch_bytes(cfg, T);
}

size_t dsv2_mla_train_scratch_bytes(const Dsv2MlaConfig *cfg, int T) {
    int C = cfg->n_embd;
    int NH = cfg->n_head;
    int hs = C / NH;
    int r = cfg->kv_lora_rank;
    size_t n = 0;
    n += (size_t)T * C;             /* x_in */
    n += (size_t)T * C;             /* q */
    n += (size_t)NH * T * hs * 2;   /* k,v heads */
    n += (size_t)T * (size_t)r;     /* c_kv all tokens */
    n += (size_t)NH * T * T * 2;    /* preatt, att */
    n += (size_t)T * C;             /* pre_wo */
    n += (size_t)C;                 /* head tmp */
    return n * sizeof(float);
}

size_t dsv2_mla_kv_cache_bytes_per_token(const Dsv2MlaConfig *cfg) {
    return (size_t)cfg->kv_lora_rank * sizeof(float);
}

size_t dsv2_mha_kv_cache_bytes_per_token(const Dsv2MlaConfig *cfg) {
    int hs = cfg->n_embd / cfg->n_head;
    return (size_t)(2 * cfg->n_head * hs) * sizeof(float);
}

/* -------------------------------------------------------------------------- */
/* MLA forward (B=1 only in this educational port)                             */
/* -------------------------------------------------------------------------- */

void dsv2_mla_forward(
    float *out,
    float *c_kv_out,
    const float *x,
    const Dsv2MlaConfig *cfg,
    int T,
    const float *wq,
    const float *w_dkv,
    const float *w_uk,
    const float *w_uv,
    const float *wo,
    float *scratch) {
    const int C = cfg->n_embd;
    const int NH = cfg->n_head;
    const int hs = C / NH;
    const int r = cfg->kv_lora_rank;
    const float scale = 1.0f / sqrtf((float)hs);

    float *q_flat = scratch;
    float *k_heads = q_flat + T * C;
    float *v_heads = k_heads + NH * T * hs;
    float *c_kv_buf = v_heads + NH * T * hs;
    float *preatt = c_kv_buf + r;
    float *att = preatt + NH * T * T;
    float *head_out = att + NH * T * T;

    memset(out, 0, (size_t)T * C * sizeof(float));

    /*
     * Pass A — build Q, latent c_kv, and per-head K/V for every time step.
     *
     * GPT-2 (llm.c): one matmul produces Q,K,V at once (3C wide).
     * MLA: separate paths — Q direct, KV through low-rank bottleneck c_kv.
     */
    for (int t = 0; t < T; t++) {
        const float *xt = x + t * C;
        float *qt = q_flat + t * C;

        linear(wq, xt, qt, C, C);

        /* Down-project: c_kv = W_dkv @ x  shape (r,) */
        linear(w_dkv, xt, c_kv_buf, r, C);
        if (c_kv_out != NULL) {
            memcpy(c_kv_out + t * r, c_kv_buf, (size_t)r * sizeof(float));
        }

        /* Up-project latent to full-width keys/values, then split heads */
        float k_full[C];
        float v_full[C];
        linear(w_uk, c_kv_buf, k_full, C, r);
        linear(w_uv, c_kv_buf, v_full, C, r);

        for (int h = 0; h < NH; h++) {
            memcpy(k_heads + h * T * hs + t * hs, k_full + h * hs, (size_t)hs * sizeof(float));
            memcpy(v_heads + h * T * hs + t * hs, v_full + h * hs, (size_t)hs * sizeof(float));
        }
    }

    /*
     * Pass B — causal attention per head (same structure as llm.c attention_forward).
     *
     * For each (t, h): softmax over t2 <= t of (q_t · k_{t2}) / sqrt(hs), then sum v_{t2}.
     */
    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            const float *qt = q_flat + t * C + h * hs;
            float *preatt_bth = preatt + h * T * T + t * T;
            float *att_bth = att + h * T * T + t * T;

            float maxval = -1e9f;
            for (int t2 = 0; t2 <= t; t2++) {
                const float *kt2 = k_heads + h * T * hs + t2 * hs;
                float val = dot(qt, kt2, hs) * scale;
                preatt_bth[t2] = val;
                if (val > maxval) {
                    maxval = val;
                }
            }

            float expsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                float ev = expf(preatt_bth[t2] - maxval);
                att_bth[t2] = ev;
                expsum += ev;
            }
            float inv = expsum > 0.0f ? 1.0f / expsum : 0.0f;
            for (int t2 = 0; t2 < T; t2++) {
                att_bth[t2] = (t2 <= t) ? att_bth[t2] * inv : 0.0f;
            }

            memset(head_out, 0, (size_t)hs * sizeof(float));
            for (int t2 = 0; t2 <= t; t2++) {
                const float *vt2 = v_heads + h * T * hs + t2 * hs;
                float w = att_bth[t2];
                for (int i = 0; i < hs; i++) {
                    head_out[i] += w * vt2[i];
                }
            }

            memcpy(out + t * C + h * hs, head_out, (size_t)hs * sizeof(float));
        }
    }

    /*
     * Pass C — output projection wo @ out (per token), like GPT-2 attn output linear.
     */
    for (int t = 0; t < T; t++) {
        float tmp[C];
        linear(wo, out + t * C, tmp, C, C);
        memcpy(out + t * C, tmp, (size_t)C * sizeof(float));
    }
}

void dsv2_mla_forward_train(
    float *out,
    const float *x,
    const Dsv2MlaConfig *cfg,
    int T,
    const float *wq,
    const float *w_dkv,
    const float *w_uk,
    const float *w_uv,
    const float *wo,
    float *scratch) {
    const int C = cfg->n_embd;
    float *x_in = scratch;
    float *q_flat = x_in + (size_t)T * (size_t)C;
    float *k_heads = q_flat + (size_t)T * (size_t)C;
    const int NH = cfg->n_head;
    const int hs = C / NH;
    const int r = cfg->kv_lora_rank;
    float *v_heads = k_heads + (size_t)NH * (size_t)T * (size_t)hs;
    float *c_kv_all = v_heads + (size_t)NH * (size_t)T * (size_t)hs;
    float *preatt = c_kv_all + (size_t)T * (size_t)r;
    float *att = preatt + (size_t)NH * (size_t)T * (size_t)T;
    float *pre_wo = att + (size_t)NH * (size_t)T * (size_t)T;
    float *head_out = pre_wo + (size_t)T * (size_t)C;
    const float scale = 1.0f / sqrtf((float)hs);

    memset(pre_wo, 0, (size_t)T * (size_t)C * sizeof(float));
    for (int t = 0; t < T; t++) {
        dsv2_linear_forward(wq, x_in + (size_t)t * (size_t)C, q_flat + (size_t)t * (size_t)C, C, C);
        dsv2_linear_forward(w_dkv, x_in + (size_t)t * (size_t)C, c_kv_all + (size_t)t * (size_t)r, r, C);
        float k_full[C], v_full[C];
        dsv2_linear_forward(w_uk, c_kv_all + (size_t)t * (size_t)r, k_full, C, r);
        dsv2_linear_forward(w_uv, c_kv_all + (size_t)t * (size_t)r, v_full, C, r);
        for (int h = 0; h < NH; h++) {
            memcpy(k_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t * (size_t)hs, k_full + (size_t)h * (size_t)hs, (size_t)hs * sizeof(float));
            memcpy(v_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t * (size_t)hs, v_full + (size_t)h * (size_t)hs, (size_t)hs * sizeof(float));
        }
    }
    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            const float *qt = q_flat + (size_t)t * (size_t)C + (size_t)h * (size_t)hs;
            float *preatt_bth = preatt + (size_t)h * (size_t)T * (size_t)T + (size_t)t * (size_t)T;
            float *att_bth = att + (size_t)h * (size_t)T * (size_t)T + (size_t)t * (size_t)T;
            float maxv = -1e9f;
            for (int t2 = 0; t2 <= t; t2++) {
                float val = dot(q_flat + (size_t)t * (size_t)C + (size_t)h * (size_t)hs, k_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t2 * (size_t)hs, hs) * scale;
                preatt_bth[t2] = val;
                if (val > maxv) {
                    maxv = val;
                }
            }
            float expsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                float ev = expf(preatt_bth[t2] - maxv);
                att_bth[t2] = ev;
                expsum += ev;
            }
            float inv = expsum > 0.0f ? 1.0f / expsum : 0.0f;
            for (int t2 = 0; t2 < T; t2++) {
                att_bth[t2] = (t2 <= t) ? att_bth[t2] * inv : 0.0f;
            }
            memset(head_out, 0, (size_t)hs * sizeof(float));
            for (int t2 = 0; t2 <= t; t2++) {
                float w = att_bth[t2];
                const float *vt2 = v_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t2 * (size_t)hs;
                for (int i = 0; i < hs; i++) {
                    head_out[i] += w * vt2[i];
                }
            }
            memcpy(pre_wo + (size_t)t * (size_t)C + (size_t)h * (size_t)hs, head_out, (size_t)hs * sizeof(float));
        }
    }
    for (int t = 0; t < T; t++) {
        dsv2_linear_forward(wo, pre_wo + (size_t)t * (size_t)C, out + (size_t)t * (size_t)C, C, C);
    }
}

void dsv2_mla_backward(
    float *dx,
    float *dwq,
    float *dw_dkv,
    float *dw_uk,
    float *dw_uv,
    float *dwo,
    const float *dout,
    const Dsv2MlaConfig *cfg,
    int T,
    const float *wq,
    const float *w_dkv,
    const float *w_uk,
    const float *w_uv,
    const float *wo,
    float *scratch) {
    const int C = cfg->n_embd;
    const int NH = cfg->n_head;
    const int hs = C / NH;
    const int r = cfg->kv_lora_rank;
    const float scale = 1.0f / sqrtf((float)hs);

    float *x_in = scratch;
    float *q_flat = x_in + (size_t)T * (size_t)C;
    float *k_heads = q_flat + (size_t)T * (size_t)C;
    float *v_heads = k_heads + (size_t)NH * (size_t)T * (size_t)hs;
    float *c_kv_all = v_heads + (size_t)NH * (size_t)T * (size_t)hs;
    float *preatt = c_kv_all + (size_t)T * (size_t)r;
    float *att = preatt + (size_t)NH * (size_t)T * (size_t)T;
    float *pre_wo = att + (size_t)NH * (size_t)T * (size_t)T;

    float dpre_wo[T * C];
    memset(dx, 0, (size_t)T * (size_t)C * sizeof(float));
    memset(dpre_wo, 0, sizeof(dpre_wo));

    for (int t = 0; t < T; t++) {
        dsv2_linear_backward(dpre_wo + (size_t)t * (size_t)C, dwo, dout + (size_t)t * (size_t)C, pre_wo + (size_t)t * (size_t)C, C, C);
    }

    float *datt = (float *)calloc((size_t)NH * (size_t)T * (size_t)T, sizeof(float));
    float *dpreatt = (float *)calloc((size_t)NH * (size_t)T * (size_t)T, sizeof(float));
    float *dk_heads = (float *)calloc((size_t)NH * (size_t)T * (size_t)hs, sizeof(float));
    float *dv_heads = (float *)calloc((size_t)NH * (size_t)T * (size_t)hs, sizeof(float));

    for (int h = 0; h < NH; h++) {
        for (int t = 0; t < T; t++) {
            const float *att_bth = att + (size_t)h * (size_t)T * (size_t)T + (size_t)t * (size_t)T;
            float *datt_bth = datt + (size_t)h * (size_t)T * (size_t)T + (size_t)t * (size_t)T;
            float *dpreatt_bth = dpreatt + (size_t)h * (size_t)T * (size_t)T + (size_t)t * (size_t)T;
            const float *dout_h = dpre_wo + (size_t)t * (size_t)C + (size_t)h * (size_t)hs;

            for (int t2 = 0; t2 <= t; t2++) {
                const float *vt2 = v_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t2 * (size_t)hs;
                for (int i = 0; i < hs; i++) {
                    datt_bth[t2] += vt2[i] * dout_h[i];
                }
            }
            for (int t2 = 0; t2 <= t; t2++) {
                for (int t3 = 0; t3 <= t; t3++) {
                    float indicator = (t2 == t3) ? 1.0f : 0.0f;
                    dpreatt_bth[t3] += att_bth[t2] * (indicator - att_bth[t3]) * datt_bth[t2];
                }
            }
            const float *qt = q_flat + (size_t)t * (size_t)C + (size_t)h * (size_t)hs;
            for (int t2 = 0; t2 <= t; t2++) {
                float *dkt2 = dk_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t2 * (size_t)hs;
                float g = dpreatt_bth[t2] * scale;
                for (int i = 0; i < hs; i++) {
                    dkt2[i] += qt[i] * g;
                }
            }
            for (int t2 = 0; t2 <= t; t2++) {
                float *dvt2 = dv_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t2 * (size_t)hs;
                for (int i = 0; i < hs; i++) {
                    dvt2[i] += att_bth[t2] * dout_h[i];
                }
            }
        }
    }

    float *dk_full = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *dv_full = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < NH; h++) {
            memcpy(
                dk_full + (size_t)t * (size_t)C + (size_t)h * (size_t)hs,
                dk_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t * (size_t)hs,
                (size_t)hs * sizeof(float));
            memcpy(
                dv_full + (size_t)t * (size_t)C + (size_t)h * (size_t)hs,
                dv_heads + (size_t)h * (size_t)T * (size_t)hs + (size_t)t * (size_t)hs,
                (size_t)hs * sizeof(float));
        }
    }

    float *dc_kv = (float *)calloc((size_t)T * (size_t)r, sizeof(float));
    float *dq = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    for (int t = 0; t < T; t++) {
        dsv2_linear_backward(dc_kv + (size_t)t * (size_t)r, dw_uk, dk_full + (size_t)t * (size_t)C, c_kv_all + (size_t)t * (size_t)r, C, r);
        dsv2_linear_backward(dc_kv + (size_t)t * (size_t)r, dw_uv, dv_full + (size_t)t * (size_t)C, c_kv_all + (size_t)t * (size_t)r, C, r);
        dsv2_linear_backward(dx + (size_t)t * (size_t)C, dw_dkv, dc_kv + (size_t)t * (size_t)r, x_in + (size_t)t * (size_t)C, r, C);
        dsv2_linear_backward(dq + (size_t)t * (size_t)C, dwq, q_flat + (size_t)t * (size_t)C, x_in + (size_t)t * (size_t)C, C, C);
    }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            dx[(size_t)t * (size_t)C + (size_t)c] += dq[(size_t)t * (size_t)C + (size_t)c];
        }
    }

    free(dq);
    free(dk_full);
    free(dv_full);
    free(dc_kv);
    free(datt);
    free(dpreatt);
    free(dk_heads);
    free(dv_heads);
    (void)wq;
    (void)w_dkv;
    (void)w_uk;
    (void)w_uv;
    (void)wo;
}
