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

#include <math.h>
#include <stddef.h>
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
    int C = cfg->n_embd;
    int NH = cfg->n_head;
    int hs = C / NH;
    int r = cfg->kv_lora_rank;
    /* q_flat, k_heads, v_heads, preatt (NH*T*T), att same, head_out */
    size_t n = 0;
    n += (size_t)T * C;           /* q per token flat */
    n += (size_t)NH * T * hs * 2; /* k and v per head */
    n += (size_t)r;               /* c_kv one token */
    n += (size_t)NH * T * T * 2;  /* preatt + att */
    n += (size_t)C;               /* head merge buffer */
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
