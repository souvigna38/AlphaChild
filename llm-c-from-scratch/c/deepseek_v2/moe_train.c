/*
 * MoE forward_train + backward (Phase 5b)
 */
#include "moe.h"
#include "ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static size_t moe_token_cache_floats(const Dsv2MoeConfig *cfg) {
    int E = cfg->n_routed_experts;
    int C = cfg->n_embd;
    int I = cfg->moe_intermediate;
    int k = cfg->num_experts_per_tok;
    int S = cfg->n_shared_experts;
    return (size_t)E * 2 + (size_t)C + (size_t)k + (size_t)k * ((size_t)I * 3 + (size_t)C) + (size_t)S * ((size_t)I * 3 + (size_t)C);
}

static float *moe_token_ptr(float *scratch, const Dsv2MoeConfig *cfg, int T, int t) {
    return scratch + (size_t)t * moe_token_cache_floats(cfg);
}

static void topk_local(const float *probs, int *out_idx, float *out_w, int n, int k) {
    int used[32];
    if (n > 32) {
        n = 32;
    }
    for (int i = 0; i < n; i++) {
        used[i] = 0;
    }
    for (int j = 0; j < k; j++) {
        int best = 0;
        float bestv = -1.0f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && probs[i] > bestv) {
                bestv = probs[i];
                best = i;
            }
        }
        out_idx[j] = best;
        out_w[j] = bestv;
        used[best] = 1;
    }
    float s = 0.0f;
    for (int j = 0; j < k; j++) {
        s += out_w[j];
    }
    if (s > 0.0f) {
        for (int j = 0; j < k; j++) {
            out_w[j] /= s;
        }
    }
}

static void swiglu_fwd(
    float *out,
    const float *x,
    int C,
    int I,
    const float *w1,
    const float *w2,
    const float *w3,
    float *h1_pre,
    float *h3,
    float *h) {
    dsv2_linear_forward(w1, x, h1_pre, I, C);
    dsv2_linear_forward(w3, x, h3, I, C);
    for (int i = 0; i < I; i++) {
        h[i] = (h1_pre[i] / (1.0f + expf(-h1_pre[i]))) * h3[i];
    }
    dsv2_linear_forward(w2, h, out, C, I);
}

static void swiglu_bwd(
    float *dx,
    float *dw1,
    float *dw2,
    float *dw3,
    const float *dout,
    const float *x,
    int C,
    int I,
    const float *h1_pre,
    const float *h3,
    const float *h) {
    float dh[256];
    float dh1[256];
    float dh3[256];
    if (I > 256 || C > 256) {
        return;
    }
    memset(dh, 0, (size_t)I * sizeof(float));
    dsv2_linear_backward(dx, dw2, dout, h, C, I);
    for (int i = 0; i < I; i++) {
        float silu = h1_pre[i] / (1.0f + expf(-h1_pre[i]));
        dh3[i] = dh[i] * silu;
        float dsilu = h3[i] * dh[i];
        float s = 1.0f / (1.0f + expf(-h1_pre[i]));
        dh1[i] = dsilu * (s + h1_pre[i] * s * (1.0f - s));
    }
    dsv2_linear_backward(dx, dw3, dh3, x, I, C);
    dsv2_linear_backward(dx, dw1, dh1, x, I, C);
}

size_t dsv2_moe_train_scratch_bytes(const Dsv2MoeConfig *cfg, int T) {
    return moe_token_cache_floats(cfg) * (size_t)T * sizeof(float);
}

void dsv2_moe_forward_train(
    float *out,
    const float *x,
    const Dsv2MoeConfig *cfg,
    int T,
    const float *gate_w,
    const float *expert_w1,
    const float *expert_w2,
    const float *expert_w3,
    const float *shared_w1,
    const float *shared_w2,
    const float *shared_w3,
    int *topi,
    float *scratch) {
    const int C = cfg->n_embd;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t w1s = (size_t)I * (size_t)C;
    const size_t w2s = (size_t)C * (size_t)I;
    const size_t es = (size_t)I * 3 + (size_t)C;

    for (int t = 0; t < T; t++) {
        float *tok = moe_token_ptr(scratch, cfg, T, t);
        float *logits = tok;
        float *probs = tok + E;
        float *xsave = probs + E;
        float *topw = xsave + C;
        float *blob = topw + k;
        const float *xt = x + (size_t)t * (size_t)C;
        float *yt = out + (size_t)t * (size_t)C;
        int ti[32];
        float tw[32];

        memcpy(xsave, xt, (size_t)C * sizeof(float));
        dsv2_linear_forward(gate_w, xt, logits, E, C);
        dsv2_softmax_forward(logits, probs, E);
        topk_local(probs, ti, tw, E, k);
        for (int j = 0; j < k; j++) {
            topi[t * k + j] = ti[j];
            topw[j] = tw[j];
        }

        memset(yt, 0, (size_t)C * sizeof(float));
        for (int j = 0; j < k; j++) {
            int e = ti[j];
            float *h1 = blob + (size_t)j * es;
            float *h3 = h1 + I;
            float *h = h3 + I;
            float *tmp = h + I;
            swiglu_fwd(tmp, xt, C, I, expert_w1 + (size_t)e * w1s, expert_w2 + (size_t)e * w2s, expert_w3 + (size_t)e * w1s, h1, h3, h);
            for (int c = 0; c < C; c++) {
                yt[c] += tw[j] * tmp[c];
            }
        }
        for (int s = 0; s < S; s++) {
            float *h1 = blob + (size_t)k * es + (size_t)s * es;
            float *h3 = h1 + I;
            float *h = h3 + I;
            float *tmp = h + I;
            swiglu_fwd(tmp, xt, C, I, shared_w1 + (size_t)s * w1s, shared_w2 + (size_t)s * w2s, shared_w3 + (size_t)s * w1s, h1, h3, h);
            for (int c = 0; c < C; c++) {
                yt[c] += tmp[c];
            }
        }
    }
}

void dsv2_moe_backward(
    float *dx,
    float *dw_gate,
    float *dw1,
    float *dw2,
    float *dw3,
    float *dsw1,
    float *dsw2,
    float *dsw3,
    const float *dout,
    const Dsv2MoeConfig *cfg,
    int T,
    const float *gate_w,
    const float *expert_w1,
    const float *expert_w2,
    const float *expert_w3,
    const float *shared_w1,
    const float *shared_w2,
    const float *shared_w3,
    const int *topi,
    float *scratch) {
    const int C = cfg->n_embd;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t w1s = (size_t)I * (size_t)C;
    const size_t w2s = (size_t)C * (size_t)I;
    const size_t es = (size_t)I * 3 + (size_t)C;

    (void)gate_w;
    (void)expert_w1;
    (void)expert_w2;
    (void)expert_w3;

    for (int t = 0; t < T; t++) {
        float *tok = moe_token_ptr(scratch, cfg, T, t);
        float *logits = tok;
        float *probs = tok + E;
        float *xsave = probs + E;
        float *topw = xsave + C;
        float *blob = topw + k;
        float *dx_t = dx + (size_t)t * (size_t)C;
        const float *dt = dout + (size_t)t * (size_t)C;
        float dprobs[32];
        float dlogits[32];
        float dtmp[256];

        if (E > 32 || C > 256 || I > 256) {
            return;
        }

        memset(dx_t, 0, (size_t)C * sizeof(float));
        memset(dprobs, 0, (size_t)E * sizeof(float));

        for (int s = 0; s < S; s++) {
            float *h1 = blob + (size_t)k * es + (size_t)s * es;
            float *h3 = h1 + I;
            float *h = h3 + I;
            swiglu_bwd(dx_t, dsw1 + (size_t)s * w1s, dsw2 + (size_t)s * w2s, dsw3 + (size_t)s * w1s, dt, xsave, C, I, h1, h3, h);
        }

        for (int j = 0; j < k; j++) {
            int e = topi[t * k + j];
            float w = topw[j]; /* saved in forward_train */
            float *h1 = blob + (size_t)j * es;
            float *h3 = h1 + I;
            float *h = h3 + I;
            float *tmp = h + I;
            float dot = 0.0f;
            for (int c = 0; c < C; c++) {
                dtmp[c] = w * dt[c];
                dot += dt[c] * tmp[c];
            }
            dprobs[e] += dot;
            swiglu_bwd(dx_t, dw1 + (size_t)e * w1s, dw2 + (size_t)e * w2s, dw3 + (size_t)e * w1s, dtmp, xsave, C, I, h1, h3, h);
        }

        memset(dlogits, 0, (size_t)E * sizeof(float));
        dsv2_softmax_backward(dlogits, probs, dprobs, E);
        dsv2_linear_backward(dx_t, dw_gate, dlogits, xsave, E, C);
    }
}
