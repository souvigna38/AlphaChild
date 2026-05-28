#include "hash_moe_train.h"

#include "swiglu.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

size_t ds4_hash_moe_train_cache_floats(const DeepSeekV4Config *cfg) {
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    /* gate_up_act: 2*I, h_act: I */
    size_t per_expert = 3 * (size_t)I;
    return (size_t)E + (size_t)k + (size_t)k + (size_t)k * per_expert + (size_t)S * per_expert;
}

void ds4_hash_moe_forward_train(
    float *out,
    const float *x,
    int token_id,
    const DeepSeekV4Config *cfg,
    const int *tid2eid,
    const float *gate_w,
    const float *expert_gate_up,
    const float *expert_down,
    const float *shared_gate_up,
    const float *shared_down,
    float *cache) {
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int C = cfg->hidden_size;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t gu_stride = (size_t)I * 2 * (size_t)C;
    const size_t dn_stride = (size_t)C * (size_t)I;
    const size_t per_expert = 3 * (size_t)I;

    float *affinity = cache;
    size_t hdr = (size_t)E + (size_t)k + (size_t)k;
    int *topi = (int *)(affinity + E);
    float *topw = affinity + hdr;
    float *expert_cache = topw + k;

    for (int e = 0; e < E; e++) {
        float logit = 0.0f;
        const float *row = gate_w + (size_t)e * (size_t)C;
        for (int c = 0; c < C; c++) {
            logit += row[c] * x[c];
        }
        affinity[e] = ds4_sqrt_softplus(logit);
    }
    float sumw = 0.0f;
    for (int j = 0; j < k; j++) {
        topi[j] = tid2eid[(size_t)token_id * (size_t)k + (size_t)j];
        topw[j] = affinity[topi[j]];
        sumw += topw[j];
    }
    if (sumw > 0.0f) {
        for (int j = 0; j < k; j++) {
            topw[j] = (topw[j] / sumw) * cfg->routed_scaling_factor;
        }
    }

    memset(out, 0, (size_t)C * sizeof(float));
    for (int j = 0; j < k; j++) {
        int e = topi[j];
        float *ec = expert_cache + (size_t)j * per_expert;
        float *gu_act = ec;
        float *h_act = ec + 2 * (size_t)I;
        float tmp[256];
        const float *egu = expert_gate_up + (size_t)e * gu_stride;
        const float *ed = expert_down + (size_t)e * dn_stride;
        ds4_swiglu_forward_save(tmp, x, C, I, egu, ed, cfg->swiglu_limit, gu_act, h_act);
        for (int c = 0; c < C; c++) {
            out[c] += topw[j] * tmp[c];
        }
    }
    for (int s = 0; s < S; s++) {
        float *sc = expert_cache + (size_t)k * per_expert + (size_t)s * per_expert;
        float tmp[256];
        const float *sgu = shared_gate_up + (size_t)s * gu_stride;
        const float *sd = shared_down + (size_t)s * dn_stride;
        ds4_swiglu_forward_save(tmp, x, C, I, sgu, sd, cfg->swiglu_limit, sc, sc + 2 * (size_t)I);
        for (int c = 0; c < C; c++) {
            out[c] += tmp[c];
        }
    }
}

void ds4_hash_moe_backward(
    float *dx,
    float *d_gate_w,
    float *d_expert_gu,
    float *d_expert_down,
    float *d_shared_gu,
    float *d_shared_down,
    const float *dout,
    const float *x,
    int token_id,
    const DeepSeekV4Config *cfg,
    const int *tid2eid,
    const float *gate_w,
    const float *expert_gate_up,
    const float *expert_down,
    const float *shared_gate_up,
    const float *shared_down,
    float *cache,
    float swiglu_limit) {
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int C = cfg->hidden_size;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t gu_stride = (size_t)I * 2 * (size_t)C;
    const size_t dn_stride = (size_t)C * (size_t)I;
    const size_t per_expert = 3 * (size_t)I;

    float *affinity = cache;
    size_t hdr = (size_t)E + (size_t)k + (size_t)k;
    int *topi = (int *)(affinity + E);
    float *topw = affinity + hdr;
    float *expert_cache = topw + k;

    memset(dx, 0, (size_t)C * sizeof(float));
    for (int j = 0; j < k; j++) {
        int e = topi[j];
        float w = topw[j];
        float *ec = expert_cache + (size_t)j * per_expert;
        float dtmp[256];
        float dx_ex[256];
        memset(dx_ex, 0, (size_t)C * sizeof(float));
        for (int c = 0; c < C; c++) {
            dtmp[c] = dout[c] * w;
        }
        const float *egu = expert_gate_up + (size_t)e * gu_stride;
        const float *ed = expert_down + (size_t)e * dn_stride;
        ds4_swiglu_backward(
            dx_ex,
            d_expert_gu + (size_t)e * gu_stride,
            d_expert_down + (size_t)e * dn_stride,
            dtmp,
            x,
            C,
            I,
            egu,
            ed,
            ec,
            ec + 2 * (size_t)I,
            swiglu_limit);
        for (int c = 0; c < C; c++) {
            dx[c] += dx_ex[c];
        }
    }
    for (int s = 0; s < S; s++) {
        float *sc = expert_cache + (size_t)k * per_expert + (size_t)s * per_expert;
        float dx_sh[256];
        memset(dx_sh, 0, (size_t)C * sizeof(float));
        const float *sgu = shared_gate_up + (size_t)s * gu_stride;
        const float *sd = shared_down + (size_t)s * dn_stride;
        ds4_swiglu_backward(
            dx_sh,
            d_shared_gu + (size_t)s * gu_stride,
            d_shared_down + (size_t)s * dn_stride,
            dout,
            x,
            C,
            I,
            sgu,
            sd,
            sc,
            sc + 2 * (size_t)I,
            swiglu_limit);
        for (int c = 0; c < C; c++) {
            dx[c] += dx_sh[c];
        }
    }
    (void)d_gate_w;
    (void)gate_w;
    (void)tid2eid;
    (void)token_id;
    (void)affinity;
}
