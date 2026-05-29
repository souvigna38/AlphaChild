/*
 * DeepSeek-V4 MoE — hash_moe (tid2eid) or routed (sqrt-softplus + top-k + bias).
 * Reference: nano_deepseek_v4/modeling.py DeepSeekV4MoE
 */
#include "moe.h"

#include "ds4_cuda.h"
#include "hash_moe.h"
#include "swiglu.h"
#include "v4_ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define DS4_MAX_E 32
#define DS4_MAX_K 8

size_t ds4_moe_scratch_bytes(const DeepSeekV4Config *cfg) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int k = cfg->num_experts_per_tok;
    size_t per = (size_t)E + (size_t)k + (size_t)I * 2 + (size_t)C;
    return per * sizeof(float);
}

static void moe_routed_token(
    float *out,
    const float *x,
    const DeepSeekV4Config *cfg,
    const float *gate_w,
    const float *route_bias,
    const float *expert_gate_up,
    const float *expert_down,
    const float *shared_gate_up,
    const float *shared_down,
    float *scratch) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t gu_stride = (size_t)I * 2 * (size_t)C;
    const size_t dn_stride = (size_t)C * (size_t)I;

    if (E > DS4_MAX_E || k > DS4_MAX_K) {
        return;
    }

    float *logits = scratch;
    float *affinity = logits + E;
    float *tmp = affinity + E;
    int topi[DS4_MAX_K];
    float topw[DS4_MAX_K];

    ds4_linear(gate_w, x, logits, E, C);
    float sumw = 0.0f;
    for (int e = 0; e < E; e++) {
        affinity[e] = ds4_sqrt_softplus(logits[e]);
        logits[e] = affinity[e] + (route_bias != NULL ? route_bias[e] : 0.0f);
    }
    ds4_topk_select(logits, E, k, topi, topw);
    for (int j = 0; j < k; j++) {
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
        const float *egu = expert_gate_up + (size_t)e * gu_stride;
        const float *ed = expert_down + (size_t)e * dn_stride;
        ds4_swiglu_forward_cuda(tmp, x, C, I, egu, ed, cfg->swiglu_limit);
        for (int c = 0; c < C; c++) {
            out[c] += topw[j] * tmp[c];
        }
    }
    for (int s = 0; s < S; s++) {
        const float *sgu = shared_gate_up + (size_t)s * gu_stride;
        const float *sd = shared_down + (size_t)s * dn_stride;
        ds4_swiglu_forward(tmp, x, C, I, sgu, sd, cfg->swiglu_limit);
        for (int c = 0; c < C; c++) {
            out[c] += tmp[c];
        }
    }
}

void ds4_moe_forward(
    float *out,
    const float *hidden,
    const int *input_ids,
    int T,
    Ds4MlpType mlp_type,
    const DeepSeekV4Config *cfg,
    const int *tid2eid,
    const float *gate_w,
    const float *route_bias,
    const float *expert_gate_up,
    const float *expert_down,
    const float *shared_gate_up,
    const float *shared_down,
    float *scratch) {
    const int C = cfg->hidden_size;
    const int I = cfg->moe_intermediate_size;
    float *tok_scratch = scratch;

    for (int t = 0; t < T; t++) {
        const float *xt = hidden + (size_t)t * (size_t)C;
        float *yt = out + (size_t)t * (size_t)C;
        if (mlp_type == DS4_MLP_HASH_MOE) {
            ds4_hash_moe_forward_token(
                yt,
                xt,
                input_ids[t],
                cfg,
                tid2eid,
                gate_w,
                expert_gate_up,
                expert_down,
                shared_gate_up,
                shared_down,
                tok_scratch);
        } else {
            moe_routed_token(
                yt,
                xt,
                cfg,
                gate_w,
                route_bias,
                expert_gate_up,
                expert_down,
                shared_gate_up,
                shared_down,
                tok_scratch);
        }
    }
    (void)I;
}
