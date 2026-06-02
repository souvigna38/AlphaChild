/*
 * Hash-MoE bootstrap layer (DeepSeek-V4) — static tid→expert table + sqrt(softplus(gate)) weights.
 * Reference: vendor/nano-deepseek-v4/nano_deepseek_v4/modeling.py DeepSeekV4MoE
 */
#include "hash_moe.h"

#include "ds4_cuda.h"
#include "swiglu.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static void linear_logits(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
}

static float sqrt_softplus(float x) {
    if (x > 20.0f) {
        return sqrtf(x);
    }
    return sqrtf(logf(1.0f + expf(x)));
}

void ds4_hash_moe_build_table(int *tid2eid, const DeepSeekV4Config *cfg) {
    const int V = cfg->vocab_size;
    const int E = cfg->n_routed_experts;
    const int k = cfg->num_experts_per_tok;
    for (int t = 0; t < V; t++) {
        for (int j = 0; j < k; j++) {
            unsigned int h = (unsigned int)t * 1103515245u + 12345u + (unsigned int)j * 2654435761u;
            tid2eid[(size_t)t * (size_t)k + (size_t)j] = (int)(h % (unsigned int)E);
        }
    }
}

void ds4_hash_moe_forward_token(
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
    float *scratch) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int k = cfg->num_experts_per_tok;
    const int S = cfg->n_shared_experts;
    const size_t gate_up_stride = (size_t)I * 2 * (size_t)C;
    const size_t down_stride = (size_t)C * (size_t)I;

    float logits[32];
    float weights[32];
    if (E > 32 || k > 32) {
        return;
    }

    linear_logits(gate_w, x, logits, E, C);
    memset(out, 0, (size_t)C * sizeof(float));

    float sumw = 0.0f;
    for (int j = 0; j < k; j++) {
        int e = tid2eid[(size_t)token_id * (size_t)k + (size_t)j];
        weights[j] = sqrt_softplus(logits[e]);
        sumw += weights[j];
    }
    if (sumw > 0.0f) {
        for (int j = 0; j < k; j++) {
            weights[j] = (weights[j] / sumw) * cfg->routed_scaling_factor;
        }
    }

    float tmp[256];
    for (int j = 0; j < k; j++) {
        int e = tid2eid[(size_t)token_id * (size_t)k + (size_t)j];
        const float *egu = expert_gate_up + (size_t)e * gate_up_stride;
        const float *ed = expert_down + (size_t)e * down_stride;
        ds4_swiglu_forward_cuda(tmp, x, C, I, egu, ed, cfg->swiglu_limit);
        for (int c = 0; c < C; c++) {
            out[c] += weights[j] * tmp[c];
        }
    }

    for (int s = 0; s < S; s++) {
        const float *sgu = shared_gate_up + (size_t)s * gate_up_stride;
        const float *sd = shared_down + (size_t)s * down_stride;
        ds4_swiglu_forward_cuda(tmp, x, C, I, sgu, sd, cfg->swiglu_limit);
        for (int c = 0; c < C; c++) {
            out[c] += tmp[c];
        }
    }
    (void)scratch;
}
