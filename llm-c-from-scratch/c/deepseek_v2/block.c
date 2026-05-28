/*
 * DeepSeek-V2 block — educational glue (notebook 14).
 *
 * PyTorch (llmc/deepseek_v2.py):
 *   x = x + attn(ln1(x))
 *   x = x + moe(ln2(x))
 *
 * llm.c GPT-2 block: layernorm + attention + residual + layernorm + MLP + residual
 *   We swap LayerNorm->RMSNorm, MHA->MLA, dense MLP->MoE.
 */
#include "block.h"

#include "../rmsnorm.h"

#include <string.h>

size_t dsv2_block_scratch_bytes(const Dsv2BlockConfig *cfg, int T) {
    int C = cfg->attn.n_embd;
    size_t floats = (size_t)T * (size_t)C * 5; /* ln1, attn, res, ln2, moe_out */
    floats += dsv2_mla_scratch_bytes(&cfg->attn, T) / sizeof(float);
    floats += dsv2_moe_scratch_bytes(&cfg->moe, T) / sizeof(float);
    return floats * sizeof(float);
}

static void vec_add(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

void dsv2_block_forward(
    float *out,
    const float *x,
    const Dsv2BlockConfig *cfg,
    int T,
    const float *rms1_weight,
    const float *rms2_weight,
    const float *wq,
    const float *w_dkv,
    const float *w_uk,
    const float *w_uv,
    const float *wo,
    const float *gate_w,
    const float *expert_w1,
    const float *expert_w2,
    const float *expert_w3,
    const float *shared_w1,
    const float *shared_w2,
    const float *shared_w3,
    float *scratch) {
    const int C = cfg->attn.n_embd;
    const int n = T * C;

    size_t off = 0;
    float *ln1_out = scratch + off;
    off += (size_t)n;
    float *attn_out = scratch + off;
    off += (size_t)n;
    float *res1 = scratch + off;
    off += (size_t)n;
    float *ln2_out = scratch + off;
    off += (size_t)n;
    float *moe_out = scratch + off;
    off += (size_t)n;
    float *mla_scratch = scratch + off;
    off += dsv2_mla_scratch_bytes(&cfg->attn, T) / sizeof(float);
    float *moe_scratch = scratch + off;

    memcpy(res1, x, (size_t)n * sizeof(float));

    ds4_rmsnorm_forward(ln1_out, (float *)x, rms1_weight, T, C, cfg->rms_eps);
    dsv2_mla_forward(attn_out, NULL, ln1_out, &cfg->attn, T, wq, w_dkv, w_uk, w_uv, wo, mla_scratch);
    vec_add(res1, attn_out, n);

    ds4_rmsnorm_forward(ln2_out, res1, rms2_weight, T, C, cfg->rms_eps);
    dsv2_moe_forward(
        moe_out,
        ln2_out,
        &cfg->moe,
        T,
        gate_w,
        expert_w1,
        expert_w2,
        expert_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        NULL,
        NULL,
        moe_scratch);
    vec_add(res1, moe_out, n);

    memcpy(out, res1, (size_t)n * sizeof(float));
}
