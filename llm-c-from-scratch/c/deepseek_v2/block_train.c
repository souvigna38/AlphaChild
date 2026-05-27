/*
 * Block forward_train + backward (Phase 5b) — notebook 14 / 17
 */
#include "block.h"

#include "../rmsnorm.h"

#include <string.h>

size_t dsv2_block_train_scratch_bytes(const Dsv2BlockConfig *cfg, int T) {
    int C = cfg->attn.n_embd;
    size_t n = (size_t)T * (size_t)C * 10; /* 6 forward + 4 backward temps */
    n += dsv2_mla_train_scratch_bytes(&cfg->attn, T) / sizeof(float);
    n += dsv2_moe_train_scratch_bytes(&cfg->moe, T) / sizeof(float);
    return n * sizeof(float);
}

void dsv2_block_forward_train(
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
    int *moe_topi,
    float *scratch) {
    const int C = cfg->attn.n_embd;
    const int n = T * C;
    size_t off = 0;
    float *x_in = scratch + off;
    off += (size_t)n;
    float *ln1 = scratch + off;
    off += (size_t)n;
    float *attn_out = scratch + off;
    off += (size_t)n;
    float *post_attn = scratch + off;
    off += (size_t)n;
    float *ln2 = scratch + off;
    off += (size_t)n;
    float *moe_out = scratch + off;
    off += (size_t)n;
    float *mla_scr = scratch + off;
    off += dsv2_mla_train_scratch_bytes(&cfg->attn, T) / sizeof(float);
    float *moe_scr = scratch + off;

    memcpy(x_in, x, (size_t)n * sizeof(float));
    ds4_rmsnorm_forward(ln1, x_in, (float *)rms1_weight, T, C, cfg->rms_eps);
    dsv2_mla_forward_train(attn_out, ln1, &cfg->attn, T, wq, w_dkv, w_uk, w_uv, wo, mla_scr);
    memcpy(post_attn, x_in, (size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        post_attn[i] += attn_out[i];
    }
    ds4_rmsnorm_forward(ln2, post_attn, (float *)rms2_weight, T, C, cfg->rms_eps);
    dsv2_moe_forward_train(moe_out, ln2, &cfg->moe, T, gate_w, expert_w1, expert_w2, expert_w3, shared_w1, shared_w2, shared_w3, moe_topi, moe_scr);
    memcpy(out, post_attn, (size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        out[i] += moe_out[i];
    }
}

void dsv2_block_backward(
    float *dx_in,
    float *drms1,
    float *drms2,
    float *dwq,
    float *dw_dkv,
    float *dw_uk,
    float *dw_uv,
    float *dwo,
    float *dw_gate,
    float *dw1,
    float *dw2,
    float *dw3,
    float *dsw1,
    float *dsw2,
    float *dsw3,
    const float *dout,
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
    const int *moe_topi,
    float *scratch) {
    const int C = cfg->attn.n_embd;
    const int n = T * C;
    size_t off = 0;
    float *x_in = scratch + off;
    off += (size_t)n;
    float *ln1 = scratch + off;
    off += (size_t)n;
    float *attn_out = scratch + off;
    off += (size_t)n;
    float *post_attn = scratch + off;
    off += (size_t)n;
    float *ln2 = scratch + off;
    off += (size_t)n;
    float *mla_scr = scratch + off;
    off += dsv2_mla_train_scratch_bytes(&cfg->attn, T) / sizeof(float);
    float *moe_scr = scratch + off;
    off += dsv2_moe_train_scratch_bytes(&cfg->moe, T) / sizeof(float);
    float *dln2 = scratch + off;
    off += (size_t)n;
    float *dattn = scratch + off;
    off += (size_t)n;
    float *dpost_ln2 = scratch + off;
    off += (size_t)n;
    float *dln1 = scratch + off;

    /* out = post_attn + moe(ln2); post_attn = x_in + attn (matches block.c / PyTorch) */
    memset(dln2, 0, (size_t)n * sizeof(float));
    dsv2_moe_backward(dln2, dw_gate, dw1, dw2, dw3, dsw1, dsw2, dsw3, dout, &cfg->moe, T, gate_w, expert_w1, expert_w2, expert_w3, shared_w1, shared_w2, shared_w3, moe_topi, moe_scr);

    memset(dpost_ln2, 0, (size_t)n * sizeof(float));
    ds4_rmsnorm_backward(dpost_ln2, drms2, dln2, post_attn, ln2, rms2_weight, T, C, cfg->rms_eps);

    memcpy(dattn, dout, (size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        dattn[i] += dpost_ln2[i];
    }

    memset(dln1, 0, (size_t)n * sizeof(float));
    dsv2_mla_backward(dln1, dwq, dw_dkv, dw_uk, dw_uv, dwo, dattn, &cfg->attn, T, wq, w_dkv, w_uk, w_uv, wo, mla_scr);

    memset(dx_in, 0, (size_t)n * sizeof(float));
    ds4_rmsnorm_backward(dx_in, drms1, dln1, x_in, ln1, rms1_weight, T, C, cfg->rms_eps);
    for (int i = 0; i < n; i++) {
        dx_in[i] += dattn[i];
    }
    (void)attn_out;
}
