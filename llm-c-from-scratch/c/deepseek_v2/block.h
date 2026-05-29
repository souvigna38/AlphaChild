/*
 * One DeepSeek-V2 transformer block (notebook 14) — MLA + MoE with pre-norm RMSNorm.
 */
#ifndef DEEPSEEK_V2_BLOCK_H
#define DEEPSEEK_V2_BLOCK_H

#include "mla.h"
#include "moe.h"

typedef struct {
    Dsv2MlaConfig attn;
    Dsv2MoeConfig moe;
    float rms_eps;
} Dsv2BlockConfig;

size_t dsv2_block_scratch_bytes(const Dsv2BlockConfig *cfg, int T);
size_t dsv2_block_train_scratch_bytes(const Dsv2BlockConfig *cfg, int T);

/* x (T,C) -> out (T,C), in-place safe if out != x */
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
    float *scratch);

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
    float *scratch);

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
    float *scratch);

#endif /* DEEPSEEK_V2_BLOCK_H */
