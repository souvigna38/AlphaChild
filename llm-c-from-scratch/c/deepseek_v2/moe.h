/*
 * DeepSeekMoE — educational C port (notebook 13)
 *
 * GPT-2 / llm.c: gelu_forward + one MLP per layer (see train_gpt2.c ~line 870).
 * DeepSeek-V2: router + top-k SwiGLU experts + shared expert(s).
 */
#ifndef DEEPSEEK_V2_MOE_H
#define DEEPSEEK_V2_MOE_H

#include <stddef.h>

typedef struct {
    int n_embd;              /* C */
    int n_routed_experts;    /* E */
    int n_shared_experts;    /* usually 1 */
    int num_experts_per_tok; /* top-k */
    int moe_intermediate;    /* hidden dim inside each expert */
} Dsv2MoeConfig;

/* Bytes of scratch for forward over T tokens (float32) */
size_t dsv2_moe_scratch_bytes(const Dsv2MoeConfig *cfg, int T);

/* Training cache (floats) + caller supplies topi (T * topk ints) */
size_t dsv2_moe_train_scratch_bytes(const Dsv2MoeConfig *cfg, int T);

/*
 * MoE forward on x (T, C) -> out (T, C).
 *
 * Weights (row-major, PyTorch Linear layout):
 *   gate_w:     (E, C)
 *   expert w1:  (E, intermediate, C)   — expert e at offset e * intermediate * C
 *   expert w2:  (E, C, intermediate)
 *   expert w3:  (E, intermediate, C)
 *   shared: same layout with n_shared_experts slices
 *
 * last_top_expert / last_top_weight: optional per-token debug (length T * topk), may be NULL.
 */
void dsv2_moe_forward(
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
    int *last_top_expert,
    float *last_top_weight,
    float *scratch);

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
    float *scratch);

/* topi: (T, topk) from forward_train; accumulates into dx and weight grads */
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
    float *scratch);

#endif /* DEEPSEEK_V2_MOE_H */
