#ifndef V4_TRAIN_H
#define V4_TRAIN_H

#include "adamw.h"
#include "deepseek_v4_config.h"
#include "v4_model.h"

/*
 * Phase 8: SGD on lm_head only (frozen trunk).
 */
float ds4_model_train_head_step(
    Ds4ModelWeights *weights,
    const DeepSeekV4Config *cfg,
    const int *input_ids,
    const int *targets,
    int T,
    float lr,
    float *streams_a,
    float *streams_b,
    float *scratch,
    float *logits,
    float *norm_h,
    float *token_scratch);

/* Trainable final stack: embed, lm_head, final_norm, hc_head (layers frozen). */
size_t ds4_model_final_param_count(const DeepSeekV4Config *cfg);

size_t ds4_model_train_final_working_bytes(const DeepSeekV4Config *cfg, int T);

/*
 * AdamW on final params; token_scratch must hold >= vocab_size floats.
 * train_work: norm_h + hidden_pre + final_streams + flat/mixv scratch.
 */
float ds4_model_train_final_adam_step(
    Ds4ModelWeights *weights,
    const DeepSeekV4Config *cfg,
    const int *input_ids,
    const int *targets,
    int T,
    Ds4AdamW *opt,
    float *grad_buf,
    float *streams_a,
    float *streams_b,
    float *model_scratch,
    float *train_work,
    float *logits,
    float *token_scratch);

#endif /* V4_TRAIN_H */
