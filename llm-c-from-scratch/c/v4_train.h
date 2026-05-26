#ifndef V4_TRAIN_H
#define V4_TRAIN_H

#include "deepseek_v4_config.h"
#include "v4_model.h"

/*
 * Phase 8 bootstrap: SGD on lm_head only (all other weights frozen).
 * Forward full model, backprop CE through tied-style lm_head rows.
 * Returns mean loss over T positions.
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

#endif /* V4_TRAIN_H */
