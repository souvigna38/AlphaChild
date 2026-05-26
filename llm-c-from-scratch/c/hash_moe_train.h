#ifndef HASH_MOE_TRAIN_H
#define HASH_MOE_TRAIN_H

#include "deepseek_v4_config.h"

size_t ds4_hash_moe_train_cache_floats(const DeepSeekV4Config *cfg);

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
    float *cache);

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
    float swiglu_limit);

#endif /* HASH_MOE_TRAIN_H */
