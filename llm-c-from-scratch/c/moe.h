#ifndef MOE_H
#define MOE_H

#include "deepseek_v4_config.h"

/*
 * DeepSeek-V4 MoE (hash_moe or routed moe) — nano DeepSeekV4MoE, B=1.
 * Expert weights: gate_up (2*I*C) + down (C*I) per expert, row-major.
 */

size_t ds4_moe_scratch_bytes(const DeepSeekV4Config *cfg);

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
    float *scratch);

#endif /* MOE_H */
