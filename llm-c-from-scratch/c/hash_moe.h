#ifndef DEEPSEEK_V4_HASH_MOE_H
#define DEEPSEEK_V4_HASH_MOE_H

#include "deepseek_v4_config.h"

#include <stddef.h>

/* Build tid2eid table (V, topk) int32 — same formula as nano DeepSeekV4MoE._build_hash_table */
void ds4_hash_moe_build_table(int *tid2eid, const DeepSeekV4Config *cfg);

/*
 * Hash-MoE forward for one token row x (hidden,).
 * gate_up/down expert weights: per-expert (2*I*C + C*I) contiguous.
 * out (hidden), scratch >= 2*intermediate floats.
 */
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
    float *scratch);

#endif /* DEEPSEEK_V4_HASH_MOE_H */
