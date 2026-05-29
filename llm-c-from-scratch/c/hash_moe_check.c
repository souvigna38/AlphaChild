/* ctypes entry for scripts/verify_v4_nano_hash_moe.py */
#include "deepseek_v4_config.h"
#include "hash_moe.h"

void hash_moe_check_forward(
    float *out,
    const float *x,
    int token_id,
    const float *gate_w,
    const float *expert_gate_up,
    const float *expert_down,
    const float *shared_gate_up,
    const float *shared_down,
    int *tid2eid) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    ds4_hash_moe_forward_token(
        out,
        x,
        token_id,
        &cfg,
        tid2eid,
        gate_w,
        expert_gate_up,
        expert_down,
        shared_gate_up,
        shared_down,
        NULL);
}
