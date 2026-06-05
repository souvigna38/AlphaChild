/* ctypes entry for scripts/verify_v4_sliding_attn.py */
#include "deepseek_v4_config.h"
#include "sliding_attn.h"

void sliding_attn_check_forward(
    float *out,
    const float *x,
    int T,
    const float *wq_a,
    const float *w_qa_norm,
    const float *wq_b,
    const float *wkv,
    const float *w_kv_norm,
    const float *attn_sink,
    const float *wo_a,
    const float *wo_b,
    float *scratch) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    ds4_sliding_attn_forward(
        out,
        x,
        T,
        &cfg,
        wq_a,
        w_qa_norm,
        wq_b,
        wkv,
        w_kv_norm,
        attn_sink,
        wo_a,
        wo_b,
        scratch);
}
