#ifndef V4_ATTENTION_H
#define V4_ATTENTION_H

#include "deepseek_v4_config.h"

/* Full DeepSeek-V4 attention layer (sliding / CSA / HCA), cache=NULL, B=1. */

size_t ds4_attention_scratch_bytes(const DeepSeekV4Config *cfg, int T);

void ds4_attention_forward(
    float *out,
    const float *hidden,
    int T,
    Ds4AttentionType attn_type,
    const DeepSeekV4Config *cfg,
    const float *wq_a,
    const float *w_qa_norm,
    const float *wq_b,
    const float *wkv,
    const float *w_kv_norm,
    const float *attn_sink,
    const float *wo_a,
    const float *wo_b,
    const float *hca_w_kv,
    const float *hca_w_gate,
    const float *hca_pos_bias,
    const float *hca_norm,
    const float *csa_w_kv,
    const float *csa_w_gate,
    const float *csa_pos_bias,
    const float *csa_norm,
    const float *idx_wq_b,
    const float *idx_w_weights,
    const float *idx_w_kv,
    const float *idx_w_gate,
    const float *idx_pos_bias,
    const float *idx_norm,
    float *scratch);

#endif /* V4_ATTENTION_H */
