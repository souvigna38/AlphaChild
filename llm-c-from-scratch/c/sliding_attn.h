#ifndef SLIDING_ATTN_H
#define SLIDING_ATTN_H

#include "deepseek_v4_config.h"

/*
 * Sliding-window causal attention with attention sink (nano DeepSeekV4Attention,
 * layer_type == sliding_attention, cache == NULL).
 *
 * Weight layout (row-major, matches PyTorch Linear):
 *   wq_a: (q_lora_rank, hidden)
 *   wq_b: (attention_width, q_lora_rank)
 *   wkv:  (head_dim, hidden)
 *   wo_a: (o_groups, o_lora_rank, attention_width/o_groups)
 *   wo_b: (hidden, o_groups * o_lora_rank)
 *   attn_sink: (num_attention_heads)
 *   w_qa_norm, w_kv_norm: RMSNorm gamma (q_lora_rank), (head_dim)
 */

size_t ds4_sliding_attn_scratch_bytes(const DeepSeekV4Config *cfg, int T);

void ds4_sliding_attn_forward(
    float *out,
    const float *x,
    int T,
    const DeepSeekV4Config *cfg,
    const float *wq_a,
    const float *w_qa_norm,
    const float *wq_b,
    const float *wkv,
    const float *w_kv_norm,
    const float *attn_sink,
    const float *wo_a,
    const float *wo_b,
    float *scratch);

/* Core masked attention: q (NH,T,D), keys/values (NH,Tk,D), mask (T,Tk) bool as 0/1. */
void ds4_core_attention(
    float *context,
    const float *q,
    const float *keys,
    int NH,
    int Tq,
    int Tk,
    int head_dim,
    const float *attn_sink,
    const int *mask);

#endif /* SLIDING_ATTN_H */
