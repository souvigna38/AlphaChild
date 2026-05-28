#ifndef V4_ATTENTION_TRAIN_H
#define V4_ATTENTION_TRAIN_H

#include "deepseek_v4_config.h"

size_t ds4_hca_attention_train_cache_floats(const DeepSeekV4Config *cfg, int T);

void ds4_hca_attention_forward_train(
    float *out,
    const float *hidden,
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
    const float *hca_w_kv,
    const float *hca_w_gate,
    const float *hca_pos_bias,
    const float *hca_norm,
    float *scratch,
    float *cache);

void ds4_hca_attention_backward(
    float *dx,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    float *d_hca_w_kv,
    float *d_hca_w_gate,
    float *d_hca_pos_bias,
    float *d_hca_norm,
    const float *dout,
    const float *hidden,
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
    const float *hca_w_kv,
    const float *hca_w_gate,
    const float *hca_pos_bias,
    const float *hca_norm,
    float *scratch,
    float *cache);

#endif /* V4_ATTENTION_TRAIN_H */
