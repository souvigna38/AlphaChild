#ifndef SLIDING_ATTN_TRAIN_H
#define SLIDING_ATTN_TRAIN_H

#include "deepseek_v4_config.h"

size_t ds4_sliding_attn_train_cache_floats(const DeepSeekV4Config *cfg, int T);

void ds4_sliding_attn_forward_train(
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
    float *scratch,
    float *cache);

void ds4_sliding_attn_backward(
    float *dx,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    const float *dout,
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
    float *scratch,
    float *cache);

#endif /* SLIDING_ATTN_TRAIN_H */
