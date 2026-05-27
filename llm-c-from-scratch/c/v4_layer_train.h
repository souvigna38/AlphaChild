#ifndef V4_LAYER_TRAIN_H
#define V4_LAYER_TRAIN_H

#include "deepseek_v4_config.h"
#include "v4_layer.h"

size_t ds4_layer_train_cache_floats(const DeepSeekV4Config *cfg, int T);

void ds4_decoder_layer_forward_train(
    float *streams_out,
    const float *streams_in,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *attn_scratch,
    float *moe_scratch,
    float *work,
    float *layer_cache);

/* MoE/FFN frozen: backprop attention path only. */
void ds4_decoder_layer_backward_attn(
    float *dstreams_in,
    const float *dstreams_out,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    float *d_attn_hc_fn,
    float *d_attn_hc_base,
    float *d_attn_hc_scale,
    float *d_attn_norm_w,
    float *attn_scratch,
    float *layer_cache);

size_t ds4_layer_attn_param_count(const DeepSeekV4Config *cfg);

typedef struct {
    float *attn_norm_w;
    float *wq_a;
    float *w_qa_norm;
    float *wq_b;
    float *wkv;
    float *w_kv_norm;
    float *attn_sink;
    float *wo_a;
    float *wo_b;
    float *attn_hc_fn;
    float *attn_hc_base;
    float *attn_hc_scale;
} Ds4LayerAttnGrads;

void ds4_layer_attn_grad_ptrs(Ds4LayerAttnGrads *g, float *buf, const DeepSeekV4Config *cfg);

#endif /* V4_LAYER_TRAIN_H */
