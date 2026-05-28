#ifndef V4_LAYER_TRAIN_H
#define V4_LAYER_TRAIN_H

#include "deepseek_v4_config.h"
#include "v4_layer.h"

size_t ds4_layer_train_cache_floats(const DeepSeekV4Config *cfg, int T);

size_t ds4_layer_train_cache_layer(const DeepSeekV4Config *cfg, int T, int layer_idx);

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

/* MoE/FFN mHC frozen: backprop attention path only. */
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
    float *d_hca_w_kv,
    float *d_hca_w_gate,
    float *d_hca_pos_bias,
    float *d_hca_norm,
    float *d_attn_hc_fn,
    float *d_attn_hc_base,
    float *d_attn_hc_scale,
    float *d_attn_norm_w,
    float *attn_scratch,
    float *layer_cache,
    int skip_ffn_upstream);

/* Attention + hash_moe trainable; ffn mHC frozen. */
void ds4_decoder_layer_backward_full(
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
    float *d_hca_w_kv,
    float *d_hca_w_gate,
    float *d_hca_pos_bias,
    float *d_hca_norm,
    float *d_attn_hc_fn,
    float *d_attn_hc_base,
    float *d_attn_hc_scale,
    float *d_attn_norm_w,
    float *d_ffn_norm_w,
    float *d_moe_gate,
    float *d_moe_expert_gu,
    float *d_moe_expert_down,
    float *d_moe_shared_gu,
    float *d_moe_shared_down,
    float *attn_scratch,
    float *moe_scratch,
    float *layer_cache);

size_t ds4_layer_attn_param_count(const DeepSeekV4Config *cfg);

size_t ds4_layer_attn_param_count_layer(const DeepSeekV4Config *cfg, int layer_idx);

size_t ds4_layer_hash_moe_param_count(const DeepSeekV4Config *cfg);

size_t ds4_layer_train_param_count(const DeepSeekV4Config *cfg, int layer_idx);

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
    float *hca_w_kv;
    float *hca_w_gate;
    float *hca_pos_bias;
    float *hca_norm;
    float *attn_hc_fn;
    float *attn_hc_base;
    float *attn_hc_scale;
    float *ffn_norm_w;
    float *moe_gate;
    float *moe_expert_gu;
    float *moe_expert_down;
    float *moe_shared_gu;
    float *moe_shared_down;
} Ds4LayerTrainGrads;

void ds4_layer_train_grad_ptrs(Ds4LayerTrainGrads *g, float *buf, const DeepSeekV4Config *cfg, int layer_idx);

#endif /* V4_LAYER_TRAIN_H */
