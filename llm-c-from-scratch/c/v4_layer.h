#ifndef V4_LAYER_H
#define V4_LAYER_H

#include "deepseek_v4_config.h"

typedef struct {
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
    float *csa_w_kv;
    float *csa_w_gate;
    float *csa_pos_bias;
    float *csa_norm;
    float *idx_wq_b;
    float *idx_w_weights;
    float *idx_w_kv;
    float *idx_w_gate;
    float *idx_pos_bias;
    float *idx_norm;
    float *attn_hc_fn;
    float *attn_hc_base;
    float attn_hc_scale[3];
    float *ffn_hc_fn;
    float *ffn_hc_base;
    float ffn_hc_scale[3];
    float *attn_norm_w;
    float *ffn_norm_w;
    float *moe_gate;
    float *moe_route_bias;
    float *moe_expert_gu;
    float *moe_expert_down;
    float *moe_shared_gu;
    float *moe_shared_down;
    int *tid2eid;
} Ds4LayerWeights;

size_t ds4_layer_scratch_bytes(const DeepSeekV4Config *cfg, int T);

/*
 * One decoder layer (mHC + attention + mHC + MoE). streams: T x hc_mult x hidden, row-major.
 */
void ds4_decoder_layer_forward(
    float *streams_out,
    const float *streams_in,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *attn_scratch,
    float *moe_scratch,
    float *work);

#endif /* V4_LAYER_H */
