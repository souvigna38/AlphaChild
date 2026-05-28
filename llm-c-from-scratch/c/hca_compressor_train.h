#ifndef HCA_COMPRESSOR_TRAIN_H
#define HCA_COMPRESSOR_TRAIN_H

#include "deepseek_v4_config.h"

size_t ds4_hca_compress_train_cache_floats(const DeepSeekV4Config *cfg, int T);

int ds4_hca_compress_forward_train(
    float *comp_kv,
    int *end_positions,
    const float *hidden,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache,
    int *out_n);

void ds4_hca_compress_backward(
    float *dhidden,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_comp_kv,
    const float *hidden,
    int T,
    int n_comp,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache);

#endif /* HCA_COMPRESSOR_TRAIN_H */
