#ifndef INDEXER_TRAIN_H
#define INDEXER_TRAIN_H

#include "deepseek_v4_config.h"

size_t ds4_csa_indexer_train_cache_floats(const DeepSeekV4Config *cfg, int T);

int ds4_csa_indexer_forward_train(
    int *sparse_mask,
    const float *hidden,
    const float *q_residual,
    int T,
    int n_comp,
    const int *comp_end_positions,
    const float *compressed,
    const DeepSeekV4Config *cfg,
    const float *wq_b,
    const float *w_weights,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    float *cache,
    int *out_n_idx);

void ds4_csa_indexer_backward(
    float *dhidden,
    float *d_q_residual,
    float *d_wq_b,
    float *d_w_weights,
    float *d_w_kv,
    float *d_w_gate,
    float *d_pos_bias,
    float *d_norm,
    const float *d_scores,
    const float *hidden,
    const float *q_residual,
    int T,
    int n_comp,
    const DeepSeekV4Config *cfg,
    const float *wq_b,
    const float *w_weights,
    const float *w_kv,
    const float *w_gate,
    const float *norm_weight,
    float *cache);

#endif /* INDEXER_TRAIN_H */
