#ifndef CSA_COMPRESSOR_H
#define CSA_COMPRESSOR_H

#include "deepseek_v4_config.h"

/* Compressed Sparse Attention KV compressor + indexer mask (cache=NULL). */

int ds4_csa_compress_forward(
    float *comp_kv,
    int *end_positions,
    int *sparse_mask,
    const float *hidden,
    const float *q_residual,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    const float *idx_wq_b,
    const float *idx_w_weights,
    const float *idx_w_kv,
    const float *idx_w_gate,
    const float *idx_pos_bias,
    const float *idx_norm_weight,
    int *out_n);

#endif /* CSA_COMPRESSOR_H */
