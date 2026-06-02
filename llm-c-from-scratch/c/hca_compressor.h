#ifndef HCA_COMPRESSOR_H
#define HCA_COMPRESSOR_H

#include "deepseek_v4_config.h"

/*
 * Heavily Compressed Attention (HCA) KV compressor — nano HCACompressor, cache=NULL.
 * Outputs up to T/rate compressed tokens (head_dim each) and end_positions.
 */

int ds4_hca_compress_forward(
    float *comp_kv,
    int *end_positions,
    const float *hidden,
    int T,
    const DeepSeekV4Config *cfg,
    const float *w_kv,
    const float *w_gate,
    const float *pos_bias,
    const float *norm_weight,
    int *out_n);

#endif /* HCA_COMPRESSOR_H */
