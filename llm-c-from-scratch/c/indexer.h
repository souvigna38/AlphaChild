#ifndef INDEXER_H
#define INDEXER_H

#include "deepseek_v4_config.h"

/*
 * Lightning indexer sparse mask for CSA (nano CSAIndexer, cache=NULL).
 * sparse_mask[t, k] = 1 if compressed key k is selected for query t.
 */

int ds4_csa_indexer_forward(
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
    const float *norm_weight);

#endif /* INDEXER_H */
