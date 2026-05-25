#ifndef MHC_H
#define MHC_H

#include "deepseek_v4_config.h"

/*
 * Manifold-Constrained Hyper-Connection (nano HyperConnection).
 * streams: (hc_mult, hidden) per token — caller loops over T for sequences.
 */

void ds4_hyper_connection_forward(
    float *post,
    float *comb,
    float *collapsed,
    const float *streams,
    const DeepSeekV4Config *cfg,
    const float *fn_weight,
    const float *fn_base,
    const float *scale);

void ds4_hc_stream_update(
    float *streams_out,
    const float *streams_in,
    const float *post,
    const float *comb,
    const float *branch,
    int hc_mult,
    int hidden);

void ds4_hyper_head_forward(float *out, const float *streams, const DeepSeekV4Config *cfg, const float *fn_weight, const float *fn_base, const float *scale);

#endif /* MHC_H */
