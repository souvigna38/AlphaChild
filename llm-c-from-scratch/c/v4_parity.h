#ifndef V4_PARITY_H
#define V4_PARITY_H

#include "deepseek_v4_config.h"
#include "v4_model.h"

/* Deterministic LCG used for C/Python parity fixtures (seed on stack). */
unsigned ds4_parity_rng_step(unsigned *seed);

void ds4_parity_fill_buffer(float *buf, size_t n, unsigned *seed);

/* Fill model weights; norm vectors set to 1.0 after random fill where applicable. */
void ds4_parity_fill_model(Ds4ModelWeights *mw, const DeepSeekV4Config *cfg, unsigned seed);

void ds4_parity_forward_logits(
    float *logits,
    const int *input_ids,
    int T,
    const DeepSeekV4Config *cfg,
    Ds4ModelWeights *mw);

/* Sum of |stream| after each decoder layer (deterministic parity probe). */
float ds4_parity_stream_checksum(const float *streams, int T, int hc_mult, int hidden);

void ds4_parity_forward_layer_checksums(
    float *checksums,
    int n_layers_out,
    const int *input_ids,
    int T,
    const DeepSeekV4Config *cfg,
    Ds4ModelWeights *mw);

void ds4_parity_free_model(Ds4ModelWeights *mw, const DeepSeekV4Config *cfg);

#endif /* V4_PARITY_H */
