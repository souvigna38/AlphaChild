#ifndef V4_MODEL_H
#define V4_MODEL_H

#include "deepseek_v4_config.h"
#include "v4_layer.h"

typedef struct {
    float *embed;
    float *lm_head;
    float *final_norm;
    float *hc_head_fn;
    float *hc_head_base;
    float hc_head_scale[3];
    Ds4LayerWeights *layers;
} Ds4ModelWeights;

size_t ds4_model_scratch_bytes(const DeepSeekV4Config *cfg, int T);

/*
 * Full forward (B=1): input_ids[T] -> logits[T * vocab_size].
 * streams_a / streams_b: each T * hc_mult * hidden (ping-pong between layers).
 */
void ds4_model_forward(
    float *logits,
    const int *input_ids,
    int T,
    const DeepSeekV4Config *cfg,
    Ds4ModelWeights *weights,
    float *streams_a,
    float *streams_b,
    float *scratch);

#endif /* V4_MODEL_H */
