#include "v4_train.h"

#include "v4_ops.h"

#include <stddef.h>

float ds4_model_train_head_step(
    Ds4ModelWeights *weights,
    const DeepSeekV4Config *cfg,
    const int *input_ids,
    const int *targets,
    int T,
    float lr,
    float *streams_a,
    float *streams_b,
    float *scratch,
    float *logits,
    float *norm_h,
    float *token_scratch) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;

    ds4_model_forward(logits, input_ids, T, cfg, weights, streams_a, streams_b, scratch, norm_h);

    float loss_sum = 0.0f;
    float scale = 1.0f / (float)T;

    for (int t = 0; t < T; t++) {
        const float *h = norm_h + (size_t)t * (size_t)C;
        float *log = logits + (size_t)t * (size_t)V;
        float *dlog = token_scratch;
        int y = targets[t];
        loss_sum += ds4_softmax_cross_entropy_backward(dlog, log, y, V);
        for (int v = 0; v < V; v++) {
            dlog[v] *= scale;
        }
        float *dlm = weights->lm_head;
        for (int v = 0; v < V; v++) {
            float *row = dlm + (size_t)v * (size_t)C;
            float g = dlog[v];
            for (int c = 0; c < C; c++) {
                row[c] -= lr * g * h[c];
            }
        }
    }
    return loss_sum / (float)T;
}
