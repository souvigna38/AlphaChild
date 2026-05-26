#include "v4_model.h"

#include "mhc.h"
#include "moe.h"
#include "rmsnorm.h"
#include "v4_attention.h"
#include "v4_layer.h"
#include "v4_ops.h"

#include <stddef.h>
#include <string.h>

size_t ds4_model_scratch_bytes(const DeepSeekV4Config *cfg, int T) {
    return ds4_layer_scratch_bytes(cfg, T) + (size_t)T * (size_t)cfg->hidden_size * 2 * sizeof(float);
}

void ds4_model_forward(
    float *logits,
    const int *input_ids,
    int T,
    const DeepSeekV4Config *cfg,
    Ds4ModelWeights *weights,
    float *streams_a,
    float *streams_b,
    float *scratch,
    float *norm_h_out) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    const int nL = cfg->num_hidden_layers;
    const float eps = cfg->rms_norm_eps;

    if (!logits || !input_ids || !cfg || !weights || !weights->layers || !streams_a || !streams_b || !scratch) {
        return;
    }

    size_t attn_n = ds4_attention_scratch_bytes(cfg, T) / sizeof(float);
    size_t moe_n = ds4_moe_scratch_bytes(cfg) / sizeof(float);
    size_t layer_base = ds4_layer_scratch_bytes(cfg, T) / sizeof(float);

    float *attn_scratch = scratch;
    float *moe_scratch = scratch + attn_n;
    float *layer_work = scratch + attn_n + moe_n;
    float *hidden = scratch + layer_base;
    float *norm_h = hidden + (size_t)T * (size_t)C;

    for (int t = 0; t < T; t++) {
        const float *emb = weights->embed + (size_t)input_ids[t] * (size_t)C;
        for (int i = 0; i < hc; i++) {
            memcpy(streams_a + ((size_t)t * (size_t)hc + (size_t)i) * (size_t)C, emb, (size_t)C * sizeof(float));
        }
    }

    float *cur = streams_a;
    float *nxt = streams_b;
    for (int L = 0; L < nL; L++) {
        ds4_decoder_layer_forward(
            nxt, cur, input_ids, T, L, cfg, &weights->layers[L], attn_scratch, moe_scratch, layer_work);
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    const float *final_streams = (nL % 2 == 0) ? streams_a : streams_b;
    if (nL == 0) {
        final_streams = streams_a;
    }

    for (int t = 0; t < T; t++) {
        ds4_hyper_head_forward(
            hidden + (size_t)t * (size_t)C,
            final_streams + ((size_t)t * (size_t)hc) * (size_t)C,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale);
        ds4_rmsnorm_forward(norm_h + (size_t)t * (size_t)C, hidden + (size_t)t * (size_t)C, weights->final_norm, 1, C, eps);
        if (norm_h_out != NULL) {
            memcpy(norm_h_out + (size_t)t * (size_t)C, norm_h + (size_t)t * (size_t)C, (size_t)C * sizeof(float));
        }
    }

    for (int t = 0; t < T; t++) {
        const float *h = norm_h + (size_t)t * (size_t)C;
        float *log = logits + (size_t)t * (size_t)V;
        for (int v = 0; v < V; v++) {
            log[v] = ds4_dot(h, weights->lm_head + (size_t)v * (size_t)C, C);
        }
    }
}
