#include "v4_train.h"

#include "mhc.h"
#include "rmsnorm.h"
#include "v4_ops.h"

#include <stddef.h>
#include <string.h>

/* Stack buffers in train-final step (tiny config hidden_size=64, hc_mult=4). */
#define DS4_TRAIN_MAX_C 128
#define DS4_TRAIN_MAX_HC 8

size_t ds4_model_final_param_count(const DeepSeekV4Config *cfg) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    return (size_t)V * (size_t)C * 2 + (size_t)C + (size_t)hc * (size_t)hc * (size_t)C + (size_t)hc + 3;
}

size_t ds4_model_train_final_working_bytes(const DeepSeekV4Config *cfg, int T) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    size_t n = (size_t)T * (size_t)C * 3 + (size_t)T * (size_t)hc * (size_t)C + (size_t)hc * (size_t)C + (size_t)hc;
    return n * sizeof(float);
}

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

    ds4_model_forward(logits, input_ids, T, cfg, weights, streams_a, streams_b, scratch, norm_h, NULL);

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

static void zero_grad_final(float *grad, const DeepSeekV4Config *cfg) {
    memset(grad, 0, ds4_model_final_param_count(cfg) * sizeof(float));
}

static void adam_apply_final(Ds4AdamW *opt, Ds4ModelWeights *w, const DeepSeekV4Config *cfg, float *grad) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    opt->step++;
    size_t off = 0;
    ds4_adamw_step_group(opt, off, (size_t)V * (size_t)C, w->embed, grad + off, 0);
    off += (size_t)V * (size_t)C;
    ds4_adamw_step_group(opt, off, (size_t)V * (size_t)C, w->lm_head, grad + off, 0);
    off += (size_t)V * (size_t)C;
    ds4_adamw_step_group(opt, off, (size_t)C, w->final_norm, grad + off, 0);
    off += (size_t)C;
    ds4_adamw_step_group(opt, off, (size_t)hc * (size_t)hc * (size_t)C, w->hc_head_fn, grad + off, 0);
    off += (size_t)hc * (size_t)hc * (size_t)C;
    ds4_adamw_step_group(opt, off, (size_t)hc, w->hc_head_base, grad + off, 0);
    off += (size_t)hc;
    ds4_adamw_step_group(opt, off, 3, w->hc_head_scale, grad + off, 0);
}

float ds4_model_train_final_adam_step(
    Ds4ModelWeights *weights,
    const DeepSeekV4Config *cfg,
    const int *input_ids,
    const int *targets,
    int T,
    Ds4AdamW *opt,
    float *grad_buf,
    float *streams_a,
    float *streams_b,
    float *model_scratch,
    float *train_work,
    float *logits,
    float *token_scratch) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    const int nL = cfg->num_hidden_layers;
    const float eps = cfg->rms_norm_eps;

    float *norm_h = train_work;
    float *hidden_pre = norm_h + (size_t)T * (size_t)C;
    float *final_streams = hidden_pre + (size_t)T * (size_t)C;
    float *flat_buf = final_streams + (size_t)T * (size_t)hc * (size_t)C;
    float *mixv_buf = flat_buf + (size_t)hc * (size_t)C;

    ds4_model_forward(
        logits,
        input_ids,
        T,
        cfg,
        weights,
        streams_a,
        streams_b,
        model_scratch,
        norm_h,
        hidden_pre);

    const float *fstreams = (nL % 2 == 0) ? streams_a : streams_b;
    memcpy(final_streams, fstreams, (size_t)T * (size_t)hc * (size_t)C * sizeof(float));

    zero_grad_final(grad_buf, cfg);
    size_t goff = 0;
    float *g_embed = grad_buf + goff;
    goff += (size_t)V * (size_t)C;
    float *g_lm = grad_buf + goff;
    goff += (size_t)V * (size_t)C;
    float *g_fn = grad_buf + goff;
    goff += (size_t)C;
    float *g_hcfn = grad_buf + goff;
    goff += (size_t)hc * (size_t)hc * (size_t)C;
    float *g_hcb = grad_buf + goff;
    goff += (size_t)hc;
    float *g_hcs = grad_buf + goff;

    float loss_sum = 0.0f;
    if (C > DS4_TRAIN_MAX_C || hc > DS4_TRAIN_MAX_HC) {
        return 0.0f;
    }
    float dx_hc[DS4_TRAIN_MAX_HC * DS4_TRAIN_MAX_C];

    for (int t = 0; t < T; t++) {
        const float *h = norm_h + (size_t)t * (size_t)C;
        float *log = logits + (size_t)t * (size_t)V;
        float *dlog = token_scratch;
        int y = targets[t];
        loss_sum += ds4_softmax_cross_entropy_backward(dlog, log, y, V);

        float dnorm_h[DS4_TRAIN_MAX_C];
        float dhidden[DS4_TRAIN_MAX_C];
        memset(dnorm_h, 0, (size_t)C * sizeof(float));
        memset(dhidden, 0, (size_t)C * sizeof(float));
        ds4_linear_backward(dnorm_h, g_lm, dlog, h, V, C);
        ds4_rmsnorm_backward(
            dhidden,
            g_fn,
            dnorm_h,
            hidden_pre + (size_t)t * (size_t)C,
            h,
            weights->final_norm,
            1,
            C,
            eps);

        const float *st = final_streams + ((size_t)t * (size_t)hc) * (size_t)C;
        float *flat = flat_buf;
        float *mixv = mixv_buf;
        ds4_hyper_head_forward_save(
            hidden_pre + (size_t)t * (size_t)C,
            st,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale,
            flat,
            mixv);

        memset(dx_hc, 0, (size_t)hc * (size_t)C * sizeof(float));
        ds4_hyper_head_backward(
            dx_hc,
            g_hcfn,
            g_hcb,
            g_hcs,
            dhidden,
            st,
            flat,
            mixv,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale);

        for (int i = 0; i < hc; i++) {
            for (int c = 0; c < C; c++) {
                g_embed[(size_t)input_ids[t] * (size_t)C + (size_t)c] += dx_hc[(size_t)i * (size_t)C + (size_t)c];
            }
        }
    }

    adam_apply_final(opt, weights, cfg, grad_buf);
    return loss_sum / (float)T;
}
