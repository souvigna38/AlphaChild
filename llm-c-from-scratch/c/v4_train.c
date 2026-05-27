#include "v4_train.h"

#include "mhc.h"
#include "rmsnorm.h"
#include "moe.h"
#include "v4_attention.h"
#include "v4_layer.h"
#include "v4_layer_train.h"
#include "v4_model.h"
#include "v4_ops.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Stack buffers in train-final / train-1layer steps (tiny config). */
#define DS4_TRAIN_MAX_C 128
#define DS4_TRAIN_MAX_HC 8
#define DS4_TRAIN_MAX_T 128

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
        ds4_linear_backward(dnorm_h, g_lm, dlog, h, weights->lm_head, V, C);
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

size_t ds4_model_1layer_param_count(const DeepSeekV4Config *cfg) {
    return (size_t)cfg->vocab_size * (size_t)cfg->hidden_size + ds4_layer_attn_param_count(cfg);
}

size_t ds4_model_train_1layer_working_bytes(const DeepSeekV4Config *cfg, int T) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    size_t n = (size_t)T * (size_t)C * 2 + (size_t)T * (size_t)hc * (size_t)C;
    return n * sizeof(float);
}

static void sgd_update(float *params, const float *grad, size_t n, float lr) {
    for (size_t i = 0; i < n; i++) {
        params[i] -= lr * grad[i];
    }
}

float ds4_model_train_step_1layer(
    Ds4ModelWeights *weights,
    const DeepSeekV4Config *cfg,
    const int *input_ids,
    const int *targets,
    int T,
    float lr,
    float *grad_buf,
    float *streams_a,
    float *streams_b,
    float *model_scratch,
    float *layer_cache,
    float *logits,
    float *token_scratch) {
    if (cfg->num_hidden_layers != 1) {
        return 0.0f;
    }
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    const float eps = cfg->rms_norm_eps;
    Ds4LayerWeights *lw = &weights->layers[0];

    size_t attn_n = ds4_attention_scratch_bytes(cfg, T) / sizeof(float);
    size_t moe_n = ds4_moe_scratch_bytes(cfg) / sizeof(float);
    size_t layer_total = ds4_layer_scratch_bytes(cfg, T) / sizeof(float);
    float *attn_scratch = model_scratch;
    float *moe_scratch = attn_scratch + attn_n;
    float *layer_work = moe_scratch + moe_n;
    float *hidden = model_scratch + layer_total;
    float *norm_h = hidden + (size_t)T * (size_t)C;

    for (int t = 0; t < T; t++) {
        const float *emb = weights->embed + (size_t)input_ids[t] * (size_t)C;
        for (int i = 0; i < hc; i++) {
            memcpy(streams_a + ((size_t)t * (size_t)hc + (size_t)i) * (size_t)C, emb, (size_t)C * sizeof(float));
        }
    }

    ds4_decoder_layer_forward_train(
        streams_b, streams_a, input_ids, T, 0, cfg, lw, attn_scratch, moe_scratch, layer_work, layer_cache);

    const float *final_streams = streams_b;
    for (int t = 0; t < T; t++) {
        ds4_hyper_head_forward(
            hidden + (size_t)t * (size_t)C,
            final_streams + ((size_t)t * (size_t)hc) * (size_t)C,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale);
        ds4_rmsnorm_forward(norm_h + (size_t)t * (size_t)C, hidden + (size_t)t * (size_t)C, weights->final_norm, 1, C, eps);
        for (int v = 0; v < V; v++) {
            logits[(size_t)t * (size_t)V + (size_t)v] =
                ds4_dot(norm_h + (size_t)t * (size_t)C, weights->lm_head + (size_t)v * (size_t)C, C);
        }
    }

    float loss_sum = 0.0f;
    float scale = 1.0f / (float)T;
    const int mix = (2 + hc) * hc;
    float *d_streams_in = streams_a;
    if (C > DS4_TRAIN_MAX_C || hc > DS4_TRAIN_MAX_HC) {
        return 0.0f;
    }
    float *d_streams_out = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float flat_sc[DS4_TRAIN_MAX_HC * DS4_TRAIN_MAX_C];
    float mixv_sc[DS4_TRAIN_MAX_HC];
    float dx_hc[DS4_TRAIN_MAX_HC * DS4_TRAIN_MAX_C];
    if (!d_streams_out) {
        return 0.0f;
    }
    memset(grad_buf, 0, ds4_model_1layer_param_count(cfg) * sizeof(float));
    float *g_embed = grad_buf;
    float *g_layer = grad_buf + (size_t)V * (size_t)C;
    Ds4LayerAttnGrads lg;

    ds4_layer_attn_grad_ptrs(&lg, g_layer, cfg);

    for (int t = 0; t < T; t++) {
        float *dlog = token_scratch;
        int y = targets[t];
        loss_sum += ds4_softmax_cross_entropy_backward(dlog, logits + (size_t)t * (size_t)V, y, V);
        for (int v = 0; v < V; v++) {
            dlog[v] *= scale;
        }
        float dnorm_h[DS4_TRAIN_MAX_C];
        memset(dnorm_h, 0, (size_t)C * sizeof(float));
        ds4_linear_backward(dnorm_h, NULL, dlog, norm_h + (size_t)t * (size_t)C, weights->lm_head, V, C);
        float dpre[DS4_TRAIN_MAX_C];
        ds4_rmsnorm_backward(dpre, NULL, dnorm_h, hidden + (size_t)t * (size_t)C, norm_h + (size_t)t * (size_t)C, weights->final_norm, 1, C, eps);
        const float *st = final_streams + ((size_t)t * (size_t)hc) * (size_t)C;
        ds4_hyper_head_forward_save(
            hidden + (size_t)t * (size_t)C,
            st,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale,
            flat_sc,
            mixv_sc);
        memset(dx_hc, 0, (size_t)hc * (size_t)C * sizeof(float));
        ds4_hyper_head_backward(
            dx_hc,
            NULL,
            NULL,
            NULL,
            dpre,
            st,
            flat_sc,
            mixv_sc,
            cfg,
            weights->hc_head_fn,
            weights->hc_head_base,
            weights->hc_head_scale);
        memcpy(
            d_streams_out + ((size_t)t * (size_t)hc) * (size_t)C,
            dx_hc,
            (size_t)hc * (size_t)C * sizeof(float));
    }
    ds4_decoder_layer_backward_attn(
        d_streams_in,
        d_streams_out,
        input_ids,
        T,
        0,
        cfg,
        lw,
        lg.wq_a,
        lg.w_qa_norm,
        lg.wq_b,
        lg.wkv,
        lg.w_kv_norm,
        lg.attn_sink,
        lg.wo_a,
        lg.wo_b,
        lg.attn_hc_fn,
        lg.attn_hc_base,
        lg.attn_hc_scale,
        lg.attn_norm_w,
        attn_scratch,
        layer_cache);

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < hc; i++) {
            for (int c = 0; c < C; c++) {
                g_embed[(size_t)input_ids[t] * (size_t)C + (size_t)c] +=
                    d_streams_in[((size_t)t * (size_t)hc + (size_t)i) * (size_t)C + (size_t)c];
            }
        }
    }

    sgd_update(weights->embed, g_embed, (size_t)V * (size_t)C, lr);
    sgd_update(lw->attn_norm_w, lg.attn_norm_w, (size_t)C, lr);
    sgd_update(lw->wq_a, lg.wq_a, (size_t)cfg->q_lora_rank * (size_t)C, lr);
    sgd_update(lw->w_qa_norm, lg.w_qa_norm, (size_t)cfg->q_lora_rank, lr);
    sgd_update(lw->wq_b, lg.wq_b, (size_t)ds4_attention_width(cfg) * (size_t)cfg->q_lora_rank, lr);
    sgd_update(lw->wkv, lg.wkv, (size_t)C * (size_t)C, lr);
    sgd_update(lw->w_kv_norm, lg.w_kv_norm, (size_t)C, lr);
    sgd_update(lw->attn_sink, lg.attn_sink, (size_t)cfg->num_attention_heads, lr);
    sgd_update(
        lw->wo_a,
        lg.wo_a,
        (size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(ds4_attention_width(cfg) / cfg->o_groups),
        lr);
    sgd_update(lw->wo_b, lg.wo_b, (size_t)C * (size_t)cfg->o_lora_rank, lr);
    sgd_update(lw->attn_hc_fn, lg.attn_hc_fn, (size_t)mix * (size_t)C * (size_t)hc, lr);
    sgd_update(lw->attn_hc_base, lg.attn_hc_base, (size_t)mix, lr);
    sgd_update(lw->attn_hc_scale, lg.attn_hc_scale, 3, lr);

    free(d_streams_out);
    return loss_sum / (float)T;
}
