#include "v4_layer_train.h"

#include "mhc.h"
#include "moe.h"
#include "rmsnorm.h"
#include "sliding_attn_train.h"
#include "v4_attention.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void ds4_layer_attn_grad_ptrs(Ds4LayerAttnGrads *g, float *buf, const DeepSeekV4Config *cfg) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int r = cfg->q_lora_rank;
    const int attn_w = ds4_attention_width(cfg);
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    const int in_pg = attn_w / cfg->o_groups;
    size_t off = 0;
    g->attn_norm_w = buf + off;
    off += (size_t)C;
    g->wq_a = buf + off;
    off += (size_t)r * (size_t)C;
    g->w_qa_norm = buf + off;
    off += (size_t)r;
    g->wq_b = buf + off;
    off += (size_t)attn_w * (size_t)r;
    g->wkv = buf + off;
    off += (size_t)C * (size_t)C;
    g->w_kv_norm = buf + off;
    off += (size_t)C;
    g->attn_sink = buf + off;
    off += (size_t)cfg->num_attention_heads;
    g->wo_a = buf + off;
    off += (size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)in_pg;
    g->wo_b = buf + off;
    off += (size_t)C * (size_t)o_mid;
    g->attn_hc_fn = buf + off;
    off += (size_t)mix * (size_t)C * (size_t)hc;
    g->attn_hc_base = buf + off;
    off += (size_t)mix;
    g->attn_hc_scale = buf + off;
}

size_t ds4_layer_attn_param_count(const DeepSeekV4Config *cfg) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int r = cfg->q_lora_rank;
    const int attn_w = ds4_attention_width(cfg);
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    size_t n = (size_t)C;
    n += (size_t)mix * (size_t)C * (size_t)hc + (size_t)mix + 3;
    n += (size_t)r * (size_t)C + (size_t)r + (size_t)attn_w * (size_t)r;
    n += (size_t)C * (size_t)C + (size_t)C;
    n += (size_t)cfg->num_attention_heads;
    n += (size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(attn_w / cfg->o_groups);
    n += (size_t)C * (size_t)o_mid;
    return n;
}

static size_t per_token_cache_floats(const DeepSeekV4Config *cfg) {
    const int hc = cfg->hc_mult;
    const int C = cfg->hidden_size;
    const int mix = (2 + hc) * hc;
    size_t n = (size_t)hc + (size_t)hc * (size_t)hc + (size_t)hc * (size_t)C + (size_t)mix + (size_t)C + (size_t)C;
    n += (size_t)hc * (size_t)C;
    n += (size_t)hc * (size_t)C;
    n += (size_t)hc + (size_t)hc * (size_t)hc;
    return n;
}

size_t ds4_layer_train_cache_floats(const DeepSeekV4Config *cfg, int T) {
    return (size_t)T * per_token_cache_floats(cfg) + ds4_sliding_attn_train_cache_floats(cfg, T);
}

static size_t st_mid_off(const DeepSeekV4Config *cfg) {
    const int hc = cfg->hc_mult;
    const int C = cfg->hidden_size;
    const int mix = (2 + hc) * hc;
    return (size_t)hc + (size_t)hc * (size_t)hc + (size_t)hc * (size_t)C + (size_t)mix + (size_t)C + (size_t)C;
}

static size_t streams_in_off(const DeepSeekV4Config *cfg) {
    return st_mid_off(cfg) + (size_t)cfg->hc_mult * (size_t)cfg->hidden_size;
}

static size_t ffn_hc_off(const DeepSeekV4Config *cfg) {
    return streams_in_off(cfg) + (size_t)cfg->hc_mult * (size_t)cfg->hidden_size;
}

void ds4_decoder_layer_forward_train(
    float *streams_out,
    const float *streams_in,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *attn_scratch,
    float *moe_scratch,
    float *work,
    float *layer_cache) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const float eps = cfg->rms_norm_eps;
    Ds4AttentionType attn_type = cfg->layer_types[layer_idx];
    Ds4MlpType mlp_type = cfg->mlp_layer_types[layer_idx];
    const size_t ptok = per_token_cache_floats(cfg);
    const size_t slide_off = (size_t)T * ptok;

    float *collapsed = work;
    float *attn_out = collapsed + (size_t)T * (size_t)C;
    float *ffn_out = attn_out + (size_t)T * (size_t)C;
    float *norm_buf = ffn_out + (size_t)T * (size_t)C;
    float *stream_tmp = norm_buf + (size_t)T * (size_t)C;

    const size_t sin_off = streams_in_off(cfg);
    for (int t = 0; t < T; t++) {
        float *lc = layer_cache + (size_t)t * ptok;
        const float *st = streams_in + ((size_t)t * (size_t)hc) * (size_t)C;
        memcpy(lc + sin_off, st, (size_t)hc * (size_t)C * sizeof(float));
        float *post_t = lc;
        float *comb_t = lc + (size_t)hc;
        float *flat_t = comb_t + (size_t)hc * (size_t)hc;
        float *mixv_t = flat_t + (size_t)hc * (size_t)C;
        float *coll_t = collapsed + (size_t)t * (size_t)C;
        ds4_hyper_connection_forward_save(
            post_t, comb_t, coll_t, st, cfg, w->attn_hc_fn, w->attn_hc_base, w->attn_hc_scale, flat_t, mixv_t);
        memcpy(lc + (size_t)hc + (size_t)hc * (size_t)hc + (size_t)hc * (size_t)C + (size_t)mix, coll_t, (size_t)C * sizeof(float));
        ds4_rmsnorm_forward(norm_buf + (size_t)t * (size_t)C, coll_t, w->attn_norm_w, 1, C, eps);
        memcpy(lc + (size_t)hc + (size_t)hc * (size_t)hc + (size_t)hc * (size_t)C + (size_t)mix + (size_t)C, norm_buf + (size_t)t * (size_t)C, (size_t)C * sizeof(float));
    }

    if (attn_type == DS4_ATTN_SLIDING) {
        ds4_sliding_attn_forward_train(
            attn_out,
            norm_buf,
            T,
            cfg,
            w->wq_a,
            w->w_qa_norm,
            w->wq_b,
            w->wkv,
            w->w_kv_norm,
            w->attn_sink,
            w->wo_a,
            w->wo_b,
            attn_scratch,
            layer_cache + slide_off);
    } else {
        ds4_attention_forward(
            attn_out,
            norm_buf,
            T,
            attn_type,
            cfg,
            w->wq_a,
            w->w_qa_norm,
            w->wq_b,
            w->wkv,
            w->w_kv_norm,
            w->attn_sink,
            w->wo_a,
            w->wo_b,
            w->hca_w_kv,
            w->hca_w_gate,
            w->hca_pos_bias,
            w->hca_norm,
            w->csa_w_kv,
            w->csa_w_gate,
            w->csa_pos_bias,
            w->csa_norm,
            w->idx_wq_b,
            w->idx_w_weights,
            w->idx_w_kv,
            w->idx_w_gate,
            w->idx_pos_bias,
            w->idx_norm,
            attn_scratch);
    }

    for (int t = 0; t < T; t++) {
        const float *st = streams_in + ((size_t)t * (size_t)hc) * (size_t)C;
        float *st_mid = layer_cache + (size_t)t * ptok + st_mid_off(cfg);
        float *st_out = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        float *lc = layer_cache + (size_t)t * ptok;
        ds4_hc_stream_update(st_out, st, lc, lc + (size_t)hc, attn_out + (size_t)t * (size_t)C, hc, C);
        memcpy(st_mid, st_out, (size_t)hc * (size_t)C * sizeof(float));
    }

    for (int t = 0; t < T; t++) {
        float *lc = layer_cache + (size_t)t * ptok;
        const size_t ffn_off = ffn_hc_off(cfg);
        const float *st = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        float *post_f = lc + ffn_off;
        float *comb_f = post_f + (size_t)hc;
        float *coll_t = collapsed + (size_t)t * (size_t)C;
        ds4_hyper_connection_forward(post_f, comb_f, coll_t, st, cfg, w->ffn_hc_fn, w->ffn_hc_base, w->ffn_hc_scale);
        ds4_rmsnorm_forward(norm_buf + (size_t)t * (size_t)C, coll_t, w->ffn_norm_w, 1, C, eps);
    }

    ds4_moe_forward(
        ffn_out,
        norm_buf,
        input_ids,
        T,
        mlp_type,
        cfg,
        w->tid2eid,
        w->moe_gate,
        w->moe_route_bias,
        w->moe_expert_gu,
        w->moe_expert_down,
        w->moe_shared_gu,
        w->moe_shared_down,
        moe_scratch);

    for (int t = 0; t < T; t++) {
        const float *st = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        float *st_final = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        float *lc = layer_cache + (size_t)t * ptok;
        const size_t ffn_off = ffn_hc_off(cfg);
        memcpy(stream_tmp + ((size_t)t * (size_t)hc) * (size_t)C, st, (size_t)hc * (size_t)C * sizeof(float));
        ds4_hc_stream_update(
            st_final,
            stream_tmp + ((size_t)t * (size_t)hc) * (size_t)C,
            lc + ffn_off,
            lc + ffn_off + (size_t)hc,
            ffn_out + (size_t)t * (size_t)C,
            hc,
            C);
    }
}

void ds4_decoder_layer_backward_attn(
    float *dstreams_in,
    const float *dstreams_out,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *d_wq_a,
    float *d_w_qa_norm,
    float *d_wq_b,
    float *d_wkv,
    float *d_w_kv_norm,
    float *d_attn_sink,
    float *d_wo_a,
    float *d_wo_b,
    float *d_attn_hc_fn,
    float *d_attn_hc_base,
    float *d_attn_hc_scale,
    float *d_attn_norm_w,
    float *attn_scratch,
    float *layer_cache) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const float eps = cfg->rms_norm_eps;
    Ds4AttentionType attn_type = cfg->layer_types[layer_idx];
    const size_t ptok = per_token_cache_floats(cfg);
    const size_t slide_off = (size_t)T * ptok;
    const size_t ffn_off = ffn_hc_off(cfg);
    const size_t sm_off = st_mid_off(cfg);
    const size_t sin_off = streams_in_off(cfg);

    size_t stream_n = (size_t)T * (size_t)hc * (size_t)C;
    size_t seq_n = (size_t)T * (size_t)C;
    float *d_streams_mid = (float *)calloc(stream_n, sizeof(float));
    float *d_attn_out = (float *)calloc(seq_n, sizeof(float));
    float *x_ln = (float *)calloc(seq_n, sizeof(float));
    float d_ffn[64];
    if (!d_streams_mid || !d_attn_out || !x_ln) {
        free(d_streams_mid);
        free(d_attn_out);
        free(x_ln);
        return;
    }
    memset(dstreams_in, 0, stream_n * sizeof(float));

    for (int t = 0; t < T; t++) {
        float *lc = layer_cache + (size_t)t * ptok;
        const float *st_mid = lc + sm_off;
        const float *post_f = lc + ffn_off;
        const float *comb_f = post_f + (size_t)hc;
        memset(d_ffn, 0, (size_t)C * sizeof(float));
        ds4_hc_stream_update_backward(
            d_streams_mid + ((size_t)t * (size_t)hc) * (size_t)C,
            d_ffn,
            dstreams_out + ((size_t)t * (size_t)hc) * (size_t)C,
            st_mid,
            post_f,
            comb_f,
            hc,
            C);
    }

    for (int t = 0; t < T; t++) {
        float *lc = layer_cache + (size_t)t * ptok;
        const float *st_in = lc + sin_off;
        const float *post_a = lc;
        const float *comb_a = lc + (size_t)hc;
        float d_branch[64];
        ds4_hc_stream_update_backward(
            dstreams_in + ((size_t)t * (size_t)hc) * (size_t)C,
            d_branch,
            d_streams_mid + ((size_t)t * (size_t)hc) * (size_t)C,
            st_in,
            post_a,
            comb_a,
            hc,
            C);
        memcpy(d_attn_out + (size_t)t * (size_t)C, d_branch, (size_t)C * sizeof(float));
        const float *ln_out = lc + (size_t)hc + (size_t)hc * (size_t)hc + (size_t)hc * (size_t)C + (size_t)mix + (size_t)C + (size_t)C;
        memcpy(x_ln + (size_t)t * (size_t)C, ln_out, (size_t)C * sizeof(float));
    }

    if (attn_type == DS4_ATTN_SLIDING) {
        float d_ln[512];
        memset(d_ln, 0, (size_t)T * (size_t)C * sizeof(float));
        ds4_sliding_attn_backward(
            d_ln,
            d_wq_a,
            d_w_qa_norm,
            d_wq_b,
            d_wkv,
            d_w_kv_norm,
            d_attn_sink,
            d_wo_a,
            d_wo_b,
            d_attn_out,
            x_ln,
            T,
            cfg,
            w->wq_a,
            w->w_qa_norm,
            w->wq_b,
            w->wkv,
            w->w_kv_norm,
            w->attn_sink,
            w->wo_a,
            w->wo_b,
            attn_scratch,
            layer_cache + slide_off);
        for (int t = 0; t < T; t++) {
            float *lc = layer_cache + (size_t)t * ptok;
            const float *post_a = lc;
            const float *comb_a = lc + (size_t)hc;
            const float *flat_a = comb_a + (size_t)hc * (size_t)hc;
            const float *mixv_a = flat_a + (size_t)hc * (size_t)C;
            const float *coll_a = mixv_a + (size_t)mix;
            const float *ln_out = coll_a + (size_t)C;
            const float *st_in = lc + sin_off;
            float d_coll[64];
            ds4_rmsnorm_backward(d_coll, d_attn_norm_w, d_ln + (size_t)t * (size_t)C, coll_a, ln_out, w->attn_norm_w, 1, C, eps);
            ds4_hyper_connection_backward(
                dstreams_in + ((size_t)t * (size_t)hc) * (size_t)C,
                d_attn_hc_fn,
                d_attn_hc_base,
                d_attn_hc_scale,
                d_coll,
                st_in,
                post_a,
                comb_a,
                flat_a,
                mixv_a,
                cfg,
                w->attn_hc_fn,
                w->attn_hc_base,
                w->attn_hc_scale);
        }
    }
    (void)input_ids;
    (void)eps;
    free(d_streams_mid);
    free(d_attn_out);
    free(x_ln);
}
