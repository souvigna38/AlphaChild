#include "v4_layer.h"

#include "mhc.h"
#include "moe.h"
#include "rmsnorm.h"
#include "v4_attention.h"

#include <stddef.h>
#include <string.h>

size_t ds4_layer_scratch_bytes(const DeepSeekV4Config *cfg, int T) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    size_t work_floats =
        (size_t)T * (size_t)C * 4 + (size_t)T * (size_t)hc * 2 + (size_t)T * (size_t)hc * (size_t)hc * 2 +
        (size_t)T * (size_t)hc * (size_t)C;
    return ds4_attention_scratch_bytes(cfg, T) + ds4_moe_scratch_bytes(cfg) + work_floats * sizeof(float);
}

void ds4_decoder_layer_forward(
    float *streams_out,
    const float *streams_in,
    const int *input_ids,
    int T,
    int layer_idx,
    const DeepSeekV4Config *cfg,
    Ds4LayerWeights *w,
    float *attn_scratch,
    float *moe_scratch,
    float *work) {
    const int C = cfg->hidden_size;
    const int hc = cfg->hc_mult;
    const float eps = cfg->rms_norm_eps;
    Ds4AttentionType attn_type = cfg->layer_types[layer_idx];
    Ds4MlpType mlp_type = cfg->mlp_layer_types[layer_idx];

    float *collapsed = work;
    float *attn_out = collapsed + (size_t)T * (size_t)C;
    float *ffn_out = attn_out + (size_t)T * (size_t)C;
    float *norm_buf = ffn_out + (size_t)T * (size_t)C;
    float *post_a = norm_buf + (size_t)T * (size_t)C;
    float *comb_a = post_a + (size_t)T * (size_t)hc;
    float *post_f = comb_a + (size_t)T * (size_t)hc * (size_t)hc;
    float *comb_f = post_f + (size_t)T * (size_t)hc;
    float *stream_tmp = comb_f + (size_t)T * (size_t)hc * (size_t)hc;

    for (int t = 0; t < T; t++) {
        const float *st = streams_in + ((size_t)t * (size_t)hc) * (size_t)C;
        float *post_t = post_a + (size_t)t * (size_t)hc;
        float *comb_t = comb_a + (size_t)t * (size_t)hc * (size_t)hc;
        float *coll_t = collapsed + (size_t)t * (size_t)C;
        ds4_hyper_connection_forward(post_t, comb_t, coll_t, st, cfg, w->attn_hc_fn, w->attn_hc_base, w->attn_hc_scale);
        ds4_rmsnorm_forward(norm_buf + (size_t)t * (size_t)C, coll_t, w->attn_norm_w, 1, C, eps);
    }

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

    for (int t = 0; t < T; t++) {
        const float *st = streams_in + ((size_t)t * (size_t)hc) * (size_t)C;
        float *st_out = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        ds4_hc_stream_update(
            st_out,
            st,
            post_a + (size_t)t * (size_t)hc,
            comb_a + (size_t)t * (size_t)hc * (size_t)hc,
            attn_out + (size_t)t * (size_t)C,
            hc,
            C);
    }

    for (int t = 0; t < T; t++) {
        const float *st = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        float *post_t = post_f + (size_t)t * (size_t)hc;
        float *comb_t = comb_f + (size_t)t * (size_t)hc * (size_t)hc;
        float *coll_t = collapsed + (size_t)t * (size_t)C;
        ds4_hyper_connection_forward(post_t, comb_t, coll_t, st, cfg, w->ffn_hc_fn, w->ffn_hc_base, w->ffn_hc_scale);
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
        float *st_tmp = stream_tmp + ((size_t)t * (size_t)hc) * (size_t)C;
        float *st_final = streams_out + ((size_t)t * (size_t)hc) * (size_t)C;
        memcpy(st_tmp, st, (size_t)hc * (size_t)C * sizeof(float));
        ds4_hc_stream_update(
            st_final,
            st_tmp,
            post_f + (size_t)t * (size_t)hc,
            comb_f + (size_t)t * (size_t)hc * (size_t)hc,
            ffn_out + (size_t)t * (size_t)C,
            hc,
            C);
    }
}
