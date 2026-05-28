#include "v4_parity.h"

#include "hash_moe.h"
#include "v4_model.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

unsigned ds4_parity_rng_step(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return *seed;
}

void ds4_parity_fill_buffer(float *buf, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; i++) {
        unsigned s = ds4_parity_rng_step(seed);
        buf[i] = 0.002f * (float)((int)(s % 1000u) - 500);
    }
}

static void fill_norm1(float *w, unsigned *seed) {
    w[0] = 1.0f;
    (void)seed;
}

static void fill_layer(Ds4LayerWeights *w, const DeepSeekV4Config *cfg, int layer_idx, int hash, unsigned *seed) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int attn_w = cfg->num_attention_heads * cfg->head_dim;
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    const int k = cfg->num_experts_per_tok;
    const int D = cfg->head_dim;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * D;
    const int rate_csa = cfg->compress_rate_csa;
    const int rate_hca = cfg->compress_rate_hca;
    Ds4AttentionType attn = cfg->layer_types[layer_idx];

    memset(w, 0, sizeof(*w));
    w->moe_gate = (float *)malloc((size_t)E * (size_t)C * sizeof(float));
    w->moe_expert_gu = (float *)malloc((size_t)E * (size_t)I * 2 * (size_t)C * sizeof(float));
    w->moe_expert_down = (float *)malloc((size_t)E * (size_t)C * (size_t)I * sizeof(float));
    w->moe_shared_gu = (float *)malloc((size_t)cfg->n_shared_experts * (size_t)I * 2 * (size_t)C * sizeof(float));
    w->moe_shared_down = (float *)malloc((size_t)C * (size_t)I * sizeof(float));
    if (!hash) {
        w->moe_route_bias = (float *)malloc((size_t)E * sizeof(float));
        ds4_parity_fill_buffer(w->moe_route_bias, (size_t)E, seed);
    }
    w->attn_norm_w = (float *)malloc((size_t)C * sizeof(float));
    w->ffn_norm_w = (float *)malloc((size_t)C * sizeof(float));
    w->attn_hc_fn = (float *)malloc((size_t)mix * (size_t)C * (size_t)hc * sizeof(float));
    w->attn_hc_base = (float *)malloc((size_t)mix * sizeof(float));
    w->ffn_hc_fn = (float *)malloc((size_t)mix * (size_t)C * (size_t)hc * sizeof(float));
    w->ffn_hc_base = (float *)malloc((size_t)mix * sizeof(float));
    w->wq_a = (float *)malloc((size_t)cfg->q_lora_rank * (size_t)C * sizeof(float));
    w->w_qa_norm = (float *)malloc((size_t)cfg->q_lora_rank * sizeof(float));
    w->wq_b = (float *)malloc((size_t)attn_w * (size_t)cfg->q_lora_rank * sizeof(float));
    w->wkv = (float *)malloc((size_t)D * (size_t)C * sizeof(float));
    w->w_kv_norm = (float *)malloc((size_t)D * sizeof(float));
    w->attn_sink = (float *)malloc((size_t)cfg->num_attention_heads * sizeof(float));
    w->wo_a = (float *)malloc((size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(attn_w / cfg->o_groups) * sizeof(float));
    w->wo_b = (float *)malloc((size_t)C * (size_t)o_mid * sizeof(float));

    ds4_parity_fill_buffer(w->moe_gate, (size_t)E * (size_t)C, seed);
    ds4_parity_fill_buffer(w->moe_expert_gu, (size_t)E * (size_t)I * 2 * (size_t)C, seed);
    ds4_parity_fill_buffer(w->moe_expert_down, (size_t)E * (size_t)C * (size_t)I, seed);
    ds4_parity_fill_buffer(w->moe_shared_gu, (size_t)cfg->n_shared_experts * (size_t)I * 2 * (size_t)C, seed);
    ds4_parity_fill_buffer(w->moe_shared_down, (size_t)C * (size_t)I, seed);
    ds4_parity_fill_buffer(w->attn_hc_fn, (size_t)mix * (size_t)C * (size_t)hc, seed);
    ds4_parity_fill_buffer(w->attn_hc_base, (size_t)mix, seed);
    ds4_parity_fill_buffer(w->ffn_hc_fn, (size_t)mix * (size_t)C * (size_t)hc, seed);
    ds4_parity_fill_buffer(w->ffn_hc_base, (size_t)mix, seed);
    ds4_parity_fill_buffer(w->wq_a, (size_t)cfg->q_lora_rank * (size_t)C, seed);
    ds4_parity_fill_buffer(w->wq_b, (size_t)attn_w * (size_t)cfg->q_lora_rank, seed);
    ds4_parity_fill_buffer(w->wkv, (size_t)D * (size_t)C, seed);
    ds4_parity_fill_buffer(w->attn_sink, (size_t)cfg->num_attention_heads, seed);
    ds4_parity_fill_buffer(w->wo_a, (size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(attn_w / cfg->o_groups), seed);
    ds4_parity_fill_buffer(w->wo_b, (size_t)C * (size_t)o_mid, seed);
    fill_norm1(w->attn_norm_w, seed);
    fill_norm1(w->ffn_norm_w, seed);
    fill_norm1(w->w_qa_norm, seed);
    fill_norm1(w->w_kv_norm, seed);
    w->attn_hc_scale[0] = w->attn_hc_scale[1] = w->attn_hc_scale[2] = 1.0f;
    w->ffn_hc_scale[0] = w->ffn_hc_scale[1] = w->ffn_hc_scale[2] = 1.0f;

    if (attn == DS4_ATTN_HCA) {
        w->hca_w_kv = (float *)malloc((size_t)D * (size_t)C * sizeof(float));
        w->hca_w_gate = (float *)malloc((size_t)D * (size_t)C * sizeof(float));
        w->hca_pos_bias = (float *)malloc((size_t)rate_hca * (size_t)D * sizeof(float));
        w->hca_norm = (float *)malloc((size_t)D * sizeof(float));
        ds4_parity_fill_buffer(w->hca_w_kv, (size_t)D * (size_t)C, seed);
        ds4_parity_fill_buffer(w->hca_w_gate, (size_t)D * (size_t)C, seed);
        ds4_parity_fill_buffer(w->hca_pos_bias, (size_t)rate_hca * (size_t)D, seed);
        fill_norm1(w->hca_norm, seed);
    } else if (attn == DS4_ATTN_CSA) {
        w->csa_w_kv = (float *)malloc((size_t)HD2 * (size_t)C * sizeof(float));
        w->csa_w_gate = (float *)malloc((size_t)HD2 * (size_t)C * sizeof(float));
        w->csa_pos_bias = (float *)malloc((size_t)rate_csa * (size_t)HD2 * sizeof(float));
        w->csa_norm = (float *)malloc((size_t)D * sizeof(float));
        w->idx_wq_b = (float *)malloc((size_t)(cfg->index_n_heads * HD) * (size_t)cfg->q_lora_rank * sizeof(float));
        w->idx_w_weights = (float *)malloc((size_t)cfg->index_n_heads * (size_t)C * sizeof(float));
        w->idx_w_kv = (float *)malloc((size_t)(2 * HD) * (size_t)C * sizeof(float));
        w->idx_w_gate = (float *)malloc((size_t)(2 * HD) * (size_t)C * sizeof(float));
        w->idx_pos_bias = (float *)malloc((size_t)rate_csa * (size_t)(2 * HD) * sizeof(float));
        w->idx_norm = (float *)malloc((size_t)HD * sizeof(float));
        ds4_parity_fill_buffer(w->csa_w_kv, (size_t)HD2 * (size_t)C, seed);
        ds4_parity_fill_buffer(w->csa_w_gate, (size_t)HD2 * (size_t)C, seed);
        ds4_parity_fill_buffer(w->csa_pos_bias, (size_t)rate_csa * (size_t)HD2, seed);
        ds4_parity_fill_buffer(w->idx_wq_b, (size_t)(cfg->index_n_heads * HD) * (size_t)cfg->q_lora_rank, seed);
        ds4_parity_fill_buffer(w->idx_w_weights, (size_t)cfg->index_n_heads * (size_t)C, seed);
        ds4_parity_fill_buffer(w->idx_w_kv, (size_t)(2 * HD) * (size_t)C, seed);
        ds4_parity_fill_buffer(w->idx_w_gate, (size_t)(2 * HD) * (size_t)C, seed);
        ds4_parity_fill_buffer(w->idx_pos_bias, (size_t)rate_csa * (size_t)(2 * HD), seed);
        fill_norm1(w->csa_norm, seed);
        fill_norm1(w->idx_norm, seed);
    }
    if (hash) {
        w->tid2eid = (int *)malloc((size_t)cfg->vocab_size * (size_t)k * sizeof(int));
        ds4_hash_moe_build_table(w->tid2eid, cfg);
    }
}

void ds4_parity_fill_model(Ds4ModelWeights *mw, const DeepSeekV4Config *cfg, unsigned seed) {
    const int C = cfg->hidden_size;
    const int V = cfg->vocab_size;
    const int hc = cfg->hc_mult;
    const int nL = cfg->num_hidden_layers;

    mw->embed = (float *)malloc((size_t)V * (size_t)C * sizeof(float));
    mw->lm_head = (float *)malloc((size_t)V * (size_t)C * sizeof(float));
    mw->final_norm = (float *)malloc((size_t)C * sizeof(float));
    mw->hc_head_fn = (float *)malloc((size_t)hc * (size_t)hc * (size_t)C * sizeof(float));
    mw->hc_head_base = (float *)malloc((size_t)hc * sizeof(float));
    mw->layers = (Ds4LayerWeights *)calloc((size_t)nL, sizeof(Ds4LayerWeights));

    ds4_parity_fill_buffer(mw->embed, (size_t)V * (size_t)C, &seed);
    ds4_parity_fill_buffer(mw->lm_head, (size_t)V * (size_t)C, &seed);
    ds4_parity_fill_buffer(mw->hc_head_fn, (size_t)hc * (size_t)hc * (size_t)C, &seed);
    ds4_parity_fill_buffer(mw->hc_head_base, (size_t)hc, &seed);
    fill_norm1(mw->final_norm, &seed);
    mw->hc_head_scale[0] = 1.0f;

    for (int L = 0; L < nL; L++) {
        fill_layer(&mw->layers[L], cfg, L, cfg->mlp_layer_types[L] == DS4_MLP_HASH_MOE, &seed);
    }
}

void ds4_parity_forward_logits(
    float *logits,
    const int *input_ids,
    int T,
    const DeepSeekV4Config *cfg,
    Ds4ModelWeights *mw) {
    const int hc = cfg->hc_mult;
    float *sa = (float *)malloc((size_t)T * (size_t)hc * (size_t)cfg->hidden_size * sizeof(float));
    float *sb = (float *)malloc((size_t)T * (size_t)hc * (size_t)cfg->hidden_size * sizeof(float));
    float *scratch = (float *)malloc(ds4_model_scratch_bytes(cfg, T));
    if (!sa || !sb || !scratch) {
        free(scratch);
        free(sb);
        free(sa);
        return;
    }
    ds4_model_forward(logits, input_ids, T, cfg, mw, sa, sb, scratch, NULL, NULL);
    free(scratch);
    free(sb);
    free(sa);
}

void ds4_parity_free_model(Ds4ModelWeights *mw, const DeepSeekV4Config *cfg) {
    const int nL = cfg->num_hidden_layers;
    if (mw->layers) {
        for (int L = 0; L < nL; L++) {
            Ds4LayerWeights *w = &mw->layers[L];
            free(w->moe_gate);
            free(w->moe_route_bias);
            free(w->moe_expert_gu);
            free(w->moe_expert_down);
            free(w->moe_shared_gu);
            free(w->moe_shared_down);
            free(w->tid2eid);
            free(w->attn_norm_w);
            free(w->ffn_norm_w);
            free(w->attn_hc_fn);
            free(w->attn_hc_base);
            free(w->ffn_hc_fn);
            free(w->ffn_hc_base);
            free(w->wq_a);
            free(w->w_qa_norm);
            free(w->wq_b);
            free(w->wkv);
            free(w->w_kv_norm);
            free(w->attn_sink);
            free(w->wo_a);
            free(w->wo_b);
            free(w->hca_w_kv);
            free(w->hca_w_gate);
            free(w->hca_pos_bias);
            free(w->hca_norm);
            free(w->csa_w_kv);
            free(w->csa_w_gate);
            free(w->csa_pos_bias);
            free(w->csa_norm);
            free(w->idx_wq_b);
            free(w->idx_w_weights);
            free(w->idx_w_kv);
            free(w->idx_w_gate);
            free(w->idx_pos_bias);
            free(w->idx_norm);
        }
        free(mw->layers);
        mw->layers = NULL;
    }
    free(mw->embed);
    free(mw->lm_head);
    free(mw->final_norm);
    free(mw->hc_head_fn);
    free(mw->hc_head_base);
    mw->embed = mw->lm_head = mw->final_norm = mw->hc_head_fn = mw->hc_head_base = NULL;
}
