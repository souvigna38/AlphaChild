#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "v4_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_layer_min(Ds4LayerWeights *w, const DeepSeekV4Config *cfg, int layer_idx, int hash) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int attn_w = ds4_attention_width(cfg);
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    const int k = cfg->num_experts_per_tok;
    const int D = cfg->head_dim;
    const int HD = cfg->index_head_dim;
    const int HD2 = 2 * D;
    const int rate_csa = cfg->compress_rate_csa;
    const int rate_hca = cfg->compress_rate_hca;
    Ds4AttentionType attn = cfg->layer_types[layer_idx];

    memset(w, 0, sizeof(*w));
    w->moe_gate = (float *)calloc((size_t)E * (size_t)C, sizeof(float));
    w->moe_expert_gu = (float *)calloc((size_t)E * (size_t)I * 2 * (size_t)C, sizeof(float));
    w->moe_expert_down = (float *)calloc((size_t)E * (size_t)C * (size_t)I, sizeof(float));
    w->moe_shared_gu = (float *)calloc((size_t)cfg->n_shared_experts * (size_t)I * 2 * (size_t)C, sizeof(float));
    w->moe_shared_down = (float *)calloc((size_t)C * (size_t)I, sizeof(float));
    if (!hash) {
        w->moe_route_bias = (float *)calloc((size_t)E, sizeof(float));
    }
    w->attn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->ffn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->attn_norm_w[0] = w->ffn_norm_w[0] = 1.0f;
    w->attn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->attn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->ffn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->ffn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->attn_hc_scale[0] = w->attn_hc_scale[1] = w->attn_hc_scale[2] = 1.0f;
    w->ffn_hc_scale[0] = w->ffn_hc_scale[1] = w->ffn_hc_scale[2] = 1.0f;
    w->wq_a = (float *)calloc((size_t)cfg->q_lora_rank * (size_t)C, sizeof(float));
    w->w_qa_norm = (float *)calloc((size_t)cfg->q_lora_rank, sizeof(float));
    w->wq_b = (float *)calloc((size_t)attn_w * (size_t)cfg->q_lora_rank, sizeof(float));
    w->wkv = (float *)calloc((size_t)C * (size_t)C, sizeof(float));
    w->w_kv_norm = (float *)calloc((size_t)C, sizeof(float));
    w->attn_sink = (float *)calloc((size_t)cfg->num_attention_heads, sizeof(float));
    w->wo_a = (float *)calloc((size_t)cfg->o_groups * (size_t)cfg->o_lora_rank * (size_t)(attn_w / cfg->o_groups), sizeof(float));
    w->wo_b = (float *)calloc((size_t)C * (size_t)o_mid, sizeof(float));
    w->w_qa_norm[0] = w->w_kv_norm[0] = 1.0f;
    if (attn == DS4_ATTN_HCA) {
        w->hca_w_kv = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
        w->hca_w_gate = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
        w->hca_pos_bias = (float *)calloc((size_t)rate_hca * (size_t)D, sizeof(float));
        w->hca_norm = (float *)calloc((size_t)D, sizeof(float));
        w->hca_norm[0] = 1.0f;
    } else if (attn == DS4_ATTN_CSA) {
        w->csa_w_kv = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
        w->csa_w_gate = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
        w->csa_pos_bias = (float *)calloc((size_t)rate_csa * (size_t)HD2, sizeof(float));
        w->csa_norm = (float *)calloc((size_t)D, sizeof(float));
        w->csa_norm[0] = 1.0f;
        w->idx_wq_b = (float *)calloc((size_t)(cfg->index_n_heads * HD) * (size_t)cfg->q_lora_rank, sizeof(float));
        w->idx_w_weights = (float *)calloc((size_t)cfg->index_n_heads * (size_t)C, sizeof(float));
        w->idx_w_kv = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
        w->idx_w_gate = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
        w->idx_pos_bias = (float *)calloc((size_t)rate_csa * (size_t)(2 * HD), sizeof(float));
        w->idx_norm = (float *)calloc((size_t)HD, sizeof(float));
        w->idx_norm[0] = 1.0f;
    }
    if (hash) {
        w->tid2eid = (int *)malloc((size_t)cfg->vocab_size * (size_t)k * sizeof(int));
        ds4_hash_moe_build_table(w->tid2eid, cfg);
    }
}

static void free_layer_min(Ds4LayerWeights *w) {
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

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    const int nL = cfg.num_hidden_layers;

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc((size_t)nL, sizeof(Ds4LayerWeights));
    for (int i = 0; i < nL; i++) {
        Ds4LayerWeights *w = &mw.layers[i];
        const int hash = cfg.mlp_layer_types[i] == DS4_MLP_HASH_MOE;
        fill_layer_min(w, &cfg, i, hash);
    }

    int ids[8] = {1, 4, 7, 12, 3, 8, 15, 2};
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    ds4_model_forward(logits, ids, T, &cfg, &mw, sa, sb, scratch);
    printf("model forward logits[0..3] = %.6f %.6f %.6f %.6f\n", logits[0], logits[1], logits[2], logits[3]);

    free(scratch);
    free(sa);
    free(sb);
    free(logits);
    for (int i = 0; i < nL; i++) {
        free_layer_min(&mw.layers[i]);
    }
    free(mw.layers);
    free(mw.embed);
    free(mw.lm_head);
    free(mw.final_norm);
    free(mw.hc_head_fn);
    free(mw.hc_head_base);
    ds4_config_free(&cfg);
    printf("OK — full DeepSeek-V4 tiny forward (%d layers, T=%d)\n", nL, T);
    return 0;
}
