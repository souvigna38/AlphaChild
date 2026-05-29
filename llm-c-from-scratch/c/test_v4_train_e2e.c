#include "adamw.h"
#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "v4_layer_train.h"
#include "v4_model.h"
#include "v4_model.h"
#include "v4_train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_layer(Ds4LayerWeights *w, const DeepSeekV4Config *cfg, int layer_idx, int hash) {
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
    w->attn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->ffn_norm_w = (float *)calloc((size_t)C, sizeof(float));
    w->attn_norm_w[0] = w->ffn_norm_w[0] = 1.0f;
    w->attn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->attn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->ffn_hc_fn = (float *)calloc((size_t)mix * (size_t)C * (size_t)hc, sizeof(float));
    w->ffn_hc_base = (float *)calloc((size_t)mix, sizeof(float));
    w->attn_hc_scale[0] = w->ffn_hc_scale[0] = 1.0f;
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

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_train_4layer(&cfg);
    const int nL = cfg.num_hidden_layers;
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 31u;

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = NULL;
    mw.lm_head_tied = 0;
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc((size_t)nL, sizeof(Ds4LayerWeights));
    for (int L = 0; L < nL; L++) {
        fill_layer(&mw.layers[L], &cfg, L, 1);
    }
    for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
        seed = seed * 1103515245u + 12345u;
        mw.embed[i] = 0.002f * (float)((int)(seed % 1000) - 500);
    }
    ds4_model_weights_tie_lm_head(&mw);

    int ids[8];
    int targets[8];
    for (int t = 0; t < T; t++) {
        seed = seed * 1103515245u + 12345u;
        ids[t] = (int)(seed % (unsigned int)V);
        seed = seed * 1103515245u + 12345u;
        targets[t] = (int)(seed % (unsigned int)V);
    }

    size_t ngrad = ds4_model_train_e2e_param_count(&cfg);
    float *grad = (float *)calloc(ngrad, sizeof(float));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_model_train_full_layer_cache_floats(&cfg, T), sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    Ds4AdamW opt;
    ds4_adamw_init(&opt, ngrad, 0.01f, 0.0f);

    float loss0 = ds4_model_train_step_e2e(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);
    float loss1 = ds4_model_train_step_e2e(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);

    printf("train-e2e loss step0=%.4f step1=%.4f\n", loss0, loss1);
    if (!(loss1 <= loss0 + 0.2f)) {
        fprintf(stderr, "expected loss stable or decreasing with head+layers trainable\n");
        return 1;
    }

    float loss2 = ds4_model_train_step_e2e(&mw, &cfg, ids, targets, T, &opt, grad, sa, sb, model_scratch, layer_cache, logits, tok);
    Ds4E2eHeadGrads hg;
    ds4_e2e_head_grad_ptrs(&hg, grad, &cfg);
    float lm_sum = 0.0f;
    if (mw.lm_head_tied) {
        for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
            lm_sum += grad[i] * grad[i];
        }
    } else {
        for (size_t i = 0; i < (size_t)V * (size_t)C; i++) {
            lm_sum += hg.lm_head[i] * hg.lm_head[i];
        }
    }
    if (lm_sum <= 0.0f) {
        fprintf(stderr, "expected embed/lm_head grad in e2e step\n");
        return 1;
    }
    (void)loss2;

    ds4_adamw_free(&opt);
    free(tok);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(grad);
    ds4_config_free(&cfg);
    printf("OK — train-e2e (4-layer + lm_head/final_norm/hc_head AdamW)\n");
    return 0;
}
