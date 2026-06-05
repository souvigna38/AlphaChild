#include "deepseek_v4_config.h"
#include "hash_moe.h"
#include "v4_layer_train.h"
#include "v4_model.h"
#include "v4_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_layer_min(Ds4LayerWeights *w, const DeepSeekV4Config *cfg, int hash) {
    const int C = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate_size;
    const int hc = cfg->hc_mult;
    const int mix = (2 + hc) * hc;
    const int attn_w = ds4_attention_width(cfg);
    const int o_mid = cfg->o_groups * cfg->o_lora_rank;
    const int k = cfg->num_experts_per_tok;

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
    if (hash) {
        w->tid2eid = (int *)malloc((size_t)cfg->vocab_size * (size_t)k * sizeof(int));
        ds4_hash_moe_build_table(w->tid2eid, cfg);
    }
}

static void rand_small(float *p, size_t n, unsigned int *seed) {
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        p[i] = 0.002f * (float)((int)(*seed % 1000) - 500);
    }
}

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_1layer(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int hc = cfg.hc_mult;
    unsigned int seed = 7u;

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.final_norm = (float *)calloc((size_t)C, sizeof(float));
    mw.final_norm[0] = 1.0f;
    mw.hc_head_fn = (float *)calloc((size_t)hc * (size_t)hc * (size_t)C, sizeof(float));
    mw.hc_head_base = (float *)calloc((size_t)hc, sizeof(float));
    mw.hc_head_scale[0] = 1.0f;
    mw.layers = (Ds4LayerWeights *)calloc(1, sizeof(Ds4LayerWeights));
    fill_layer_min(&mw.layers[0], &cfg, 1);
    rand_small(mw.embed, (size_t)V * (size_t)C, &seed);

    int ids[8];
    int targets[8];
    for (int t = 0; t < T; t++) {
        seed = seed * 1103515245u + 12345u;
        ids[t] = (int)(seed % (unsigned int)V);
        seed = seed * 1103515245u + 12345u;
        targets[t] = (int)(seed % (unsigned int)V);
    }

    size_t ngrad = ds4_model_1layer_param_count(&cfg);
    float *grad = (float *)calloc(ngrad, sizeof(float));
    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    float *sa = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *sb = (float *)calloc((size_t)T * (size_t)hc * (size_t)C, sizeof(float));
    float *model_scratch = (float *)malloc(ds4_model_scratch_bytes(&cfg, T));
    float *layer_cache = (float *)calloc(ds4_layer_train_cache_floats(&cfg, T), sizeof(float));
    float *tok = (float *)calloc((size_t)V, sizeof(float));

    float loss0 = ds4_model_train_step_1layer(&mw, &cfg, ids, targets, T, 0.05f, grad, sa, sb, model_scratch, layer_cache, logits, tok);
    float loss1 = ds4_model_train_step_1layer(&mw, &cfg, ids, targets, T, 0.05f, grad, sa, sb, model_scratch, layer_cache, logits, tok);

    printf("train-1layer loss step0=%.4f step1=%.4f\n", loss0, loss1);
    if (!(loss1 <= loss0 + 0.1f)) {
        fprintf(stderr, "expected loss stable or decreasing\n");
        return 1;
    }

    free(tok);
    free(layer_cache);
    free(model_scratch);
    free(sa);
    free(sb);
    free(logits);
    free(grad);
    ds4_config_free(&cfg);
    printf("OK — train-1layer (sliding attn + attn mHC, MoE frozen)\n");
    return 0;
}
