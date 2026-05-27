#include "deepseek_v4_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ds4_fill_default_schedules(DeepSeekV4Config *cfg) {
    int n = cfg->num_hidden_layers;
    cfg->layer_types = (Ds4AttentionType *)malloc((size_t)n * sizeof(Ds4AttentionType));
    cfg->mlp_layer_types = (Ds4MlpType *)malloc((size_t)n * sizeof(Ds4MlpType));
    for (int i = 0; i < n; i++) {
        if (i < 2) {
            cfg->layer_types[i] = DS4_ATTN_SLIDING;
        } else {
            cfg->layer_types[i] = ((i - 2) % 2 == 0) ? DS4_ATTN_CSA : DS4_ATTN_HCA;
        }
        cfg->mlp_layer_types[i] = (i < cfg->num_hash_layers) ? DS4_MLP_HASH_MOE : DS4_MLP_MOE;
    }
}

void ds4_config_init_tiny(DeepSeekV4Config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->vocab_size = 512;
    cfg->hidden_size = 64;
    cfg->moe_intermediate_size = 96;
    cfg->num_hidden_layers = 4;
    cfg->num_attention_heads = 4;
    cfg->num_key_value_heads = 1;
    cfg->head_dim = 16;
    cfg->q_lora_rank = 32;
    cfg->num_experts_per_tok = 2;
    cfg->n_routed_experts = 8;
    cfg->n_shared_experts = 1;
    cfg->max_seq_len = 256;
    cfg->sliding_window = 8;
    cfg->compress_rate_csa = 4;
    cfg->compress_rate_hca = 128;
    cfg->index_n_heads = 4;
    cfg->index_head_dim = 8;
    cfg->index_topk = 4;
    cfg->hc_mult = 4;
    cfg->hc_sinkhorn_iters = 8;
    cfg->num_hash_layers = 3;
    cfg->o_groups = 2;
    cfg->o_lora_rank = 16;
    cfg->rms_norm_eps = 1e-6f;
    cfg->hc_eps = 1e-6f;
    cfg->rope_theta = 10000.0f;
    cfg->compress_rope_theta = 160000.0f;
    cfg->partial_rotary_factor = 0.5f;
    cfg->routed_scaling_factor = 1.5f;
    cfg->swiglu_limit = 10.0f;
    ds4_fill_default_schedules(cfg);
}

void ds4_config_init_1layer(DeepSeekV4Config *cfg) {
    ds4_config_init_tiny(cfg);
    free(cfg->layer_types);
    free(cfg->mlp_layer_types);
    cfg->num_hidden_layers = 1;
    cfg->num_hash_layers = 1;
    ds4_fill_default_schedules(cfg);
}

void ds4_config_free(DeepSeekV4Config *cfg) {
    free(cfg->layer_types);
    free(cfg->mlp_layer_types);
    cfg->layer_types = NULL;
    cfg->mlp_layer_types = NULL;
}

static const char *attn_name(Ds4AttentionType t) {
    switch (t) {
    case DS4_ATTN_SLIDING:
        return "sliding_attention";
    case DS4_ATTN_CSA:
        return "compressed_sparse_attention";
    case DS4_ATTN_HCA:
        return "heavily_compressed_attention";
    default:
        return "?";
    }
}

static const char *mlp_name(Ds4MlpType t) {
    return (t == DS4_MLP_HASH_MOE) ? "hash_moe" : "moe";
}

int ds4_qk_rope_head_dim(const DeepSeekV4Config *cfg) {
    return (int)((float)cfg->head_dim * cfg->partial_rotary_factor);
}

int ds4_attention_width(const DeepSeekV4Config *cfg) {
    return cfg->num_attention_heads * cfg->head_dim;
}

void ds4_config_print(const DeepSeekV4Config *cfg) {
    printf("DeepSeek-V4 tiny (C educational port)\n");
    printf("  hidden_size=%d layers=%d heads=%d head_dim=%d\n", cfg->hidden_size, cfg->num_hidden_layers,
           cfg->num_attention_heads, cfg->head_dim);
    printf("  MoE: routed=%d topk=%d shared=%d hash_layers=%d\n", cfg->n_routed_experts, cfg->num_experts_per_tok,
           cfg->n_shared_experts, cfg->num_hash_layers);
    printf("  mHC: hc_mult=%d sinkhorn_iters=%d\n", cfg->hc_mult, cfg->hc_sinkhorn_iters);
    printf("  compress: csa_rate=%d hca_rate=%d index_topk=%d\n", cfg->compress_rate_csa, cfg->compress_rate_hca,
           cfg->index_topk);
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        printf("  layer %d: attn=%s mlp=%s\n", i, attn_name(cfg->layer_types[i]), mlp_name(cfg->mlp_layer_types[i]));
    }
}
