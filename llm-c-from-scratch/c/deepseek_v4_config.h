/*
 * DeepSeek-V4 tiny config for educational C port (matches nano_deepseek_v4 defaults).
 * Full Flash config: vendor/nano-deepseek-v4/references/DeepSeek-V4-Flash-config.json
 */
#ifndef DEEPSEEK_V4_CONFIG_H
#define DEEPSEEK_V4_CONFIG_H

#include <stddef.h>

typedef enum {
    DS4_ATTN_SLIDING = 0,
    DS4_ATTN_CSA = 1,
    DS4_ATTN_HCA = 2,
} Ds4AttentionType;

typedef enum {
    DS4_MLP_HASH_MOE = 0,
    DS4_MLP_MOE = 1,
} Ds4MlpType;

typedef struct {
    int vocab_size;
    int hidden_size;
    int moe_intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads; /* V4: shared K=V MQA => 1 */
    int head_dim;
    int q_lora_rank;
    int num_experts_per_tok;
    int n_routed_experts;
    int n_shared_experts;
    int max_seq_len;
    int sliding_window;
    int compress_rate_csa;
    int compress_rate_hca;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int hc_mult;
    int hc_sinkhorn_iters;
    int num_hash_layers;
    int o_groups;
    int o_lora_rank;
    float rms_norm_eps;
    float hc_eps;
    float rope_theta;
    float compress_rope_theta;
    float partial_rotary_factor;
    float routed_scaling_factor;
    float swiglu_limit;
    /* per-layer schedules (length = num_hidden_layers) */
    Ds4AttentionType *layer_types;
    Ds4MlpType *mlp_layer_types;
} DeepSeekV4Config;

/* Build default tiny schedule (same logic as nano_deepseek_v4/config.py). */
void ds4_config_init_tiny(DeepSeekV4Config *cfg);
/* One sliding-attention + hash_moe layer (Phase 10 -train-1layer). */
void ds4_config_init_1layer(DeepSeekV4Config *cfg);
/* Two layers: sliding + HCA, both hash_moe (Phase 11 -train-full). */
void ds4_config_init_train_full(DeepSeekV4Config *cfg);

/* Four layers: sliding + HCA + CSA + sliding, all hash_moe (Phase 12 -train-4layer). */
void ds4_config_init_train_4layer(DeepSeekV4Config *cfg);
void ds4_config_free(DeepSeekV4Config *cfg);
void ds4_config_print(const DeepSeekV4Config *cfg);

/* Derived dims (match nano_deepseek_v4.config properties). */
int ds4_qk_rope_head_dim(const DeepSeekV4Config *cfg);
int ds4_attention_width(const DeepSeekV4Config *cfg);

#endif /* DEEPSEEK_V4_CONFIG_H */
