/*
 * Tiny DeepSeek-V2 model — forward, loss, sample (notebook 14–16).
 * Phase 5/5b C training helpers; use PyTorch notebook 15 for long training runs.
 */
#ifndef DEEPSEEK_V2_MODEL_H
#define DEEPSEEK_V2_MODEL_H

#include "adamw.h"
#include "block.h"

#include <stddef.h>
#include <stdint.h>

#define DSV2_CKPT_MAGIC 0x32565344u /* 'DSV2' little-endian */

typedef struct {
    int vocab_size;
    int block_size;
    int n_layer;
    int n_embd;
    int n_head;
    int kv_lora_rank;
    int n_routed_experts;
    int n_shared_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    float rms_eps;
} Dsv2ModelConfig;

typedef struct {
    Dsv2ModelConfig cfg;
    float *wte; /* (V, C) tied with lm_head */
    float *rms_f;
    /* Per-layer pointers (length n_layer each, contiguous backing) */
    float *rms1_w;
    float *rms2_w;
    float *wq;
    float *w_dkv;
    float *w_uk;
    float *w_uv;
    float *wo;
    float *gate;
    float *expert_w1;
    float *expert_w2;
    float *expert_w3;
    float *shared_w1;
    float *shared_w2;
    float *shared_w3;
    float *memory; /* owns all */
} Dsv2Model;

size_t dsv2_model_param_bytes(const Dsv2ModelConfig *cfg);
size_t dsv2_model_activations_bytes(const Dsv2ModelConfig *cfg, int B, int T);

void dsv2_model_init(Dsv2Model *m, const Dsv2ModelConfig *cfg, unsigned int seed);
void dsv2_model_free(Dsv2Model *m);

/* idx/targets (B,T) int32 row-major; logits (B,T,V) */
void dsv2_model_forward(
    float *logits,
    const int *idx,
    const Dsv2Model *m,
    int B,
    int T,
    float *activations);

float dsv2_cross_entropy_loss(const float *logits, const int *targets, int B, int T, int V, int C);

/* Greedy decode one token from logits at last time step */
int dsv2_sample_greedy_last(const float *logits, int B, int T, int V);

/* Head-only SGD step (tied wte/lm_head) — demo C training */
void dsv2_train_head_step(
    Dsv2Model *m,
    const int *idx,
    const int *targets,
    int B,
    int T,
    float lr,
    float *activations,
    float *logits);

/* Load/save checkpoint (same layout as scripts/export_v2_tiny.py) */
int dsv2_model_load_checkpoint(Dsv2Model *m, const char *path);
int dsv2_model_save_checkpoint(const Dsv2Model *m, const char *path);

typedef struct {
    float lr;
    float grad_clip;   /* 0 = no clip */
    Dsv2AdamW *adam;   /* non-NULL → AdamW; else SGD on all params */
} Dsv2TrainConfig;

/*
 * Phase 5: one training step (n_layer must be 1).
 * Updates MLA + RMSNorm + tied wte; MoE forward only (frozen in backward).
 * grad_memory must hold 2 * dsv2_model_param_bytes.
 */
float dsv2_model_train_step_1layer(
    Dsv2Model *m,
    const int *idx,
    const int *targets,
    int B,
    int T,
    float lr,
    float *activations,
    float *logits,
    float *grad_memory);

/* Phase 5b/6: all layers, MLA + MoE backward; B>=1 (gradients averaged over batch) */
size_t dsv2_model_train_working_bytes(const Dsv2ModelConfig *cfg, int B, int T);
float *dsv2_train_grad_ptr(char *work, const Dsv2ModelConfig *cfg, int B, int T);
float dsv2_model_train_step_full(
    Dsv2Model *m,
    const int *idx,
    const int *targets,
    int B,
    int T,
    const Dsv2TrainConfig *tc,
    float *activations,
    float *logits,
    float *grad_memory,
    int *moe_topi);

#endif /* DEEPSEEK_V2_MODEL_H */
