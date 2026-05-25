/*
 * Multi-Head Latent Attention (MLA) — educational C port
 *
 * Mirrors: llmc/deepseek_v2.py :: MultiHeadLatentAttention
 * Notebook: 12.MLA.ipynb
 * llm.c analogue: attention_forward() in vendor/llm.c/train_gpt2.c
 *
 * We use B=1 and small T/C so you can printf-debug every buffer.
 */
#ifndef DEEPSEEK_V2_MLA_H
#define DEEPSEEK_V2_MLA_H

#include <stddef.h>

/* Tiny dims — match DeepSeekV2Config.tiny() defaults in Python */
typedef struct {
    int n_embd;       /* C  — model width */
    int n_head;       /* NH — number of heads */
    int kv_lora_rank; /* latent KV size (what MLA caches at inference) */
    int block_size;   /* max T for causal mask */
} Dsv2MlaConfig;

/*
 * Workspace sizes (float32). Call once at startup.
 *
 * For one forward with sequence length T:
 *   - c_kv: (T, kv_lora_rank)  ← the compressed cache DeepSeek-V2 advertises
 *   - scratch for Q,K,V, attention scores, output
 */
size_t dsv2_mla_c_kv_bytes(const Dsv2MlaConfig *cfg, int T);
size_t dsv2_mla_scratch_bytes(const Dsv2MlaConfig *cfg, int T);

/*
 * Forward MLA on input x (T, C), write output out (T, C).
 *
 * Weights are stored row-major like PyTorch nn.Linear:
 *   y = x @ W^T   (same as F.linear / matmul in notebooks)
 *
 * wq:      (C, C)
 * w_dkv:   (kv_lora_rank, C)   down-project to latent
 * w_uk:    (C, kv_lora_rank)   up-project latent → keys
 * w_uv:    (C, kv_lora_rank)   up-project latent → values
 * wo:      (C, C)
 *
 * scratch: buffer from dsv2_mla_scratch_bytes(); c_kv_out may be NULL.
 */
void dsv2_mla_forward(
    float *out,
    float *c_kv_out,
    const float *x,
    const Dsv2MlaConfig *cfg,
    int T,
    const float *wq,
    const float *w_dkv,
    const float *w_uk,
    const float *w_uv,
    const float *wo,
    float *scratch);

/* Compare MLA vs classic MHA KV cache bytes per token (all layers = multiply by n_layer externally) */
size_t dsv2_mla_kv_cache_bytes_per_token(const Dsv2MlaConfig *cfg);
size_t dsv2_mha_kv_cache_bytes_per_token(const Dsv2MlaConfig *cfg);

#endif /* DEEPSEEK_V2_MLA_H */
