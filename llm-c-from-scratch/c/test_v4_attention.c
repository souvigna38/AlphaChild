#include "deepseek_v4_config.h"
#include "v4_attention.h"

#include <stdio.h>
#include <stdlib.h>

static void smoke_layer(Ds4AttentionType type, const char *name) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    if (type == DS4_ATTN_HCA) {
        cfg.compress_rate_hca = 4;
    }
    const int T = 16;
    const int C = cfg.hidden_size;
    const int NH = cfg.num_attention_heads;
    const int D = cfg.head_dim;
    const int r = cfg.q_lora_rank;
    const int attn_w = ds4_attention_width(&cfg);
    const int o_mid = cfg.o_groups * cfg.o_lora_rank;
    const int HD = cfg.index_head_dim;
    const int HD2 = 2 * D;

    size_t scratch_n = ds4_attention_scratch_bytes(&cfg, T);
    float *scratch = (float *)malloc(scratch_n);
    float *x = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *wq_a = (float *)calloc((size_t)r * (size_t)C, sizeof(float));
    float *wqa_n = (float *)calloc((size_t)r, sizeof(float));
    float *wq_b = (float *)calloc((size_t)attn_w * (size_t)r, sizeof(float));
    float *wkv = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
    float *wkvn = (float *)calloc((size_t)D, sizeof(float));
    float *sink = (float *)calloc((size_t)NH, sizeof(float));
    float *wo_a = (float *)calloc((size_t)cfg.o_groups * (size_t)cfg.o_lora_rank * (size_t)(attn_w / cfg.o_groups), sizeof(float));
    float *wo_b = (float *)calloc((size_t)C * (size_t)o_mid, sizeof(float));
    float *hca_kv = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
    float *hca_g = (float *)calloc((size_t)D * (size_t)C, sizeof(float));
    float *hca_pb = (float *)calloc((size_t)cfg.compress_rate_hca * (size_t)D, sizeof(float));
    float *hca_n = (float *)calloc((size_t)D, sizeof(float));
    float *csa_kv = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *csa_g = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *csa_pb = (float *)calloc((size_t)cfg.compress_rate_csa * (size_t)HD2, sizeof(float));
    float *csa_n = (float *)calloc((size_t)D, sizeof(float));
    float *idx_qb = (float *)calloc((size_t)(cfg.index_n_heads * HD) * (size_t)r, sizeof(float));
    float *idx_w = (float *)calloc((size_t)cfg.index_n_heads * (size_t)C, sizeof(float));
    float *idx_kv = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
    float *idx_g = (float *)calloc((size_t)(2 * HD) * (size_t)C, sizeof(float));
    float *idx_pb = (float *)calloc((size_t)cfg.compress_rate_csa * (size_t)(2 * HD), sizeof(float));
    float *idx_n = (float *)calloc((size_t)HD, sizeof(float));

    for (int i = 0; i < r; i++) {
        wqa_n[i] = 1.0f;
    }
    for (int i = 0; i < D; i++) {
        wkvn[i] = 1.0f;
        hca_n[i] = 1.0f;
        csa_n[i] = 1.0f;
    }
    for (int i = 0; i < HD; i++) {
        idx_n[i] = 1.0f;
    }
    for (size_t i = 0; i < (size_t)r * (size_t)C; i++) {
        wq_a[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)attn_w * (size_t)r; i++) {
        wq_b[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)D * (size_t)C; i++) {
        wkv[i] = 0.001f;
        hca_kv[i] = 0.001f;
        hca_g[i] = 0.001f;
    }
    for (size_t i = 0; i < (size_t)C * (size_t)o_mid; i++) {
        wo_b[i] = 0.001f;
    }

    ds4_attention_forward(
        out,
        x,
        T,
        type,
        &cfg,
        wq_a,
        wqa_n,
        wq_b,
        wkv,
        wkvn,
        sink,
        wo_a,
        wo_b,
        hca_kv,
        hca_g,
        hca_pb,
        hca_n,
        csa_kv,
        csa_g,
        csa_pb,
        csa_n,
        idx_qb,
        idx_w,
        idx_kv,
        idx_g,
        idx_pb,
        idx_n,
        scratch);
    printf("%s out[0] = %.6f\n", name, out[0]);

    free(scratch);
    free(x);
    free(out);
    free(wq_a);
    free(wqa_n);
    free(wq_b);
    free(wkv);
    free(wkvn);
    free(sink);
    free(wo_a);
    free(wo_b);
    free(hca_kv);
    free(hca_g);
    free(hca_pb);
    free(hca_n);
    free(csa_kv);
    free(csa_g);
    free(csa_pb);
    free(csa_n);
    free(idx_qb);
    free(idx_w);
    free(idx_kv);
    free(idx_g);
    free(idx_pb);
    free(idx_n);
    ds4_config_free(&cfg);
}

int main(void) {
    smoke_layer(DS4_ATTN_SLIDING, "sliding");
    smoke_layer(DS4_ATTN_HCA, "hca");
    smoke_layer(DS4_ATTN_CSA, "csa");
    printf("OK — v4 attention paths (sliding / HCA / CSA)\n");
    return 0;
}
