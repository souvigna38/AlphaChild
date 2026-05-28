#include "deepseek_v4_config.h"
#include "indexer.h"
#include "indexer_train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = 8;
    const int C = cfg.hidden_size;
    const int r = cfg.q_lora_rank;
    const int NH = cfg.index_n_heads;
    const int HD = cfg.index_head_dim;
    const int HD2 = 2 * HD;
    const int rate = cfg.compress_rate_csa;
    const int n_comp = T / rate;

    float *hidden = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *q_mid = (float *)calloc((size_t)T * (size_t)r, sizeof(float));
    float *wq_b = (float *)calloc((size_t)(NH * HD) * (size_t)r, sizeof(float));
    float *w_weights = (float *)calloc((size_t)NH * (size_t)C, sizeof(float));
    float *w_kv = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *w_gate = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *pos_bias = (float *)calloc((size_t)rate * (size_t)HD2, sizeof(float));
    float *norm_w = (float *)calloc((size_t)HD, sizeof(float));
    norm_w[0] = 1.0f;
    for (size_t i = 0; i < (size_t)T * (size_t)C; i++) {
        hidden[i] = 0.01f * (float)((int)i % 7 - 3);
    }
    for (size_t i = 0; i < (size_t)(NH * HD) * (size_t)r; i++) {
        wq_b[i] = 0.02f;
    }
    for (size_t i = 0; i < (size_t)NH * (size_t)C; i++) {
        w_weights[i] = 0.02f;
    }
    for (size_t i = 0; i < (size_t)HD2 * (size_t)C; i++) {
        w_kv[i] = 0.02f;
        w_gate[i] = 0.02f;
    }
    for (size_t i = 0; i < (size_t)T * (size_t)r; i++) {
        q_mid[i] = 0.05f;
    }

    int *mask_ref = (int *)calloc((size_t)T * (size_t)n_comp, sizeof(int));
    int *mask_tr = (int *)calloc((size_t)T * (size_t)n_comp, sizeof(int));
    int *comp_end = (int *)malloc((size_t)n_comp * sizeof(int));
    float *comp_kv = (float *)calloc((size_t)n_comp * (size_t)cfg.head_dim, sizeof(float));
    size_t csz = ds4_csa_indexer_train_cache_floats(&cfg, T);
    float *cache = (float *)calloc(csz, sizeof(float));

    int n_ref = ds4_csa_indexer_forward(
        mask_ref,
        hidden,
        q_mid,
        T,
        n_comp,
        comp_end,
        comp_kv,
        &cfg,
        wq_b,
        w_weights,
        w_kv,
        w_gate,
        pos_bias,
        norm_w);
    int n_tr = 0;
    ds4_csa_indexer_forward_train(
        mask_tr,
        hidden,
        q_mid,
        T,
        n_comp,
        comp_end,
        comp_kv,
        &cfg,
        wq_b,
        w_weights,
        w_kv,
        w_gate,
        pos_bias,
        norm_w,
        cache,
        &n_tr);

    if (n_ref != n_tr) {
        fprintf(stderr, "n_idx mismatch ref=%d train=%d\n", n_ref, n_tr);
        return 1;
    }
    int mism = 0;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < n_comp; k++) {
            if (mask_ref[(size_t)t * (size_t)n_comp + (size_t)k] != mask_tr[(size_t)t * (size_t)n_comp + (size_t)k]) {
                mism++;
            }
        }
    }
    if (mism > 0) {
        fprintf(stderr, "sparse_mask mismatch count=%d\n", mism);
        return 1;
    }

    float d_scores[128];
    memset(d_scores, 0, sizeof(d_scores));
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < n_comp; k++) {
            if (mask_tr[(size_t)t * (size_t)n_comp + (size_t)k]) {
                d_scores[(size_t)t * 64 + (size_t)k] = 0.01f;
            }
        }
    }
    float *dhidden = (float *)calloc((size_t)T * (size_t)C, sizeof(float));
    float *dq = (float *)calloc((size_t)T * (size_t)r, sizeof(float));
    float *dwq = (float *)calloc((size_t)(NH * HD) * (size_t)r, sizeof(float));
    float *dww = (float *)calloc((size_t)NH * (size_t)C, sizeof(float));
    float *dkv = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *dg = (float *)calloc((size_t)HD2 * (size_t)C, sizeof(float));
    float *dpb = (float *)calloc((size_t)rate * (size_t)HD2, sizeof(float));
    float *dn = (float *)calloc((size_t)HD, sizeof(float));

    ds4_csa_indexer_backward(
        dhidden,
        dq,
        dwq,
        dww,
        dkv,
        dg,
        dpb,
        dn,
        d_scores,
        hidden,
        q_mid,
        T,
        n_comp,
        &cfg,
        wq_b,
        w_weights,
        w_kv,
        w_gate,
        norm_w,
        cache);

    float sum = 0.0f;
    for (size_t i = 0; i < (size_t)NH * (size_t)C; i++) {
        sum += dww[i] * dww[i];
    }
    for (size_t i = 0; i < (size_t)HD2 * (size_t)C; i++) {
        sum += dkv[i] * dkv[i] + dg[i] * dg[i];
    }
    if (sum <= 0.0f) {
        fprintf(stderr, "expected non-zero indexer grads\n");
        return 1;
    }

    free(dn);
    free(dpb);
    free(dg);
    free(dkv);
    free(dww);
    free(dwq);
    free(dq);
    free(dhidden);
    free(cache);
    free(comp_kv);
    free(comp_end);
    free(mask_tr);
    free(mask_ref);
    free(norm_w);
    free(pos_bias);
    free(w_gate);
    free(w_kv);
    free(w_weights);
    free(wq_b);
    free(q_mid);
    free(hidden);
    ds4_config_free(&cfg);
    printf("OK — indexer train forward matches ref; backward non-zero\n");
    return 0;
}
