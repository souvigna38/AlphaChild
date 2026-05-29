#include "deepseek_v4_config.h"
#include "mhc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int hc = cfg.hc_mult;
    const int C = cfg.hidden_size;
    const int mix = (2 + hc) * hc;
    const int flat = hc * C;

    float *streams = (float *)calloc((size_t)hc * (size_t)C, sizeof(float));
    float *fn = (float *)calloc((size_t)mix * (size_t)flat, sizeof(float));
    float *base = (float *)calloc((size_t)mix, sizeof(float));
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float *post = (float *)calloc((size_t)hc, sizeof(float));
    float *comb = (float *)calloc((size_t)hc * (size_t)hc, sizeof(float));
    float *collapsed = (float *)calloc((size_t)C, sizeof(float));
    float *out = (float *)calloc((size_t)C, sizeof(float));

    for (int i = 0; i < hc; i++) {
        for (int c = 0; c < C; c++) {
            streams[(size_t)i * (size_t)C + (size_t)c] = 0.01f * (float)(i + c);
        }
    }
    for (size_t i = 0; i < (size_t)mix * (size_t)flat; i++) {
        fn[i] = 0.0001f;
    }

    ds4_hyper_connection_forward(post, comb, collapsed, streams, &cfg, fn, base, scale);
    ds4_hyper_head_forward(out, streams, &cfg, fn, base, scale);
    printf("mHC collapsed[0..3] = %.6f %.6f %.6f %.6f\n", collapsed[0], collapsed[1], collapsed[2], collapsed[3]);
    printf("HyperHead out[0..3] = %.6f %.6f %.6f %.6f\n", out[0], out[1], out[2], out[3]);

    free(streams);
    free(fn);
    free(base);
    free(post);
    free(comb);
    free(collapsed);
    free(out);
    ds4_config_free(&cfg);
    printf("OK — mHC Sinkhorn + stream collapse (nano HyperConnection)\n");
    return 0;
}
