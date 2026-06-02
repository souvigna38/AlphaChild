#include "deepseek_v4_config.h"
#include "v4_model.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int C = cfg.hidden_size;
    const int V = cfg.vocab_size;

    Ds4ModelWeights mw = {0};
    mw.embed = (float *)calloc((size_t)V * (size_t)C, sizeof(float));
    mw.lm_head = (float *)calloc((size_t)V * (size_t)C, sizeof(float));

    ds4_model_weights_tie_lm_head(&mw);
    if (mw.lm_head != mw.embed || !mw.lm_head_tied) {
        fprintf(stderr, "tie_lm_head did not alias pointers\n");
        free(mw.embed);
        ds4_config_free(&cfg);
        return 1;
    }
    if (ds4_model_lm_matrix(&mw) != mw.embed) {
        fprintf(stderr, "lm_matrix should return embed when tied\n");
        free(mw.embed);
        ds4_config_free(&cfg);
        return 1;
    }

    printf("OK — tied lm_head uses embed matrix (V=%d C=%d)\n", V, C);
    free(mw.embed);
    ds4_config_free(&cfg);
    return 0;
}
