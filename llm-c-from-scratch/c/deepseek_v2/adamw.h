/* AdamW optimizer state for educational C training (Phase 6). */
#ifndef DEEPSEEK_V2_ADAMW_H
#define DEEPSEEK_V2_ADAMW_H

#include <stddef.h>

typedef struct {
    size_t n;
    float *m;
    float *v;
    int step;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
} Dsv2AdamW;

void dsv2_adamw_init(Dsv2AdamW *opt, size_t n, float lr, float weight_decay);
void dsv2_adamw_free(Dsv2AdamW *opt);
void dsv2_adamw_step(Dsv2AdamW *opt, float *params, const float *grads);

#endif /* DEEPSEEK_V2_ADAMW_H */
