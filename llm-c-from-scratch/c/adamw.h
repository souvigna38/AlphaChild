#ifndef ADAMW_H
#define ADAMW_H

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
} Ds4AdamW;

void ds4_adamw_init(Ds4AdamW *opt, size_t n, float lr, float weight_decay);
void ds4_adamw_free(Ds4AdamW *opt);
void ds4_adamw_step(Ds4AdamW *opt, float *params, const float *grads);

/* Update n params; m/v indexed at offset in optimizer state (shared step counter). */
void ds4_adamw_step_group(Ds4AdamW *opt, size_t offset, size_t n, float *params, const float *grads, int bump_step);

#endif /* ADAMW_H */
