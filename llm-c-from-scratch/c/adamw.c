#include "adamw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ds4_adamw_init(Ds4AdamW *opt, size_t n, float lr, float weight_decay) {
    memset(opt, 0, sizeof(*opt));
    opt->n = n;
    opt->m = (float *)calloc(n, sizeof(float));
    opt->v = (float *)calloc(n, sizeof(float));
    opt->lr = lr;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->weight_decay = weight_decay;
}

void ds4_adamw_free(Ds4AdamW *opt) {
    free(opt->m);
    free(opt->v);
    memset(opt, 0, sizeof(*opt));
}

void ds4_adamw_step_group(Ds4AdamW *opt, size_t offset, size_t n, float *params, const float *grads, int bump_step) {
    if (bump_step) {
        opt->step++;
    }
    float t = (float)opt->step;
    float bc1 = 1.0f - powf(opt->beta1, t);
    float bc2 = 1.0f - powf(opt->beta2, t);
    for (size_t i = 0; i < n; i++) {
        size_t j = offset + i;
        float g = grads[i];
        float p = params[i];
        opt->m[j] = opt->beta1 * opt->m[j] + (1.0f - opt->beta1) * g;
        opt->v[j] = opt->beta2 * opt->v[j] + (1.0f - opt->beta2) * g * g;
        float m_hat = opt->m[j] / bc1;
        float v_hat = opt->v[j] / bc2;
        float upd = m_hat / (sqrtf(v_hat) + opt->eps) + opt->weight_decay * p;
        params[i] -= opt->lr * upd;
    }
}

void ds4_adamw_step(Ds4AdamW *opt, float *params, const float *grads) {
    ds4_adamw_step_group(opt, 0, opt->n, params, grads, 1);
}
