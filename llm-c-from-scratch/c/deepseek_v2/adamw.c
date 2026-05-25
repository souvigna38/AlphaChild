/*
 * AdamW — same defaults as PyTorch / notebook 15 Trainer (betas 0.9, 0.999).
 */
#include "adamw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void dsv2_adamw_init(Dsv2AdamW *opt, size_t n, float lr, float weight_decay) {
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

void dsv2_adamw_free(Dsv2AdamW *opt) {
    free(opt->m);
    free(opt->v);
    memset(opt, 0, sizeof(*opt));
}

void dsv2_adamw_step(Dsv2AdamW *opt, float *params, const float *grads) {
    opt->step++;
    float t = (float)opt->step;
    float bc1 = 1.0f - powf(opt->beta1, t);
    float bc2 = 1.0f - powf(opt->beta2, t);
    for (size_t i = 0; i < opt->n; i++) {
        float g = grads[i];
        float p = params[i];
        opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * g;
        opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * g * g;
        float m_hat = opt->m[i] / bc1;
        float v_hat = opt->v[i] / bc2;
        float upd = m_hat / (sqrtf(v_hat) + opt->eps) + opt->weight_decay * p;
        params[i] -= opt->lr * upd;
    }
}
