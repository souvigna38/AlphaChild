#ifndef DS4_RMSNORM_H
#define DS4_RMSNORM_H

/* RMSNorm (DeepSeek-V4) — replaces GPT-2 LayerNorm in llm.c */

void ds4_rmsnorm_forward(float *out, float *inp, const float *weight, int n, int C, float eps);

/* Backward: dinp, dweight accumulate; dout is upstream grad w.r.t. out */
void ds4_rmsnorm_backward(
    float *dinp,
    float *dweight,
    const float *dout,
    const float *inp,
    const float *out,
    const float *weight,
    int n,
    int C,
    float eps);

#endif
