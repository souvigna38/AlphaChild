#ifndef DS4_RMSNORM_H
#define DS4_RMSNORM_H

/* RMSNorm (DeepSeek-V4) — replaces GPT-2 LayerNorm in llm.c */

void ds4_rmsnorm_forward(float *out, float *inp, const float *weight, int n, int C, float eps);

#endif
