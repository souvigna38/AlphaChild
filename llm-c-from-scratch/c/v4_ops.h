#ifndef V4_OPS_H
#define V4_OPS_H

/* Small BLAS-free helpers shared by DeepSeek-V4 C modules. */

void ds4_linear(const float *W, const float *x, float *y, int out_dim, int in_dim);
void ds4_softmax(float *out, const float *in, int n);
float ds4_dot(const float *a, const float *b, int n);
float ds4_sqrt_softplus(float x);

/* Top-k on scores (E <= 32). Writes expert indices and unnormalized weights. */
void ds4_topk_select(const float *scores, int n, int k, int *out_idx, float *out_w);

#endif /* V4_OPS_H */
