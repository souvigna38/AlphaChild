#ifndef V4_OPS_H
#define V4_OPS_H

/* Small BLAS-free helpers shared by DeepSeek-V4 C modules. */

void ds4_linear(const float *W, const float *x, float *y, int out_dim, int in_dim);
void ds4_softmax(float *out, const float *in, int n);
float ds4_dot(const float *a, const float *b, int n);

#endif /* V4_OPS_H */
