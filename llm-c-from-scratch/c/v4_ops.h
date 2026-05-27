#ifndef V4_OPS_H
#define V4_OPS_H

/* Small BLAS-free helpers shared by DeepSeek-V4 C modules. */

void ds4_linear(const float *W, const float *x, float *y, int out_dim, int in_dim);
void ds4_linear_backward(
    float *dx,
    float *dW,
    const float *dy,
    const float *x,
    const float *W,
    int out_dim,
    int in_dim);
void ds4_softmax(float *out, const float *in, int n);
void ds4_softmax_backward(float *dscores, const float *probs, const float *dprobs, int n);
/* Stable softmax CE: writes dlogits, returns mean loss over n positions. */
float ds4_softmax_cross_entropy_backward(float *dlogits, const float *logits, int target, int vocab);
float ds4_dot(const float *a, const float *b, int n);
float ds4_sqrt_softplus(float x);

/* Top-k on scores (E <= 32). Writes expert indices and unnormalized weights. */
void ds4_topk_select(const float *scores, int n, int k, int *out_idx, float *out_w);

#endif /* V4_OPS_H */
