/* Shared forward/backward ops for DeepSeek-V2 C training (Phase 5). */
#ifndef DEEPSEEK_V2_OPS_H
#define DEEPSEEK_V2_OPS_H

/* y = W @ x, W (out,in) row-major */
void dsv2_linear_forward(const float *W, const float *x, float *y, int out_dim, int in_dim);

/* dx = W^T @ dy, dW += dy @ x^T (accumulate) */
void dsv2_linear_backward(
    float *dx,
    float *dW,
    const float *dy,
    const float *x,
    int out_dim,
    int in_dim);

void dsv2_silu_forward(const float *x, float *y, int n);
void dsv2_silu_backward(float *dx, const float *x, const float *dy, int n);

/* Stable softmax + backward into logits */
void dsv2_softmax_forward(const float *logits, float *probs, int n);
void dsv2_softmax_backward(float *dlogits, const float *probs, const float *dprobs, int n);

#endif /* DEEPSEEK_V2_OPS_H */
