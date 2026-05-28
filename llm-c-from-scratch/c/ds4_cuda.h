#ifndef DS4_CUDA_H
#define DS4_CUDA_H

/* Phase 16–17: optional CUDA kernels. Falls back to CPU when unavailable. */

int ds4_cuda_available(void);

int ds4_cuda_device_count(void);

/* RMSNorm forward on device/host buffers (n rows, C cols). Uses GPU when built with CUDA. */
void ds4_rmsnorm_forward_cuda(float *out, const float *inp, const float *weight, int n, int C, float eps);

/* SwiGLU expert forward (single token). Uses GPU when built with CUDA. */
void ds4_swiglu_forward_cuda(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit);

#endif /* DS4_CUDA_H */
