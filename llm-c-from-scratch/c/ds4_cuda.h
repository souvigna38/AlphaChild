#ifndef DS4_CUDA_H
#define DS4_CUDA_H

/* Phase 16: optional CUDA kernels (RMSNorm first). Falls back to CPU when unavailable. */

int ds4_cuda_available(void);

int ds4_cuda_device_count(void);

/* RMSNorm forward on device/host buffers (n rows, C cols). Uses GPU when built with CUDA. */
void ds4_rmsnorm_forward_cuda(float *out, const float *inp, const float *weight, int n, int C, float eps);

#endif /* DS4_CUDA_H */
