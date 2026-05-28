#include <cuda_runtime.h>

#include <math.h>

/* Single-expert SwiGLU forward (tiny: hidden, intermediate <= 256). */
__device__ float ds4_clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

__global__ void ds4_swiglu_forward_kernel(
    float *__restrict__ out,
    const float *__restrict__ x,
    int hidden,
    int intermediate,
    const float *__restrict__ gate_up_w,
    const float *__restrict__ down_w,
    float swiglu_limit) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (intermediate > 256 || hidden > 256 || hidden <= 0 || intermediate <= 0) {
        return;
    }

    float gate[256];
    float up[256];
    float h[256];

    for (int o = 0; o < intermediate; o++) {
        float s = 0.0f;
        const float *row = gate_up_w + (size_t)o * (size_t)hidden;
        for (int i = 0; i < hidden; i++) {
            s += row[i] * x[i];
        }
        gate[o] = s;
    }
    const float *up_w = gate_up_w + (size_t)intermediate * (size_t)hidden;
    for (int o = 0; o < intermediate; o++) {
        float s = 0.0f;
        const float *row = up_w + (size_t)o * (size_t)hidden;
        for (int i = 0; i < hidden; i++) {
            s += row[i] * x[i];
        }
        up[o] = s;
    }

    for (int i = 0; i < intermediate; i++) {
        float g = ds4_clampf(gate[i], -1e30f, swiglu_limit);
        float u = ds4_clampf(up[i], -swiglu_limit, swiglu_limit);
        float silu = g / (1.0f + expf(-g));
        h[i] = silu * u;
    }

    for (int o = 0; o < hidden; o++) {
        float s = 0.0f;
        const float *row = down_w + (size_t)o * (size_t)intermediate;
        for (int i = 0; i < intermediate; i++) {
            s += row[i] * h[i];
        }
        out[o] = s;
    }
}

extern "C" int ds4_cuda_swiglu_forward_launch(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit) {
    if (hidden <= 0 || intermediate <= 0 || hidden > 256 || intermediate > 256) {
        return -1;
    }
    size_t bytes_x = (size_t)hidden * sizeof(float);
    size_t bytes_out = bytes_x;
    size_t bytes_gu = (size_t)intermediate * 2u * (size_t)hidden * sizeof(float);
    size_t bytes_down = (size_t)hidden * (size_t)intermediate * sizeof(float);

    float *d_out = NULL;
    float *d_x = NULL;
    float *d_gu = NULL;
    float *d_down = NULL;

    if (cudaMalloc(&d_out, bytes_out) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc(&d_x, bytes_x) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&d_gu, bytes_gu) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&d_down, bytes_down) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_x, x, bytes_x, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_gu, gate_up_w, bytes_gu, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_down, down_w, bytes_down, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }

    ds4_swiglu_forward_kernel<<<1, 1>>>(d_out, d_x, hidden, intermediate, d_gu, d_down, swiglu_limit);
    if (cudaGetLastError() != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto fail;
    }

    cudaFree(d_down);
    cudaFree(d_gu);
    cudaFree(d_x);
    cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_down);
    cudaFree(d_gu);
    cudaFree(d_x);
    cudaFree(d_out);
    return -1;
}
