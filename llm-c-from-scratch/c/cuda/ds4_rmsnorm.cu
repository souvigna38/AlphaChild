#include <cuda_runtime.h>

#include <math.h>

/* One CUDA block per row; tiny config uses C<=128. */
__global__ void ds4_rmsnorm_forward_kernel(
    float *__restrict__ out,
    const float *__restrict__ inp,
    const float *__restrict__ weight,
    int n,
    int C,
    float eps) {
    int row = (int)blockIdx.x;
    if (row >= n) {
        return;
    }
    const float *x = inp + (size_t)row * (size_t)C;
    float *y = out + (size_t)row * (size_t)C;
    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = x[c];
        sum_sq += v * v;
    }
    float scale = rsqrtf(sum_sq / (float)C + eps);
    for (int c = 0; c < C; c++) {
        y[c] = x[c] * scale * weight[c];
    }
}

extern "C" int ds4_cuda_device_count_impl(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

extern "C" int ds4_cuda_rmsnorm_forward_launch(
    float *out,
    const float *inp,
    const float *weight,
    int n,
    int C,
    float eps) {
    if (n <= 0 || C <= 0) {
        return 0;
    }
    float *d_out = NULL;
    float *d_inp = NULL;
    float *d_w = NULL;
    size_t bytes_inp = (size_t)n * (size_t)C * sizeof(float);
    size_t bytes_w = (size_t)C * sizeof(float);
    cudaError_t err = cudaMalloc(&d_out, bytes_inp);
    if (err != cudaSuccess) {
        return -1;
    }
    err = cudaMalloc(&d_inp, bytes_inp);
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return -1;
    }
    err = cudaMalloc(&d_w, bytes_w);
    if (err != cudaSuccess) {
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }
    if (cudaMemcpy(d_inp, inp, bytes_inp, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    ds4_rmsnorm_forward_kernel<<<n, 1>>>(d_out, d_inp, d_w, n, C, eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(out, d_out, bytes_inp, cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto fail;
    }
    cudaFree(d_w);
    cudaFree(d_inp);
    cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_w);
    cudaFree(d_inp);
    cudaFree(d_out);
    return -1;
}
