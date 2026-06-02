#include <cuda_runtime.h>

#include <math.h>

#define DS4_CUDA_MAX_T 128
#define DS4_CUDA_MAX_SCORES (DS4_CUDA_MAX_T + 1)

__device__ float ds4_cuda_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

__device__ void ds4_cuda_softmax(float *out, const float *in, int n) {
    float maxv = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > maxv) {
            maxv = in[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - maxv);
        sum += out[i];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] *= inv;
    }
}

__global__ void ds4_core_attention_kernel(
    float *__restrict__ context,
    const float *__restrict__ q,
    const float *__restrict__ keys,
    int NH,
    int Tq,
    int Tk,
    int head_dim,
    const float *__restrict__ attn_sink,
    const int *__restrict__ mask) {
    int h = (int)blockIdx.x / Tq;
    int tq = (int)blockIdx.x % Tq;
    if (h >= NH || tq >= Tq) {
        return;
    }
    if (Tq > DS4_CUDA_MAX_T || Tk > DS4_CUDA_MAX_T || head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores[DS4_CUDA_MAX_SCORES];
    float probs[DS4_CUDA_MAX_SCORES];

    const float *qh = q + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;
    float *ch = context + ((size_t)h * (size_t)Tq + (size_t)tq) * (size_t)head_dim;

    int n_scores = 0;
    for (int tk = 0; tk < Tk; tk++) {
        if (!mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
            scores[n_scores] = -1e30f;
        } else {
            const float *kh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
            scores[n_scores] = ds4_cuda_dot(qh, kh, head_dim) * scale;
        }
        n_scores++;
    }
    scores[n_scores] = attn_sink[h];
    n_scores++;
    ds4_cuda_softmax(probs, scores, n_scores);

    for (int d = 0; d < head_dim; d++) {
        ch[d] = 0.0f;
    }
    int idx = 0;
    for (int tk = 0; tk < Tk; tk++) {
        if (mask[(size_t)tq * (size_t)Tk + (size_t)tk]) {
            const float *vh = keys + ((size_t)h * (size_t)Tk + (size_t)tk) * (size_t)head_dim;
            float p = probs[idx];
            for (int d = 0; d < head_dim; d++) {
                ch[d] += p * vh[d];
            }
        }
        idx++;
    }
}

extern "C" int ds4_cuda_core_attention_launch(
    float *context,
    const float *q,
    const float *keys,
    int NH,
    int Tq,
    int Tk,
    int head_dim,
    const float *attn_sink,
    const int *mask) {
    if (NH <= 0 || Tq <= 0 || Tk <= 0 || head_dim <= 0 || Tq > DS4_CUDA_MAX_T || Tk > DS4_CUDA_MAX_T) {
        return -1;
    }

    size_t bytes_q = (size_t)NH * (size_t)Tq * (size_t)head_dim * sizeof(float);
    size_t bytes_k = (size_t)NH * (size_t)Tk * (size_t)head_dim * sizeof(float);
    size_t bytes_ctx = bytes_q;
    size_t bytes_mask = (size_t)Tq * (size_t)Tk * sizeof(int);
    size_t bytes_sink = (size_t)NH * sizeof(float);

    float *d_ctx = NULL;
    float *d_q = NULL;
    float *d_k = NULL;
    int *d_mask = NULL;
    float *d_sink = NULL;

    if (cudaMalloc(&d_ctx, bytes_ctx) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc(&d_q, bytes_q) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&d_k, bytes_k) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&d_mask, bytes_mask) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&d_sink, bytes_sink) != cudaSuccess) {
        goto fail;
    }

    if (cudaMemcpy(d_q, q, bytes_q, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_k, keys, bytes_k, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(d_sink, attn_sink, bytes_sink, cudaMemcpyHostToDevice) != cudaSuccess) {
        goto fail;
    }

    ds4_core_attention_kernel<<<NH * Tq, 1>>>(d_ctx, d_q, d_k, NH, Tq, Tk, head_dim, d_sink, d_mask);
    if (cudaGetLastError() != cudaSuccess) {
        goto fail;
    }
    if (cudaMemcpy(context, d_ctx, bytes_ctx, cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto fail;
    }

    cudaFree(d_sink);
    cudaFree(d_mask);
    cudaFree(d_k);
    cudaFree(d_q);
    cudaFree(d_ctx);
    return 0;
fail:
    cudaFree(d_sink);
    cudaFree(d_mask);
    cudaFree(d_k);
    cudaFree(d_q);
    cudaFree(d_ctx);
    return -1;
}
