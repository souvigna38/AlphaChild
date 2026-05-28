# DeepSeek-V4 CUDA (Phase 16+)

Hot paths to port from `llm.c/train_gpt2.cu` once CPU backward matches nano tiny:

- **RMSNorm forward** (`cuda/ds4_rmsnorm.cu`) — Phase 16
- **SwiGLU expert forward** (`cuda/ds4_swiglu.cu`) — Phase 17
- SwiGLU batched GEMV (multi-token)
- Sliding-window attention (QK^T softmax V)
- Hash-MoE gather / routed MoE top-k

CPU reference: `make test_v4`, `-train-1layer`, `-train-full`, `-train-4layer`, and `-train-e2e` in `train_deepseek_v4_tiny.c`.

## Build

Host API: `ds4_cuda.h` (`ds4_rmsnorm_forward_cuda`, `ds4_swiglu_forward_cuda` fall back to CPU when CUDA is not built or launch fails).

```bash
cd llm-c-from-scratch/c
make bin/test_cuda_rmsnorm bin/test_cuda_swiglu   # links CUDA when nvcc is on PATH
make cuda-stub                                    # runs RMSNorm + SwiGLU CPU vs GPU checks
./bin/test_cuda_rmsnorm
./bin/test_cuda_swiglu
```

Without `nvcc`, both CUDA tests still pass using the CPU fallback path.
