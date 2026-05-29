# DeepSeek-V4 CUDA (Phase 16+)

Hot paths to port from `llm.c/train_gpt2.cu` once CPU backward matches nano tiny:

- **RMSNorm forward** (`cuda/ds4_rmsnorm.cu`) — Phase 16
- **SwiGLU expert forward** (`cuda/ds4_swiglu.cu`) — Phase 17
- **Core masked attention** (`cuda/ds4_core_attention.cu`) — Phase 18 (QK^T + sink softmax V)
- SwiGLU batched GEMV (multi-token)
- Full sliding_attn forward on GPU (projections still CPU; Phase 19 wires RMSNorm + core_attention)
- Hash-MoE gather / routed MoE top-k

CPU reference: `make test_v4`, `-train-1layer`, `-train-full`, `-train-4layer`, and `-train-e2e` in `train_deepseek_v4_tiny.c`.

## Build

Host API: `ds4_cuda.h` (`ds4_rmsnorm_forward_cuda`, `ds4_swiglu_forward_cuda`, `ds4_core_attention_cuda` fall back to CPU when CUDA is not built or launch fails).

```bash
cd llm-c-from-scratch/c
make bin/test_cuda_rmsnorm bin/test_cuda_swiglu bin/test_cuda_core_attention
make cuda-stub
./bin/test_cuda_rmsnorm
./bin/test_cuda_swiglu
./bin/test_cuda_core_attention
```

Without `nvcc`, all CUDA tests still pass using the CPU fallback path.
