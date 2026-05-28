# DeepSeek-V4 CUDA (Phase 12 stub)

Hot paths to port from `llm.c/train_gpt2.cu` once CPU backward matches nano tiny:

- Fused RMSNorm + residual
- SwiGLU expert batched GEMV
- Sliding-window attention (QK^T softmax V)
- Hash-MoE gather / routed MoE top-k

CPU reference: `make test_v4`, `-train-1layer`, `-train-full`, and `-train-4layer` in `train_deepseek_v4_tiny.c`.

Optional stub build (requires `nvcc`):

```bash
cd llm-c-from-scratch/c && make cuda-stub
```
