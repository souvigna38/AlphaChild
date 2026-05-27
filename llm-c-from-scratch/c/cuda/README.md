# DeepSeek-V4 CUDA (Phase 10 stub)

Hot paths to port from `llm.c/train_gpt2.cu` once CPU backward matches nano tiny:

- Fused RMSNorm + residual
- SwiGLU expert batched GEMV
- Sliding-window attention (QK^T softmax V)
- Hash-MoE gather / routed MoE top-k

CPU reference: `make test_v4` and `-train-1layer` in `train_deepseek_v4_tiny.c`.
