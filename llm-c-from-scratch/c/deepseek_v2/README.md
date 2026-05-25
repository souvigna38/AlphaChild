# DeepSeek-V2 in C (piece-by-piece)

This folder mirrors **notebooks 11–16** in commented C, the same way `vendor/llm.c/train_gpt2.c` mirrors notebooks 1–10.

## Order (do not skip)

| Step | File | Notebook |
|------|------|----------|
| 1 | `mla.c` | 12 — MLA |
| 2 | `moe.c` | 13 — DeepSeekMoE |
| 3 | `block.c` | 14 — full block |
| 4 | `train_v2_tiny.c` | 15–16 train + sample |
| 5 | `mla.c` backward | 17 — `-train-1layer` |
| 5b | `moe_train.c`, `block_train.c` | 17 — `-train-full` |

Build MLA smoke test from repo root:

```bash
cd llm-c-from-scratch/c
make test_v2           # mla + moe + block + trainer smoke
make bin/train_v2_tiny # full trainer binary
make test_mla
```

Reuse `../rmsnorm.c` for pre-norm in a later step (same math as V4 skeleton).
