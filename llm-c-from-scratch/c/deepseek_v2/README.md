# DeepSeek-V2 in C (piece-by-piece)

This folder mirrors **notebooks 11–16** in commented C, the same way `vendor/llm.c/train_gpt2.c` mirrors notebooks 1–10.

## Order (do not skip)

| Step | File | Notebook |
|------|------|----------|
| 1 | `mla.c` | 12 — MLA |
| 2 | `moe.c` | 13 — DeepSeekMoE |
| 3 | `block.c` | 14 — full block |
| 4 | `train_v2_tiny.c` (TODO) | 15–16 |

Build MLA smoke test from repo root:

```bash
cd llm-c-from-scratch/c
make test_v2    # mla + moe + block
make test_mla
```

Reuse `../rmsnorm.c` for pre-norm in a later step (same math as V4 skeleton).
