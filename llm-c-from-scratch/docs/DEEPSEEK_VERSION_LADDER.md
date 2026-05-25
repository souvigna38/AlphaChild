# Learning ladder: llm.c (GPT-2) → DeepSeek

## Why not “DeepSeek V1”?

There is **no widely used public “DeepSeek-V1 architecture”** in the same sense as V2/V3/V4:

- Early **DeepSeek** releases were strong models, but the **named architectural jump** most tutorials reference is **DeepSeek-V2** (May 2024): **MLA** + **DeepSeekMoE**.
- **V3** keeps MLA + MoE and adds training tweaks (aux-loss-free routing, MTP, etc.).
- **V4** replaces much of the stack again (mHC, CSA/HCA, hash-MoE bootstrap).

So for **piece-by-piece learning** after `llm.c`, we recommend:

```
Notebooks 1–10   GPT-2 / llm.c ideas (PyTorch)
Notebooks 11–17  DeepSeek-V2 (MLA + MoE + C train/backward)
Notebook 18       DeepSeek-V4 kickoff (hash-MoE, SwiGLU) — `c/swiglu.c`, `c/hash_moe.c`
(later)          V4 sliding / CSA / HCA / mHC in c/
```

If you literally need the **first** DeepSeek checkpoint family, treat it as “pre-V2 dense LM” and skip to V2 for new mechanisms.

## Map to llm.c C files

| Notebook | Concept | llm.c analogue |
|----------|---------|----------------|
| 11 | Roadmap | `train_gpt2.c` overview |
| 12 | MLA | `attention` + KV cache size |
| 13 | DeepSeekMoE | (no MoE in GPT-2 — new) | `moe.c` ✓ |
| 14 | Full V2 block | transformer loop in `train_gpt2.c` | `block.c` ✓ |
| 15 | Training | main training loop | `train_v2_tiny.c` ✓ |
| 16 | Sampling | generation + export | `-sample -ckpt` ✓ |
| 17 | V2 backward / full train | `block_train.c`, `-train-full` | ✓ |
| 18 | V4 roadmap + hash-MoE | `swiglu.c`, `hash_moe.c` | `make test_v4` ✓ |

Matching **commented C** ports live under `c/deepseek_v2/` (V2) and top-level `c/` (V4 phases 0–2).

**Phase 5:** MLA backward + `train_v2_tiny -train-1layer` (MoE frozen).

**Phase 5b:** MoE + block backward (`moe_train.c`, `block_train.c`) + `-train-full` (2 layers, B=1, global grad clip). PyTorch `Trainer` in notebook 15 remains the path for long runs.

**Phase 6:** AdamW (`adamw.c`), batched gradients (`-batch B`), C checkpoint save/load (`-save` / `-ckpt`), `-train-adam`.
