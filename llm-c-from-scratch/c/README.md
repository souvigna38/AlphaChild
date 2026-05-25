# C ports (educational)

Two ladders, both piece-by-piece:

1. **DeepSeek-V2** (after notebooks 11–16): `deepseek_v2/mla.c` (+ MoE later) — **start here** after GPT-2/llm.c.
2. **DeepSeek-V4** (advanced): `deepseek_v4_config.c`, `rmsnorm.c` — see V4 scope doc.

## Build & run

```bash
cd llm-c-from-scratch/c
make test_v2     # DeepSeek-V2: MLA + MoE + block (notebooks 12–14)
make test_mla    # piece 1 only
make test_moe    # piece 2 only
make test_block  # piece 3 only
make test        # DeepSeek-V4 phase-0 skeleton
```

## Prerequisites

```bash
cd llm-c-from-scratch
./scripts/setup_vendor.sh   # clones llm.c + nano-deepseek-v4 into vendor/
```

## Read first

- [../docs/V4_SOURCES_AND_SCOPE.md](../docs/V4_SOURCES_AND_SCOPE.md) — what V4 source exists, phased plan  
- [../docs/DEEPSEEK_ROADMAP.md](../docs/DEEPSEEK_ROADMAP.md) — original PyTorch notebook plan (parallel track)

## Files

| File | Status |
|------|--------|
| `deepseek_v4_config.c` | Tiny config + per-layer schedules |
| `rmsnorm.c` | RMSNorm forward |
| `train_deepseek_v4_tiny.c` | Skeleton main (smoke tests only) |
| `deepseek_v2/mla.c` | MLA (notebook 12) |
| `deepseek_v2/moe.c` | MoE (notebook 13) |
| `deepseek_v2/block.c` | One block (notebook 14) |
| `train_v2_tiny.c` | Train + sample (15–16); `-train-1layer` Phase 5 |
| `mla.c` backward | `dsv2_mla_backward` (notebook 17) |
| `mhc.c`, … | V4 only |
