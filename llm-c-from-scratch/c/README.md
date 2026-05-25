# DeepSeek-V4 in C (educational port)

This directory is the **C track** for turning the [llm.c](https://github.com/karpathy/llm.c) style codebase into a **DeepSeek-V4** trainer — step by step, not in one jump.

## Build & run (phase 0)

```bash
cd llm-c-from-scratch/c
make test
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
| `mhc.c`, `moe.c`, `csa.c`, … | **TODO** per scope doc |
