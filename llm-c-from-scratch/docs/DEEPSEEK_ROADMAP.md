# DeepSeek feature roadmap (for your fork)

This repo teaches **GPT-2 / llm.c** in PyTorch notebooks. A fork is the right place to add **DeepSeek** ideas without confusing beginners on `main`.

References:

- [DeepSeek-V2](https://arxiv.org/html/2405.04434) — MLA + DeepSeekMoE  
- [DeepSeekMoE](https://arxiv.org/html/2401.06066) — fine-grained experts + shared experts  
- [DeepSeek-V3 / R1](https://github.com/deepseek-ai) — production stacks (optional later)

---

## What to add (notebook order)

| # | Proposed notebook | Topic | Replaces / extends |
|---|-------------------|--------|---------------------|
| 11 | `11.MLA.ipynb` | Multi-head Latent Attention (low-rank KV) | `4.Attention.ipynb` |
| 12 | `12.DeepSeekMoE.ipynb` | Routed experts + shared experts in FFN | `5.GPTBlock.ipynb` MLP |
| 13 | `13.GPTDeepSeek.ipynb` | Full model: MLA + MoE block + GPT skeleton | `6.GPT.ipynb` |
| 14 | `14.TrainDeepSeek.ipynb` | Train tiny MoE on Shakespeare (CPU/GPU) | `8.Train.ipynb` |
| 15 | `15.SampleDeepSeek.ipynb` | Generation with MoE model | `9.Sample.ipynb` |
| 16 | `16.CompareGPT2DeepSeek.ipynb` | Param count, active params, vs llm.c / GPT-2 | `10.GPT2AndLlmc.ipynb` |

Optional later:

- RMSNorm + SwiGLU (if matching DeepSeek-V2 blocks)  
- YaRN / long context (inference only)  
- Load DeepSeek **weights** via Hugging Face (separate from “from scratch” path)

---

## Implementation notes (for agents)

### MLA (simplified educational version)

- Compress KV with low-rank projections (latent dim `d_latent` < `d_model`).  
- Keep causal mask identical to notebook 4.  
- Compare **KV cache size**: MHA vs MLA (plot bytes per token).

### DeepSeekMoE (simplified)

- Split FFN into `n_routed_experts` small MLPs + `n_shared_experts`.  
- Router: linear → softmax → top-`k` experts per token.  
- Load-balancing aux loss (optional in v1; add in v2).  
- Report **active parameters** per forward pass vs total parameters.

### Training

- Start with `GPTConfig.tiny()` scale; MoE with 4 experts, top-2.  
- Same `tiny_shakespeare` data as notebooks 1–8.  
- Do not require DeepSeek API or official checkpoints for the tutorial path.

---

## Package layout (fork)

```
llmc/
  model.py          # GPT (baseline) — keep
  deepseek.py       # MLA, MoE, GPTDeepSeek (new)
  data.py, train.py, sample.py
```

Tests:

```
tests/test_deepseek_mla.py
tests/test_deepseek_moe.py
```

---

## What *not* to merge back to upstream without discussion

- Heavy dependencies (`transformers`, official DeepSeek CUDA kernels).  
- Huge downloaded checkpoints in git.  
- Breaking changes to notebooks 1–10.

---

## Cursor agent one-liner

```
On fork <YOUR_USER>/llm-c-from-scratch, branch feature/deepseek:
implement docs/DEEPSEEK_ROADMAP.md starting with 11.MLA.ipynb and llmc/deepseek.py;
keep notebooks 1-10 unchanged; pytest must pass for old + new tests.
```
