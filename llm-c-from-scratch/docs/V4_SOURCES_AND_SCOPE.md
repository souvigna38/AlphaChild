# DeepSeek-V4 source code map (for the C port)

## Can we jump straight from llm.c → full DeepSeek-V4 in C?

**No — not in one step.** DeepSeek-V4 is a different architecture from GPT-2 (what `llm.c` implements today).

| Piece | GPT-2 (`llm.c`) | DeepSeek-V4 |
|-------|-----------------|-------------|
| Attention | Dense MHA + full KV cache | Sliding + CSA + HCA + Lightning Indexer |
| Residual | `x + sublayer(x)` | **mHC** (manifold hyper-connections, Sinkhorn) |
| FFN | Dense MLP | **Hash-MoE** bootstrap → routed **MoE** |
| Norm | LayerNorm | **RMSNorm** |
| Extra | — | MTP heads, partial RoPE, grouped output proj |

A direct “transform” would replace almost every layer in `train_gpt2.c`. The practical path is **phased C ports** with a **tiny** config first (`deepseek_v4_tiny`), then scale toward Flash/Pro shapes.

---

## Where the real V4 logic lives today (Python)

There is **no** official `deepseek-ai/DeepSeek-V4` training repo in C/CUDA (unlike V3’s inference repo).

| Source | URL | Use for C port |
|--------|-----|----------------|
| **Hugging Face Transformers** | [modeling_deepseek_v4.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) | Ground-truth behavior (large, generated) |
| **nano-deepseek-v4** | https://github.com/hebo1221/nano-deepseek-v4 | **Best readable reference** (~1.3k LOC `modeling.py`) |
| **Config JSON** | `vendor/nano-deepseek-v4/references/DeepSeek-V4-Flash-config.json` | Flash 43-layer schedule |
| **llm.c** (baseline) | https://github.com/karpathy/llm.c | C style, training loop, tokenizer `.bin` |
| **DeepSeek-V3** | https://github.com/deepseek-ai/DeepSeek-V3 | MLA/MoE ideas; **not** V4 architecture |

**Not architecture code:** `DeepSeek-V4/deepseek-V4` on GitHub is an installer/downloader, not a model implementation.

---

## Recommended C roadmap (this repo `c/`)

| Phase | C module | Reference |
|-------|----------|-----------|
| 0 | `train_deepseek_v4_tiny.c` skeleton + Makefile | `llm.c/train_gpt2.c` |
| 1 | `rmsnorm.c`, `swiglu.c` | V4 vs GPT-2 LayerNorm/GELU |
| 2 | `hash_moe.c` | `mlp_layer_types: hash_moe` |
| 3 | `sliding_attn.c` | local window + attention sink |
| 4 | `hca_compressor.c`, `csa_compressor.c`, `indexer.c` | CSA/HCA + Lightning indexer |
| 5 | `mhc.c` | hyper-connections (Sinkhorn) |
| 6 | `v4_attention.c` | unified sliding / CSA / HCA forward |
| 7 | `moe.c` (routed) + full `train_deepseek_v4_tiny.c` | Match `nano_deepseek_v4` tiny config |
| 8 | CUDA kernels (`c/cuda/`) | `llm.c/train_gpt2.cu` pattern |

Educational numbering mirrors notebooks → **`c/01_*.c` …** (optional rename as files land).

---

## Verification strategy

1. **Tiny CPU C** forward matches **nano-deepseek-v4** tiny config (float tolerance).  
2. Optional: compare one layer against HF Transformers on CPU.  
3. Only then port hot paths to CUDA.

Run reference (Python, after `./scripts/setup_vendor.sh`):

```bash
cd llm-c-from-scratch
python3 scripts/verify_v4_reference.py
```
