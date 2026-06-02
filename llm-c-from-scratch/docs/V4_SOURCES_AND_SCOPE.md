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
| 7 | `moe.c` (routed) + full forward (`v4_model.c`) | Match `nano_deepseek_v4` tiny config |
| 8 | `v4_train.c` train-head SGD (`-train-head`) | V2 `train_v2_tiny -train-head` pattern |
| 9 | `hash_moe_train.c`, HyperHead backward, `-train-adam` | `block_train.c` / AdamW |
| 10 | `sliding_attn_train.c`, `v4_layer_train.c`, `-train-1layer` | V2 `-train-1layer` (MoE frozen) |
| 11 | `hca_compressor_train.c`, `v4_attention_train.c`, `-train-full` | HCA backward + 2-layer hash_moe block train |
| 12 | `csa_compressor_train.c`, CSA train path, `-train-4layer`; CUDA stub | `llm.c/train_gpt2.cu` |
| 13 | `indexer_train.c`, indexer backward + AdamW on CSA layer | Lightning indexer in nano |
| 14 | `ds4_model_train_step_e2e`, `-train-e2e` | Unfreeze lm_head + final_norm + hc_head on 4-layer block |
| 15 | `v4_parity.c`, `test_v4_parity`, `verify_v4_parity.py` | Deterministic forward golden + config vs nano |
| 16 | `cuda/ds4_rmsnorm.cu`, `ds4_cuda.c`, `test_cuda_rmsnorm` | First CUDA kernel (RMSNorm forward) + CPU fallback |
| 17 | `cuda/ds4_swiglu.cu`, `test_cuda_swiglu` | SwiGLU expert forward CUDA + CPU fallback |
| 18 | `cuda/ds4_core_attention.cu`, `test_cuda_core_attention` | Sliding masked attention core (QK^T + sink) CUDA |
| 19 | CUDA dispatch in `sliding_attn.c`, `v4_attention.c`, `ds4_cuda.c` in `V4_CORE` | Wire RMSNorm + core_attention into sliding/HCA/CSA forward |
| 20 | Priorities 1–5 (Mac-friendly) | Layer checksum parity, train-e2e logging, SwiGLU cuda in MoE, tied embed/lm_head, llmc hash_moe check |
| 21 | `verify_v4_sliding_attn.py`, `sliding_attn_fill_local_mask` | C sliding attention forward matches nano (CPU/MPS) |

Educational numbering mirrors notebooks → **`c/01_*.c` …** (optional rename as files land).

### Mac (Apple Silicon) workflow

CPU builds work without `nvcc`; CUDA wrappers fall back automatically.

```bash
cd llm-c-from-scratch/c && make test_v4
cd llm-c-from-scratch && pytest tests/test_deepseek_v4.py -q
python3 scripts/verify_v4_parity.py
python3 scripts/verify_v4_nano_hash_moe.py   # C hash_moe vs llmc (PyTorch CPU/MPS)
python3 scripts/verify_v4_sliding_attn.py  # C sliding_attn vs nano DeepSeekV4Attention
./c/bin/train_deepseek_v4_tiny -train-e2e 50 -log-every 5 -data data/tiny_shakespeare.txt
```

---

## Verification strategy

1. **Tiny CPU C** forward matches **nano-deepseek-v4** tiny config (float tolerance).  
2. Optional: compare one layer against HF Transformers on CPU.  
3. Only then port hot paths to CUDA.

Run reference (Python, after `./scripts/setup_vendor.sh`):

```bash
cd llm-c-from-scratch
python3 scripts/verify_v4_reference.py
python3 scripts/verify_v4_parity.py
```
