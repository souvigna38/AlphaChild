# llm.c From Scratch (Educational Notebooks)

A **step-by-step Jupyter tutorial** that teaches the ideas behind [karpathy/llm.c](https://github.com/karpathy/llm.c) in **PyTorch**, using the same progressive style as [AlphaZeroFromScratch](https://github.com/foersterrobert/AlphaZero) / this repo’s numbered notebooks.

> **llm.c** trains GPT-2 in fast C/CUDA. **This repo** helps you *understand* the model and training loop before (or alongside) reading the C code.

## Notebook path

| # | Notebook | Topic |
|---|----------|--------|
| 1 | `1.Tokens.ipynb` | Text → token IDs (character level) |
| 2 | `2.Bigram.ipynb` | Simplest language model |
| 3 | `3.Embeddings.ipynb` | Token + position embeddings |
| 4 | `4.Attention.ipynb` | Causal self-attention |
| 5 | `5.GPTBlock.ipynb` | One transformer block |
| 6 | `6.GPT.ipynb` | Full GPT module |
| 7 | `7.BatchAndLoss.ipynb` | Batches + cross-entropy |
| 8 | `8.Train.ipynb` | Training loop (tiny Shakespeare) |
| 9 | `9.Sample.ipynb` | Generate text |
| 10 | `10.GPT2AndLlmc.ipynb` | GPT-2 124M config + map to llm.c |

## Quick start

```bash
git clone https://github.com/souvigna38/llm-c-from-scratch.git
cd llm-c-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/ -v
jupyter lab
```

Open **`1.Tokens.ipynb`** and run cells in order through **`10`**.

## Clone upstream llm.c (optional)

To run the original C/CUDA trainer after the notebooks:

```bash
git clone https://github.com/karpathy/llm.c.git
cd llm.c
pip install -r requirements.txt
./dev/download_starter_pack.sh   # tiny Shakespeare + GPT-2 tokenizer bins
make train_gpt2fp32cu            # or train_gpt2 for CPU
```

See **[AGENTS.md](AGENTS.md)** for full agent/human setup instructions.

### Forking for DeepSeek (or other architectures)

Keep the original tutorial on `main`; experiment on a **GitHub fork** and branch `feature/deepseek`.

- **[FORK.md](FORK.md)** — how to fork (GitHub UI, CLI, local copy)  
- **[docs/DEEPSEEK_ROADMAP.md](docs/DEEPSEEK_ROADMAP.md)** — planned notebooks 11–16 (MLA, MoE, …)  
- **[CURSOR_AGENT_FORK_DEEPSEEK.md](CURSOR_AGENT_FORK_DEEPSEEK.md)** — paste-ready task for another Cursor agent / OpenClaw

## Project layout

```
llm-c-from-scratch/
├── 1.Tokens.ipynb … 10.GPT2AndLlmc.ipynb
├── llmc/                 # importable package (used in tests + later notebooks)
├── data/tiny_shakespeare.txt
├── tests/
├── pyproject.toml
└── AGENTS.md
```

## Credits

- [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy (MIT)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — PyTorch structure inspiration
- Tutorial format inspired by AlphaZero-from-scratch notebooks
