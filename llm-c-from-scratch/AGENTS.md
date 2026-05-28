# AGENTS.md — llm.c From Scratch

Instructions for **Cursor agents** cloning and running this educational GPT / llm.c notebook project.

---

## What this project is

- **Not** a fork of llm.c C/CUDA code.
- **Is** a numbered Jupyter curriculum (like AlphaZero notebooks 1–10) that teaches GPT-2 training concepts in **PyTorch**.
- **`llmc/`** package mirrors notebook code for tests.

Upstream reference: https://github.com/karpathy/llm.c

---

## 1. Clone this project

```bash
git clone https://github.com/souvigna38/llm-c-from-scratch.git
cd llm-c-from-scratch
```

If nested inside another monorepo:

```bash
cd llm-c-from-scratch
```

---

## 2. Python environment

Python **3.10+** required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
export PATH="$HOME/.local/bin:$PATH"
```

---

## 3. Verify install

```bash
ruff check llmc/ tests/
python3 -m pytest tests/ -v
```

All tests should pass (~10s on CPU).

---

## 4. Run notebooks

```bash
python3 -m ipykernel install --user --name llmc --display-name "Python (llmc)"
jupyter lab
```

Run in order: **`1.Tokens.ipynb`** → **`10.GPT2AndLlmc.ipynb`**.

**Working directory:** repo root (so `data/tiny_shakespeare.txt` resolves).

**Skip in automation:** notebook **8** `trainer.train()` (slow) — tests skip it.

---

## 5. Clone karpathy/llm.c (separate repo)

Agents setting up **full C/CUDA training** should clone llm.c **next to** this project:

```bash
cd ..
git clone https://github.com/karpathy/llm.c.git
cd llm.c
pip install -r requirements.txt
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
```

CPU demo:

```bash
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

GPU (if available):

```bash
make train_gpt2cu
./train_gpt2cu
```

Map notebooks → llm.c files using **notebook 10**.

---

## 6. Notebook ↔ llm.c map

| Notebook | llm.c / nanoGPT equivalent |
|----------|----------------------------|
| 1.Tokens | `dev/data/tinyshakespeare.py` |
| 4.Attention | `dev/cuda` attention kernels |
| 6.GPT | model forward in `train_gpt2.c` |
| 8.Train | `train_gpt2.py` / `train_gpt2.cu` loop |
| 9.Sample | generation at end of training |
| 10.GPT2AndLlmc | GPT-2 124M hyperparameters |

---

## 7. One-shot agent script

```bash
git clone https://github.com/souvigna38/llm-c-from-scratch.git && cd llm-c-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/ -v
python3 -m ipykernel install --user --name llmc --display-name "Python (llmc)"
jupyter lab
```

---

## 8. Dependencies

| Extra | Packages |
|-------|----------|
| (core) | numpy, torch, matplotlib, tqdm |
| `dev` | pytest, ruff, ipykernel, nbformat, nbclient |
| `gpt2` | tiktoken (optional, for future BPE notebook) |

Character-level tokens are used in notebooks 1–9 so **tiktoken is not required** for the default path.
