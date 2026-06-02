#!/usr/bin/env python3
"""Rewrite notebooks 1–10 for Cursor workbook (setup cell, markdown, path fixes)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = """\
# --- Setup: find repo root (llm-c-from-scratch or Cursor workbook) ---
import sys
from pathlib import Path


def find_llm_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "llmc" / "__init__.py").is_file():
            return base
        nested = base / "llm-c-from-scratch"
        if (nested / "llmc" / "__init__.py").is_file():
            return nested
    return Path.cwd()


ROOT = find_llm_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmc.notebook_utils import data_path, checkpoint_path

DATA = data_path(ROOT)
CHECKPOINT = checkpoint_path(ROOT)
print("ROOT", ROOT.resolve())
print("data", "OK" if DATA.is_file() else "missing")
"""

VERIFY_MODEL = """\
# --- Fast verify (seconds) ---
import subprocess

tests = ROOT / "tests" / "test_model.py"
if tests.is_file():
    cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "-k", "gpt_forward or generate"]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print("pytest OK")
else:
    print("Skip pytest:", tests)
"""

VERIFY_TRAIN = """\
# --- Fast verify (seconds) ---
import subprocess

tests = ROOT / "tests" / "test_train.py"
if tests.is_file():
    cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "-k", "trainer_short"]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print("pytest OK — ready for dojo-grade --lesson C2-L08")
else:
    print("Skip pytest:", tests)
"""


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in src.strip("\n").split("\n")],
    }


def nb(*cells) -> dict:
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


NOTEBOOKS = {
    "1.Tokens.ipynb": nb(
        md(
            "# 1 — Tokens",
            "",
            "**Before:** nothing.",
            "",
            "**This notebook:** text → character token IDs (llm.c uses BPE; we start simpler).",
            "",
            "**Cursor workbook:** run top-to-bottom. Works from `llm-c-from-scratch/` or a student workspace.",
            "",
            "**Dojo (optional):** `dojo-grade --lesson C2-L01`",
        ),
        code(SETUP),
        code(
            """\
import numpy as np
import torch

print(np.__version__)
text = DATA.read_text(encoding="utf-8")
print(f"Characters in corpus: {len(text):,}")
print(text[:200])"""
        ),
        code(
            """\
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

sample = "ROMEO:"
print("encoded:", encode(sample))
print("decoded:", decode(encode(sample)))"""
        ),
        code(
            """\
tokens = torch.tensor(encode(text[:1000]), dtype=torch.long)
print("Tensor shape:", tokens.shape)
print("First 20 token ids:", tokens[:20].tolist())"""
        ),
    ),
    "2.Bigram.ipynb": nb(
        md(
            "# 2 — Bigram language model",
            "",
            "**Before:** notebook **1** (tokens).",
            "",
            "**This notebook:** count bigrams and sample — simplest next-char predictor.",
            "",
            "**Dojo (optional):** run cells, then continue to notebook **3**.",
        ),
        code(SETUP),
        code(
            """\
import torch
import torch.nn.functional as F

text = DATA.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)"""
        ),
        code(
            """\
# Bigram model: predict next character from previous character only.
N = vocab_size
counts = torch.zeros((N, N), dtype=torch.float32)
for t in range(len(data) - 1):
    i, j = data[t].item(), data[t + 1].item()
    counts[i, j] += 1

bigram = counts + 1  # smoothing
bigram /= bigram.sum(dim=1, keepdim=True)
print("Bigram table shape:", bigram.shape)"""
        ),
        code(
            """\
# Sample from the bigram model
g = torch.Generator().manual_seed(0)
idx = torch.zeros(1, 1, dtype=torch.long)
for _ in range(200):
    logits = bigram[idx[:, -1]]
    idx_next = torch.multinomial(logits, num_samples=1, generator=g)
    idx = torch.cat([idx, idx_next], dim=1)

print(decode(idx.squeeze().tolist()))"""
        ),
    ),
    "3.Embeddings.ipynb": nb(
        md(
            "# 3 — Token + position embeddings",
            "",
            "**Before:** notebooks **1–2**.",
            "",
            "**This notebook:** `wte` + `wpe` — same idea as GPT-2 / llm.c.",
        ),
        code(SETUP),
        code(
            """\
import torch
import torch.nn as nn

text = DATA.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

block_size = 8
batch_size = 4"""
        ),
        code(
            """\
n_embd = 32

token_emb = nn.Embedding(vocab_size, n_embd)
pos_emb = nn.Embedding(block_size, n_embd)

ix = torch.randint(0, len(data) - block_size, (batch_size,))
x = torch.stack([data[i : i + block_size] for i in ix])

tok = token_emb(x)
pos = pos_emb(torch.arange(block_size))
hidden = tok + pos

print("token_emb:", tok.shape)
print("hidden state:", hidden.shape)"""
        ),
    ),
    "4.Attention.ipynb": nb(
        md(
            "# 4 — Causal self-attention",
            "",
            "**Before:** notebook **3** (embeddings).",
            "",
            "**This notebook:** one head, then multi-head attention (GPT-2 style).",
            "",
            "Uses synthetic tensors — no data file required after setup.",
        ),
        code(SETUP),
        code(
            """\
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
B, T, C = 2, 8, 16
n_head = 4
head_dim = C // n_head

x = torch.randn(B, T, C)"""
        ),
        code(
            """\
# Single head
q = k = v = x
wei = q @ k.transpose(-2, -1) / math.sqrt(C)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ v
print("attention output:", out.shape)

# Multi-head (GPT-2 style)
c_attn = nn.Linear(C, 3 * C)
qkv = c_attn(x)
q, k, v = qkv.split(C, dim=2)
q = q.view(B, T, n_head, head_dim).transpose(1, 2)
k = k.view(B, T, n_head, head_dim).transpose(1, 2)
v = v.view(B, T, n_head, head_dim).transpose(1, 2)
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
att = att.masked_fill(tril.view(1, 1, T, T) == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
print("multi-head output:", y.shape)"""
        ),
    ),
    "5.GPTBlock.ipynb": nb(
        md(
            "# 5 — One GPT transformer block",
            "",
            "**Before:** notebook **4** (attention).",
            "",
            "**This notebook:** `llmc.model.Block` — LayerNorm + attention + MLP + residuals.",
        ),
        code(SETUP),
        code(
            """\
import torch
from llmc.model import GPTConfig, Block

text = DATA.read_text(encoding="utf-8")
vocab_size = len(sorted(set(text)))

config = GPTConfig.tiny(vocab_size=vocab_size, block_size=64)
block = Block(config)
x = torch.randn(2, 16, config.n_embd)
y = block(x)
print("GPT block in/out:", x.shape, "->", y.shape)"""
        ),
    ),
    "6.GPT.ipynb": nb(
        md(
            "# 6 — Full GPT module",
            "",
            "**Before:** notebook **5** (one block).",
            "",
            "**This notebook:** stack blocks into `GPT`, forward pass + loss.",
        ),
        code(SETUP),
        code(
            """\
import torch
from llmc.data import CharTokenizer, load_text
from llmc.model import GPT, GPTConfig

text = load_text(DATA)
tokenizer = CharTokenizer.from_text(text)
config = GPTConfig.tiny(vocab_size=tokenizer.vocab_size, block_size=64)
model = GPT(config)

print(model)
print(f"Parameters: {model.count_parameters():,}")"""
        ),
        code(
            """\
x = torch.randint(0, tokenizer.vocab_size, (4, 32))
logits, loss = model(x, x)
print("logits:", logits.shape)
print("loss (random targets):", loss.item())"""
        ),
        code(VERIFY_MODEL),
    ),
    "7.BatchAndLoss.ipynb": nb(
        md(
            "# 7 — Batches and cross-entropy loss",
            "",
            "**Before:** notebook **6** (full GPT).",
            "",
            "**This notebook:** `get_batch` + one training-style forward on real token IDs.",
        ),
        code(SETUP),
        code(
            """\
import torch
from llmc.data import CharTokenizer, get_batch, load_text, train_val_split
from llmc.model import GPT, GPTConfig

text = load_text(DATA)
train_text, val_text = train_val_split(text)
tok = CharTokenizer.from_text(text)
train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

config = GPTConfig.tiny(vocab_size=tok.vocab_size, block_size=64)
model = GPT(config)"""
        ),
        code(
            """\
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x, y = get_batch(train_ids, batch_size=8, block_size=config.block_size, device=device)
_, loss = model(x, y)
print(f"device={device}, batch loss={loss.item():.4f}")"""
        ),
    ),
    "8.Train.ipynb": nb(
        md(
            "# 8 — Training loop",
            "",
            "**Before:** notebook **7** (batches + loss).",
            "",
            "**This notebook:** `Trainer` — same structure as llm.c / `train_gpt2.py`.",
            "",
            "Set `RUN_TRAIN = True` to run 300 steps (~seconds on tiny GPT). Default is a quick 50-step demo.",
            "",
            "**Dojo (optional):** `dojo-grade --lesson C2-L08`",
        ),
        code(SETUP),
        code(
            """\
import torch
from llmc.data import CharTokenizer, load_text, train_val_split
from llmc.model import GPT, GPTConfig
from llmc.train import Trainer, TrainConfig

text = load_text(DATA)
train_text, val_text = train_val_split(text)
tok = CharTokenizer.from_text(text)
train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

config = GPTConfig.tiny(vocab_size=tok.vocab_size, block_size=64)
model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"

RUN_TRAIN = False  # True → 300 steps; False → 50-step quick demo
max_steps = 300 if RUN_TRAIN else 50

trainer = Trainer(
    model,
    train_ids,
    val_ids,
    TrainConfig(max_steps=max_steps, batch_size=32, eval_interval=25, learning_rate=3e-3),
    device=device,
)
print(f"device={device}, max_steps={max_steps}")"""
        ),
        code(
            """\
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
history = trainer.train()
for row in history:
    print(f"step {row['step']:4d} | train {row['train']:.4f} | val {row['val']:.4f}")

torch.save({"model": trainer.model.state_dict(), "config": config}, CHECKPOINT)
print("saved", CHECKPOINT)"""
        ),
        code(VERIFY_TRAIN),
    ),
    "9.Sample.ipynb": nb(
        md(
            "# 9 — Generate text",
            "",
            "**Before:** notebook **8** (training). Loads `checkpoints/tiny_gpt.pt` if you trained in nb 8.",
            "",
            "**This notebook:** `generate()` — inference loop from llm.c / nanoGPT.",
        ),
        code(SETUP),
        code(
            """\
import torch
from llmc.data import CharTokenizer, load_text
from llmc.model import GPT, GPTConfig
from llmc.sample import generate

text = load_text(DATA)
tok = CharTokenizer.from_text(text)
config = GPTConfig.tiny(vocab_size=tok.vocab_size, block_size=64)
model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"

if CHECKPOINT.is_file():
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint", CHECKPOINT)
else:
    print("no checkpoint — untrained model (gibberish). Run notebook 8 first.")

model = model.to(device)
prompt = "ROMEO:"
ctx = tok.encode_tensor(prompt, device=device).unsqueeze(0)
out = generate(model, ctx, max_new_tokens=200, temperature=0.8, top_k=40)
print(tok.decode(out.squeeze().tolist()))"""
        ),
    ),
    "10.GPT2AndLlmc.ipynb": nb(
        md(
            "# 10 — GPT-2 124M and llm.c map",
            "",
            "**Before:** notebooks **1–9** (tiny GPT track).",
            "",
            "**This notebook:** GPT-2 hyperparameters + parameter count vs our tiny model.",
            "",
            "Cell 2 allocates ~124M params in RAM (no training) — skip on very low-memory hosts.",
        ),
        code(SETUP),
        code(
            """\
from llmc.model import GPTConfig

cfg = GPTConfig.gpt2_small(vocab_size=50257, block_size=1024)
print("GPT-2 small config:")
print(f"  layers={cfg.n_layer}, heads={cfg.n_head}, embd={cfg.n_embd}")
print(f"  block_size={cfg.block_size}, vocab={cfg.vocab_size}")"""
        ),
        code(
            """\
print(\"\"\"
Our notebooks (PyTorch, educational)     llm.c (C/CUDA, production speed)
------------------------------------     ---------------------------------
1.Tokens                                 dev/data/*.py tokenizes to .bin
2.Bigram                                 (baseline)
3.Embeddings                             wte + wpe in GPT
4.Attention                              dev/cuda attention kernels
5.GPTBlock                               one transformer block
6.GPT                                    full model forward
7.BatchAndLoss                           cross-entropy + batches
8.Train                                  train_gpt2.cu main loop
9.Sample                                 inference / generation
10.GPT2AndLlmc                           GPT-2 124M + compare checkpoints

Upstream: https://github.com/karpathy/llm.c
Clone llm.c separately for CUDA training; use THIS repo to understand the math.
\"\"\")"""
        ),
        code(
            """\
from llmc.model import GPT
from llmc.data import CharTokenizer, load_text

text = load_text(DATA)
tok = CharTokenizer.from_text(text)
tiny = GPT(GPTConfig.tiny(tok.vocab_size, 64))
gpt2 = GPT(GPTConfig.gpt2_small(50257, 1024))
print(f"tiny params:   {tiny.count_parameters():,}")
print(f"gpt2 params:   {gpt2.count_parameters():,}")"""
        ),
    ),
}


def main() -> None:
    for name, notebook in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(notebook, indent=2) + "\n")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
