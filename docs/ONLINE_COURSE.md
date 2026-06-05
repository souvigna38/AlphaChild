# AlphaChild — Online Course Guide

A **beginner-friendly**, notebook-first curriculum for learning reinforcement learning and language models from scratch.

## Two tracks

| Track | Notebooks | Goal | Time (rough) |
|-------|-----------|------|--------------|
| **A — AlphaZero** | `1.TicTacToe` … `16.MuZeroAtariTrain` (repo root) | MCTS → AlphaZero → MuZero → Gym/Atari | ~20–40 hours |
| **B — LLM (llm.c)** | `llm-c-from-scratch/1.Tokens` … `18.DeepSeekV4Path` | Tokens → GPT → DeepSeek-V2 → V4 C port | ~25–50 hours |

You can take **either track first**. Track B assumes basic Python; Track A assumes basic Python + NumPy.

## Quick start (students)

```bash
git clone https://github.com/souvigna38/AlphaChild.git
cd AlphaChild
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,atari]"
pip install -e "llm-c-from-scratch/.[dev]"
python3 -m pytest tests/ llm-c-from-scratch/tests/ -q
jupyter lab
```

Open notebooks **in numeric order** within each track.

## How notebooks are structured

Every course-ready notebook includes:

1. **Markdown intro** — prerequisites, summary, and **learning objectives** (bullet list)
2. **Setup cell** — finds the repo root automatically (works in Cursor workbook / nested clones)
3. **Code cells** — run top-to-bottom; no hidden state required
4. **Gated slow cells** — training, C compiles, and human `input()` play are **off by default**

Objectives live in each notebook’s first markdown cell. To regenerate them from source:

```bash
python3 scripts/add_learning_objectives.py
```

### Track A flags (setup cell)

| Flag | Default | When to enable |
|------|---------|----------------|
| `RUN_TRAIN` | `False` | Self-play / MuZero training (minutes to hours) |
| `PLAY_INTERACTIVE` | `False` | Play human vs AI in the terminal |
| `DEMO_SEARCHES` | `100` | MCTS simulations for demos (raise for stronger play) |

### Track B flags (setup cell)

| Flag | Default | When to enable |
|------|---------|----------------|
| `RUN_TRAIN` | `False` | Full GPT / DeepSeek training loops |
| `RUN_GPT2` | `False` | Allocate GPT-2 124M (~500MB RAM) in nb 10 |
| `RUN_C` | `False` | Compile/run C smoke tests in terminal cells |
| `RUN_EXPORT` | `False` | Export V2 weights in nb 16 |

## Track A — AlphaZero (notebooks 1–16)

### Module 1: Foundations (1–4)

| # | Notebook | Learning objectives |
|---|----------|---------------------|
| 1 | TicTacToe | Game API: state, moves, terminal, reward |
| 2 | MCTS | Selection, expansion, simulation, backprop |
| 3 | Model | Policy + value head, board encoding |
| 4 | AlphaMCTS | Neural-guided tree search |

### Module 2: AlphaZero (5–10)

| # | Notebook | Learning objectives |
|---|----------|---------------------|
| 5 | AlphaSelfPlay | Replay buffer, self-play loop |
| 6 | AlphaTrain | Multi-epoch training on Tic-Tac-Toe |
| 7 | AlphaTweaks | LR decay, temperature, Dirichlet noise |
| 8 | ConnectFour | Harder game, same pipeline |
| 9 | AlphaParallel | Parallel self-play for speed |
| 10 | Eval | Benchmarking, optional Kaggle env |

### Module 3: MuZero (11–16)

| # | Notebook | Learning objectives |
|---|----------|---------------------|
| 11 | MuZeroModel | Dynamics + prediction networks |
| 12 | MuZeroMCTS | Latent-space planning |
| 13 | MuZeroTrain | MuZero training loop |
| 14 | MuZeroGym | Classic control (CartPole) |
| 15 | MuZeroAtariModel | Visual encoder, frames |
| 16 | MuZeroAtariTrain | Train CartPole; Atari optional |

**Checkpoints:** `model_2.pt` and `model_7_ConnectFour.pt` may ship in the repo for eval notebooks. If missing, train in notebooks 5–8 with `RUN_TRAIN=True`.

## Track B — LLM / llm.c (notebooks 1–18)

See **`llm-c-from-scratch/README.md`** for the full table.

### Module 1: GPT foundations (1–10)

Character tokens → bigram → embeddings → attention → GPT block → full GPT → train → sample → GPT-2 scale.

### Module 2: DeepSeek-V2 (11–17)

Roadmap → MLA → MoE → full model → train → sample/export → C backward.

### Module 3: DeepSeek-V4 (18)

Hash-MoE, SwiGLU, C port map — `make test_v4`.

**Verify as you go:**

```bash
cd llm-c-from-scratch
pytest tests/test_model.py -q          # after nb 6
pytest tests/test_train.py -q          # after nb 8
pytest tests/test_deepseek_v2.py -q    # after nb 14–15
pytest tests/test_deepseek_v4.py -q    # after nb 18
```

## Agentic Dojo (optional cohort)

Discord-based gates with cryptographic proofs. See **`docs/AGENTIC_DOJO.md`**.

| Track | Example gate |
|-------|----------------|
| LLM | `dojo-grade --lesson C2-L08` |
| V2 | `dojo-grade --lesson C2-L15` |
| V4 | `dojo-grade --lesson C2-L18` |

Pin curriculum: `git checkout course-2026.1`

## Instructor checklist

- [ ] Students clone AlphaChild and install both packages (see Quick start)
- [ ] Walk through setup cell — must print `ROOT` and `data OK` (Track B)
- [ ] Emphasize **Run All** works with default flags (no training, no keyboard input)
- [ ] Assign `RUN_TRAIN=True` as homework after notebook 5 (Track A) or 8 (Track B)
- [ ] Point advanced students to C ports: `llm-c-from-scratch/c/` and `make test_v2` / `make test_v4`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: data/tiny_shakespeare.txt` | Re-run setup cell; cwd should not matter |
| `ModuleNotFoundError: llmc` | `pip install -e llm-c-from-scratch/.[dev]` |
| `ModuleNotFoundError: alphazero` | `pip install -e ".[dev,atari]"` from repo root |
| Notebook hangs on `input()` | Set `PLAY_INTERACTIVE=False` in setup |
| Training too slow | Keep `RUN_TRAIN=False` for demos; use tiny configs |
| Missing `model_2.pt` | Train in nb 5–6 or use random weights for code walkthrough |

## Regenerate course notebooks

After editing patch templates or learning objectives (`scripts/learning_objectives.py`):

```bash
python3 scripts/add_learning_objectives.py
python3 llm-c-from-scratch/scripts/patch_notebooks_course.py
python3 scripts/patch_alphazero_notebooks.py
```
