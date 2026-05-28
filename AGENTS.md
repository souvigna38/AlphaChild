# AGENTS.md

Instructions for **Cursor agents** (and humans) cloning this repo and running it locally.

---

## Quick checklist (do this in order)

1. Clone the repository and `cd` into it.
2. Use **Python 3.10+** (3.11 or 3.12 recommended).
3. Create a virtual environment and activate it.
4. Install dependencies with `pip install -e ".[dev,atari]"` (or a smaller set — see below).
5. Put `~/.local/bin` on your `PATH` if needed.
6. Run `pytest tests/ -v` to verify the install.
7. Start Jupyter and open notebooks in numeric order (`1` → `16`).

There are **no servers** to start (no Docker, no database). Everything is notebooks + Python.

---

## 1. Clone the repository

```bash
git clone https://github.com/souvigna38/AlphaChild.git
cd AlphaChild
```

To use the branch that includes **MuZero + Gym/Atari** (notebooks 11–16):

```bash
git fetch origin
git checkout cursor/muzero-atari-gym-64d3
```

For the default AlphaZero-only tree, stay on `main`:

```bash
git checkout main
```

---

## 2. Python environment

**Requirement:** Python **3.10 or newer**.

```bash
python3 --version   # must be >= 3.10
```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

## 3. Install dependencies

Install from the repo root (`AlphaChild/`). The project is defined in `pyproject.toml`.

### Option A — Full local setup (recommended for agents)

Includes core libs, tests, linter, Jupyter kernel, Gym, and Atari:

```bash
pip install -e ".[dev,atari]"
```

This installs:

| Extra | Purpose |
|-------|---------|
| (core) | `numpy`, `torch`, `matplotlib`, `tqdm` |
| `dev` | `pytest`, `ruff`, `nbformat`, `nbclient`, `ipykernel`, `gymnasium`, `opencv-python` |
| `atari` | `gymnasium[atari,accept-rom-license]` for `ALE/Pong-v5` etc. |

### Option B — Minimal (board games only, notebooks 1–10)

```bash
pip install -e ".[dev]"
```

Skips Atari ROMs; MuZero Gym notebooks 14–16 Atari cells will fail until you add `[atari]`.

### Option C — MuZero CartPole only (no Atari ROMs)

```bash
pip install -e ".[dev,gym]"
```

Enough for **notebook 16** CartPole training; Atari cells in 14–15 need Option A.

### PATH

Pip may install CLI tools to `~/.local/bin`. If `jupyter`, `pytest`, or `ruff` are not found:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add that line to your shell profile if you use this machine often.

---

## 4. Accept Atari ROM license (first time only)

If you installed `[atari]`, the **first** import of an ALE environment may require accepting the ROM license:

```bash
python3 -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"
```

Follow any prompt, or ensure `gymnasium[atari,accept-rom-license]` was installed so the license is pre-accepted.

---

## 5. Verify the install

From the repo root:

```bash
ruff check alphazero/ tests/
python3 -m pytest tests/ -v
```

**Expected:** all tests pass (80+). CartPole MuZero tests need `gymnasium`; Atari is not required for CI-style tests.

If tests fail:

- **`ModuleNotFoundError: gymnasium`** → run `pip install -e ".[dev,gym]"` or `.[dev,atari]`.
- **`torch` errors** → install a CPU build: `pip install torch` (or follow https://pytorch.org for CUDA).
- **`opencv` missing** → `pip install opencv-python`.

---

## 6. Run Jupyter locally

Register a kernel (once per venv):

```bash
python3 -m ipykernel install --user --name alphazero --display-name "Python (alphazero)"
```

Start Jupyter from the repo root:

```bash
jupyter notebook
# or
jupyter lab
```

In the UI, open notebooks **in numeric order**:

| Range | Topic |
|-------|--------|
| `1`–`10` | AlphaZero (TicTacToe → Connect Four → training → eval) |
| `11`–`13` | MuZero on board games |
| `14`–`16` | Gym wrapper → Atari model → training (CartPole + optional Pong) |

**Kernel:** Notebooks default to a kernel named `myenv`. Either:

- Select **Python (alphazero)** (after `ipykernel install` above), or  
- Select any Python 3 kernel that uses the same venv where you ran `pip install`.

**Non-interactive execution (optional):**

```bash
jupyter nbconvert --to notebook --execute 1.TicTacToe.ipynb \
  --ExecutePreprocessor.kernel_name=python3
```

---

## 7. What agents should not run blindly

- Cells with **`input()`** — human vs AI play (notebooks 2, 4, 8, 10, 12, 13, etc.). Skip or run manually.
- **`muzero.learn()`** / long **`alphaZero.learn()`** — training loops; can take a long time. Use small `args` in the notebook or skip unless the task is training.
- **Notebook 16 Atari cell** — needs `[atari]` and ROM license; CartPole cells work without Atari.

Pre-trained weights in the repo (optional):

```python
import torch
model.load_state_dict(torch.load("model_2.pt", map_location="cpu", weights_only=True))
```

---

## 8. Project layout (reference)

```
AlphaChild/
├── 1.TicTacToe.ipynb … 10.Eval.ipynb    # AlphaZero tutorials
├── 11.MuZeroModel.ipynb … 13.MuZeroTrain.ipynb
├── 14.MuZeroGym.ipynb … 16.MuZeroAtariTrain.ipynb
├── alphazero/                           # importable package
│   ├── games.py, model.py, mcts.py
│   ├── gym_env.py, muzero_atari.py, muzero_gym.py
├── tests/
├── pyproject.toml
├── model_*.pt                           # checkpoints (optional)
└── AGENTS.md                            # this file
```

---

## 9. Commands summary (copy-paste for agents)

```bash
git clone https://github.com/souvigna38/AlphaChild.git
cd AlphaChild
git checkout cursor/muzero-atari-gym-64d3   # optional: MuZero + Gym branch
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,atari]"
export PATH="$HOME/.local/bin:$PATH"
python3 -m pytest tests/ -v
python3 -m ipykernel install --user --name alphazero --display-name "Python (alphazero)"
jupyter lab
```

---

## Cursor Cloud specific notes

This is a pure-Python Jupyter notebook project. No Docker compose, no background services.

### Linting

```bash
ruff check alphazero/ tests/
```

### Testing

```bash
python3 -m pytest tests/ -v
```

Notebook syntax tests skip `input()`, `.learn()`, `AtariGym`, and `ALE/` cells when ROMs are unavailable.

### Notebook caveats

- Override kernel: `--ExecutePreprocessor.kernel_name=python3`
- Training cells are skipped in automated notebook tests to keep CI fast.
