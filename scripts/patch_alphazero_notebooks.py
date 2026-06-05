#!/usr/bin/env python3
"""Prepare AlphaZero / MuZero notebooks 1–16 for online beginner course."""

from __future__ import annotations

import json
import re
from pathlib import Path

from learning_objectives import ALPHA_OBJECTIVES, format_objectives

ROOT = Path(__file__).resolve().parents[1]

SETUP = """\
# --- Course setup (AlphaChild repo root) ---
import sys
from pathlib import Path


def find_repo_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "alphazero").is_dir() and (base / "1.TicTacToe.ipynb").is_file():
            return base
        if (base / "alphazero").is_dir() and (base / "pyproject.toml").is_file():
            return base
    return Path.cwd()


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero.notebook_utils import checkpoint_path

RUN_TRAIN = False           # True: run self-play training (slow — minutes+)
PLAY_INTERACTIVE = False    # True: human vs AI in terminal (needs keyboard input)
DEMO_SEARCHES = 100         # MCTS searches for demos; increase when curious

print("ROOT", ROOT.resolve())
print("RUN_TRAIN", RUN_TRAIN, "| PLAY_INTERACTIVE", PLAY_INTERACTIVE, "| DEMO_SEARCHES", DEMO_SEARCHES)
"""

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

COURSE = {
    "1.TicTacToe.ipynb": (
        "1 — Tic-Tac-Toe game rules",
        "nothing — start here.",
        "implement the game board, legal moves, and win detection.",
    ),
    "2.MCTS.ipynb": (
        "2 — Monte Carlo Tree Search",
        "notebook **1** (game rules).",
        "build MCTS from scratch and play vs random on Tic-Tac-Toe.",
    ),
    "3.Model.ipynb": (
        "3 — Neural network for policy + value",
        "notebooks **1–2**.",
        "ResNet-style model that guides MCTS (AlphaZero-style).",
    ),
    "4.AlphaMCTS.ipynb": (
        "4 — AlphaZero MCTS",
        "notebooks **1–3**.",
        "combine the neural net with MCTS search.",
    ),
    "5.AlphaSelfPlay.ipynb": (
        "5 — Self-play training loop",
        "notebooks **1–4**.",
        "the core AlphaZero loop: self-play → train on replay buffer.",
    ),
    "6.AlphaTrain.ipynb": (
        "6 — Train on Tic-Tac-Toe",
        "notebook **5**.",
        "multi-iteration training; loads `model_2.pt` if present.",
    ),
    "7.AlphaTweaks.ipynb": (
        "7 — Training tweaks",
        "notebook **6**.",
        "learning-rate schedule, temperature, Dirichlet noise.",
    ),
    "8.ConnectFour.ipynb": (
        "8 — Connect Four",
        "notebooks **1–7**.",
        "scale up to a harder game — same AlphaZero recipe.",
    ),
    "9.AlphaParallel.ipynb": (
        "9 — Parallel self-play",
        "notebook **8**.",
        "speed up data collection with parallel games.",
    ),
    "10.Eval.ipynb": (
        "10 — Evaluation",
        "notebooks **1–9**.",
        "benchmark agents; optional Kaggle Connect Four env.",
    ),
    "11.MuZeroModel.ipynb": (
        "11 — MuZero model",
        "AlphaZero track **1–10**.",
        "dynamics + prediction network (MuZero).",
    ),
    "12.MuZeroMCTS.ipynb": (
        "12 — MuZero MCTS",
        "notebook **11**.",
        "planning in learned latent space.",
    ),
    "13.MuZeroTrain.ipynb": (
        "13 — MuZero training",
        "notebooks **11–12**.",
        "full MuZero learn loop on board games.",
    ),
    "14.MuZeroGym.ipynb": (
        "14 — MuZero + Gymnasium",
        "MuZero notebooks **11–13**.",
        "intro to classic control (CartPole). Requires `pip install -e \".[dev,atari]\"`.",
    ),
    "15.MuZeroAtariModel.ipynb": (
        "15 — MuZero Atari model",
        "notebook **14**.",
        "visual encoder + frame stacking for Atari (optional — needs ROMs).",
    ),
    "16.MuZeroAtariTrain.ipynb": (
        "16 — MuZero Atari training",
        "notebook **15**.",
        "train on CartPole by default; Atari in try/except.",
    ),
}


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_src(cell: dict, text: str) -> None:
    cell["source"] = [ln + "\n" for ln in text.strip("\n").split("\n")]


def has_setup(nb: dict) -> bool:
    return any(
        cell.get("cell_type") == "code" and "find_repo_root" in _src(cell) for cell in nb.get("cells", [])
    )


def intro_cell(title: str, before: str, body: str, filename: str) -> dict:
    objectives = ALPHA_OBJECTIVES.get(filename, [])
    obj_block = ("\n\n" + format_objectives(objectives)) if objectives else ""
    text = (
        f"# {title}\n\n"
        f"**Before:** {before}\n\n"
        f"**This notebook:** {body}"
        f"{obj_block}\n\n"
        "**Online course:** run cells top-to-bottom. In setup, keep `RUN_TRAIN=False` "
        "until you want a long training run. Set `PLAY_INTERACTIVE=True` only to play in the terminal.\n\n"
        "**Install:** `pip install -e \".[dev,atari]\"` from the AlphaChild repo root.\n\n"
        "Curriculum: `docs/ONLINE_COURSE.md`"
    )
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in text.split("\n")]}


def setup_cell() -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": []}


def patch_source(src: str) -> str:
    # Guard checkpoint loads
    for name in ("model_2.pt", "model_7_ConnectFour.pt", "model_0_ConnectFour.pt"):
        pat = f"torch.load('{name}'"
        if pat in src and f"checkpoint_path(ROOT, '{name}')" not in src:
            src = src.replace(
                f"torch.load('{name}'",
                f"torch.load(checkpoint_path(ROOT, '{name}')",
            )
        pat2 = f'torch.load("{name}"'
        if pat2 in src:
            src = src.replace(
                f'torch.load("{name}"',
                f'torch.load(checkpoint_path(ROOT, "{name}")',
            )

    # Wrap .learn() calls
    if re.search(r"^\s*\w+\.learn\(\)\s*$", src, re.M) and "RUN_TRAIN" not in src:
        src = re.sub(
            r"^(\s*)(\w+\.learn\(\))\s*$",
            r"\1if RUN_TRAIN:\n\1    \2\n\1else:\n\1    print(\"Training skipped — set RUN_TRAIN=True in setup cell\")",
            src,
            flags=re.M,
        )

    # Wrap interactive play (cells with input for player action)
    if 'input(f"{player}:")' in src and "PLAY_INTERACTIVE" not in src:
        src = (
            "if PLAY_INTERACTIVE:\n"
            + "\n".join("    " + line for line in src.split("\n"))
            + "\nelse:\n"
            + '    print("Interactive play skipped — set PLAY_INTERACTIVE=True in setup cell")'
        )

    # Softer MCTS for human-play demos
    if "PLAY_INTERACTIVE" in src or "input(f" in src:
        src = re.sub(r"'num_searches':\s*1000", "'num_searches': DEMO_SEARCHES", src)
        src = re.sub(r"'num_searches':\s*600", "'num_searches': DEMO_SEARCHES", src)

    return src


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    meta = COURSE.get(path.name)
    if meta:
        if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown":
            _set_src(nb["cells"][0], _src(intro_cell(*meta, path.name)))
        else:
            nb["cells"].insert(0, intro_cell(*meta, path.name))
    if not has_setup(nb):
        idx = 1 if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown" else 0
        nb["cells"].insert(idx, setup_cell())
        _set_src(nb["cells"][idx], SETUP)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            _set_src(cell, patch_source(_src(cell)))
            cell["outputs"] = []
            cell["execution_count"] = None

    nb["metadata"].update(KERNEL)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 4
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched", path.name)


def main() -> None:
    for path in sorted(ROOT.glob("[0-9]*.ipynb")):
        patch_notebook(path)
    print("done")


if __name__ == "__main__":
    main()
