"""Helpers for AlphaZero / MuZero Jupyter notebooks."""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    """Return AlphaChild repo root from cwd or parent directories."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "alphazero").is_dir() and (base / "1.TicTacToe.ipynb").is_file():
            return base
        if (base / "alphazero").is_dir() and (base / "pyproject.toml").is_file():
            return base
    return Path.cwd()


def checkpoint_path(root: Path | None = None, name: str = "model_2.pt") -> Path:
    return (root or find_repo_root()) / name
