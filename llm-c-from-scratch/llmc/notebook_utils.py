"""Helpers for Jupyter notebooks — path discovery in Cursor workbook or repo clone."""

from __future__ import annotations

from pathlib import Path


def find_llm_root() -> Path:
    """Return llm-c-from-scratch root from cwd, parent, or nested AlphaChild layout."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "llmc" / "__init__.py").is_file():
            return base
        nested = base / "llm-c-from-scratch"
        if (nested / "llmc" / "__init__.py").is_file():
            return nested
    return Path.cwd()


def data_path(root: Path | None = None) -> Path:
    return (root or find_llm_root()) / "data" / "tiny_shakespeare.txt"


def checkpoint_path(root: Path | None = None, name: str = "tiny_gpt.pt") -> Path:
    return (root or find_llm_root()) / "checkpoints" / name
