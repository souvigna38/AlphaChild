#!/usr/bin/env python3
"""Inject learning objectives into the first markdown cell of every course notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path

from learning_objectives import ALPHA_OBJECTIVES, LLM_OBJECTIVES, format_objectives

ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "llm-c-from-scratch"


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_src(cell: dict, text: str) -> None:
    cell["source"] = [ln + "\n" for ln in text.strip("\n").split("\n")]


def upsert_objectives(md: str, objectives: list[str]) -> str:
    block = format_objectives(objectives)
    if "**Learning objectives**" in md:
        md = re.sub(
            r"\*\*Learning objectives\*\*.*?(?=\n\*\*|\n## |\Z)",
            block + "\n\n",
            md,
            flags=re.S,
        )
        return md.rstrip() + "\n"
    # Insert after **This notebook:** paragraph
    m = re.search(r"(\*\*This notebook:\*\*[^\n]*\n)", md)
    if m:
        pos = m.end()
        return md[:pos] + "\n" + block + "\n\n" + md[pos:].lstrip("\n")
    # Fallback: after title
    lines = md.split("\n")
    if lines and lines[0].startswith("#"):
        return lines[0] + "\n\n" + block + "\n\n" + "\n".join(lines[1:]).lstrip("\n") + "\n"
    return block + "\n\n" + md


def patch_notebook(path: Path, objectives: list[str]) -> None:
    nb = json.loads(path.read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            _set_src(cell, upsert_objectives(_src(cell), objectives))
            path.write_text(json.dumps(nb, indent=2) + "\n")
            print("updated", path.name)
            return
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": []})
    _set_src(nb["cells"][0], upsert_objectives("", objectives))
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("inserted intro", path.name)


def main() -> None:
    for name, objs in ALPHA_OBJECTIVES.items():
        p = ROOT / name
        if p.is_file():
            patch_notebook(p, objs)
    for name, objs in LLM_OBJECTIVES.items():
        p = LLM_ROOT / name
        if p.is_file():
            patch_notebook(p, objs)
    print("done")


if __name__ == "__main__":
    main()
