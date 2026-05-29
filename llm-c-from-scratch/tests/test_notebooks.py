import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(ROOT.glob("*.ipynb"))

SKIP = [
    r"trainer\.train\(\)",
    r"history = trainer\.train",
]


def _skip(source: str) -> bool:
    return any(re.search(p, source) for p in SKIP)


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[n.name for n in NOTEBOOKS])
def test_notebook_syntax(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if _skip(source):
            continue
        compile(source, f"{nb_path.name}:cell[{i}]", "exec")
