"""Test that non-interactive notebook cells execute without errors.

Cells containing input() are skipped since they require interactive stdin.
Training cells (alphaZero.learn()) are also skipped to keep tests fast.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

SKIP_PATTERNS = [
    r'\binput\s*\(',
    r'\.learn\(\)',
    r'alphaZero\.learn',
    r'alphaZeroParallel\.learn',
    r'muzero\.learn',
    r'muzero_atari\.learn',
    r'AtariGym\(',
    r'ALE/',
]


def _cell_should_skip(source: str) -> bool:
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, source):
            return True
    return False


def _get_notebook_cells(notebook_path: Path):
    with open(notebook_path) as f:
        nb = json.load(f)
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cells.append((i, source))
    return cells


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[nb.name for nb in NOTEBOOKS],
)
def test_notebook_non_interactive_cells_parse(notebook_path):
    """Verify all non-interactive cells are valid Python syntax."""
    cells = _get_notebook_cells(notebook_path)
    assert len(cells) > 0, f"No code cells found in {notebook_path.name}"

    for cell_idx, source in cells:
        if _cell_should_skip(source):
            continue
        try:
            compile(source, f"{notebook_path.name}:cell[{cell_idx}]", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {notebook_path.name} cell {cell_idx}: {e}")


def test_notebook_1_tictactoe_game_logic():
    """Execute non-interactive cells from notebook 1 to verify TicTacToe works."""
    nb_path = NOTEBOOK_DIR / "1.TicTacToe.ipynb"
    cells = _get_notebook_cells(nb_path)

    namespace = {}
    executed = 0
    for cell_idx, source in cells:
        if _cell_should_skip(source):
            continue
        exec(compile(source, f"notebook1:cell[{cell_idx}]", "exec"), namespace)
        executed += 1

    assert executed > 0
    assert "TicTacToe" in namespace


def test_notebook_3_model_inference():
    """Execute non-interactive cells from notebook 3 to verify ResNet works."""
    import matplotlib
    matplotlib.use("Agg")

    nb_path = NOTEBOOK_DIR / "3.Model.ipynb"
    cells = _get_notebook_cells(nb_path)

    namespace = {}
    executed = 0
    for cell_idx, source in cells:
        if _cell_should_skip(source):
            continue
        exec(compile(source, f"notebook3:cell[{cell_idx}]", "exec"), namespace)
        executed += 1

    assert executed >= 4
    assert "ResNet" in namespace
    assert "model" in namespace
