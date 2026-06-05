"""Run packaged dojo gate tests (fast subset)."""

import subprocess
import sys
from pathlib import Path

GATES = Path(__file__).resolve().parents[1] / "dojo" / "gates"
FAST = [
    "test_c1_l01_tictactoe.py",
    "test_c1_l02_mcts.py",
    "test_c2_l01_tokens.py",
    "test_c2_l04_attention.py",
]


def test_dojo_gates_fast():
    repo = Path(__file__).resolve().parents[1]
    llm = repo / "llm-c-from-scratch"
    env = {"PYTHONPATH": f"{repo}{':'}{llm}"}
    for name in FAST:
        path = GATES / name
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q"],
            cwd=str(repo / "dojo"),
            env={**dict(**__import__("os").environ), **env},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
