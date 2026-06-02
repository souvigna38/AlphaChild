"""Run lesson gate tests and emit cryptographic PASS proof."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dojo.lessons import DOJO_ROOT, get_lesson
from dojo.proof import make_proof


def run_pytest_gate(gate_file: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(gate_file),
        "-q",
        "--tb=short",
    ]
    env = os.environ.copy()
    # Repo roots for editable / monorepo dev installs
    repo_root = DOJO_ROOT.parent
    llm_root = repo_root / "llm-c-from-scratch"
    py_path = [str(repo_root), str(llm_root)]
    prev = env.get("PYTHONPATH", "")
    if prev:
        py_path.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(py_path)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(DOJO_ROOT))
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def grade_lesson(discord_user_id: str, lesson_id: str, *, secret: str) -> int:
    lesson = get_lesson(lesson_id)
    if lesson is None:
        print(f"Unknown lesson: {lesson_id}", file=sys.stderr)
        print("Valid:", ", ".join(sorted(__import__("dojo.lessons", fromlist=["load_lessons"]).load_lessons())))
        return 2

    gate = lesson.gate_path
    if not gate.is_file():
        print(f"Gate file missing: {gate}", file=sys.stderr)
        return 2

    ok, output = run_pytest_gate(gate)
    if not ok:
        print("pytest FAILED — fix your environment before requesting proof.", file=sys.stderr)
        print(output[-4000:] if len(output) > 4000 else output, file=sys.stderr)
        return 1

    proof = make_proof(
        discord_user_id,
        lesson.lesson_id,
        secret,
        pytest_target=lesson.pytest_target,
    )
    print("=== Gate passed ===")
    print(proof)
    print(f"\nPaste only the line above into Discord lesson thread for {lesson.title}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Dojo lesson gate and print PASS-* cryptographic proof",
    )
    parser.add_argument("--discord-user-id", required=True, help="Your Discord user ID (numeric)")
    parser.add_argument("--lesson", required=True, help="Lesson ID e.g. C2-L12")
    args = parser.parse_args()

    secret = os.environ.get("DOJO_PROOF_SECRET", "")
    if not secret or len(secret) < 16:
        print(
            "DOJO_PROOF_SECRET must be set (16+ chars). Copy from your student .env / dojo bot config.",
            file=sys.stderr,
        )
        return 2

    return grade_lesson(args.discord_user_id, args.lesson.upper(), secret=secret)


if __name__ == "__main__":
    raise SystemExit(main())
