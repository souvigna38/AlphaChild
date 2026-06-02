"""15-minute gate assessment — administered by gatekeeper bot in DM."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

ASSESSMENT_MINUTES = 15
PASS_FRACTION = 0.8


@dataclass(frozen=True)
class Question:
    prompt: str
    choices: tuple[str, ...]
    correct_index: int


QUESTIONS: tuple[Question, ...] = (
    Question(
        "What does `git fork` primarily give you?",
        ("A read-only copy of issues", "Your own remote copy to push changes", "A local branch only", "CI secrets"),
        1,
    ),
    Question(
        "In Python, what does `pytest` do?",
        ("Formats code", "Runs automated tests", "Builds Docker images", "Creates virtualenvs"),
        1,
    ),
    Question(
        "Why do we use a student workspace repo separate from the curriculum repo?",
        (
            "Faster clones",
            "Avoid merge conflicts when curriculum updates",
            "Discord requires it",
            "PyTorch limitation",
        ),
        1,
    ),
    Question(
        "A causal language model mask prevents attending to ___ tokens.",
        ("previous", "future", "padding only", "all other batch items"),
        1,
    ),
    Question(
        "MCTS balances exploration and exploitation using a ___ term.",
        ("loss", "UCB", "dropout", "embedding"),
        1,
    ),
    Question(
        "What is stored in DOJO_PROOF_SECRET?",
        ("Your Discord password", "Shared HMAC key for lesson proofs", "GitHub PAT", "Model weights"),
        1,
    ),
    Question(
        "If your Mac has 8GB RAM and no GPU, which track tier is appropriate for tiny LLM notebooks?",
        ("gpu", "standard", "lightweight only forever", "none"),
        1,
    ),
    Question(
        "Copying another student's PASS-* hash will fail because the proof binds to ___.",
        ("notebook name", "Discord user id", "GPU serial", "pytest version only"),
        1,
    ),
    Question(
        "MoE routes each token through ___ experts (typical DeepSeek-style).",
        ("all", "top-k", "zero", "exactly one always"),
        1,
    ),
    Question(
        "After `pip install` curriculum upgrade, you should re-run ___ before requesting a new proof.",
        ("discord", "dojo-grade gate", "format C: drive", "delete .venv"),
        1,
    ),
    Question(
        "AlphaZero improves over vanilla MCTS by learning ___ from self-play.",
        ("only rules", "policy and value", "tokenizer", "CUDA kernels"),
        1,
    ),
    Question(
        "The command `dojo-profile` reports hardware tier to gate ___ tracks.",
        ("Atari-only", "LLM vs board-game", "Discord Nitro", "Git LFS"),
        1,
    ),
)


@dataclass
class AssessmentSession:
    user_id: int
    questions: list[Question]
    answers: list[int | None]
    index: int
    started_at: float
    deadline_at: float

    @property
    def finished(self) -> bool:
        return self.index >= len(self.questions)

    def expired(self) -> bool:
        return time.time() > self.deadline_at

    def score_fraction(self) -> float:
        correct = 0
        for q, a in zip(self.questions, self.answers, strict=False):
            if a is not None and a == q.correct_index:
                correct += 1
        return correct / len(self.questions) if self.questions else 0.0

    def passed(self) -> bool:
        return self.score_fraction() >= PASS_FRACTION


def start_session(user_id: int, *, num_questions: int = 10) -> AssessmentSession:
    pool = list(QUESTIONS)
    random.shuffle(pool)
    picked = pool[: min(num_questions, len(pool))]
    now = time.time()
    return AssessmentSession(
        user_id=user_id,
        questions=picked,
        answers=[None] * len(picked),
        index=0,
        started_at=now,
        deadline_at=now + ASSESSMENT_MINUTES * 60,
    )


def format_question(session: AssessmentSession) -> str:
    q = session.questions[session.index]
    lines = [
        f"**Question {session.index + 1}/{len(session.questions)}**",
        q.prompt,
        "",
    ]
    for i, choice in enumerate(q.choices):
        lines.append(f"{i + 1}. {choice}")
    lines.append("")
    lines.append("Reply with a number `1`–`4` (time limit applies).")
    remaining = int(session.deadline_at - time.time())
    lines.append(f"_Time left: ~{max(0, remaining // 60)} min_")
    return "\n".join(lines)
