"""
Socratic Sparring Partner — 24/7 TA via OpenClaw or any OpenAI-compatible API.

Guardrails: never writes student code; rate-limited; post-filter blocks solution dumps.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("dojo.sparring_partner")

SOCRATIC_SYSTEM_PROMPT = """You are the Agentic Dojo Sparring Partner — a Socratic tutor for ML and RL notebooks.

RULES (non-negotiable):
1. NEVER write complete code solutions, full functions, or copy-paste homework answers.
2. NEVER output PASS-* proof tokens or tell students how to forge dojo-grade proofs.
3. You MAY: explain concepts, clarify error messages, describe tensor shapes in words, suggest
   what to print/debug, ask leading questions, point to a specific line if the student pasted code.
4. Keep answers under 400 words. End with one guiding question when appropriate.
5. If the student asks for the full answer, refuse politely and offer a smaller hint instead.

Context: students use AlphaChild / llm-c-from-scratch notebooks, pytest gates, and dojo-grade locally.
"""

# Patterns that suggest the model violated guardrails
_FORBIDDEN_PATTERNS = [
    re.compile(r"here(?:'s| is) the (?:full |complete )?solution", re.I),
    re.compile(r"copy(?:\s+and)?\s*paste this", re.I),
    re.compile(r"PASS-[A-Z0-9]+-L\d{2}-[0-9a-f]{12}", re.I),
    re.compile(r"DOJO_PROOF_SECRET", re.I),
]

_REFUSAL = (
    "🥋 I can only tutor Socratically — I won't write the solution. "
    "Tell me what you've tried, the exact error, and which notebook cell you're on."
)


@dataclass(frozen=True)
class TAConfig:
    enabled: bool
    api_url: str
    api_key: str
    model: str
    max_requests_per_hour: int
    timeout_sec: float


def load_ta_config() -> TAConfig:
    import os

    enabled = os.environ.get("DOJO_TA_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    api_url = (
        os.environ.get("OPENCLAW_API_URL", "").strip()
        or os.environ.get("DOJO_TA_API_URL", "http://127.0.0.1:18789/v1/chat/completions").strip()
    )
    api_key = os.environ.get("OPENCLAW_API_KEY", "").strip() or os.environ.get("DOJO_TA_API_KEY", "").strip()
    model = os.environ.get("DOJO_TA_MODEL", "openclaw").strip()
    max_rph = int(os.environ.get("DOJO_TA_MAX_REQUESTS_PER_HOUR", "12"))
    timeout = float(os.environ.get("DOJO_TA_TIMEOUT_SEC", "90"))
    return TAConfig(
        enabled=enabled,
        api_url=api_url,
        api_key=api_key,
        model=model,
        max_requests_per_hour=max_rph,
        timeout_sec=timeout,
    )


def extract_question(content: str, bot_user_id: int) -> str | None:
    """Strip mention markup; return None if not directed at the bot."""
    if not content:
        return None
    mention = f"<@{bot_user_id}>"
    if mention not in content and f"<@!{bot_user_id}>" not in content:
        return None
    text = re.sub(rf"<@!?{bot_user_id}>", "", content).strip()
    return text or None


def _count_code_lines(text: str) -> int:
    blocks = re.findall(r"```[\s\S]*?```", text)
    return sum(block.count("\n") for block in blocks)


def guardrail_check(reply: str) -> str | None:
    """Return refusal message if reply violates policy; else None."""
    if _count_code_lines(reply) > 20:
        return _REFUSAL
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(reply):
            return _REFUSAL
    return None


def build_messages(student_question: str, *, extra_context: str = "") -> list[dict[str, str]]:
    user = student_question
    if extra_context:
        user = f"{extra_context}\n\n---\nStudent question:\n{student_question}"
    return [
        {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def chat_completion_sync(cfg: TAConfig, messages: list[dict[str, str]]) -> str:
    """POST to OpenAI-compatible chat/completions (OpenClaw, vLLM, Ollama, etc.)."""
    body = json.dumps(
        {
            "model": cfg.model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 800,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    req = urllib.request.Request(cfg.api_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        log.error("TA API error: %s", e)
        raise

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected TA API response: {data!r}") from e


async def socratic_reply(
    cfg: TAConfig,
    student_question: str,
    *,
    channel_name: str = "",
    lesson_hint: str = "",
) -> str:
    """Async wrapper — runs sync HTTP in executor to avoid blocking discord event loop."""
    import asyncio

    ctx_parts = []
    if channel_name:
        ctx_parts.append(f"Discord channel: #{channel_name}")
    if lesson_hint:
        ctx_parts.append(f"Likely lesson context: {lesson_hint}")
    extra = "\n".join(ctx_parts)

    loop = asyncio.get_running_loop()
    messages = build_messages(student_question, extra_context=extra)

    def _call() -> str:
        return chat_completion_sync(cfg, messages)

    raw = await loop.run_in_executor(None, _call)
    violation = guardrail_check(raw)
    if violation:
        return violation
    return raw


def check_rate_limit(user_id: int, max_per_hour: int) -> bool:
    """Return True if request is allowed."""
    from dojo.store import ta_rate_allow

    return ta_rate_allow(user_id, max_per_hour)


def record_rate_limit(user_id: int) -> None:
    from dojo.store import ta_rate_record

    ta_rate_record(user_id)


def chunk_for_discord(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks
