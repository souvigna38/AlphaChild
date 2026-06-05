"""Cryptographic lesson proofs (HMAC). User-bound; not copy-pasteable across accounts."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import NamedTuple

PROOF_RE = re.compile(
    r"^PASS-(?P<lesson>[A-Z0-9]+-(?:L\d{2}|HW))-(?P<token>[0-9a-f]{12})$",
    re.IGNORECASE,
)
HARDWARE_RE = re.compile(
    r"^HW-(?P<tier>lightweight|standard|gpu)-(?P<token>[0-9a-f]{12})$",
    re.IGNORECASE,
)


class ParsedProof(NamedTuple):
    lesson_id: str
    token: str


def manifest_digest(pytest_target: str) -> str:
    """Stable fingerprint of the exact test gate (curriculum version binding)."""
    return hashlib.sha256(pytest_target.encode("utf-8")).hexdigest()[:16]


def make_proof(
    discord_user_id: int | str,
    lesson_id: str,
    secret: str | bytes,
    *,
    pytest_target: str = "",
) -> str:
    """Build PASS-{lesson}-{token} for posting in Discord."""
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    manifest = manifest_digest(pytest_target) if pytest_target else ""
    msg = f"{discord_user_id}|{lesson_id.upper()}|{manifest}"
    token = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"PASS-{lesson_id.upper()}-{token}"


def verify_proof(
    discord_user_id: int | str,
    lesson_id: str,
    token: str,
    secret: str | bytes,
    *,
    pytest_target: str = "",
) -> bool:
    expected = make_proof(
        discord_user_id,
        lesson_id,
        secret,
        pytest_target=pytest_target,
    )
    expected_token = expected.rsplit("-", 1)[-1]
    return hmac.compare_digest(expected_token.lower(), token.lower())


def parse_proof_line(text: str) -> ParsedProof | None:
    for word in text.strip().split():
        m = PROOF_RE.match(word.strip())
        if m:
            return ParsedProof(lesson_id=m.group("lesson").upper(), token=m.group("token").lower())
    return None


def make_hardware_proof(
    discord_user_id: int | str,
    tier: str,
    secret: str | bytes,
    *,
    profile_blob: str,
) -> str:
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    msg = f"{discord_user_id}|HW|{tier.lower()}|{profile_blob}"
    token = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"HW-{tier.lower()}-{token}"


def verify_hardware_proof(
    discord_user_id: int | str,
    tier: str,
    token: str,
    secret: str | bytes,
    *,
    profile_blob: str,
) -> bool:
    expected = make_hardware_proof(
        discord_user_id,
        tier,
        secret,
        profile_blob=profile_blob,
    )
    expected_token = expected.rsplit("-", 1)[-1]
    return hmac.compare_digest(expected_token.lower(), token.lower())


def parse_hardware_line(text: str) -> tuple[str, str] | None:
    for word in text.strip().split():
        m = HARDWARE_RE.match(word.strip())
        if m:
            return m.group("tier").lower(), m.group("token").lower()
    return None
