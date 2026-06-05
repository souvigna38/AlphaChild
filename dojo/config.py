"""Load Dojo bot configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DojoConfig:
    token: str
    guild_id: int
    proof_secret: str
    channel_waiting_room_id: int | None
    channel_verify_setup_id: int | None
    role_verified: str
    role_white_belt: str
    role_track_c1: str
    role_track_c2: str
    role_track_c3: str
    admin_user_ids: frozenset[int]
    ta_channel_ids: frozenset[int]
    ta_channel_name_prefixes: tuple[str, ...]

    @classmethod
    def from_env(cls) -> DojoConfig:
        token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN is required")

        secret = os.environ.get("DOJO_PROOF_SECRET", "").strip()
        if len(secret) < 16:
            raise ValueError("DOJO_PROOF_SECRET must be at least 16 characters")

        guild_id = int(os.environ.get("DOJO_GUILD_ID", "0"))
        if guild_id <= 0:
            raise ValueError("DOJO_GUILD_ID is required")

        def _cid(name: str) -> int | None:
            raw = os.environ.get(name, "").strip()
            return int(raw) if raw else None

        admins = os.environ.get("DOJO_ADMIN_USER_IDS", "")
        admin_ids = frozenset(int(x.strip()) for x in admins.split(",") if x.strip())

        ta_ch = os.environ.get("DOJO_TA_CHANNEL_IDS", "")
        ta_channel_ids = frozenset(int(x.strip()) for x in ta_ch.split(",") if x.strip())
        prefixes_raw = os.environ.get("DOJO_TA_CHANNEL_PREFIXES", "ask-,dojo-debug,sparring")
        ta_prefixes = tuple(p.strip().lower() for p in prefixes_raw.split(",") if p.strip())

        return cls(
            token=token,
            guild_id=guild_id,
            proof_secret=secret,
            channel_waiting_room_id=_cid("DOJO_CHANNEL_WAITING_ROOM"),
            channel_verify_setup_id=_cid("DOJO_CHANNEL_VERIFY_SETUP"),
            role_verified=os.environ.get("DOJO_ROLE_VERIFIED", "Verified"),
            role_white_belt=os.environ.get("DOJO_ROLE_WHITE_BELT", "White Belt"),
            role_track_c1=os.environ.get("DOJO_ROLE_TRACK_C1", "Track-Alpha"),
            role_track_c2=os.environ.get("DOJO_ROLE_TRACK_C2", "Track-LLM"),
            role_track_c3=os.environ.get("DOJO_ROLE_TRACK_C3", "Track-Systems"),
            admin_user_ids=admin_ids,
            ta_channel_ids=ta_channel_ids,
            ta_channel_name_prefixes=ta_prefixes,
        )

    def ta_allowed_in_channel(self, channel_id: int, channel_name: str) -> bool:
        if channel_id in self.ta_channel_ids:
            return True
        name = channel_name.lower()
        return any(name.startswith(p) for p in self.ta_channel_name_prefixes)
