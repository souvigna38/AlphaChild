#!/usr/bin/env python3
"""
Provision a Discord server for Agentic Dojo Season 1.

YOU must run this locally — the cloud agent cannot access your Discord account.

Requirements:
  - Bot created at https://discord.com/developers/applications
  - Bot invited with Administrator (one-time setup), or Manage Roles + Manage Channels
  - DISCORD_BOT_TOKEN, DOJO_GUILD_ID, DISCORD_APPLICATION_ID in environment

Usage:
  pip install -e ".[dojo]"
  set -a && source dojo/.env && set +a
  python dojo/scripts/discord_provision.py
  python dojo/scripts/discord_provision.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import discord
import yaml

ROOT = Path(__file__).resolve().parents[2]
PINS = Path(__file__).resolve().parents[1] / "discord-pins"
LESSONS = Path(__file__).resolve().parents[1] / "lessons.yaml"

# Roles created bottom → top (Discord inserts new roles at bottom; we reposition at end)
AUTOMATED_ROLES = [
    ("Verified", 0x95A5A6),
    ("White Belt", 0xF1C40F),
    ("Track-Alpha", 0x3498DB),
    ("Track-LLM", 0x9B59B6),
    ("Track-Systems", 0x1ABC9C),
    ("Tier-Lightweight", 0x7F8C8D),
    ("Tier-Standard", 0x2ECC71),
    ("Tier-Gpu", 0xE74C3C),
]

STAFF_ROLE = ("Staff", 0x992D22)


def _lesson_role_names() -> list[str]:
    data = yaml.safe_load(LESSONS.read_text(encoding="utf-8"))
    names: list[str] = []
    for track in data.get("tracks", {}).values():
        for entry in track:
            names.append(entry.get("role", entry["id"]))
    return names


def _load_pin(name: str) -> str:
    path = PINS / name
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    # Discord markdown is close enough; strip # titles to bold for readability
    lines = []
    for line in text.splitlines():
        if line.startswith("# "):
            lines.append(f"**{line[2:].strip()}**")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def invite_url(application_id: str, permissions: int = 8) -> str:
    """Administrator=8 for initial provision; re-invite with least privilege later."""
    return (
        f"https://discord.com/api/oauth2/authorize"
        f"?client_id={application_id}&permissions={permissions}&scope=bot"
    )


def _save_channel_ids(
    waiting_id: int,
    verify_id: int,
) -> None:
    try:
        from dojo.env_file import merge_env

        merge_env(
            {
                "DOJO_CHANNEL_WAITING_ROOM": str(waiting_id),
                "DOJO_CHANNEL_VERIFY_SETUP": str(verify_id),
            }
        )
        print(f"  ✓ saved channel IDs to dojo/.env")
    except Exception as e:
        print(f"  ! could not update dojo/.env: {e}")
        print(f"    DOJO_CHANNEL_WAITING_ROOM={waiting_id}")
        print(f"    DOJO_CHANNEL_VERIFY_SETUP={verify_id}")


async def provision(dry_run: bool) -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    guild_id = int(os.environ.get("DOJO_GUILD_ID", "0"))
    app_id = os.environ.get("DISCORD_APPLICATION_ID", "").strip()

    if not token or guild_id <= 0:
        print("Set DISCORD_BOT_TOKEN and DOJO_GUILD_ID", file=sys.stderr)
        sys.exit(2)

    if app_id:
        print("Bot invite URL (Administrator for setup):\n", invite_url(app_id), "\n")

    intents = discord.Intents.default()
    intents.guilds = True
    intents.members = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        guild = client.get_guild(guild_id)
        if guild is None:
            guild = await client.fetch_guild(guild_id)
        print(f"Connected to guild: {guild.name} ({guild.id})")

        if dry_run:
            print("[dry-run] Would create roles:", [r[0] for r in AUTOMATED_ROLES] + _lesson_role_names())
            print("[dry-run] Would create channels — see discord_provision.py CHANNEL_PLAN")
            await client.close()
            return

        role_map: dict[str, discord.Role] = {}
        existing = {r.name: r for r in guild.roles}

        async def ensure_role(name: str, color: int) -> discord.Role:
            if name in existing:
                role_map[name] = existing[name]
                return existing[name]
            role = await guild.create_role(name=name, color=discord.Color(color), mentionable=True)
            role_map[name] = role
            print(f"  + role {name}")
            return role

        await ensure_role(*STAFF_ROLE)
        for spec in AUTOMATED_ROLES:
            await ensure_role(*spec)
        for name in _lesson_role_names():
            await ensure_role(name, 0x5865F2)

        # Reposition: Staff top, then bot, then lesson/track/tier, then verified at bottom of our stack
        me = guild.me
        if me and me.top_role:
            bot_role = me.top_role
            staff = role_map.get("Staff")
            order: list[discord.Role] = []
            if staff:
                order.append(staff)
            order.append(bot_role)
            for name in reversed([r[0] for r in AUTOMATED_ROLES] + _lesson_role_names()):
                if name in role_map:
                    order.append(role_map[name])
            try:
                await guild.edit_role_positions(positions=order, reason="Agentic Dojo provision")
                print("  ✓ role order updated (Staff > Bot > milestones)")
            except discord.Forbidden:
                print("  ! Could not reorder roles — drag @DojoBot above student roles manually")

        everyone = guild.default_role
        verified = role_map.get("Verified")
        white = role_map.get("White Belt")

        def ow(**kwargs: bool) -> discord.PermissionOverwrite:
            return discord.PermissionOverwrite(**kwargs)

        async def ensure_category(name: str) -> discord.CategoryChannel:
            for c in guild.categories:
                if c.name == name:
                    return c
            return await guild.create_category(name)

        async def ensure_text(
            name: str,
            category: discord.CategoryChannel | None,
            *,
            topic: str = "",
            overwrites: dict[discord.abc.Snowflake, discord.PermissionOverwrite] | None = None,
        ) -> discord.TextChannel:
            for ch in guild.text_channels:
                if ch.name == name:
                    return ch
            return await guild.create_text_channel(
                name,
                category=category,
                topic=topic,
                overwrites=overwrites,
            )

        async def post_pin(channel: discord.TextChannel, body: str) -> None:
            if not body:
                return
            for i in range(0, len(body), 1900):
                chunk = body[i : i + 1900]
                if i == 0:
                    msg = await channel.send(chunk)
                else:
                    msg = await channel.send(chunk)
            try:
                await msg.pin()
            except discord.HTTPException:
                pass
            print(f"  ✓ pinned {channel.name}")

        cat_start = await ensure_category("📌 START HERE")
        cat_help = await ensure_category("🆘 HELP")
        cat_c1 = await ensure_category("🎮 TRACK 1 — AlphaZero")
        cat_c2 = await ensure_category("🧠 TRACK 2 — LLM")
        cat_admin = await ensure_category("🔒 STAFF")

        # @everyone: only see entry channels
        public_entry = {
            everyone: ow(view_channel=True, send_messages=False),
        }
        waiting = {
            everyone: ow(view_channel=True, send_messages=True),
        }
        verified_ch = {}
        if verified:
            verified_ch[verified] = ow(view_channel=True, send_messages=True)
            verified_ch[everyone] = ow(view_channel=False)
        white_only = {}
        if white:
            white_only[white] = ow(view_channel=True, send_messages=True)
            white_only[everyone] = ow(view_channel=False)

        staff_only = {everyone: ow(view_channel=False)}
        if role_map.get("Staff"):
            staff_only[role_map["Staff"]] = ow(view_channel=True, send_messages=True)

        ch_rules = await ensure_text("rules", cat_start, overwrites=public_entry)
        ch_welcome = await ensure_text("welcome", cat_start, overwrites=public_entry)
        ch_setup = await ensure_text("setup-guide", cat_start, overwrites=verified_ch or public_entry)
        ch_wait = await ensure_text("waiting-room", cat_start, overwrites=waiting)
        ch_verify = await ensure_text("verify-setup", cat_start, overwrites=verified_ch or public_entry)
        ch_track = await ensure_text("choose-your-track", cat_start, overwrites=white_only or verified_ch)

        ask_ow = dict(white_only or verified_ch)
        await ensure_text(
            "ask-track-1",
            cat_help,
            topic="@DojoBot for Socratic help — Track 1",
            overwrites=ask_ow,
        )
        await ensure_text(
            "ask-track-2",
            cat_help,
            topic="@DojoBot for Socratic help — Track 2",
            overwrites=ask_ow,
        )
        ch_debug = await ensure_text("dojo-debugging", cat_help, overwrites=ask_ow)
        ch_admin = await ensure_text("admin-logs", cat_admin, overwrites=staff_only)

        # Milestone lesson channels (text)
        for lesson_id in _lesson_role_names():
            slug = lesson_id.lower().replace("-", "")
            if lesson_id.startswith("C1"):
                cat = cat_c1
            else:
                cat = cat_c2
            ch_ow: dict[discord.abc.Snowflake, discord.PermissionOverwrite] = {
                everyone: ow(view_channel=False),
            }
            if verified:
                ch_ow[verified] = ow(view_channel=True, send_messages=True)
            if lesson_id in role_map:
                ch_ow[role_map[lesson_id]] = ow(view_channel=True, send_messages=True)
            await ensure_text(slug, cat, topic=f"Post PASS-* for {lesson_id}", overwrites=ch_ow)

        await post_pin(ch_rules, _load_pin("rules.md"))
        await post_pin(ch_welcome, _load_pin("welcome.md"))
        await post_pin(ch_setup, _load_pin("setup-guide.md"))
        await post_pin(ch_wait, _load_pin("waiting-room.md"))
        await post_pin(ch_track, _load_pin("choose-your-track.md"))

        _save_channel_ids(ch_wait.id, ch_verify.id)

        print("\n✅ Provision complete.")
        print("Next:")
        print("  1. Confirm @DojoBot is above milestone roles in Server Settings → Roles")
        print("  2. Channel IDs saved to dojo/.env automatically")
        print("  3. ./dojo/scripts/launch_check.sh && dojo-gatekeeper")
        await client.close()

    await client.start(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision Discord server for Agentic Dojo")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without API calls")
    args = parser.parse_args()
    asyncio.run(provision(args.dry_run))


if __name__ == "__main__":
    main()
