"""
Agentic Dojo gatekeeper — single custom bot (no Carl-bot / MEE6 / Zira).
"""

from __future__ import annotations

import logging
import re

import discord
from discord.ext import commands

from dojo.assessment import format_question, start_session
from dojo.config import DojoConfig
from dojo.hardware import tier_allows
from dojo.lessons import get_lesson, prerequisite_met
from dojo.proof import parse_hardware_line, parse_proof_line, verify_hardware_proof, verify_proof
from dojo.sparring_partner import (
    check_rate_limit,
    chunk_for_discord,
    extract_question,
    load_ta_config,
    record_rate_limit,
    socratic_reply,
)
from dojo.store import (
    completed_lessons,
    get_hardware_tier,
    mark_lesson_complete,
    set_hardware_tier,
)

log = logging.getLogger("dojo.gatekeeper")

TRACK_ROLES = {
    "c1": "role_track_c1",
    "c2": "role_track_c2",
    "c3": "role_track_c3",
    "1": "role_track_c1",
    "2": "role_track_c2",
    "3": "role_track_c3",
}

PROFILE_BLOB_RE = re.compile(r"profile_blob:([0-9.|+-]+)", re.I)


class GatekeeperBot(commands.Bot):
    def __init__(self, cfg: DojoConfig):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.cfg = cfg
        self.assessments: dict[int, object] = {}
        self.ta_cfg = load_ta_config()

    async def _find_role(self, guild: discord.Guild, name: str) -> discord.Role | None:
        for role in guild.roles:
            if role.name.lower() == name.lower():
                return role
        return None

    async def _grant(self, member: discord.Member, role_name: str) -> bool:
        role = await self._find_role(member.guild, role_name)
        if role is None:
            log.warning("Role missing on server: %s", role_name)
            return False
        if role not in member.roles:
            await member.add_roles(role, reason="Agentic Dojo gatekeeper")
        return True

    def _in_guild(self, message: discord.Message) -> bool:
        return message.guild is not None and message.guild.id == self.cfg.guild_id

    async def _handle_assessment_dm(self, message: discord.Message) -> bool:
        session = self.assessments.get(message.author.id)
        if session is None or session.finished:
            return False

        if session.expired():
            del self.assessments[message.author.id]
            await message.channel.send("⏱ Time expired. Run `!start_assessment` again in #waiting-room.")
            return True

        m = re.match(r"^[1-4]$", (message.content or "").strip())
        if not m:
            await message.channel.send("Reply with `1`, `2`, `3`, or `4`.")
            return True

        choice = int(m.group()) - 1
        session.answers[session.index] = choice
        session.index += 1

        if session.finished:
            del self.assessments[message.author.id]
            pct = session.score_fraction() * 100
            if session.passed():
                guild = self.get_guild(self.cfg.guild_id)
                if guild:
                    member = guild.get_member(message.author.id)
                    if member:
                        await self._grant(member, self.cfg.role_verified)
                        await self._grant(member, self.cfg.role_white_belt)
                await message.channel.send(
                    f"✅ Passed ({pct:.0f}%). You have @Verified + @White Belt. "
                    "Pick `!track c1|c2|c3`, run `dojo-profile`, then start lessons."
                )
            else:
                await message.channel.send(
                    f"❌ Failed ({pct:.0f}%). Study docs/AGENTIC_DOJO.md and retry `!start_assessment`."
                )
            return True

        await message.channel.send(format_question(session))
        return True

    async def _handle_hardware_post(self, message: discord.Message) -> bool:
        hw = parse_hardware_line(message.content or "")
        if hw is None:
            return False
        tier, token = hw
        blob_m = PROFILE_BLOB_RE.search(message.content or "")
        if blob_m is None:
            await message.reply("Include the `profile_blob:...` line from `dojo-profile` output.")
            return True
        blob = blob_m.group(1)
        if not verify_hardware_proof(
            message.author.id,
            tier,
            token,
            self.cfg.proof_secret,
            profile_blob=blob,
        ):
            await message.reply("Invalid HW proof.")
            return True
        set_hardware_tier(message.author.id, tier, blob)
        await self._grant(message.author, f"Tier-{tier.capitalize()}")
        await message.add_reaction("✅")
        await message.reply(f"Hardware tier **{tier}** recorded.")
        return True

    async def _handle_lesson_proof(self, message: discord.Message) -> bool:
        proof = parse_proof_line(message.content or "")
        if proof is None:
            return False

        lesson = get_lesson(proof.lesson_id)
        if lesson is None:
            await message.reply(f"Unknown lesson id `{proof.lesson_id}`.")
            return True

        if not verify_proof(
            message.author.id,
            lesson.lesson_id,
            proof.token,
            self.cfg.proof_secret,
            pytest_target=lesson.pytest_target,
        ):
            await message.reply("Invalid or forged proof (wrong user, lesson, or curriculum gate).")
            return True

        hw = get_hardware_tier(message.author.id)
        if hw is None:
            await message.reply("Complete hardware profiling (#verify-setup) before lesson proofs.")
            return True
        student_tier, _ = hw
        if not tier_allows(lesson.min_tier, student_tier):
            await message.reply(
                f"Hardware tier `{student_tier}` cannot unlock `{lesson.lesson_id}` "
                f"(requires `{lesson.min_tier}`).",
            )
            return True

        done = completed_lessons(message.author.id)
        if not prerequisite_met(done, lesson.lesson_id):
            await message.reply("Complete the previous lesson in this track first.")
            return True

        await self._grant(message.author, lesson.role)
        mark_lesson_complete(message.author.id, lesson.lesson_id)
        await message.add_reaction("✅")
        await message.reply(f"**{lesson.title}** cleared. Role `@{lesson.role}` granted.")
        return True

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        if message.guild is None:
            if await self._handle_assessment_dm(message):
                return
            return

        await self.process_commands(message)

        if not self._in_guild(message) or not isinstance(message.author, discord.Member):
            return

        if (
            self.cfg.channel_verify_setup_id
            and message.channel.id == self.cfg.channel_verify_setup_id
            and await self._handle_hardware_post(message)
        ):
            return

        if await self._handle_sparring_partner(message):
            return

        await self._handle_lesson_proof(message)

    async def _member_has_white_belt(self, member: discord.Member) -> bool:
        role = await self._find_role(member.guild, self.cfg.role_white_belt)
        return role is not None and role in member.roles

    async def _handle_sparring_partner(self, message: discord.Message) -> bool:
        """Socratic TA when @mentioned in help channels (Season 1 — Sparring Partner)."""
        if not self.ta_cfg.enabled or self.user is None:
            return False
        if not isinstance(message.author, discord.Member):
            return False
        if not self.cfg.ta_allowed_in_channel(message.channel.id, message.channel.name or ""):
            return False

        question = extract_question(message.content or "", self.user.id)
        if question is None:
            return False

        if not await self._member_has_white_belt(message.author):
            await message.reply("Pass `!start_assessment` first to unlock the Sparring Partner.")
            return True

        if not check_rate_limit(message.author.id, self.ta_cfg.max_requests_per_hour):
            await message.reply(
                f"Rate limit: {self.ta_cfg.max_requests_per_hour} questions/hour. "
                "Try again later or narrow your question."
            )
            return True

        async with message.channel.typing():
            try:
                record_rate_limit(message.author.id)
                reply = await socratic_reply(
                    self.ta_cfg,
                    question,
                    channel_name=message.channel.name or "",
                )
            except Exception:
                log.exception("Sparring Partner API failure")
                await message.reply(
                    "Sparring Partner is offline. Check `OPENCLAW_API_URL` on the bot host, "
                    "or post in #dojo-debugging for a human."
                )
                return True

        for chunk in chunk_for_discord(reply):
            await message.reply(chunk, mention_author=False)
        return True

    @commands.command(name="start_assessment")
    async def start_assessment(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("Run this in the server.")
            return
        if (
            self.cfg.channel_waiting_room_id
            and ctx.channel.id != self.cfg.channel_waiting_room_id
        ):
            await ctx.reply("Use this command in #waiting-room.")
            return
        member = ctx.author
        if not isinstance(member, discord.Member):
            return
        session = start_session(member.id)
        self.assessments[member.id] = session
        try:
            dm = await member.create_dm()
            await dm.send(
                "🥋 **Agentic Dojo — Gate Assessment**\n"
                "15 minutes, 80% to pass. Reply `1`–`4` per question.\n\n"
                + format_question(session)
            )
            await ctx.reply("Check your DMs — assessment started.")
        except discord.Forbidden:
            await ctx.reply("Enable DMs from server members.")

    @commands.command(name="track")
    async def choose_track(self, ctx: commands.Context, track: str = "") -> None:
        member = ctx.author
        if not isinstance(member, discord.Member):
            return
        attr = TRACK_ROLES.get(track.strip().lower())
        if attr is None:
            await ctx.reply("Usage: `!track c1` | `!track c2` | `!track c3`")
            return
        white = await self._find_role(member.guild, self.cfg.role_white_belt)
        if white and white not in member.roles:
            await ctx.reply("Pass the gate assessment first.")
            return
        await self._grant(member, getattr(self.cfg, attr))
        await ctx.reply(f"Track granted. Run `dojo-grade` locally and post PASS-* proofs.")

    @commands.command(name="dojo_help")
    async def dojo_help(self, ctx: commands.Context) -> None:
        await ctx.reply(
            "`!start_assessment` · `!track c1|c2|c3` · "
            "`dojo-profile` → #verify-setup · `dojo-grade` → PASS-* in lesson channels · "
            "@mention me in #ask-* for Socratic help (no full solutions)"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = DojoConfig.from_env()
    bot = GatekeeperBot(cfg)

    @bot.event
    async def on_ready() -> None:
        log.info("Gatekeeper online as %s", bot.user)

    bot.run(cfg.token)


if __name__ == "__main__":
    main()
