"""
Interactive Discord setup wizard for beginners.

Run on YOUR Mac (not the cloud agent):

    pip install -e ".[dojo]"
    dojo-setup

This walks you through Discord Developer Portal steps, writes dojo/.env,
creates roles/channels, and saves channel IDs automatically.
"""

from __future__ import annotations

import asyncio
import getpass
import secrets
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOJO = Path(__file__).resolve().parent
SCRIPTS = DOJO / "scripts"


def _banner() -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║        Agentic Dojo — Discord Setup Wizard                   ║
║        (built for Discord beginners)                         ║
╚══════════════════════════════════════════════════════════════╝

Discord quick map:
  • SERVER  = your classroom (one per course)
  • CHANNEL = a room (#rules, #waiting-room)
  • ROLE    = a badge (@White Belt) the bot assigns automatically
  • BOT     = a robot account (DojoBot) — NOT your personal login

The cloud Cursor agent cannot log into Discord for you.
This wizard runs on your machine and does the mechanical setup.

"""
    )


def _pause(msg: str = "Press Enter when done…") -> None:
    try:
        input(f"\n→ {msg} ")
    except EOFError:
        pass


def _ask(label: str, *, secret: bool = False, default: str = "") -> str:
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    if secret:
        val = getpass.getpass(prompt)
    else:
        val = input(prompt).strip()
    return val or default


def _open_url(url: str, label: str) -> None:
    print(f"\n  Opening: {label}")
    print(f"  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        print("  (Could not open browser — copy the URL above manually)")


def _step_create_server() -> None:
    print("\n━━━ STEP 1/6 — Create a Discord server ━━━")
    print(
        """
1. Install Discord: https://discord.com/download
2. Sign up / log in
3. Click the + button on the left → "Create My Own"
4. Choose "For me and my friends" (or Community)
5. Name it e.g. "Agentic Dojo Season 1"
"""
    )
    _pause("Created your server?")


def _step_developer_mode() -> None:
    print("\n━━━ STEP 2/6 — Turn on Developer Mode ━━━")
    print(
        """
Developer Mode lets you copy IDs (numbers Discord uses internally).

Desktop app:
  User Settings (gear) → Advanced → Developer Mode → ON

You will need IDs later for your server and channels.
"""
    )
    _pause("Developer Mode is ON?")


def _step_create_bot() -> tuple[str, str]:
    print("\n━━━ STEP 3/6 — Create the DojoBot ━━━")
    _open_url("https://discord.com/developers/applications", "Discord Developer Portal")

    print(
        """
In the Developer Portal:
  1. Click "New Application" → name it "Agentic Dojo Bot"
  2. Open the "Bot" tab on the left
  3. Click "Reset Token" → COPY the token (you only see it once!)
  4. Under "Privileged Gateway Intents" turn ON:
       ☑ Message Content Intent
  5. Open the "General Information" tab → copy "Application ID"
"""
    )
    app_id = _ask("Paste Application ID (long number)")
    while not app_id.isdigit():
        app_id = _ask("Application ID must be numeric — try again")
    token = _ask("Paste Bot Token", secret=True)
    while len(token) < 20:
        token = _ask("Token looks too short — paste the full Bot token", secret=True)
    return app_id, token


def _step_invite_bot(app_id: str) -> None:
    print("\n━━━ STEP 4/6 — Invite the bot to YOUR server ━━━")
    invite = (
        f"https://discord.com/api/oauth2/authorize"
        f"?client_id={app_id}&permissions=8&scope=bot"
    )
    _open_url(invite, "Bot invite link (Administrator — needed once for setup)")

    print(
        """
1. Select YOUR server (Agentic Dojo) from the dropdown
2. Click Authorize
3. Complete the captcha if asked

You should see "Agentic Dojo Bot" appear in the member list (offline until we start it).
"""
    )
    _pause("Bot is in your server?")


def _step_guild_id() -> str:
    print("\n━━━ STEP 5/6 — Copy your Server ID ━━━")
    print(
        """
In the Discord app:
  Right-click your SERVER NAME (top-left icon) → "Copy Server ID"

(Paste it below — it is a long number, not the server name)
"""
    )
    guild_id = _ask("Paste Server ID")
    while not guild_id.isdigit():
        guild_id = _ask("Server ID must be numeric")
    return guild_id


def _write_env(app_id: str, token: str, guild_id: str, proof_secret: str) -> Path:
    from dojo.env_file import merge_env

    path = merge_env(
        {
            "DISCORD_BOT_TOKEN": token,
            "DISCORD_APPLICATION_ID": app_id,
            "DOJO_GUILD_ID": guild_id,
            "DOJO_PROOF_SECRET": proof_secret,
        }
    )
    return path


def _run_provision() -> int:
    print("\n━━━ STEP 6/6 — Auto-build roles & channels ━━━")
    print("Running discord_provision.py (may take 30–60 seconds)…\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "discord_provision.py")],
        cwd=str(ROOT),
    )
    return proc.returncode


def _run_launch_check() -> None:
    script = SCRIPTS / "launch_check.sh"
    if script.is_file():
        subprocess.run(["bash", str(script)], cwd=str(ROOT))


def _print_finish(env_path: Path, proof_secret: str) -> None:
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ Discord structure created!                               ║
╚══════════════════════════════════════════════════════════════╝

Saved config: {env_path}

ONE manual check in Discord:
  Server Settings → Roles → drag "Agentic Dojo Bot" ABOVE
  Verified, White Belt, and lesson roles.

Assign yourself the "Staff" role (right-click your name → Roles).

Start the bot (keep this terminal open):

  cd {ROOT}
  set -a && source dojo/.env && set +a
  dojo-gatekeeper

Share with students (on Zoom, NOT public Discord):
  DOJO_PROOF_SECRET={proof_secret[:8]}…  (full value is in dojo/.env)

Test in Discord:
  #waiting-room → type:  !start_assessment
  #ask-track-2  → @YourBotName why do tensor shapes matter?

Docs: docs/DISCORD_SETUP.md · docs/SEASON1_COHORT_COMMANDER.md
"""
    )


def main() -> int:
    _banner()

    if not (DOJO / ".env.example").is_file():
        print("Error: run this from the AlphaChild repo root.", file=sys.stderr)
        return 2

    try:
        import discord  # noqa: F401
    except ImportError:
        print("Installing dojo dependencies…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dojo]"], cwd=str(ROOT))

    _step_create_server()
    _step_developer_mode()
    app_id, token = _step_create_bot()
    _step_invite_bot(app_id)
    guild_id = _step_guild_id()

    proof_secret = secrets.token_urlsafe(32)
    print(f"\nGenerated DOJO_PROOF_SECRET (saved to dojo/.env)")

    env_path = _write_env(app_id, token, guild_id, proof_secret)

    if _ask("Run auto-provision now? (y/n)", default="y").lower().startswith("y"):
        # Export for subprocess
        import os

        os.environ["DISCORD_BOT_TOKEN"] = token
        os.environ["DISCORD_APPLICATION_ID"] = app_id
        os.environ["DOJO_GUILD_ID"] = guild_id
        os.environ["DOJO_PROOF_SECRET"] = proof_secret
        rc = _run_provision()
        if rc != 0:
            print("\nProvision failed. Check bot is in server and has Administrator.")
            print("Retry: set -a && source dojo/.env && set +a && python dojo/scripts/discord_provision.py")
            return rc

    _run_launch_check()
    _print_finish(env_path, proof_secret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
