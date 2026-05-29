# Agentic Dojo — Automated Discord + Cryptographic Grading

This is **not** a manual “react ✅ for roles” community. One custom bot (`gatekeeper.py`) runs onboarding, hardware gating, and lesson progression.

## Architecture

| Layer | Tool | Human work |
|-------|------|------------|
| Onboarding | `!start_assessment` → 15-min DM quiz | Zero |
| Hardware | `dojo-profile` → `HW-*` + `profile_blob:` | Zero |
| Lessons | `dojo-grade` → `PASS-*` in channel | Zero |
| Curriculum | pip-installed AlphaChild | You ship tags |
| Student code | Forked `student-template` | Student owns repo |

**Do not install** Carl-bot, MEE6, or Zira for roles.

## Bot setup (instructor)

1. Create Discord application → Bot → enable **Message Content Intent**.
2. Invite bot with `Manage Roles`, `Send Messages`, `Read Message History`.
3. Create roles (names must match): `Verified`, `White Belt`, `Track-Alpha`, `Track-LLM`, `Track-Systems`, `Tier-Lightweight`, `Tier-Standard`, `Tier-Gpu`, plus lesson roles from `dojo/lessons.yaml` (`C1-L01`, `C2-L12`, …).
4. Channel permissions:
   - `#waiting-room` — everyone; only bot + `!start_assessment`
   - `#choose-your-track` — `@White Belt` only
   - `#verify-setup` — `@Verified`
   - Lesson forums — `@Track-*` + prior lesson role
5. Copy `dojo/.env.example` → `dojo/.env` (never commit).
6. Generate shared secret for students:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Publish `DOJO_PROOF_SECRET` to enrolled students (separate from bot token).

7. Run bot:

```bash
cd /path/to/AlphaChild
pip install -e ".[dojo]"
set -a && source dojo/.env && set +a
dojo-gatekeeper
```

## Student flow

1. Fork `dojo/student-template` → clone locally → `pip install -e ".[dev,curriculum]"`
2. Discord: join server → `#waiting-room` → `!start_assessment`
3. Pass quiz → `@Verified` + `@White Belt`
4. `!track c2` (example)
5. `dojo-profile --discord-user-id YOUR_ID` → post output in `#verify-setup`
6. Open curriculum notebooks from installed packages / upstream repo links
7. `dojo-grade --discord-user-id YOUR_ID --lesson C2-L12` → paste `PASS-C2-L12-xxxxxxxxxxxx` in lesson thread

## Cryptographic proofs

```
HMAC-SHA256(DOJO_PROOF_SECRET, "{user_id}|{lesson_id}|{gate_manifest}")[:12]
→ PASS-C2-L12-8f92a4b1c3d0
```

- Binds to **Discord user id** — copying another student’s line fails.
- Binds to **gate file** — curriculum updates change manifest → new proofs required.
- `dojo-grade` only prints proof after **pytest passes** locally.

## Hardware tiers

| Tier | Typical hardware | Unlocks |
|------|------------------|---------|
| `lightweight` | 4GB+ RAM, no GPU | Track 1 board games |
| `standard` | 8GB+ RAM | Track 2 LLM / DeepSeek tiny |
| `gpu` | 16GB+ RAM, 4GB+ VRAM | Atari / heavy MuZero |

## Lesson registry

Edit `dojo/lessons.yaml` and add gate tests under `dojo/gates/`. Each gate is a small pytest file shipped with the package (always available after pip install).

## Discord channel map (recommended)

```
📌 START HERE
  #welcome
  #rules
  #waiting-room          ← !start_assessment
  #verify-setup          ← HW-* proofs
  #choose-your-track     ← !track c1|c2|c3

🎮 TRACK 1 — forum: c1-lessons
🧠 TRACK 2 — forum: c2-lessons
⚙️ TRACK 3 — forum: c3-labs

🆘 #dojo-debugging
```

## Instructor upgrades

Tag releases: `git tag course-2026.1 && git push origin course-2026.1`

Students: `pip install --upgrade "alphazero-from-scratch[dojo] @ git+...@course-2026.1"`

No merge conflicts in student forks.
