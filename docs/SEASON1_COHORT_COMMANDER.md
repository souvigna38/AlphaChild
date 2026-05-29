# Season 1 — Cohort Commander Playbook

Operational sequence: staging Discord → live Dojo. The codebase is locked at **`course-2026.1`**. Your job is server topology, pinned messages, and the Zoom kickoff.

---

## Discord automation (you run locally)

The cloud agent **cannot** log into Discord. Run once:

```bash
python dojo/scripts/discord_provision.py
```

See [`DISCORD_SETUP.md`](DISCORD_SETUP.md).

---

## Pre-flight (you, 30 minutes before doors open)

```bash
# On the bot host — all three must print non-empty values
echo "${DISCORD_BOT_TOKEN:+BOT_TOKEN=set}"
echo "${DOJO_PROOF_SECRET:+PROOF_SECRET=set}"
echo "${OPENCLAW_API_URL:-MISSING}"

pip install -e ".[dojo]"
set -a && source dojo/.env && set +a
./dojo/scripts/launch_check.sh   # exits 1 if misconfigured

# Terminal A: OpenClaw gateway
# Terminal B:
dojo-gatekeeper
```

Merge [PR #15](https://github.com/souvigna38/AlphaChild/pull/15) if not on `main` yet. Students use tag **`course-2026.1`** only.

---

## Phase 1: Role & permission matrix

**Discord rule:** drag `@DojoBot` **above** every role it must assign. Bot needs **Manage Roles**, **Manage Nicknames**, **View Channels**, **Send Messages**, **Read Message History**, **Add Reactions**.

| Role | Type | Purpose |
|------|------|---------|
| `@Staff` | Human | Administrator. `#admin-logs`, break-glass, ban hammer. |
| `@DojoBot` | Bot | Above all student roles. Grants automated roles. |
| `@Tier-Gpu` | Automated | `dojo-profile` — Atari / heavy MuZero |
| `@Tier-Standard` | Automated | `dojo-profile` — LLM / DeepSeek tiny (CPU) |
| `@Tier-Lightweight` | Automated | `dojo-profile` — board games only |
| `@C2-L12`, `@C1-L01`, … | Automated | `dojo-grade` milestones (`dojo/lessons.yaml`) |
| `@Track-Systems` | Automated | `!track c3` |
| `@Track-LLM` | Automated | `!track c2` |
| `@Track-Alpha` | Automated | `!track c1` |
| `@White Belt` | Automated | Pass `!start_assessment` |
| `@Verified` | Automated | Pass quiz (paired with White Belt today) |
| `@everyone` | Default | **Deny** View Channel everywhere except `#rules`, `#welcome`, `#waiting-room` |

### Channel permission sketch

| Channel | @everyone | @Verified | @White Belt | Notes |
|---------|-----------|-----------|-------------|-------|
| `#rules`, `#welcome` | View | View | View | Read-only for students |
| `#waiting-room` | View + Send | View + Send | View + Send | `!start_assessment` only |
| `#setup-guide` | — | View | View | Pin curriculum commands |
| `#verify-setup` | — | View + Send | View + Send | `HW-*` + `profile_blob:` |
| `#choose-your-track` | — | — | View + Send | `!track c1\|c2\|c3` |
| `#ask-track-1`, `#ask-track-2` | — | — | Track role + Send | `@DojoBot` TA |
| `#dojo-debugging` | — | — | White Belt + Send | Escalation |
| `#admin-logs` | — | — | — | Staff only |
| Lesson forums | — | — | Prior lesson role | One post per gate |

Copy-paste channel bodies: [`dojo/discord-pins/`](../dojo/discord-pins/).

---

## Phase 2: Onboarding message stack

Pin these **before** inviting students. Source files in `dojo/discord-pins/`:

1. **`rules.md`** → `#rules`
2. **`setup-guide.md`** → `#setup-guide` (uses **student template fork** + `@course-2026.1` — not a dev branch)
3. **`welcome.md`** → `#welcome`
4. **`waiting-room.md`** → `#waiting-room`

**Do not** publish `DOJO_PROOF_SECRET` in a public channel. Hand it on Zoom or a 1:1 enrollment DM after White Belt assessment (see Phase 3).

---

## Phase 3: 60-minute Zoom orientation agenda

| Time | Block | Script |
|------|-------|--------|
| **00:00–00:15** | Vibe check & blueprint | LLMs are **engines of state**, not search bars. The Dojo mimics production: wrong commands → compiler/bot rejects you. Integrity is cryptographic, not honor-system. |
| **00:15–00:35** | Local environment | Live-share: fork template → `pip install` PEP 508 URL → `dojo-profile`. Everyone posts HW lines in `#verify-setup` before quiz. |
| **00:35–00:50** | Sparring Partner demo | `@DojoBot` a shape-mismatch question. Show **Socratic pushback**, not code dumps. "At 2am this replaces pinging the instructor." |
| **00:50–01:00** | Unlock the gates | Paste `DOJO_PROOF_SECRET` in **Zoom chat** (not Discord). "Go to `#waiting-room`, type `!start_assessment`. Fifteen minutes. Survivors get White Belt." |

### Instructor lines (optional verbatim)

- *"If you cannot follow instructions exactly, the toolchain will reject you. This Dojo is intentional."*
- *"Never paste another student's `PASS-*` line — the HMAC is bound to your Discord ID."*
- *"The bot will not write your homework. Ask for debugging vectors."*

---

## Phase 4: Launch day timeline

| T−24h | Bot online 24h soak test; run `launch_check.sh`; smoke-test all gates |
| T−1h | Paste Discord pins; verify role hierarchy |
| T−0 | Zoom opens; orientation |
| T+0 | `DOJO_PROOF_SECRET` in Zoom chat; `!start_assessment` live |
| T+1d | First lesson forum unlocked (`C1-L01` or `C2-L01` per track) |
| T+7d | Add next gates in `dojo/lessons.yaml` ahead of cohort pace |

---

## Student curriculum install (canonical)

Students **do not** clone `AlphaChild` directly for daily work. They fork the template and pip-install the engine:

```bash
# Fork on GitHub: AlphaChild → dojo/student-template → "Use this template"
git clone https://github.com/YOUR_USER/dojo-workspace.git
cd dojo-workspace
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,curriculum]"
cp .env.example .env   # DISCORD_USER_ID + DOJO_PROOF_SECRET (after Zoom)
export $(grep -v '^#' .env | xargs)
dojo-profile --discord-user-id YOUR_DISCORD_ID
```

Curriculum line (immutable):

```bash
pip install "alphazero-from-scratch[dojo] @ git+https://github.com/souvigna38/AlphaChild.git@course-2026.1"
```

---

## Enforcement

| Violation | Action |
|-----------|--------|
| Copied `PASS-*` hash | Bot rejects; Staff review → ban |
| TA abuse (solution fishing) | Post-filter + warn; repeat → mute |
| Wrong tier for LLM lessons | Bot blocks gate; upgrade hardware or stay Track 1 |

---

## Quick links

- Architecture: [`AGENTIC_DOJO.md`](AGENTIC_DOJO.md)
- Technical launch: [`SEASON1_LAUNCH.md`](SEASON1_LAUNCH.md)
- Discord pins: [`dojo/discord-pins/`](../dojo/discord-pins/)
- Lessons / roles: [`dojo/lessons.yaml`](../dojo/lessons.yaml)

**The codebase is locked. The architecture is autonomous. Open the doors.**
