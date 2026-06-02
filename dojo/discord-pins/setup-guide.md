# ⚙️ Setup Guide — Cross the Threshold

## 0. Prerequisites

- Python **3.10+**
- `git`, `pip`, a terminal (macOS / Linux / WSL)
- Discord **Developer Mode** on → right-click your name → **Copy User ID**

## 1. Create your workspace (fork — do not edit curriculum in-place)

1. On GitHub, open **`souvigna38/AlphaChild`** → folder **`dojo/student-template`** → **Use this template** → create **your** repo.
2. Clone **your** fork:

```bash
git clone https://github.com/YOUR_USER/dojo-workspace.git
cd dojo-workspace
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,curriculum]"
```

The `curriculum` extra installs the locked engine:

```text
alphazero-from-scratch[dojo] @ git+https://github.com/souvigna38/AlphaChild.git@course-2026.1
```

## 2. Configure secrets (instructor provides `DOJO_PROOF_SECRET` on Zoom)

```bash
cp .env.example .env
# Edit .env: DISCORD_USER_ID and DOJO_PROOF_SECRET
set -a && source .env && set +a
```

## 3. Hardware profiler

```bash
dojo-profile --discord-user-id YOUR_DISCORD_ID
```

Post **both** output lines in `#verify-setup`:

```text
HW-standard-xxxxxxxxxxxx
profile_blob:16.00|-1.00|0
```

## 4. White Belt assessment

Go to `#waiting-room` and type:

```text
!start_assessment
```

You have **15 minutes** in DMs. Need **80%** to pass → `@Verified` + `@White Belt`.

## 5. Pick a track

```text
!track c1    # AlphaZero → MuZero
!track c2    # GPT → DeepSeek (needs Tier-Standard+)
!track c3    # V4 C systems lab
```

## 6. Complete a lesson

```bash
dojo-grade --discord-user-id YOUR_DISCORD_ID --lesson C2-L01
```

Paste only the `PASS-C2-L01-xxxxxxxxxxxx` line in the lesson channel. The bot assigns your role.

---

*Help: `!dojo_help` · Stuck? `@DojoBot` in `#ask-track-*` · Season 1 tag: `course-2026.1`*
