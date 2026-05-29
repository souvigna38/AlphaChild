# Agentic Dojo — Student Workspace

This is your **private workspace**. The curriculum lives in pip packages — you never edit it directly.

## Setup

1. Fork this template on GitHub (blank repo).
2. Clone **your** fork:

```bash
git clone https://github.com/YOUR_USER/dojo-workspace.git
cd dojo-workspace
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

3. Copy `.env.example` → `.env` and set:
   - `DOJO_PROOF_SECRET` — same value the dojo instructor published (NOT the bot token).
   - `DISCORD_USER_ID` — your numeric Discord user id.

4. Install curriculum (pin updates via tag):

```bash
pip install "alphachild-dojo @ git+https://github.com/souvigna38/AlphaChild.git@cursor/v4-sliding-attn-64d3"
```

Or split installs:

```bash
pip install "git+https://github.com/souvigna38/AlphaChild.git@cursor/v4-sliding-attn-64d3#egg=alphachild-dojo"
```

5. Gate assessment on Discord: `!start_assessment` in #waiting-room.

6. Hardware:

```bash
dojo-profile --discord-user-id YOUR_ID
```

Post both lines in #verify-setup.

7. Per lesson:

```bash
dojo-grade --discord-user-id YOUR_ID --lesson C2-L12
```

Paste the `PASS-*` line in the lesson channel — the bot assigns your role automatically.

## Your files

- `notebooks/` — your experiments (import `llmc`, `alphazero` from installed packages).
- `workspace/` — scratch code, assignments.

## Upgrading curriculum

```bash
pip install --upgrade "alphachild-dojo @ git+https://github.com/souvigna38/AlphaChild.git@course-2026.1"
```

No merge conflicts — your notebooks stay untouched.
