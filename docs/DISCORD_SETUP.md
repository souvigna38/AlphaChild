# Discord setup — what the agent can and cannot do

## Can the cloud agent set up Discord for you?

**No.** This environment does not have:

- Your Discord login
- Your server admin access
- Your bot token or application secrets

Never paste `DISCORD_BOT_TOKEN` into Cursor chat or a public channel.

## What you can run in ~15 minutes

We ship a **provision script** that does the mechanical work once you create the bot.

### Step 1 — Create the bot (5 min, one-time)

1. Open https://discord.com/developers/applications → **New Application** → name it `Agentic Dojo`.
2. **Bot** → **Reset Token** → copy token → `dojo/.env` as `DISCORD_BOT_TOKEN`.
3. Copy **Application ID** → `DISCORD_APPLICATION_ID` in `dojo/.env`.
4. Enable **Message Content Intent** under Bot.
5. **OAuth2 → URL Generator** → scope `bot`, permission **Administrator** (setup only) → open URL → add bot to your server.
6. In Discord: **Server Settings → Widget** or right-click server → **Copy Server ID** → `DOJO_GUILD_ID`.

### Step 2 — Run provisioner (2 min)

```bash
cd AlphaChild
pip install -e ".[dojo]"
cp dojo/.env.example dojo/.env
# fill DISCORD_BOT_TOKEN, DOJO_GUILD_ID, DISCORD_APPLICATION_ID, DOJO_PROOF_SECRET

set -a && source dojo/.env && set +a
python dojo/scripts/discord_provision.py --dry-run
python dojo/scripts/discord_provision.py
```

This creates:

- Roles (`Verified`, `White Belt`, tracks, tiers, lesson milestones)
- Categories and channels (`#rules`, `#waiting-room`, `#ask-track-*`, …)
- Pinned messages from `dojo/discord-pins/`

### Step 3 — Wire channel IDs (2 min)

Developer Mode on → right-click channels → **Copy ID** → add to `dojo/.env`:

```env
DOJO_CHANNEL_WAITING_ROOM=...
DOJO_CHANNEL_VERIFY_SETUP=...
```

### Step 4 — Start bot + OpenClaw (2 min)

```bash
./dojo/scripts/launch_check.sh
dojo-gatekeeper
```

### Step 5 — Manual checks (2 min)

| Check | Action |
|-------|--------|
| Role order | `@DojoBot` **above** all roles it grants |
| Staff | Assign yourself `@Staff` |
| Re-invite bot | Optional: re-invite without Administrator after provision |

## Optional: I can help you live

If you run the provision script and paste **non-secret** output (e.g. error messages, role list screenshot description), we can debug together. Do **not** share tokens.

## Full cohort launch

After Discord exists: [`SEASON1_COHORT_COMMANDER.md`](SEASON1_COHORT_COMMANDER.md)
