# Discord setup for beginners

**The cloud agent cannot log into Discord.** Run this on your Mac — one wizard does almost everything.

## Fastest path (recommended)

```bash
git clone https://github.com/souvigna38/AlphaChild.git
cd AlphaChild
git checkout cursor/agentic-dojo-64d3

pip install -e ".[dojo]"
dojo-setup
```

Or:

```bash
bash dojo/scripts/setup.sh
```

The wizard will:

1. Explain Discord in plain English (server, channels, roles, bot)
2. Open the Developer Portal in your browser
3. Ask you to paste **Bot Token**, **Application ID**, and **Server ID**
4. Open the bot invite link
5. Write `dojo/.env` (including auto-generated `DOJO_PROOF_SECRET`)
6. Create all roles, channels, and pinned messages
7. Save channel IDs into `dojo/.env` automatically

**Time:** ~15 minutes if you are new to Discord.

---

## What you do manually (cannot be automated)

| Step | Why |
|------|-----|
| Create a Discord account | Needs your identity |
| Create a server (click +) | Your classroom |
| Copy/paste 3 numbers from Discord | Token, App ID, Server ID |
| Drag bot role above student roles | Discord UI only |
| Assign yourself `@Staff` | So you see `#admin-logs` |
| Run `dojo-gatekeeper` on your Mac | Bot must stay online on your machine |

**Never paste your Bot Token in Cursor chat or any public channel.**

---

## After the wizard

```bash
cd AlphaChild
set -a && source dojo/.env && set +a
dojo-gatekeeper
```

In Discord test:

- `#waiting-room` → `!start_assessment`
- `#ask-track-2` → `@YourBotName explain MLA in one sentence`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Missing Access" | Re-invite bot with Administrator (wizard gives URL) |
| Bot offline | Run `dojo-gatekeeper` in a terminal |
| "Cannot message you" | User Settings → allow DMs from server members |
| Provision failed | Bot must be in the server before running provision |

---

## What gets created

**Roles:** Staff, Verified, White Belt, Track-*, Tier-*, C1-L01, C2-L12, …

**Channels:** #rules, #welcome, #setup-guide, #waiting-room, #verify-setup, #ask-track-*, lesson channels

**Pins:** Text from `dojo/discord-pins/`

---

## Next: open Season 1

[`SEASON1_COHORT_COMMANDER.md`](SEASON1_COHORT_COMMANDER.md)
