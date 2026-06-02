# Season 1 — Opening the Dojo

**Cohort commander playbook (roles, Zoom, Discord pins):** [`SEASON1_COHORT_COMMANDER.md`](SEASON1_COHORT_COMMANDER.md)

Checklist before inviting White Belts.

## 1. Curriculum tag

**`course-2026.1`** is published on GitHub. Students **only** install that tag — never moving branches.

## 2. Discord server

| Item | Action |
|------|--------|
| Roles | `Verified`, `White Belt`, `Track-*`, `Tier-*`, lesson roles from `dojo/lessons.yaml` |
| Channels | `#waiting-room`, `#verify-setup`, `#ask-track-1`, `#ask-track-2`, `#dojo-debugging` |
| Bot intents | Message Content + Server Members |
| Permissions | `#choose-your-track` → White Belt only |

## 3. Secrets

- `DISCORD_BOT_TOKEN` — bot host only
- `DOJO_PROOF_SECRET` — publish to enrolled students (for `dojo-grade`)
- `OPENCLAW_API_KEY` — if your gateway requires it

## 4. Run stack on bot host

```bash
pip install -e ".[dojo]"
set -a && source dojo/.env && set +a
./dojo/scripts/launch_check.sh
# Terminal A: OpenClaw gateway
# Terminal B:
dojo-gatekeeper
```

## 5. Smoke test (you)

1. `!start_assessment` → pass DM quiz → White Belt
2. `dojo-profile` → post HW lines in `#verify-setup`
3. `!track c2`
4. `dojo-grade --lesson C2-L01` → post `PASS-*`
5. `@DojoBot why would MLA shrink KV cache?` in `#ask-track-2` → Socratic reply (no code dump)

## 6. Discord pins

Copy from [`dojo/discord-pins/`](../dojo/discord-pins/) into `#rules`, `#setup-guide`, `#welcome`, `#waiting-room`, `#choose-your-track`.

## Deferred (post–Season 1)

- Full 1–18 lesson gates (add weekly ahead of cohort pace)
- GitHub Actions on student forks
- Certificate PDF automation
