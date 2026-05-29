# Season 1 — Opening the Dojo

Checklist before inviting White Belts.

## 1. Merge and tag curriculum

```bash
git checkout main   # after merging PR #15 (Agentic Dojo)
git pull
git tag -a course-2026.1 -m "Season 1 immutable curriculum (Agentic Dojo)"
git push origin course-2026.1
```

Students **only** install `@course-2026.1`, never moving branches.

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
# Terminal A: OpenClaw gateway (your usual start command)
# Terminal B:
set -a && source dojo/.env && set +a
dojo-gatekeeper
```

## 5. Smoke test (you)

1. `!start_assessment` → pass DM quiz → White Belt
2. `dojo-profile` → post HW lines in `#verify-setup`
3. `!track c2`
4. `dojo-grade --lesson C2-L01` → post `PASS-*`
5. `@DojoBot why would MLA shrink KV cache?` in `#ask-track-2` → Socratic reply (no code dump)

## 6. Student onboarding message (paste in #welcome)

```
Welcome to Season 1.
1. Fork the student template (link in pins).
2. pip install @course-2026.1
3. #waiting-room → !start_assessment
4. dojo-profile → #verify-setup
5. Stuck at 2am? @DojoBot in #ask-track-* — hints only, never full solutions.
```

## Deferred (post–Season 1)

- Full 1–18 lesson gates (add weekly ahead of cohort pace)
- GitHub Actions on student forks
- Certificate PDF automation
