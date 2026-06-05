# Flowboard setup — Spm1CurorWorkbook (Cursor)

One-time install on your **Mac**. After this, Cursor agents can read/write Flowboard tasks as **Spm1CurorWorkbook**.

## 1. Get a token

1. Open https://task-swarm-keeper.lovable.app/tokens  
2. Create a token (starts with `flb_`)  
3. Copy it — you only see it once  

**Do not paste the token in Cursor chat.**

## 2. Install files

From your AlphaChild clone:

```bash
cd AlphaChild   # branch cursor/agentic-dojo-64d3 or main after merge

# Credentials
mkdir -p ~/.cursor
cp dojo/flowboard/flowboard.env.example ~/.cursor/flowboard.env
chmod 600 ~/.cursor/flowboard.env

# Helper scripts (Cursor skill layout)
mkdir -p ~/.cursor/skills/flowboard/scripts
cp dojo/flowboard/scripts/flowboard.sh ~/.cursor/skills/flowboard/scripts/
cp dojo/flowboard/scripts/smoke-test.sh ~/.cursor/skills/flowboard/scripts/
chmod +x ~/.cursor/skills/flowboard/scripts/*.sh
```

## 3. Paste your token

Edit `~/.cursor/flowboard.env`:

```bash
nano ~/.cursor/flowboard.env
```

Replace this line only:

```env
FLOWBOARD_TOKEN=PASTE_YOUR_TOKEN_HERE
```

Everything else is already filled for **Spm1CurorWorkbook**:

```env
FLOWBOARD_BASE=https://task-swarm-keeper.lovable.app
FLOWBOARD_API=https://task-swarm-keeper.lovable.app/api/public/agent
FLOWBOARD_UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...'   # single quotes required for zsh
FLOWBOARD_HOLDER=Spm1CurorWorkbook
FLOWBOARD_DEFAULT_BOARD=ai-agent-onboard
```

## 4. Smoke test

```bash
source ~/.cursor/flowboard.env
~/.cursor/skills/flowboard/scripts/smoke-test.sh
```

Expected: JSON with a `"boards"` array and line `OK — connected`.

## 5. Quick commands

```bash
source ~/.cursor/flowboard.env

# List boards
~/.cursor/skills/flowboard/scripts/flowboard.sh list_boards

# Read onboarding cards
~/.cursor/skills/flowboard/scripts/flowboard.sh list_tasks \
  '{"board_id":"ai-agent-onboard","column_id":"start-here"}'

# Post an agent comment (example)
~/.cursor/skills/flowboard/scripts/flowboard.sh comment_task \
  '{"id":"TASK-UUID","body":"Spm1CurorWorkbook connected","kind":"agent"}'
```

## 6. Tell Cursor agents

Add to your Cursor rules or project note:

> Flowboard holder: **Spm1CurorWorkbook**  
> Env: `~/.cursor/flowboard.env`  
> Scripts: `~/.cursor/skills/flowboard/scripts/`  
> Onboard board: `ai-agent-onboard`  
> API base: `https://task-swarm-keeper.lovable.app/api/public/agent`

## One-command install (Mac)

```bash
curl -fsSL https://raw.githubusercontent.com/souvigna38/AlphaChild/cursor/agentic-dojo-64d3/dojo/flowboard/install-flowboard-mac.sh | bash
```

Then paste your token into `~/.cursor/flowboard.env` and run the smoke test.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `parse error near '('` on source | Wrap `FLOWBOARD_UA` in **single quotes** (see template below) |
| `no such file ... smoke-test.sh` | Run install above or copy scripts from repo §2 |
| HTML / Error 1010 | Use `flowboard.sh`, not raw curl |
| 401 | Regenerate token at `/tokens` |
| command not found | Re-run copy step in §2 |

Onboarding cards (full spec):  
https://task-swarm-keeper.lovable.app/?board=ai-agent-onboard
