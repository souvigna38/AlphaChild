# Task for Cursor agent: publish `llm-c-from-scratch` to GitHub

Copy this entire file (or the user’s message pointing here) into another Cursor agent session.

---

## Goal

Publish the project **`llm-c-from-scratch`** as its **own public GitHub repository**:

- **Owner:** `souvigna38`
- **Repo name:** `llm-c-from-scratch`
- **Target URL:** https://github.com/souvigna38/llm-c-from-scratch

Do **not** merge this into `AlphaChild`. It is a separate educational repo (GPT / llm.c notebooks).

---

## What you are publishing

| Item | Details |
|------|---------|
| Purpose | Jupyter tutorials (1–10) teaching GPT / [karpathy/llm.c](https://github.com/karpathy/llm.c) in PyTorch |
| Package | `llmc/` (`data`, `model`, `train`, `sample`) |
| Data | `data/tiny_shakespeare.txt` |
| Tests | `pytest tests/ -v` (16 tests, all should pass) |
| Docs | `README.md`, `AGENTS.md`, `LICENSE` (MIT) |

---

## Step 1 — Locate the source tree

The project may exist in one of these places:

**A) Standalone git repo (preferred)**

```bash
cd llm-c-from-scratch   # folder with its own .git
git status
git log --oneline -3
```

You should see commits including `Initial commit: llm.c educational notebooks (PyTorch)`.

**B) Inside a parent workspace (e.g. cloud VM)**

```bash
cd /workspace/llm-c-from-scratch
# or search: find ~ -maxdepth 4 -type d -name llm-c-from-scratch 2>/dev/null
```

**C) Only inside AlphaChild branch (no standalone copy)**

If the folder is missing, check out branch `cursor/muzero-atari-gym-64d3` on `souvigna38/AlphaChild` or ask the user for the path. If it was never pushed, recover from the agent transcript or re-create from instructions in `AGENTS.md`.

---

## Step 2 — Verify before push

```bash
cd llm-c-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
export PATH="$HOME/.local/bin:$PATH"
ruff check llmc/ tests/
python3 -m pytest tests/ -v
```

All tests must pass before publishing.

---

## Step 3 — Publish to GitHub (try in order)

### 3a) `gh` CLI (best if authenticated with `repo` scope)

```bash
cd llm-c-from-scratch
gh auth status
gh repo create llm-c-from-scratch --public \
  --description "Educational GPT/llm.c tutorials in Jupyter notebooks (PyTorch)" \
  --source=. --remote=origin --push
```

If the repo already exists:

```bash
git remote add origin https://github.com/souvigna38/llm-c-from-scratch.git 2>/dev/null || \
  git remote set-url origin https://github.com/souvigna38/llm-c-from-scratch.git
git branch -M main
git push -u origin main
```

### 3b) Bundled script in the repo

```bash
cd llm-c-from-scratch
chmod +x scripts/publish_github.sh
./scripts/publish_github.sh
```

### 3c) Manual (human or token with repo access)

1. Open: https://github.com/new?name=llm-c-from-scratch&description=Educational+GPT%2Fllm.c+tutorials&visibility=public  
2. Owner: **souvigna38**, name: **llm-c-from-scratch**, **empty** repo (no README / no .gitignore).  
3. Push:

```bash
cd llm-c-from-scratch
git remote add origin https://github.com/souvigna38/llm-c-from-scratch.git
git branch -M main
git push -u origin main
```

### 3d) If you get `403 Resource not accessible by integration` or `404 Repository not found`

The Cursor cloud token often **cannot create repos**. Stop retrying the same API call. Go to **Step 4 (OpenClaw)**.

---

## Step 4 — Delegate to OpenClaw (if you cannot publish)

If Steps 3a–3c fail, **hand off to OpenClaw** with the message below (copy verbatim, fill in `SOURCE_PATH`).

### Message to paste into OpenClaw

```
Task: Create and push a new public GitHub repository.

Repository:
  Owner: souvigna38
  Name: llm-c-from-scratch
  Visibility: public
  Description: Educational GPT/llm.c tutorials in Jupyter notebooks (PyTorch), inspired by karpathy/llm.c
  URL: https://github.com/souvigna38/llm-c-from-scratch

Source directory (local git repo):
  SOURCE_PATH=<absolute path to llm-c-from-scratch, e.g. /workspace/llm-c-from-scratch>

Requirements:
  1. Create the empty repo on GitHub under souvigna38 (if it does not exist).
  2. Add remote origin https://github.com/souvigna38/llm-c-from-scratch.git
  3. Push branch main (must include all notebooks 1-10, llmc/, data/, tests/, README.md, AGENTS.md, LICENSE).
  4. Do NOT include .venv, __pycache__, .pytest_cache, or *.pt checkpoints.
  5. Confirm push succeeded and reply with the commit SHA on main.

Pre-push verification (run in SOURCE_PATH):
  pip install -e ".[dev]" && python3 -m pytest tests/ -v

If you cannot access SOURCE_PATH, ask the user to zip the folder or grant GitHub PAT with repo + workflow scopes.
```

Replace `SOURCE_PATH` with the real path on the machine OpenClaw uses.

---

## Step 5 — Confirm success

Reply to the user with:

1. **Repo URL:** https://github.com/souvigna38/llm-c-from-scratch  
2. **Latest commit SHA** on `main`  
3. **Test result:** `16 passed` (or paste pytest summary)  
4. **Clone command:**

```bash
git clone https://github.com/souvigna38/llm-c-from-scratch.git
```

---

## Step 6 — Optional follow-up

- Enable GitHub Actions / CI later (not required for first publish).  
- Clone upstream [karpathy/llm.c](https://github.com/karpathy/llm.c) separately for C/CUDA training (`scripts/clone_llmc.sh`).  
- Do **not** change notebook numbering without updating `AGENTS.md` and `tests/test_notebooks.py`.

---

## Failure modes

| Error | Action |
|-------|--------|
| `createRepository` 403 | Use OpenClaw or human creates empty repo, then `git push` |
| `Repository not found` on push | Create repo first (3c or OpenClaw) |
| `Permission denied` | User must add PAT or deploy key with push access to `souvigna38/llm-c-from-scratch` |
| No source folder | Search workspace or ask user for zip / branch name |

---

## Done when

- [ ] https://github.com/souvigna38/llm-c-from-scratch exists and is public  
- [ ] `main` contains `1.Tokens.ipynb` … `10.GPT2AndLlmc.ipynb`  
- [ ] `git clone` URL works for a fresh directory  
- [ ] User notified with URL + commit SHA  
