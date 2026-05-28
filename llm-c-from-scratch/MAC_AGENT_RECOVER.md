# Mac agent: get llm-c-from-scratch on your machine

The standalone repo **https://github.com/souvigna38/llm-c-from-scratch** may not exist yet.  
The full project is published on **AlphaChild** branch `cursor/llm-c-from-scratch-64d3`.

---

## Option 1 — Clone branch folder (easiest)

```bash
cd ~/projects
git clone https://github.com/souvigna38/AlphaChild.git
cd AlphaChild
git fetch origin cursor/llm-c-from-scratch-64d3
git checkout cursor/llm-c-from-scratch-64d3
cd llm-c-from-scratch
```

Verify:

```bash
git log --oneline -3
# cbcc334 Document GitHub fork workflow and DeepSeek feature roadmap
# fbf958f Initial commit: llm.c educational notebooks (PyTorch)

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/ -v   # expect 16 passed
```

Publish as its own repo (you are logged in as `souvigna38`):

```bash
gh repo create llm-c-from-scratch --public \
  --description "Educational GPT/llm.c tutorials in Jupyter notebooks (PyTorch)" \
  --source=. --remote=origin --push
```

---

## Option 2 — Restore standalone git from bundle (same branch)

From `AlphaChild` repo root (after checkout above):

```bash
mkdir -p ~/projects/llm-c-from-scratch
cd ~/projects/llm-c-from-scratch
git init
git pull /path/to/AlphaChild/llm-c-from-scratch.bundle main
# Or:
# git clone /path/to/AlphaChild/llm-c-from-scratch.bundle llm-c-from-scratch
```

Then `pytest` and `gh repo create` as in Option 1.

---

## Option 3 — Sparse checkout (only this subfolder)

```bash
mkdir -p ~/projects/llm-c-from-scratch && cd ~/projects/llm-c-from-scratch
git init
git remote add origin https://github.com/souvigna38/AlphaChild.git
git fetch origin cursor/llm-c-from-scratch-64d3
git checkout cursor/llm-c-from-scratch-64d3 -- llm-c-from-scratch
mv llm-c-from-scratch/* llm-c-from-scratch/.* . 2>/dev/null || true
```

---

## After publish

Report to user:

- **Repo:** https://github.com/souvigna38/llm-c-from-scratch  
- **Commit SHA** on `main`  
- **Tests:** 16 passed  

Then fork for DeepSeek: see `FORK.md` and `CURSOR_AGENT_FORK_DEEPSEEK.md`.
