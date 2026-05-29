# Paste into Cursor agent: fork + DeepSeek features

---

## Part 1 — Publish or locate upstream (if needed)

Upstream (intended public repo):

- https://github.com/souvigna38/llm-c-from-scratch

If **404**: follow `CURSOR_AGENT_TASK.md` or delegate repo creation to **OpenClaw**, then continue.

---

## Part 2 — Create the fork

### GitHub UI

1. Open upstream repo → **Fork** → your account.  
2. Clone your fork:

```bash
git clone https://github.com/<YOUR_GITHUB_USER>/llm-c-from-scratch.git
cd llm-c-from-scratch
git remote add upstream https://github.com/souvigna38/llm-c-from-scratch.git
git checkout -b feature/deepseek
```

### CLI

```bash
gh repo fork souvigna38/llm-c-from-scratch --clone --remote
cd llm-c-from-scratch
git checkout -b feature/deepseek
```

If `gh repo fork` fails with **403**, use GitHub **Fork** button manually, then clone.

---

## Part 3 — Add DeepSeek features (on the fork only)

**Do not rewrite notebooks 1–10.** Add new lessons per `docs/DEEPSEEK_ROADMAP.md`:

1. `11.MLA.ipynb` — Multi-head Latent Attention  
2. `12.DeepSeekMoE.ipynb` — mixture-of-experts FFN  
3. `13.GPTDeepSeek.ipynb` — wire MLA + MoE into one model  
4. `14.TrainDeepSeek.ipynb` — short training run  
5. `15.SampleDeepSeek.ipynb` — text generation  
6. `16.CompareGPT2DeepSeek.ipynb` — compare active params vs GPT-2  

Code:

- Add `llmc/deepseek.py` (MLA, MoE block, `GPTDeepSeek` config)  
- Add tests under `tests/test_deepseek_*.py`  
- Run: `pip install -e ".[dev]" && pytest tests/ -v`

**Optional:** Hugging Face / official DeepSeek weights — separate notebook, not required for v1.

---

## Part 4 — Push and report

```bash
git add -A
git commit -m "Add DeepSeek educational path (MLA + MoE notebooks)"
git push -u origin feature/deepseek
```

Tell the user:

- Fork URL: `https://github.com/<YOUR_GITHUB_USER>/llm-c-from-scratch`  
- Branch: `feature/deepseek`  
- PR optional: `feature/deepseek` → `main` on the **fork**  
- Upstream sync: `git fetch upstream && git merge upstream/main`

---

## Part 5 — OpenClaw fallback

If you cannot fork or push:

```
OpenClaw: Fork https://github.com/souvigna38/llm-c-from-scratch to <USER>.
Create branch feature/deepseek.
Implement docs/DEEPSEEK_ROADMAP.md (notebooks 11-16 + llmc/deepseek.py + tests).
Push and return fork URL + commit SHA.
```

---

## References for implementer

- DeepSeek-V2: MLA + DeepSeekMoE — https://arxiv.org/html/2405.04434  
- DeepSeekMoE paper — https://arxiv.org/html/2401.06066  
- Baseline in this repo: notebooks 1–10, package `llmc/model.py`
