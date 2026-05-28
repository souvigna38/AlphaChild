# Forking this project (for DeepSeek and other experiments)

You **should fork** if you want to add DeepSeek-style features (MLA, MoE, etc.) without changing the upstream tutorial line.

---

## Prerequisite: upstream repo must exist

Forking on GitHub requires a **published** parent repo:

**https://github.com/souvigna38/llm-c-from-scratch**

If you get `404`, publish first using `CURSOR_AGENT_TASK.md` or OpenClaw, then return here.

---

## Option A — GitHub Fork (recommended)

1. Open https://github.com/souvigna38/llm-c-from-scratch  
2. Click **Fork** → choose your account or org.  
3. Clone **your** fork:

```bash
git clone https://github.com/<YOUR_USER>/llm-c-from-scratch.git
cd llm-c-from-scratch
git remote add upstream https://github.com/souvigna38/llm-c-from-scratch.git
```

4. Create a feature branch:

```bash
git checkout -b feature/deepseek
```

5. Push to your fork:

```bash
git push -u origin feature/deepseek
```

**Sync later from upstream:**

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

---

## Option B — New repo (copy, not a GitHub “Fork” link)

Use this if you want a **renamed** project (e.g. `llm-c-deepseek-edu`) or upstream is not published yet.

```bash
cp -r llm-c-from-scratch llm-c-deepseek-edu
cd llm-c-deepseek-edu
rm -rf .git
git init
git add -A && git commit -m "Initial commit: fork for DeepSeek experiments"
gh repo create llm-c-deepseek-edu --public --source=. --push
```

Update `README.md` to credit upstream:  
https://github.com/souvigna38/llm-c-from-scratch

---

## Option C — Fork from local copy (this workspace)

If you only have the folder locally (not on GitHub yet):

```bash
cd /path/to/llm-c-from-scratch
git remote -v
# Publish to YOUR account:
gh repo create <YOUR_USER>/llm-c-from-scratch --public --source=. --push
# Or push to a new name:
gh repo create <YOUR_USER>/llm-c-deepseek-edu --public --source=. --push
```

---

## Suggested branch layout for DeepSeek work

| Branch | Purpose |
|--------|---------|
| `main` | Stable tutorial notebooks 1–10 (GPT / llm.c baseline) |
| `feature/deepseek` | All DeepSeek experiments |
| `feature/deepseek-mla` | Multi-head Latent Attention only |
| `feature/deepseek-moe` | DeepSeekMoE FFN only |

Keep **notebook numbers 1–10** stable on `main`; add **11+** on the fork for DeepSeek (see `docs/DEEPSEEK_ROADMAP.md`).

---

## License

Upstream is MIT. Your fork stays MIT; note DeepSeek model weights/API may have **separate** terms if you integrate their checkpoints or API.
