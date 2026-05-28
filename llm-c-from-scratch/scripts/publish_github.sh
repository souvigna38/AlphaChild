#!/usr/bin/env bash
# Publish llm-c-from-scratch to its own GitHub repository.
# Requires: gh CLI logged in with repo scope, OR an empty repo already created on GitHub.
set -euo pipefail

OWNER="${GITHUB_OWNER:-souvigna38}"
REPO="${GITHUB_REPO:-llm-c-from-scratch}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "https://github.com/${OWNER}/${REPO}.git"
fi

if gh auth status &>/dev/null; then
  if ! gh repo view "${OWNER}/${REPO}" &>/dev/null; then
    echo "Creating public repo ${OWNER}/${REPO} ..."
    gh repo create "${REPO}" --public \
      --description "Educational GPT/llm.c tutorials in Jupyter notebooks (PyTorch)" \
      --source=. --remote=origin --push
    exit 0
  fi
  echo "Repo exists; pushing ..."
  git push -u origin main
  exit 0
fi

echo "gh not authenticated. Create an empty repo first:"
echo "  https://github.com/new  -> name: ${REPO}, public, no README"
echo "Then run:"
echo "  git push -u origin main"
exit 1
