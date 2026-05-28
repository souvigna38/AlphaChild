#!/usr/bin/env bash
# Clone upstream llm.c for C/CUDA training after finishing the notebooks.
set -euo pipefail
TARGET="${1:-../llm.c}"
if [ -d "$TARGET/.git" ]; then
  echo "Already cloned: $TARGET"
  exit 0
fi
git clone https://github.com/karpathy/llm.c.git "$TARGET"
echo "Cloned to $TARGET"
echo "Next: cd $TARGET && pip install -r requirements.txt && ./dev/download_starter_pack.sh"
