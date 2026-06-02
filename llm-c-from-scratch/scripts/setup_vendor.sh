#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"
clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "exists: $dir"
  else
    git clone --depth 1 "$url" "$dir"
  fi
}
clone_if_missing https://github.com/karpathy/llm.c.git "$VENDOR/llm.c"
clone_if_missing https://github.com/hebo1221/nano-deepseek-v4.git "$VENDOR/nano-deepseek-v4"
echo "Vendor ready under $VENDOR"
