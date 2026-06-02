#!/usr/bin/env bash
# One-shot install: Flowboard scripts + zsh-safe env template for Mac.
# Usage: bash install-flowboard-mac.sh
set -euo pipefail

REPO="${FLOWBOARD_INSTALL_REPO:-$HOME/AlphaChild}"
BRANCH="${FLOWBOARD_INSTALL_BRANCH:-cursor/agentic-dojo-64d3}"

if [[ ! -d "$REPO/dojo/flowboard/scripts/flowboard.sh" ]]; then
  echo "Cloning AlphaChild to $REPO ..."
  git clone --branch "$BRANCH" --depth 1 https://github.com/souvigna38/AlphaChild.git "$REPO" 2>/dev/null || {
    echo "Clone failed. If you already have the repo, set:"
    echo "  FLOWBOARD_INSTALL_REPO=/path/to/AlphaChild bash install-flowboard-mac.sh"
    exit 1
  }
fi

mkdir -p "$HOME/.cursor/skills/flowboard/scripts"
cp "$REPO/dojo/flowboard/scripts/flowboard.sh" "$HOME/.cursor/skills/flowboard/scripts/"
cp "$REPO/dojo/flowboard/scripts/load-flowboard-env.sh" "$HOME/.cursor/skills/flowboard/scripts/"
cp "$REPO/dojo/flowboard/scripts/smoke-test.sh" "$HOME/.cursor/skills/flowboard/scripts/"
chmod +x "$HOME/.cursor/skills/flowboard/scripts/"*.sh

if [[ ! -f "$HOME/.cursor/flowboard.env" ]]; then
  cp "$REPO/dojo/flowboard/flowboard.env.example" "$HOME/.cursor/flowboard.env"
  chmod 600 "$HOME/.cursor/flowboard.env"
  echo "Created $HOME/.cursor/flowboard.env — paste your flb_ token, then re-run smoke test."
else
  echo "Keeping existing $HOME/.cursor/flowboard.env"
  echo "If you saw 'parse error near (', fix FLOWBOARD_UA line — use single quotes around the UA string."
fi

echo ""
echo "Installed scripts to ~/.cursor/skills/flowboard/scripts/"
echo "Next:"
echo "  nano ~/.cursor/flowboard.env    # paste FLOWBOARD_TOKEN=flb_..."
echo "  ~/.cursor/skills/flowboard/scripts/smoke-test.sh   # no need to source .env"
