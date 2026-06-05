#!/usr/bin/env bash
# Install Flowboard scripts on Mac (no git required if curl works).
# Usage: bash install-flowboard-mac.sh
set -euo pipefail

REPO="${FLOWBOARD_INSTALL_REPO:-$HOME/AlphaChild}"
BRANCH="${FLOWBOARD_INSTALL_BRANCH:-cursor/agentic-dojo-64d3}"
RAW="https://raw.githubusercontent.com/souvigna38/AlphaChild/${BRANCH}/dojo/flowboard"
SCRIPT_DIR="$HOME/.cursor/skills/flowboard/scripts"

install_from_curl() {
  echo "Downloading scripts from GitHub (no git)..."
  mkdir -p "$SCRIPT_DIR"
  for f in flowboard.sh load-flowboard-env.sh smoke-test.sh; do
    curl -fsSL "${RAW}/scripts/${f}" -o "${SCRIPT_DIR}/${f}"
  done
  chmod +x "$SCRIPT_DIR/"*.sh
  if [[ ! -f "$HOME/.cursor/flowboard.env" ]]; then
    curl -fsSL "${RAW}/flowboard.env.example" -o "$HOME/.cursor/flowboard.env"
    chmod 600 "$HOME/.cursor/flowboard.env"
    echo "Created $HOME/.cursor/flowboard.env — paste your flb_ token."
  fi
}

install_from_repo() {
  mkdir -p "$SCRIPT_DIR"
  cp "$REPO/dojo/flowboard/scripts/flowboard.sh" "$SCRIPT_DIR/"
  cp "$REPO/dojo/flowboard/scripts/load-flowboard-env.sh" "$SCRIPT_DIR/"
  cp "$REPO/dojo/flowboard/scripts/smoke-test.sh" "$SCRIPT_DIR/"
  chmod +x "$SCRIPT_DIR/"*.sh
  if [[ ! -f "$HOME/.cursor/flowboard.env" ]]; then
    cp "$REPO/dojo/flowboard/flowboard.env.example" "$HOME/.cursor/flowboard.env"
    chmod 600 "$HOME/.cursor/flowboard.env"
    echo "Created $HOME/.cursor/flowboard.env — paste your flb_ token."
  fi
}

if [[ -f "$REPO/dojo/flowboard/scripts/flowboard.sh" ]]; then
  echo "Using existing repo: $REPO"
  install_from_repo
elif [[ -d "$REPO/.git" ]]; then
  echo "Updating $REPO ..."
  git -C "$REPO" fetch origin "$BRANCH" 2>/dev/null || true
  git -C "$REPO" checkout "$BRANCH" 2>/dev/null || git -C "$REPO" pull 2>/dev/null || true
  if [[ -f "$REPO/dojo/flowboard/scripts/flowboard.sh" ]]; then
    install_from_repo
  else
    echo "Repo present but flowboard scripts missing — using curl fallback."
    install_from_curl
  fi
else
  echo "Trying git clone to $REPO ..."
  if git clone --branch "$BRANCH" --depth 1 https://github.com/souvigna38/AlphaChild.git "$REPO" 2>/dev/null; then
    install_from_repo
  else
    echo "Git clone skipped or failed — using curl fallback."
    install_from_curl
  fi
fi

echo ""
echo "Installed to $SCRIPT_DIR"
echo "Next:"
echo "  nano ~/.cursor/flowboard.env"
echo "  $SCRIPT_DIR/smoke-test.sh"
