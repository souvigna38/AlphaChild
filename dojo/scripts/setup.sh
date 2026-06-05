#!/usr/bin/env bash
# One-command Discord setup for beginners (run on your Mac in the AlphaChild repo).
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "Installing Agentic Dojo tools…"
python3 -m pip install -q -e ".[dojo]"
exec python3 "$(dirname "$0")/../setup_wizard.py"
