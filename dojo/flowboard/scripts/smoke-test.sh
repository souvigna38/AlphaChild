#!/usr/bin/env bash
# Verify Flowboard token + headers for Spm1CurorWorkbook / Cursor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLOWBOARD_ENV="${FLOWBOARD_ENV:-$HOME/.cursor/flowboard.env}"

echo "=== Flowboard smoke test ==="
echo "Env file: ${FLOWBOARD_ENV}"
echo "Holder:   ${FLOWBOARD_HOLDER:-Spm1CurorWorkbook}"
echo ""

out=$("$SCRIPT_DIR/flowboard.sh" list_boards)

if echo "$out" | grep -qi '<html\|Error 1010\|Access denied'; then
  echo "FAIL: Cloudflare blocked the request (HTML response)."
  echo "Fix: set FLOWBOARD_UA, Origin, Referer in flowboard.env — or use flowboard.sh."
  exit 1
fi

if echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'boards' in d" 2>/dev/null; then
  echo "$out" | python3 -m json.tool 2>/dev/null || echo "$out"
  echo ""
  if echo "$out" | grep -q 'ai-agent-onboard'; then
    echo "OK — connected. ai-agent-onboard board visible."
  else
    echo "OK — connected (ai-agent-onboard not in list — check board access)."
  fi
  exit 0
fi

if echo "$out" | grep -q 'missing_bearer_token\|401'; then
  echo "FAIL: bad or missing FLOWBOARD_TOKEN."
  echo "Get a token at: https://task-swarm-keeper.lovable.app/tokens"
  exit 1
fi

echo "Unexpected response:"
echo "$out"
exit 1
