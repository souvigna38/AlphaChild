#!/usr/bin/env bash
# Pre-launch verification for cohort commander (run on bot host).
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail() { echo -e "${RED}FAIL:${NC} $1" >&2; exit 1; }
ok() { echo -e "${GREEN}OK:${NC} $1"; }

[[ -n "${DISCORD_BOT_TOKEN:-}" ]] || fail "DISCORD_BOT_TOKEN unset"
ok "DISCORD_BOT_TOKEN set"

[[ -n "${DOJO_PROOF_SECRET:-}" ]] || fail "DOJO_PROOF_SECRET unset"
[[ ${#DOJO_PROOF_SECRET} -ge 16 ]] || fail "DOJO_PROOF_SECRET too short (need 16+)"
ok "DOJO_PROOF_SECRET set"

[[ -n "${DOJO_GUILD_ID:-}" ]] || fail "DOJO_GUILD_ID unset"
ok "DOJO_GUILD_ID=${DOJO_GUILD_ID}"

if [[ "${DOJO_TA_ENABLED:-1}" =~ ^(1|true|yes)$ ]]; then
  [[ -n "${OPENCLAW_API_URL:-}${DOJO_TA_API_URL:-}" ]] || fail "TA enabled but OPENCLAW_API_URL / DOJO_TA_API_URL unset"
  ok "TA URL=${OPENCLAW_API_URL:-${DOJO_TA_API_URL:-}}"
else
  ok "TA disabled (DOJO_TA_ENABLED)"
fi

command -v dojo-gatekeeper >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || fail "dojo-gatekeeper not on PATH (pip install -e '.[dojo]')"
ok "dojo CLI available"

if command -v curl >/dev/null 2>&1 && [[ -n "${OPENCLAW_API_URL:-}" ]]; then
  base="${OPENCLAW_API_URL%/chat/completions}"
  if curl -sf --max-time 3 "${base%/v1}/" >/dev/null 2>&1 || curl -sf --max-time 3 -o /dev/null -w "%{http_code}" "${OPENCLAW_API_URL}" -X POST -H "Content-Type: application/json" -d '{}' | grep -qE '^[245]'; then
    ok "OpenClaw endpoint reachable (or returned expected HTTP)"
  else
    echo -e "${RED}WARN:${NC} OpenClaw not reachable — Sparring Partner will fail until gateway is up"
  fi
fi

echo ""
echo "Launch check passed. Start: set -a && source dojo/.env && set +a && dojo-gatekeeper"
