#!/usr/bin/env bash
# Flowboard agent API helper — loads ~/.cursor/flowboard.env and sets Cloudflare-safe headers.
set -euo pipefail

ENV_FILE="${FLOWBOARD_ENV:-$HOME/.cursor/flowboard.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ENV_FILE" ]]; then
  # Do NOT use `source` — FLOWBOARD_UA contains () and breaks zsh
  eval "$("$SCRIPT_DIR/load-flowboard-env.sh" "$ENV_FILE")"
fi

: "${FLOWBOARD_BASE:?Set FLOWBOARD_BASE in $ENV_FILE}"
: "${FLOWBOARD_TOKEN:?Set FLOWBOARD_TOKEN in $ENV_FILE — get one at ${FLOWBOARD_BASE}/tokens}"

API="${FLOWBOARD_API:-${FLOWBOARD_BASE}/api/public/agent}"
UA="${FLOWBOARD_UA:-Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36}"

curl_common() {
  curl -sS \
    -H "Authorization: Bearer ${FLOWBOARD_TOKEN}" \
    -H "User-Agent: ${UA}" \
    -H "Origin: ${FLOWBOARD_BASE}" \
    -H "Referer: ${FLOWBOARD_BASE}/" \
    -H "Accept: application/json" \
    "$@"
}

usage() {
  cat <<'EOF'
Usage: flowboard.sh <tool> [json_body]

Tools:
  list_boards              GET  /boards
  list_tasks               GET  /tasks?board_id=&column_id=
  get_task                 GET  /tasks/:id
  create_task              POST /tasks
  patch_task               PATCH /tasks/:id
  move_task                POST /tasks/:id/move
  claim_task               POST /tasks/:id/claim
  release_task             POST /tasks/:id/release
  comment_task             POST /tasks/:id/comment
  delete_task              DELETE /tasks/:id
  tools                    GET  /tools

Examples:
  flowboard.sh list_boards
  flowboard.sh list_tasks '{"board_id":"ai-agent-onboard","column_id":"start-here"}'
  flowboard.sh create_task '{"board_id":"ai-agent-onboard","column_id":"backlog","title":"Test from Spm1CurorWorkbook"}'
EOF
}

tool="${1:-}"
shift || true

case "$tool" in
  list_boards)
    curl_common "${API}/boards"
    ;;
  tools)
    curl_common "${API}/tools"
    ;;
  list_tasks)
    body="${1:-{}}"
    board_id=$(echo "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('board_id',''))" 2>/dev/null || true)
    column_id=$(echo "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('column_id',''))" 2>/dev/null || true)
    url="${API}/tasks"
    sep="?"
    [[ -n "$board_id" ]] && url="${url}${sep}board_id=${board_id}" && sep="&"
    [[ -n "$column_id" ]] && url="${url}${sep}column_id=${column_id}"
    curl_common "$url"
    ;;
  get_task)
    id="${1:?task id required}"
    curl_common "${API}/tasks/${id}"
    ;;
  create_task)
    body="${1:?json body required}"
    curl_common -X POST -H "Content-Type: application/json" -d "$body" "${API}/tasks"
    ;;
  patch_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    curl_common -X PATCH -H "Content-Type: application/json" -d "$1" "${API}/tasks/${id}"
    ;;
  move_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    curl_common -X POST -H "Content-Type: application/json" -d "$1" "${API}/tasks/${id}/move"
    ;;
  claim_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    holder="${FLOWBOARD_HOLDER:-Spm1CurorWorkbook}"
    payload=$(echo "$1" | python3 -c "import json,sys,os; d=json.load(sys.stdin); d.setdefault('holder',os.environ.get('FLOWBOARD_HOLDER','Spm1CurorWorkbook')); print(json.dumps(d))")
    curl_common -X POST -H "Content-Type: application/json" -d "$payload" "${API}/tasks/${id}/claim"
    ;;
  release_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    curl_common -X POST -H "Content-Type: application/json" -d "$1" "${API}/tasks/${id}/release"
    ;;
  comment_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    curl_common -X POST -H "Content-Type: application/json" -d "$1" "${API}/tasks/${id}/comment"
    ;;
  delete_task)
    id=$(echo "${1:?json with id required}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    curl_common -X DELETE -H "Content-Type: application/json" -d "$1" "${API}/tasks/${id}"
    ;;
  ""|-h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown tool: $tool" >&2
    usage
    exit 1
    ;;
esac

echo ""
