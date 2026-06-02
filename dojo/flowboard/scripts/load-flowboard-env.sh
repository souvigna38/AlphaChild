#!/usr/bin/env bash
# Safe loader for flowboard.env — avoids zsh/bash parse errors from parentheses in UA string.
# Usage: eval "$(load-flowboard-env.sh "$HOME/.cursor/flowboard.env")"

path="${1:?env file path required}"
python3 - "$path" <<'PY'
import shlex
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        print(f"export {key}={shlex.quote(val)}")
PY
