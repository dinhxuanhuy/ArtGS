#!/usr/bin/env bash
# Claude Code Stop/SessionEnd hook: warn when Colab runtimes are still running (they burn
# compute units). Never blocks; prints a systemMessage JSON only when something is active.
out=$(timeout 25 colab sessions 2>/dev/null | grep -v "new version of Colab CLI\|colab update\|update check")
[ -z "$out" ] && exit 0
echo "$out" | grep -qi "no active sessions" && exit 0
python3 -c 'import json,sys; print(json.dumps({"systemMessage": "⚠ Colab sessions still active (burning compute units). Stop with scripts/colab_down.sh <name> or --all:\n" + sys.argv[1]}))' "$out"
