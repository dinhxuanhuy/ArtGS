#!/usr/bin/env bash
# Stop Colab sessions (idle runtimes burn compute units).
#   scripts/colab_down.sh <session>...     stop the named sessions
#   scripts/colab_down.sh --all            stop every session known to the local CLI state
# Always ends by printing what is still running on the server.
set -uo pipefail
quiet() { grep -v "new version of Colab CLI\|colab update\|update check" || true; }
names=("$@")
if [ "${1:-}" = "--all" ]; then
  colab sessions >/dev/null 2>&1   # prunes stale local entries
  mapfile -t names < <(python3 - <<'EOF'
import json, os
p = os.path.expanduser("~/.config/colab-cli/sessions.json")
try:
    d = json.load(open(p))
except Exception:
    d = {}
def names(x):
    if isinstance(x, dict):
        if isinstance(x.get("name"), str):
            yield x["name"]
        for k, v in x.items():
            if isinstance(v, dict) and "name" not in v and k not in ("settings",):
                yield from ([k] if any(f in v for f in ("endpoint", "accelerator", "kernel_id", "notebook_id")) else names(v))
            elif isinstance(v, (dict, list)):
                yield from names(v)
    elif isinstance(x, list):
        for v in x:
            yield from names(v)
print("\n".join(sorted(set(names(d)))))
EOF
)
fi
for s in "${names[@]}"; do
  [ -n "$s" ] && colab stop -s "$s" 2>&1 | quiet
done
echo "--- still active on server:"
colab sessions 2>&1 | quiet
