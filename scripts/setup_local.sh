#!/usr/bin/env bash
# Local (orchestrator) setup for running ArtGS on Google Colab via the Colab CLI.
# Idempotent: safe to re-run. Does NOT authenticate (that is interactive, see below).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

log() { echo "[setup_local] $*"; }

# 1. Platform check (Colab CLI supports Linux/macOS only)
case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo "Unsupported OS $(uname -s): Colab CLI needs Linux or macOS" >&2; exit 1 ;;
esac

# 2. Required tools
for t in git curl tar; do
  command -v "$t" >/dev/null || { echo "missing required tool: $t" >&2; exit 1; }
done
if ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Colab CLI (install or upgrade in place)
# keyrings.alt: headless Linux has no OS keyring, and CLI >= 0.8 stores its OAuth
# token via `keyring`. Without a backend you'd re-login on every call. The token then
# lives in ~/.local/share/python_keyring/ (user-only, outside the repo).
log "installing/upgrading google-colab-cli (+ keyrings.alt)"
uv tool install -U google-colab-cli --with keyrings.alt >/dev/null
log "colab $(colab version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1)"

# 4. Project skill for Claude Code (refreshed from the installed CLI each run)
mkdir -p .claude/skills/colab
colab skill 2>/dev/null > .claude/skills/colab/SKILL.md.tmp \
  && mv .claude/skills/colab/SKILL.md.tmp .claude/skills/colab/SKILL.md
log "skill -> .claude/skills/colab/SKILL.md"

# 5. Local dirs used by the workflow (all gitignored)
mkdir -p cache runs

# 6. Auth status (never triggers the interactive flow)
# (CLI >= 0.8 keeps the oauth2 token in the OS keyring; probe with stdin closed so
#  an unauthenticated CLI aborts instead of waiting for a code.)
auth_out="$(timeout 60 colab --auth=oauth2 sessions </dev/null 2>&1 || true)"
if grep -qi "authorize colab-cli\|authorization code" <<<"$auth_out"; then
  log "NOT AUTHENTICATED. Run once, interactively:  colab --auth=oauth2 sessions"
else
  log "colab oauth2 auth OK"
fi
log "done"
