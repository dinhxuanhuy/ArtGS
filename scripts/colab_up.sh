#!/usr/bin/env bash
# Start (or reuse) a Colab session and make it ready for ArtGS:
#   new session -> rclone + Drive credentials -> rsync code -> bootstrap (venv, extensions, data)
# Usage: scripts/colab_up.sh <session> [--gpu T4|L4|A100|H100] [--sync-only]
# Env forwarded to the VM bootstrap: DATASETS (default "paris"), FORCE_BUILD, ARTGS_REMOTE.
# Log: runs/_sessions/<session>_<timestamp>.log        Stop with: scripts/colab_down.sh <session>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
S="${1:?usage: colab_up.sh <session> [--gpu T4] [--sync-only]}"; shift
GPU=T4; SYNC_ONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --gpu) GPU="$2"; shift 2;; --sync-only) SYNC_ONLY=1; shift;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
REMOTE_NAME="${RCLONE_REMOTE_NAME:-nckh}"
SSH=(ssh -F "$REPO/scripts/ssh_config" "$S.colab")
RSYNC=(rsync -az -e "ssh -F $REPO/scripts/ssh_config")
quiet() { grep -v "new version of Colab CLI\|colab update\|update check" || true; }
mkdir -p runs/_sessions; LOG="runs/_sessions/${S}_$(date +%Y%m%d_%H%M%S).log"
t0=$(date +%s); ts() { echo "[colab_up +$(( $(date +%s)-t0 ))s] $*" | tee -a "$LOG"; }

if colab sessions 2>/dev/null | grep -qw -- "$S"; then
  ts "reusing session $S"
else
  ts "creating session $S (--gpu $GPU)"
  colab new -s "$S" --gpu "$GPU" 2>&1 | quiet | tee -a "$LOG"
  colab sessions 2>/dev/null | grep -qw -- "$S" || { echo "session $S not created (no $GPU quota?)" >&2; exit 1; }
fi

# Drive access via rclone (headless; `colab drivemount` needs a human click per VM).
# Only the [$REMOTE_NAME] section is sent; it lives on the VM until the VM is stopped.
if rclone listremotes 2>/dev/null | grep -qx "$REMOTE_NAME:"; then
  ts "installing rclone + '$REMOTE_NAME' remote on VM"
  "${RSYNC[@]}" "$(command -v rclone)" "$S.colab:/usr/local/bin/rclone" </dev/null
  rclone config show "$REMOTE_NAME" 2>/dev/null | "${SSH[@]}" \
    'umask 077; mkdir -p /root/.config/rclone; cat > /root/.config/rclone/rclone.conf'
  REMOTE="${ARTGS_REMOTE:-$REMOTE_NAME:}"
else
  ts "WARN: rclone remote '$REMOTE_NAME' not configured locally -> no Drive cache (see CLAUDE.md)"
  REMOTE=none
fi

# HuggingFace token (gated models such as facebook/sam3): copied only if it exists locally,
# never written into the repo; it lives on the VM until the VM is stopped.
HF_TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [ -s "$HF_TOKEN_FILE" ]; then
  ts "installing HF token on VM"
  "${SSH[@]}" 'umask 077; mkdir -p /root/.cache/huggingface; cat > /root/.cache/huggingface/token' < "$HF_TOKEN_FILE"
fi

ts "syncing code"
"${RSYNC[@]}" --exclude-from=scripts/rsync_excludes.txt ./ "$S.colab:/content/ArtGS/" </dev/null
[ "$SYNC_ONLY" = 1 ] && { ts "sync done"; exit 0; }

ts "bootstrapping VM"
set +e
"${SSH[@]}" "DATASETS='${DATASETS-paris}' ARTGS_REMOTE='$REMOTE' FORCE_BUILD='${FORCE_BUILD:-}' \
  bash -c '[ -n \"\$FORCE_BUILD\" ] || unset FORCE_BUILD; bash /content/ArtGS/scripts/bootstrap_colab.sh'" \
  </dev/null 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
"${RSYNC[@]}" "$S.colab:/content/bootstrap_timing.tsv" "${LOG%.log}_timing.tsv" </dev/null 2>/dev/null || true
ts "bootstrap exit=$rc  (log: $LOG)"
exit "$rc"
