#!/usr/bin/env bash
# Independent finisher for a job launched by scripts/colab_job.sh: wait until the job on the VM writes
# its exit code (or a deadline passes), pull the record, upload full outputs to Drive, STOP the VM.
# Safe to run alongside colab_job.sh (both cleanups are idempotent). Run it detached so it survives
# the launching shell:
#   setsid nohup bash scripts/colab_finish.sh <RUN_ID> <session> [deadline_epoch] > runs/<RUN_ID>/finish.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
RUN_ID=$1; S=$2; DEADLINE=${3:-$(( $(date +%s) + 4*3600 ))}
REC="runs/$RUN_ID"; mkdir -p "$REC"
SSH=(ssh -F "$REPO/scripts/ssh_config" "$S.colab")
RSYNC=(rsync -az -e "ssh -F $REPO/scripts/ssh_config")
log() { echo "[finish $(date +%T)] $*"; }
alive() { colab sessions 2>/dev/null | grep -qw -- "$S"; }

while alive; do
  "${SSH[@]}" "test -f /content/runs/$RUN_ID.exit_code" </dev/null 2>/dev/null && { log "job finished"; break; }
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "deadline reached -> killing job"
    "${SSH[@]}" "pkill -f job_$RUN_ID.sh; sleep 2; echo 124 > /content/runs/$RUN_ID.exit_code" </dev/null
    break
  fi
  sleep 60
done
if alive; then
  for i in 1 2 3; do "${RSYNC[@]}" --max-size=20m "$S.colab:/content/runs/$RUN_ID/" "$REC/" </dev/null && break; sleep 10; done
  "${RSYNC[@]}" "$S.colab:/content/runs/$RUN_ID.exit_code" "$REC/exit_code" </dev/null
  # deform weights exceed 20 MB but are needed for figures
  "${RSYNC[@]}" --include='*/' --include='deform.pth' --exclude='*' "$S.colab:/content/runs/$RUN_ID/" "$REC/" </dev/null
  if "${SSH[@]}" 'test -s /root/.config/rclone/rclone.conf' </dev/null; then
    R=$("${SSH[@]}" "sed -n 's/^\[\(.*\)\]$/\1/p' /root/.config/rclone/rclone.conf | head -1" </dev/null)
    log "uploading to $R:experiments/$RUN_ID/"
    "${SSH[@]}" "rclone copy /content/runs/$RUN_ID $R:experiments/$RUN_ID --transfers 16 -q" </dev/null \
      && echo "$R:experiments/$RUN_ID" > "$REC/drive_path"
  fi
  colab stop -s "$S" 2>&1 | tail -1
fi
python3 scripts/collect_metrics.py "$REC" >/dev/null 2>&1 || true
log "done: exit=$(cat "$REC/exit_code" 2>/dev/null)  sessions now: $(colab sessions 2>/dev/null | grep -c '^\[')"
