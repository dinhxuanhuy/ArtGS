#!/usr/bin/env bash
# Run one experiment on a fresh Colab GPU, record it, pull results, ALWAYS stop the VM.
#
#   scripts/colab_job.sh <job.sh> [--name NAME] [--gpu T4] [--datasets "paris"] \
#                        [--max-cu 5] [--max-hours 3] [--keep] [--session NAME]
#
# <job.sh> is a bash script run ON THE VM from /content/ArtGS with the venv active and:
#   RUN_ID   e.g. 20261112_0530_smoke_00c3b84
#   RUN_OUT  /content/runs/$RUN_ID   -> write ALL outputs here (model_path, logs, metrics)
# Record kept locally in runs/$RUN_ID/: job.sh, args, git_sha, git_diff.patch, untracked.txt,
# env.txt, job.log, exit_code, timing, and every file < 20 MB from RUN_OUT. If the Drive
# remote is configured, the complete RUN_OUT (checkpoints) goes to <remote>experiments/$RUN_ID/.
# Guards: refuses to start if balance < max-cu + 10; kills the job at the CU or time cap.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
JOB="${1:?usage: colab_job.sh <job.sh> [--name N] [--gpu T4] [--datasets ...] [--max-cu 5] [--max-hours 3] [--keep]}"; shift
[ -f "$JOB" ] || { echo "no such job script: $JOB" >&2; exit 2; }
NAME="$(basename "$JOB" .sh)"; GPU=T4; DATASETS_ARG="paris"; MAX_CU=5; MAX_HOURS=3; KEEP=0; S=""; REMOTE=""
while [ $# -gt 0 ]; do case "$1" in
  --name) NAME="$2"; shift 2;; --gpu) GPU="$2"; shift 2;; --datasets) DATASETS_ARG="$2"; shift 2;;
  --max-cu) MAX_CU="$2"; shift 2;; --max-hours) MAX_HOURS="$2"; shift 2;; --keep) KEEP=1; shift;;
  --session) S="$2"; shift 2;;   # reuse an existing session (still stopped at the end unless --keep)
  *) echo "unknown arg $1" >&2; exit 2;; esac; done

quiet() { grep -v "new version of Colab CLI\|colab update\|update check" || true; }
balance() { colab usage 2>/dev/null | awk '/Current balance/{print $3}'; }
rate()    { colab usage 2>/dev/null | awk '/Usage rate/{sub("/hr","",$3); print $3}'; }

# ---- budget guard
BAL=$(balance); [ -n "$BAL" ] || { echo "cannot read compute-unit balance (auth?)" >&2; exit 1; }
if python3 -c "import sys; sys.exit(not ($BAL < $MAX_CU + 10))"; then
  echo "balance $BAL CU < max-cu $MAX_CU + 10 reserve; refusing to start" >&2; exit 1
fi

# ---- run record
SHA=$(git rev-parse --short HEAD); git diff --quiet HEAD 2>/dev/null || SHA="$SHA-dirty"
RUN_ID="$(date +%Y%m%d_%H%M)_${NAME}_${SHA}"
REC="runs/$RUN_ID"; mkdir -p "$REC"
cp "$JOB" "$REC/job.sh"
printf 'job=%s\nname=%s\ngpu=%s\ndatasets=%s\nmax_cu=%s\nmax_hours=%s\nstart_balance=%s\n' \
  "$JOB" "$NAME" "$GPU" "$DATASETS_ARG" "$MAX_CU" "$MAX_HOURS" "$BAL" > "$REC/args.txt"
git rev-parse HEAD > "$REC/git_sha"
git diff HEAD > "$REC/git_diff.patch" || true
git ls-files --others --exclude-standard > "$REC/untracked.txt" || true
S="${S:-job-$(date +%H%M%S)}"
SSH=(ssh -F "$REPO/scripts/ssh_config" "$S.colab")
RSYNC=(rsync -az -e "ssh -F $REPO/scripts/ssh_config")
log() { echo "[colab_job $(date +%T)] $*" | tee -a "$REC/driver.log"; }
log "run $RUN_ID on session $S ($GPU), caps: ${MAX_CU} CU / ${MAX_HOURS} h"

cleanup() {
  local rc=$?
  set +e
  if colab sessions 2>/dev/null | grep -qw -- "$S"; then
    log "pulling results"
    "${RSYNC[@]}" --max-size=20m "$S.colab:/content/runs/$RUN_ID/" "$REC/" </dev/null 2>/dev/null
    "${RSYNC[@]}" "$S.colab:/content/runs/$RUN_ID.exit_code" "$REC/exit_code" </dev/null 2>/dev/null
    "${SSH[@]}" "source /content/venv/bin/activate 2>/dev/null; python -c 'import torch;print(\"torch\",torch.__version__,torch.cuda.get_device_name(0))'; \
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; pip freeze 2>/dev/null" > "$REC/env.txt" </dev/null 2>/dev/null
    if "${SSH[@]}" 'test -s /root/.config/rclone/rclone.conf' </dev/null 2>/dev/null; then
      log "uploading full outputs to ${REMOTE}experiments/$RUN_ID/"
      "${SSH[@]}" "rclone copy /content/runs/$RUN_ID ${REMOTE}experiments/$RUN_ID --transfers 8 -q" </dev/null \
        && echo "${REMOTE}experiments/$RUN_ID" > "$REC/drive_path"
    fi
    if [ "$KEEP" = 1 ]; then log "--keep: session $S left running. Stop it: scripts/colab_down.sh $S"
    else colab stop -s "$S" 2>&1 | quiet | tee -a "$REC/driver.log"; fi
  fi
  echo "end_balance=$(balance)" >> "$REC/args.txt"
  [ -f "$REC/exit_code" ] || echo "$rc" > "$REC/exit_code"
  log "done: exit=$(cat "$REC/exit_code")  record: $REC"
  [ -f "$REPO/scripts/collect_metrics.py" ] && python3 "$REPO/scripts/collect_metrics.py" "$REC" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM

DATASETS="$DATASETS_ARG" bash scripts/colab_up.sh "$S" --gpu "$GPU" 2>&1 | tee "$REC/bootstrap.log"
[ "${PIPESTATUS[0]}" = 0 ] || { log "bootstrap failed"; exit 1; }
REMOTE=$(grep -q "persistent remote:" "$REC/bootstrap.log" && sed -n 's/.*persistent remote: //p' "$REC/bootstrap.log" | head -1 || echo "")

# ---- launch detached (survives ssh disconnects)
"${RSYNC[@]}" "$JOB" "$S.colab:/content/job_$RUN_ID.sh" </dev/null
"${SSH[@]}" "mkdir -p /content/runs/$RUN_ID && cd /content/ArtGS && \
  ( setsid nohup bash -c 'source /content/venv/bin/activate; export RUN_ID=$RUN_ID RUN_OUT=/content/runs/$RUN_ID PYTHONNOUSERSITE=1; \
  s=\$(date +%s); bash /content/job_$RUN_ID.sh > /content/runs/$RUN_ID/job.log 2>&1; rc=\$?; \
  echo \"job_seconds \$((\$(date +%s)-s))\" > /content/runs/$RUN_ID/timing.txt; echo \$rc > /content/runs/$RUN_ID.exit_code' \
  </dev/null >/dev/null 2>&1 & ) ; echo launched" </dev/null
# ^ double fork: Colab's sshd waits for every child of the session, even setsid+nohup+redirects;
#   only a re-parented grandchild lets ssh return immediately (verified on a CPU VM, 2026-11-12).
log "job launched; tail: ssh -F scripts/ssh_config $S.colab tail -f /content/runs/$RUN_ID/job.log"

# ---- monitor with caps
R=$(rate); R=${R:-2}; python3 -c "import sys; sys.exit(not float('$R') > 0)" || R=2
LIMIT=$(python3 -c "print(int(min($MAX_HOURS*3600, $MAX_CU/float('$R')*3600)))")
log "usage rate ${R} CU/h -> job time limit ${LIMIT}s"
t0=$(date +%s)
while :; do
  if "${SSH[@]}" "test -f /content/runs/$RUN_ID.exit_code" </dev/null 2>/dev/null; then
    log "job finished (exit $("${SSH[@]}" "cat /content/runs/$RUN_ID.exit_code" </dev/null))"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$LIMIT" ]; then
    log "CAP REACHED after ${el}s -> killing job"
    "${SSH[@]}" "pkill -f job_$RUN_ID.sh; sleep 2; echo 124 > /content/runs/$RUN_ID.exit_code" </dev/null; break
  fi
  last=$("${SSH[@]}" "tail -c 2000 /content/runs/$RUN_ID/job.log 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -1" </dev/null 2>/dev/null | cut -c1-150)
  log "+${el}s  ${last}"
  sleep 60
done
