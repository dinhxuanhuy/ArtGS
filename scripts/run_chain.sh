#!/usr/bin/env bash
# Run colab jobs one after another (one VM at a time). Detach it:
#   setsid nohup bash scripts/run_chain.sh "<job> <datasets>" ... > runs/_sessions/chain.log 2>&1 &
cd "$(dirname "$0")/.."
for spec in "$@"; do
  set -- $spec
  echo "$(date -u +%FT%T) start $1"
  bash scripts/colab_job.sh "scripts/jobs/$1" --gpu T4 --datasets "${*:2}" --max-cu 5 --max-hours 4.3 2>&1 | tail -3
  echo "$(date -u +%FT%T) end $1; sessions: $(colab sessions 2>/dev/null | grep -c '^\[')"
done
echo "chain done"
