#!/usr/bin/env bash
# SemArt job body, sourced by a small per-experiment job file that sets the CONFIG variables:
#   SCENES="dta/sapien/storage_47254 artgs/sapien/oven_101908"   (dataset/subset/scene)
#   METHODS="artgs,sem_k,sem_mask,sem_full"   SEGMENTER=sam3|qwen_sam2   VLM=Qwen/Qwen3-VL-4B-Instruct
#   ITERS=20000 RES=1 PREDICT_ITERS= (default per dataset) SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1
#   VIEW_STRIDE=1 EXTRA_TRAIN_ARGS=""
# Runs on the VM from /content/ArtGS (venv active, RUN_ID/RUN_OUT set by scripts/colab_job.sh).
set -euo pipefail
: "${SCENES:?set SCENES}"; : "${METHODS:=artgs,sem_k,sem_mask,sem_full}"
: "${SEGMENTER:=sam3}"; : "${VLM:=Qwen/Qwen3-VL-4B-Instruct}"; : "${ITERS:=20000}"; : "${RES:=1}"
: "${PREDICT_ITERS:=}"; : "${SEEDS:=0}"; : "${K_SOURCE:=vlm}"; : "${LAMBDA_SEM:=0.1}"
: "${VIEW_STRIDE:=1}"; : "${EXTRA_TRAIN_ARGS:=}"; : "${SEM_UNTIL:=15000}"
REMOTE=$(sed -n 's/^\[\(.*\)\]$/\1/p' /root/.config/rclone/rclone.conf 2>/dev/null | head -1)
REMOTE=${REMOTE:+$REMOTE:}

ln -sfn "$RUN_OUT" outputs
export ARTGS_CACHE=/content/cache
T=$RUN_OUT/stage_seconds.tsv; touch "$T"
stage() { local n=$1; shift; local s=$(date +%s); echo "=== $n: $*"; "$@"; printf '%s\t%d\n' "$n" $(( $(date +%s)-s )) >> "$T"; }

# ---- S0 env: Colab's own python (modern torch) for the VLM / segmenter
SYS_PY=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3)
echo "system python: $SYS_PY $($SYS_PY -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
stage s0_pip env -u VIRTUAL_ENV PYTHONNOUSERSITE= "$SYS_PY" -m pip install -q -U "transformers>=4.57" accelerate
echo "transformers $($SYS_PY -c 'import transformers;print(transformers.__version__)')"

: "${PARALLEL:=0}"   # 1 = after the shared stages, run the scenes' pipelines concurrently (big GPU)
P_ARGS=(); [ -n "$PREDICT_ITERS" ] && P_ARGS=(--predict_iters "$PREDICT_ITERS")
pipe() {  # <ds> <sub> <name> <semseg> <seed> [extra run_pipeline args]
  python -m semart.run_pipeline --dataset "$1" --subset "$2" --scene "$3" --semseg_dir "$4" --seed "$5" \
    --methods "$METHODS" --k_source "$K_SOURCE" --iters "$ITERS" --res "$RES" --lambda_sem "$LAMBDA_SEM" \
    --sem_until_iter "$SEM_UNTIL" "${P_ARGS[@]}" --extra_train_args "$EXTRA_TRAIN_ARGS" "${@:6}"
}
declare -A SEMSEGS
for SC in $SCENES; do
  IFS=/ read -r DS SUB NAME <<< "$SC"
  echo "################ $SC"
  # ---- S0 preprocess (Drive-cached by config hash)
  if [ -n "$REMOTE" ]; then rclone copy "${REMOTE}cache/semseg/$SC" "$ARTGS_CACHE/semseg/$SC" -q || true; fi
  stage "s0_$NAME" env -u VIRTUAL_ENV PYTHONNOUSERSITE= "$SYS_PY" -m semart.preprocess \
    --scene_dir "data/$SC" --object_id "$SC" --vlm "$VLM" --segmenter "$SEGMENTER" \
    --view_stride "$VIEW_STRIDE" 2>&1 | tee "$RUN_OUT/preprocess_$NAME.log"
  SEMSEG=$(sed -n 's/^SEMSEG_DIR=//p' "$RUN_OUT/preprocess_$NAME.log" | tail -1)
  [ -n "$SEMSEG" ] || { echo "preprocess failed for $SC"; exit 1; }
  if [ -n "$REMOTE" ]; then rclone copy "$ARTGS_CACHE/semseg/$SC" "${REMOTE}cache/semseg/$SC" -q || true; fi
  mkdir -p "$RUN_OUT/$SC/semseg"; cp -r "$SEMSEG"/{vlm.json,summary.json,meta.json,vis} "$RUN_OUT/$SC/semseg/"
  SEMSEGS[$SC]=$SEMSEG
  if [ "$PARALLEL" = 1 ]; then
    # coarse + lift sequentially: train_coarse.py rewrites a shared json in arguments/
    stage "shared_$NAME" pipe "$DS" "$SUB" "$NAME" "$SEMSEG" "${SEEDS%% *}" --only_shared
  else
    # ---- S1..S5 per seed (coarse + lift are shared across methods and seeds; train varies with seed)
    for SEED in $SEEDS; do stage "pipe_${NAME}_s$SEED" pipe "$DS" "$SUB" "$NAME" "$SEMSEG" "$SEED"; done
  fi
done

if [ "$PARALLEL" = 1 ]; then
  s0=$(date +%s)
  : "${MAXPAR:=99}"   # at most this many scene pipelines at once (T4 VMs have 2 vCPUs: use 2)
  for SC in $SCENES; do
    IFS=/ read -r DS SUB NAME <<< "$SC"
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n || true; done
    ( rc=0; for SEED in $SEEDS; do pipe "$DS" "$SUB" "$NAME" "${SEMSEGS[$SC]}" "$SEED" || rc=$?; done
      echo $rc > "$RUN_OUT/pipe_$NAME.rc" ) > "$RUN_OUT/pipe_$NAME.log" 2>&1 &
    echo "launched $SC (log pipe_$NAME.log)"
  done
  wait
  printf 'parallel_pipelines\t%d\n' $(( $(date +%s)-s0 )) >> "$T"
  FAIL=0; for SC in $SCENES; do n=$(basename "$SC"); echo "$n rc=$(cat "$RUN_OUT/pipe_$n.rc")"
    [ "$(cat "$RUN_OUT/pipe_$n.rc")" = 0 ] || FAIL=1; done
fi
find "$RUN_OUT" -name result.csv | sort
cat "$T"
exit ${FAIL:-0}
