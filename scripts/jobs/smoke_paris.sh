#!/usr/bin/env bash
# Smoke test: full pipeline on ONE small PARIS object with short stages.
# Not a benchmark. Coarse must run 10k iters (train.py loads coarse_gs/.../iteration_10000).
# Runs on the VM via scripts/colab_job.sh (provides RUN_OUT, venv active).
set -euo pipefail
SCENE=${SCENE:-blade_103706}; DS=paris; SUB=sapien
OUT="$RUN_OUT/$DS/$SUB/$SCENE"
T=$RUN_OUT/stage_seconds.tsv; : > "$T"
stage() { local n=$1; shift; local s=$(date +%s); echo "=== $n: $*"; "$@"; printf '%s\t%d\n' "$n" $(( $(date +%s)-s )) >> "$T"; }
COMMON=(--dataset $DS --subset $SUB --scene_name $SCENE --random_bg_color)

stage coarse  python train_coarse.py  "${COMMON[@]}" --model_path "$OUT/coarse_gs" \
  --resolution 2 --iterations 10000 --opacity_reg_weight 0.1 --quiet
stage predict python train_predict.py "${COMMON[@]}" --model_path "$OUT/pred" --eval \
  --resolution 8 --iterations 500 --densify_grad_threshold 0.001 --coarse_name coarse_gs --quiet
stage train   python train.py         "${COMMON[@]}" --model_path "$OUT/artgs" --eval \
  --resolution 2 --iterations 2000 --coarse_name coarse_gs --seed 0 --use_art_type_prior \
  --densify_grad_threshold 0.001 --quiet
stage render  python render.py --dataset $DS --subset $SUB --scene_name $SCENE \
  --model_path "$OUT/artgs" --resolution 2 --iteration 2000 --skip_test
find "$OUT/artgs" -maxdepth 4 \( -name '*.csv' -o -name 'joint_info.json' \) -print
cat "$T"
