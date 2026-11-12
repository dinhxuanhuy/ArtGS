#!/usr/bin/env bash
# Baselines: re-evaluate the official ArtGS checkpoints in OUR environment (no training).
# Needs datasets: "paris dta artgs ckpt". Each scene's checkpoint is copied into
# $RUN_OUT/<dataset>/<scene>/ and rendered there (render.py writes next to the model).
# SCENES="paris/blade_103706 dta/fridge_10489" limits the set; default = every sapien scene.
set -uo pipefail
CK=data/ArtGS_ckpt; RES=${RES:-1}
if [ -z "${SCENES:-}" ]; then
  SCENES=$(cd "$CK" && ls -d */* | grep -v '/real_')
fi
T=$RUN_OUT/stage_seconds.tsv; : > "$T"; fail=0
for ds_scene in $SCENES; do
  ds=${ds_scene%%/*}; scene=${ds_scene#*/}
  case "$scene" in real_*) sub=realscan;; *) sub=sapien;; esac
  dst="$RUN_OUT/$ds/$scene"; mkdir -p "$dst"; cp -r "$CK/$ds/$scene/." "$dst/"
  s=$(date +%s); echo "=== $ds/$sub/$scene"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py --dataset "$ds" --subset "$sub" \
       --scene_name "$scene" --model_path "$dst" --resolution "$RES" --iteration best --skip_test; then
    printf '%s/%s\t%d\n' "$ds" "$scene" $(( $(date +%s)-s )) >> "$T"
  else
    echo "!!! FAILED $ds/$scene"; fail=$((fail+1))
  fi
  rm -rf "$dst/deform" "$dst/point_cloud"   # keep results, drop the copied checkpoint
done
echo "failed scenes: $fail"; exit $fail
