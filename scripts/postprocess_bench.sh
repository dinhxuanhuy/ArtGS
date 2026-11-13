#!/usr/bin/env bash
# After a benchmark run: fetch deform weights from Drive, render a part figure per scene (seed 0),
# rebuild the benchmark report over all benchmark runs.
#   bash scripts/postprocess_bench.sh runs/<RUN_ID> [method]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$1; M=${2:-sem_v2_mask}; ID=$(basename "$RUN")
FIGS=$HOME/ArtGS/research/semart/figs_bench; mkdir -p "$FIGS"
PY="$HOME/ArtGS/envs/artgs/bin/python"; export PYTHONNOUSERSITE=1
for d in $(cd "$RUN" && ls -d */*/*/${M}_s0 2>/dev/null); do
  sc=$(dirname "$d")
  rclone copy "nckh:experiments/$ID/$d/deform/iteration_20000" "$RUN/$d/deform/iteration_20000" -q 2>/dev/null
  "$PY" -m semart.vis_parts --run_dir "$RUN" --scene "$sc" --methods "$M" --views 0000,0030 \
    --out "$FIGS/${sc//\//_}.jpg" 2>&1 | grep -E "figure ->|Error" | tail -1
done
python3 semart/report_bench.py $HOME/ArtGS/runs/*_semart_bench_* --method "$M" \
  --out $HOME/ArtGS/research/semart/REPORT_BENCH.md --figs "$FIGS" > /dev/null && echo "report -> research/semart/REPORT_BENCH.md"
