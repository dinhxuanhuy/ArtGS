---
name: analyze-results
description: Compare ArtGS run results against baselines and write up findings. Use after an experiment finishes, when asked how a change performed, or to update the leaderboard/journal.
---

# Analyze results

1. `python3 scripts/collect_metrics.py runs/<RUN_ID>` (colab_job.sh already does this) then
   `python3 scripts/leaderboard.py` -> `results/leaderboard.md`.
2. Baseline = official ArtGS checkpoints re-evaluated in OUR environment (`model == official`,
   from `scripts/jobs/eval_official.sh`). Compare against that, not against numbers quoted from
   the paper (different env/seed can shift them). If a scene has no baseline yet, say so.
3. Metrics (lower is better): `avg_angle` axis angle error (deg), `avg_distance` axis position
   error (prismatic joints report 0), `avg_theta_diff` part motion error, `avg_CD_static`,
   `avg_CD_dynamic`, `avg_CD_whole` Chamfer distances. PSNR/SSIM/LPIPS = -1 means appearance eval
   was skipped (`--skip_test`).
4. Be honest about noise: one seed is anecdotal. Before claiming an improvement, run >= 3 seeds
   (`--seed`) on the affected scenes and report mean +- std. Report regressions on other scenes too.
5. Smoke-test runs (short iterations) are NOT comparable to baselines; never put them in a claim.
6. Write the entry in `research/journal.md`:
   `## <date> <idea>` / Hypothesis / Change (branch, commit) / Runs (ids) / Result (table vs
   baseline) / Conclusion + next step.
