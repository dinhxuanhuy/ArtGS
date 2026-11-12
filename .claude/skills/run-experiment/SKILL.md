---
name: run-experiment
description: Run an ArtGS experiment on a Colab GPU end to end (write a job script, launch with scripts/colab_job.sh, monitor, pull results, stop the VM). Use for any training/evaluation run on Colab, smoke tests, ablations or baseline evals.
---

# Run an experiment on Colab

1. **Branch**: research code changes go on `exp/<idea>` (never on `main`). Commit or at least
   save the diff: every run records `git_sha` + `git_diff.patch`, so uncommitted edits are captured
   but committed ones are easier to reproduce.
2. **Job script**: copy `scripts/jobs/smoke_paris.sh` to `scripts/jobs/<name>.sh`. Rules:
   - start with `ln -sfn "$RUN_OUT" outputs` and use model paths `outputs/<dataset>/<subset>/<scene>/<model>`:
     ArtGS hard-codes the coarse location as `source_path.replace('data/','outputs/')`, and the
     symlink makes everything land in `$RUN_OUT` (so `collect_metrics.py` finds `result.csv`);
   - append `<stage>\t<seconds>` lines to `$RUN_OUT/stage_seconds.tsv`;
   - coarse stage must run 10000 iters (train.py loads `coarse_gs/point_cloud/iteration_10000`);
   - `render.py --iteration best` only works if training passed iteration 15000 (best is
     tracked after 15k); for short runs render `--iteration <last>`;
   - intermediate products (masks, labels, init Gaussians) go through `tools/cache.py`.
3. **Budget**: check `colab usage`. Default caps: `--max-cu 5 --max-hours 3`. Ask the user before
   anything that needs more than 5 CU, an A100/H100, or a download > 5 GB.
4. **Launch** (blocks until done; run it in the background and wait for the notification):
   ```bash
   bash scripts/colab_job.sh scripts/jobs/<name>.sh --name <name> --gpu T4 --datasets "paris"
   ```
   It always stops the VM (trap), even on failure. `--keep` leaves it running: then YOU must
   run `scripts/colab_down.sh <session>`.
5. **Check**: `runs/<RUN_ID>/{exit_code,job.log,metrics.json}`. Failed? Read `job.log`, fix,
   rerun once. Same step fails twice -> stop and report to the user (what failed, what you tried).
6. **Record**: `python3 scripts/leaderboard.py`, then add an entry to `research/journal.md`
   (hypothesis, run ids, result vs baseline, conclusion). Commit code + journal, never `runs/`.
7. **Before ending the session**: `colab sessions` must show nothing (`scripts/colab_down.sh --all`).
