# ArtGS research repo: agent guide

ArtGS (ICLR 2025): articulated-object reconstruction with 3D Gaussians. Fork of
YuLiu-LY/ArtGS at github.com/dinhxuanhuy/ArtGS (**public**). Orchestration runs on this
machine; GPU compute on Google Colab via the `colab` CLI; persistent storage on Google Drive
folder **NCKH-2027** via rclone.

## Start a session (exact commands)
```bash
cd ~/ArtGS
bash scripts/setup_local.sh              # idempotent: CLI install/upgrade, skill, auth check
colab usage                              # compute-unit balance (refuse big jobs if low)
colab sessions                           # must be empty before you start
# one-shot experiment (new VM -> bootstrap -> job -> pull results -> ALWAYS stops VM):
bash scripts/colab_job.sh scripts/jobs/smoke_paris.sh --gpu T4 --datasets "paris"
# or an interactive VM you manage yourself:
bash scripts/colab_up.sh artgs-t4 --gpu T4          # env + data ready on the VM
ssh -F scripts/ssh_config artgs-t4.colab 'cd /content/ArtGS && source /content/venv/bin/activate && ...'
bash scripts/colab_down.sh artgs-t4                 # or --all
```
If `setup_local.sh` says NOT AUTHENTICATED, ask the user to run `! colab --auth=oauth2 sessions`.
If `rclone listremotes` lacks `nckh:`, Drive caching is off (everything still works, slower);
ask the user to run `! rclone config create nckh drive scope=drive root_folder_id=<NCKH-2027 folder ID: ask the user>`.

## Environment (pinned; matches the local env that passes verify_env.py)
| | Version |
|---|---|
| Python | 3.10 (uv venv `/content/venv` on Colab; Colab's own Python 3.13 / torch 2.14 is NOT used) |
| PyTorch | 2.1.2+cu121, torchvision 0.16.2 |
| CUDA (build only) | 12.1 via micromamba (`nvidia/label/cuda-12.1.1`), host compiler g++-11 (g++-12 breaks on pybind11 headers with nvcc 12.1) |
| Must-pin | numpy 1.26.4, pillow 10.4.0 (`dataset_readers.py` breaks on Pillow >= 11), open3d 0.18 |
| Extensions | diff-gaussian-rasterization (ArtGS fork), simple-knn, pointnet2 (utils/pointnet_lib), pytorch3d V0.7.8 @75ebeea, tinycudann @3daa6e5 |
All pins: `requirements-colab.txt`, `scripts/bootstrap_colab.sh` (top). Local machine: conda env
`envs/artgs` (Quadro RTX 4000 8 GB, sm_75; `PYTHONNOUSERSITE=1` required, see SETUP_NOTES.md).

## Where things live
| What | Where |
|---|---|
| Code on VM | `/content/ArtGS` (rsynced from here, excludes in `scripts/rsync_excludes.txt`) |
| Data on VM | `/content/data/<dataset>/<subset>/<scene>` (local disk; `ArtGS/data` symlinks to it) |
| Drive (`nckh:` = NCKH-2027) | `data/archives/*.zip` (+SHA256SUMS), `data/processed/`, `data/raw/`, `checkpoints/ArtGS_ckpt/`, `_env_cache/wheels/<key>/`, `experiments/<RUN_ID>/`, `resources/papers/` |
| Local data | `data/` (gitignored), originals in `data/_raw/` |
| Run records | `runs/<RUN_ID>/` (job.sh, git diff, env, logs, metrics.json; files < 20 MB); full outputs on Drive `experiments/<RUN_ID>/` |
| Results | `results/leaderboard.md` (generated), `research/journal.md` (private notes) |
| Related papers | `docs/RELATED_WORK.md`; PDFs + text (grep-able `paper.txt`) locally in `paper_resource/` and on Drive `resources/papers/` |
| Dataset registry | `configs/datasets.json` (URLs, sizes, sha256, archive -> dataset mapping) |

Datasets: `python3 scripts/load_data.py <paris|dta|artgs|ckpt|paris_raw|dta_raw|realscan_raw>`
(verifies sha256, skips if present, refuses archives > 5 GB without `--yes`).
Training reads only from VM local disk, never directly from Drive.

## Train / eval
Pipeline per scene (`--dataset paris|dta|artgs --subset sapien|realscan --scene_name X`):
1. `train_coarse.py` (10000 iters, res 2; must be 10000: train.py loads `coarse_gs/point_cloud/iteration_10000`)
2. `train_predict.py` (joint-type prediction, res 8)
3. `train.py` (20000 iters, `--use_art_type_prior`; joint types read from `arguments/joint_types_*.json`)
4. `render.py --iteration best --skip_test` -> `<model>/train/ours_best/result.csv`
   (`best` only exists after iteration > 15000; use `--iteration <last>` for short runs)
Reference flags: `scripts/{coarse,predict,train,eval}.sh`. Job templates: `scripts/jobs/`
(`smoke_paris.sh` = quick pipeline check, `eval_official.sh` = baselines from official ckpts).
Metrics: `scripts/collect_metrics.py runs/<RUN_ID>`; table: `python3 scripts/leaderboard.py`.
Skills: `.claude/skills/{colab,run-experiment,analyze-results}`.

## Cache convention (intermediate results)
`cache/<stage>/<object_id>/<config_hash>/` via `tools/cache.py` (`CacheEntry`). `config_hash` =
sha256 of canonical JSON of EVERY parameter that affects the stage (model + version/checkpoint,
prompt, thresholds, resolution, upstream hashes). Write outputs, then `mark_complete()`.
Never reuse an entry whose hash differs; incomplete entries are wiped and recomputed.
Built CUDA wheels follow the same idea: key `py310-torch2.1.2-cu121-sm<arch>-<srchash>`.

## Rules
- **Always stop Colab runtimes** (`colab stop -s <name>` / `scripts/colab_down.sh`) when a job
  finishes or fails; idle runtimes burn compute units. Before ending any session run
  `colab sessions` and stop everything listed. `colab_job.sh` stops its VM automatically.
- Budget: default cap 5 CU per job (`--max-cu`). Ask the user before exceeding it, before using
  A100/H100, and before downloading anything > 5 GB (state the size).
- Never commit secrets, tokens or credentials (rclone.conf, OAuth tokens, `.env`). Use env vars;
  they are gitignored. The rclone config is copied to a VM only for the session.
- Never commit datasets, checkpoints, caches, runs, or results (`data/ outputs/ cache/ runs/ results/`).
- The repo is public: unpublished research (core-code edits, `improve_h1/`, review notes,
  `paper_resource/`, `research/journal.md`) stays local unless the user says otherwise.
- Do not modify ArtGS core method code for setup/infra work. Only add scripts/configs. If a
  source change is unavoidable, explain why and ask first. Research changes go on `exp/<idea>`.
- If any step fails twice, stop and report the error and what you tried instead of looping.
- `colab drivemount` cannot be used headlessly (needs a browser click per VM); use rclone.
- This machine is shared: the gcloud account here is NOT the user's; never use it for Colab/GCP.
