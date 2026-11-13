# SemArt: semantic part priors for two-state articulated reconstruction

SemArt extends **ArtGS** (ICLR 2025). ArtGS needs the number of parts K (from a per-object list), initialises
part centres by spectral clustering of moving Gaussians, and predicts joint types from a short motion-only run.
SemArt replaces these priors with evidence from a vision-language model (**Qwen3-VL**) and a promptable
segmenter (**SAM3**), **verified by the observed motion between the two states**, and adds a 2D
part-consistency loss during training. No ground-truth part count or joint type is used.

The same code trains **plain ArtGS** or **SemArt**: the method is chosen by a config (or `--methods`).

| | ArtGS | SemArt (`sem_v2_mask`) |
|---|---|---|
| Number of parts K | given (`arguments/num_slots.json`) | Qwen3-VL count, corrected by the eigengap of the motion/mask affinity (silhouette) |
| Part centres | spectral clustering of moving Gaussians (positions only) | SAM3 masks from all views lifted onto moving Gaussians (`semart/lift.py`) |
| Joint types | `train_predict.py` (learned rotation > 10° → revolute) | Qwen3-VL, overridden where the learned rotation clearly contradicts it |
| Extra loss | – | 2D part consistency on rendered slot probabilities (`semart/mask_loss.py`) |
| Checkpoint for evaluation | last iteration (`--no_gt_select`) | last iteration |

Design notes: [`docs/semart/DESIGN.md`](docs/semart/DESIGN.md).

## 1. Environments

Two Python environments are used:

| Stage | Environment | Why |
|---|---|---|
| VLM + SAM3 preprocessing (`semart.preprocess`) | Python ≥ 3.11, torch ≥ 2.5, `transformers` ≥ 4.57, `accelerate` | SAM3 / Qwen3-VL need recent torch and transformers |
| Everything else (ArtGS training, lift, evaluation) | the ArtGS env: Python 3.10, torch 2.1.2+cu121 (`requirements-colab.txt`, `scripts/bootstrap_colab.sh`) | unchanged ArtGS CUDA extensions |

On Google Colab the preprocessing runs in Colab's own Python and training in the ArtGS venv; the job scripts
below handle both. Locally, create a second env for preprocessing, e.g.
`python3.12 -m venv ~/semart-pre && ~/semart-pre/bin/pip install torch "transformers>=4.57" accelerate pillow numpy`.

**SAM3 is gated on HuggingFace**: request access at <https://huggingface.co/facebook/sam3>, then
`hf auth login` (the token is read from `~/.cache/huggingface/token`; never commit it).
Without access use `--segmenter qwen_sam2` (Qwen3-VL boxes + SAM2.1; slower, ungated).

Data: `python3 scripts/load_data.py paris|dta|artgs` (see the main README).

## 2. Train ArtGS or SemArt on one object

Run from the repo root. Paths follow ArtGS: data in `data/<dataset>/<subset>/<scene>`, outputs in
`outputs/<dataset>/<subset>/<scene>/<method>_s<seed>`.

**ArtGS** (no preprocessing needed):
```bash
python -m semart.run_pipeline --config configs/semart/artgs.json \
    --dataset dta --subset sapien --scene fridge_10489
```

**SemArt**: first the VLM + SAM3 masks (preprocessing env), then the pipeline (ArtGS env):
```bash
# 1) part list + per-view masks -> prints SEMSEG_DIR=<cache entry>
~/semart-pre/bin/python -m semart.preprocess --scene_dir data/dta/sapien/fridge_10489 \
    --object_id dta/sapien/fridge_10489 --segmenter sam3 --vlm Qwen/Qwen3-VL-4B-Instruct

# 2) coarse -> lift -> joint types -> train -> render -> evaluate
python -m semart.run_pipeline --config configs/semart/semart.json \
    --dataset dta --subset sapien --scene fridge_10489 --semseg_dir <SEMSEG_DIR>
```

**Both on the same coarse model** (paired comparison): `--config configs/semart/compare.json`.

Any config key can be overridden on the command line, e.g. `--seed 1 --iters 10000 --res 1`.

### Methods (`--methods a,b,...`)

Each method is a preset in `METHODS` in [`semart/run_pipeline.py`](semart/run_pipeline.py); a new variant
is one dict entry.

| Method | K / centres | Joint types | 2D loss |
|---|---|---|---|
| `artgs` | GT K / spectral | motion (`train_predict.py`) | – |
| `sem_v2_mask` (**SemArt**) | VLM + motion eigengap / lift | fused VLM + motion | yes (no static term, λ 0.03, until 10k) |
| `sem_v2` | as above | fused | – |
| `sem_k`, `sem_vlm`, `sem_mask`, `sem_full` | v1 ablations (K from VLM only) | motion / VLM | –/yes |
| `gtk_mask` | GT K / spectral | motion | yes |

### What each stage writes

| Stage | Output |
|---|---|
| `semart.preprocess` | `cache/semseg/<object>/<hash>/`: `vlm.json` (parts, counts, joint types), `masks/<state>/<frame>.npz`, `vis/` |
| coarse | `coarse_gs/` (ArtGS) |
| `semart.lift` | `lift_<k_source>/`: `slots.json` (K, joint types), `center_info.npy`, `masks_filtered/`, `lift_report.json` |
| train / render | `<method>_s<seed>/` (ArtGS layout), meshes in `train/ours_<iters>/meshes/` |
| `semart.eval_robust` | `train/ours_<iters>/semart_eval.json` |

## 3. Evaluation

`semart/eval_robust.py` matches predicted and GT joints with the Hungarian algorithm (cost: axis angle +
axis distance + state error, mesh-centroid tie-break). Unlike ArtGS' own evaluation it also works when the
predicted K or joint types differ from GT; it reports part recall, joint-type accuracy, axis angle / position,
state error and Chamfer distances (static, per movable part, whole), in the same units as ArtGS.
It is applied to every method, ArtGS included.

Tables and figures:
```bash
python3 semart/compare.py runs/<RUN_ID>                       # method table of a run
python3 semart/report_bench.py runs/<RUN>... --out report.md  # vs ArtGS / REArtGS / REArtGS++ paper numbers
python -m semart.vis_parts --run_dir runs/<RUN_ID> --scene artgs/sapien/storage_47648 \
    --methods artgs,sem_v2_mask --out fig.jpg                 # input | GT parts | SAM3 | each method
```
Paper numbers used by `report_bench.py` and their protocols: [`semart/paper_baselines.json`](semart/paper_baselines.json).

## 4. Running on Google Colab

Job files in `scripts/jobs/` set a few variables and source `scripts/jobs/semart_lib.sh`
(preprocess in Colab's Python with Drive caching, then the pipeline in the ArtGS venv):
```bash
bash scripts/colab_job.sh scripts/jobs/semart_smoke.sh --gpu T4 --datasets "dta"        # quick check
bash scripts/colab_job.sh scripts/jobs/semart_bench_multi.sh --gpu T4 --datasets "artgs" # 5 objects × 3 seeds
```
Variables: `SCENES`, `METHODS`, `SEEDS`, `ITERS`, `RES`, `SEGMENTER`, `VLM`, `PARALLEL=1` + `MAXPAR=2`
(several objects at once on one GPU; coarse and lift stay sequential). `colab_job.sh` records the run in
`runs/<RUN_ID>/`, uploads full outputs to Drive and always stops the VM; `scripts/run_chain.sh` runs several
jobs one after another. See the main README for the Colab setup.

## 5. Code map

| File | Role |
|---|---|
| `semart/preprocess.py` | Qwen3-VL part list (8 views of both states), SAM3 (fp16, shared vision features) or Qwen3-VL + SAM2.1 masks |
| `semart/lift.py` | 2D motion filter, count-1 merging, co-occurrence affinity, spectral clustering, K selection, centres, joint types |
| `semart/mask_loss.py` | id-free 2D part-consistency loss |
| `semart/args.py` | flags shared by `train.py` / `train_predict.py` (defaults = original ArtGS) |
| `semart/run_pipeline.py` | method presets, stage runner, `--config` |
| `semart/eval_robust.py`, `dump_joints.py`, `gt_masks.py` | evaluation, joints from checkpoints, ray-cast GT part masks |
| `semart/compare.py`, `report.py`, `report_bench.py`, `vis_parts.py` | tables and figures |

Changes to ArtGS files are small and off by default: `train.py` / `train_predict.py` (new flags),
`scene/artgs.py` (returns soft slot probabilities, records predicted rotation angles), `render.py`
(writes `pred_joints.pkl`).

## Known limitations

- K still depends on the VLM naming the parts: on `oven_101908` Qwen3-VL reported one oven door where there are
  two, and merging the instances of a "count 1" category fused them.
- The preprocessing needs a GPU with ≥ 15 GB (Qwen3-VL-4B fp16 + SAM3).

## Credits

ArtGS: Liu et al., ICLR 2025 (<https://github.com/YuLiu-LY/ArtGS>, Apache-2.0). SAM3 (Meta), Qwen3-VL (Qwen team).
