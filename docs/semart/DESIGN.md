# SemArt: semantic part priors (VLM + promptable segmentation) for two-state ArtGS

Status: v1 design, 2026-11-12. Branch `exp/semart` (local only: unpublished research).

## Claim (paper thesis, option 1 "automation")
ArtGS needs three pieces of prior knowledge per object: the number of parts K
(`arguments/num_slots.json`), a part-centre initialisation (spectral clustering on moving
Gaussians, which the ArtGS authors fix by hand on hard scenes) and joint types
(`train_predict.py`). We replace all three with evidence from a VLM (Qwen3-VL) and a promptable
segmenter (SAM3, or Qwen3-VL boxes + SAM2.1), **verified by the observed two-state motion**, and
add a 2D part-consistency loss during training. Target: no GT priors, accuracy equal to or better
than ArtGS with priors.

Difference from Rest2Art (ECCV'26), which also pairs a VLM with SAM3: Rest2Art has a single rest
state and must imagine motion with video diffusion. We have real motion from two states, so
(a) motion rejects semantic parts that did not move and (b) semantics split parts with identical
motion (an ArtGS failure case, its Appendix C).

## Pipeline
```
S0 preprocess (Colab system python, modern torch)   semart/preprocess.py
   VLM on a multi-view sheet of both states -> part list {name, count, joint}
   segmenter per train view and per part name -> instance label maps + scores
   -> cache/semseg/<ds>/<sub>/<scene>/<hash>/{vlm.json, masks/<state>/<frame>.npz, summary.json}
S1 coarse (unchanged ArtGS train_coarse.py)
S2 lift (ArtGS venv)                                  semart/lift.py
   moving Gaussians of the larger-motion state, recomputed with a FIXED ratio (no GT K)
   -> project into every view of that state with a z-buffer -> per-view instance ids
   -> co-occurrence affinity (same mask in views where both are visible) -> spectral clustering
      into K-1 parts (K-1 from semantic counts) -> drop parts with too few moving points
   -> center_info.npy (same format as ArtGS) + slots.json {num_slots, joint_types, categories}
S3 predict (optional)   train_predict.py --slot_init <dir>   (joint types from motion, as ArtGS)
S4 train                train.py --slot_init <dir> [--joint_source vlm] [--lambda_sem w]
S5 render/eval          render.py --iteration <last>    (no GT-selected "best" checkpoint)
```

## Semantic consistency loss (S4)
Per training view with masks: render the soft slot probabilities of the Gaussians (K channels,
geometry detached, so only the segmentation field gets gradients), normalise by alpha.
For every 2D instance j (pixel set Omega_j, mean distribution p_j):
- intra: mean over Omega_j of ||P - p_j||^2 (one instance = one slot)
- inter: sum over pairs j<l of <p_j, p_l> (two movable instances = different slots)
- static: mean_j p_j[0] (a movable instance is not the static base)
Permutation invariant: no cross-view instance ids needed.

## Methods (one flag: `python -m semart.run_pipeline --methods a,b,...`)
| method | K | centres | joint types | sem loss |
|---|---|---|---|---|
| `artgs` | GT json | spectral | predict | - |
| `sem_k` | semantic | lifted | predict (sem init) | - |
| `sem_vlm` | semantic | lifted | VLM | - |
| `sem_mask` | semantic | lifted | predict | yes |
| `sem_full` | semantic | lifted | VLM | yes |
All methods share one coarse model per scene, evaluated at the last iteration, seed 0 (3 seeds
for the paper table).

## Metrics
ArtGS `result.csv` (axis angle, axis position, state error, CD static/dynamic/whole) plus
lift diagnostics: K accuracy, per-part assignment purity against GT part meshes.
