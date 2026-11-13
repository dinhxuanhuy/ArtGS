"""Evaluation that tolerates a predicted part count / joint types different from GT.

ArtGS' eval_axis_and_state_all assumes len(pred) == len(GT) and a permutation with identical
joint types; otherwise it returns nothing or crashes. Here predicted and GT joints are matched by
the Hungarian algorithm on angle + distance (type mismatch adds a large cost but is allowed).
Per GT part we report axis angle (deg), axis distance (x10, as ArtGS), state error, type
correctness and dynamic CD (x1000, as ArtGS); unmatched GT parts are reported as missing.
Applied to every method, including the baseline, so all numbers share one protocol.

    python -m semart.eval_robust --source_path data/dta/sapien/storage_47254 \
        --result_dir outputs/.../<method>_s0/train/ours_2000
Writes <result_dir>/semart_eval.json.
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.metrics import compute_recon_error, eval_axis_and_state, read_gt
from pytorch_lightning import seed_everything

TYPE_MISMATCH_COST = 1000.0


def chamfer(pred, gt, n_trials):
    v = 0.0
    for seed in range(n_trials):
        seed_everything(seed)
        v += compute_recon_error(pred, gt, n_samples=10000, vis=False) * 1000
    return v / n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n_trials", type=int, default=3)
    a = ap.parse_args()

    pj = pickle.load(open(os.path.join(a.result_dir, "pred_joints.pkl"), "rb"))
    ptypes, pred = pj["types"], pj["joints"]
    gt = read_gt(os.path.join(a.source_path, "gt", "trans.json"))
    gtypes = ["r" if g["type"] == "revolute" else "p" for g in gt]
    P, G = len(pred), len(gt)

    C = np.zeros((P, G))
    cache = {}
    for i in range(P):
        for j in range(G):
            if ptypes[i] == gtypes[j]:
                ang, dist, st = eval_axis_and_state(pred[i], gt[j], gtypes[j])
            else:   # wrong type: only the axis direction is comparable
                d1, d2 = pred[i]["axis_direction"], gt[j]["axis_direction"]
                c = abs(np.dot(d1, d2)) / np.linalg.norm(d1) / np.linalg.norm(d2)
                ang, dist, st = np.rad2deg(np.arccos(np.clip(c, -1, 1))), 0.0, np.nan
            cache[i, j] = (float(ang), float(dist), float(st))
            # as ArtGS' eval_axis_and_state_all: angle + distance + state; parallel prismatic joints
            # (two drawers) are otherwise tied and could be matched crosswise
            C[i, j] = ang + dist + (st if st == st else 0.0) + (TYPE_MISMATCH_COST if ptypes[i] != gtypes[j] else 0.0)
    # tie-break by part position: distance between predicted and GT part mesh centroids (start state)
    mesh = os.path.join(a.result_dir, "meshes")
    gt_dir = os.path.join(a.source_path, "gt")
    dyn = [""] if G == 1 else [f"_{j}" for j in range(G)]
    try:
        import trimesh
        pc = [trimesh.load(f"{mesh}/start_{i + 1}.ply").vertices.mean(0) for i in range(P)]
        gc = [trimesh.load(f"{gt_dir}/start/start_dynamic{dyn[j]}_rotate.ply").vertices.mean(0) for j in range(G)]
        C = C + 0.1 * np.array([[np.linalg.norm(pc[i] - gc[j]) for j in range(G)] for i in range(P)])
    except Exception as e:  # meshes missing: joint-only matching
        print("centroid tie-break skipped:", e)
    rows, cols = linear_sum_assignment(C) if P and G else ([], [])
    match = {int(j): int(i) for i, j in zip(rows, cols)}

    parts = []
    for j in range(G):
        e = {"gt_part": j, "gt_type": gtypes[j]}
        if j in match:
            i = match[j]
            ang, dist, st = cache[i, j]
            e.update(pred_part=i, pred_type=ptypes[i], type_ok=ptypes[i] == gtypes[j],
                     angle=ang, distance=dist, theta_diff=st)
            cds = []
            for s in ("start", "end"):
                pm = f"{mesh}/{s}_{i + 1}.ply"
                cds.append(chamfer(pm, f"{gt_dir}/{s}/{s}_dynamic{dyn[j]}_rotate.ply", a.n_trials)
                           if os.path.exists(pm) else np.nan)
            e["CD_dynamic"] = float(np.mean(cds))
        else:
            e.update(missing=True)
        parts.append(e)

    out = {"K_pred": P + 1, "K_gt": G + 1, "part_recall": len(match) / G if G else 1.0,
           "extra_parts": max(P - G, 0), "parts": parts}
    ok = [p for p in parts if "angle" in p]
    out["type_acc"] = float(np.mean([p["type_ok"] for p in ok])) if ok else 0.0
    for k in ("angle", "distance", "theta_diff", "CD_dynamic"):
        v = [p[k] for p in ok if k in p and np.isfinite(p[k])]
        out[f"mean_{k}"] = float(np.mean(v)) if v else None
    for name, gfile, pid in (("CD_static", "static_rotate", 0), ("CD_whole", "rotate", -1)):
        vals = []
        for s in ("start", "end"):
            pm = f"{mesh}/{s}_{pid}.ply"
            if os.path.exists(pm):
                vals.append(chamfer(pm, f"{gt_dir}/{s}/{s}_{gfile}.ply", a.n_trials))
        out[name] = float(np.mean(vals)) if vals else None
    json.dump(out, open(os.path.join(a.result_dir, "semart_eval.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "parts"}))


if __name__ == "__main__":
    main()
