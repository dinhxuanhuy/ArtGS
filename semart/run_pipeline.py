"""Run several methods on one scene with a shared coarse model (ArtGS venv).

    python -m semart.run_pipeline --dataset dta --subset sapien --scene storage_47254 \
        --methods artgs,sem_k,sem_mask,sem_full --semseg_dir <preprocess entry> [--iters 20000]

Every method is a preset of four switches, so new variants are one dict entry:
    k        'gt'  : K from arguments/num_slots.json (ArtGS)        | 'sem': K + centres from lift
    joints   'pred': train_predict.py on the method's slot init     | 'vlm': types from the VLM
    sem_loss  lambda of the 2D part-consistency loss (0 = off)
Model dirs: outputs/<ds>/<sub>/<scene>/{coarse_gs, lift_<k_source>, <method>_s<seed>}.
Evaluation always uses the last iteration (no GT-selected checkpoint).
"""
import argparse
import json
import os
import subprocess
import sys
import time

METHODS = {
    "artgs":    {"k": "gt",  "joints": "pred", "sem_loss": 0.0},
    "sem_k":    {"k": "sem", "joints": "pred", "sem_loss": 0.0},
    "sem_vlm":  {"k": "sem", "joints": "vlm",  "sem_loss": 0.0},
    "sem_mask": {"k": "sem", "joints": "pred", "sem_loss": None},   # None -> --lambda_sem
    "sem_full": {"k": "sem", "joints": "vlm",  "sem_loss": None},
    "gtk_mask": {"k": "gt",  "joints": "pred", "sem_loss": None},   # ablation: loss only
    # v2 (2026-11-12): K = VLM count corrected by the motion/mask eigengap (silhouette), joint types =
    # VLM unless the learned rotation clearly contradicts it, safer 2D loss (no static term, small, early off)
    "sem_v2":      {"k": "sem", "joints": "fuse", "sem_loss": 0.0, "ksrc": "vlm_motion"},
    "sem_v2_mask": {"k": "sem", "joints": "fuse", "sem_loss": 0.03, "ksrc": "vlm_motion",
                    "sem_args": ["--sem_w_static", "0", "--sem_until_iter", "10000"]},
}


def fuse_types(lift_dir):
    """VLM joint types, overridden where the motion-only prediction is unambiguous:
    VLM 'p' but learned rotation > 30 deg -> 'r'; VLM 'r' but rotation < 3 deg -> 'p'."""
    vlm = json.load(open(f"{lift_dir}/slots.json"))["joint_types"].split(",")
    ang = json.load(open(f"{lift_dir}/joint_types_pred.txt.json"))["angles_deg"]
    out, why = ["s"], []
    for i, (t, a) in enumerate(zip(vlm[1:], ang)):
        f = "r" if t == "p" and a > 30 else "p" if t == "r" and a < 3 else t
        if f != t:
            why.append(f"slot {i + 1}: VLM {t} -> {f} (rotation {a:.1f} deg)")
        out.append(f)
    with open(f"{lift_dir}/joint_types_fused.txt", "w") as fh:
        fh.write(",".join(out) + "\n")
    print("fused joint types:", ",".join(out), "|", "; ".join(why) or "VLM kept everywhere")
    return f"{lift_dir}/joint_types_fused.txt"


def run(name, cmd, log_tsv, fatal=True):
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    t = time.time()
    rc = subprocess.call(cmd)
    with open(log_tsv, "a") as f:
        f.write(f"{name}\t{int(time.time() - t)}\t{rc}\n")
    if rc != 0 and fatal:
        raise RuntimeError(f"stage {name} failed (exit {rc})")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--methods", default="artgs,sem_k,sem_mask,sem_full")
    ap.add_argument("--semseg_dir", default="")
    ap.add_argument("--k_source", default="vlm", choices=["vlm", "vlm_motion", "sam", "gt", "eigengap"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--coarse_iters", type=int, default=10000)
    ap.add_argument("--predict_iters", type=int, default=None, help="default: 5000 artgs, 3000 others")
    ap.add_argument("--res", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_sem", type=float, default=0.1)
    ap.add_argument("--sem_until_iter", type=int, default=15000)
    ap.add_argument("--extra_train_args", default="", help="appended to every train.py call")
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--only_shared", action="store_true", help="stop after coarse + lift (for staged runs)")
    ap.add_argument("--config", default="", help="JSON with any of the options above (CLI flags override it)")
    pre, _ = ap.parse_known_args()
    if pre.config:
        cfg = json.load(open(pre.config))
        unknown = set(cfg) - {x.dest for x in ap._actions} - {"_comment"}
        if unknown:
            sys.exit(f"unknown keys in {pre.config}: {sorted(unknown)}")
        ap.set_defaults(**{k: v for k, v in cfg.items() if k != "_comment"})
        for x in ap._actions:            # values given by the config satisfy required=True
            if x.dest in cfg:
                x.required = False
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method {m}; known: {', '.join(METHODS)}")
    py = sys.executable
    root = f"outputs/{a.dataset}/{a.subset}/{a.scene}"
    os.makedirs(root, exist_ok=True)
    tsv = f"{root}/stage_seconds.tsv"
    common = ["--dataset", a.dataset, "--subset", a.subset, "--scene_name", a.scene, "--random_bg_color"]
    p_iters = a.predict_iters or (5000 if a.dataset == "artgs" else 3000)
    needs_sem = any(METHODS[m]["k"] == "sem" or METHODS[m]["sem_loss"] != 0.0 for m in methods)
    if needs_sem and not a.semseg_dir:
        sys.exit("--semseg_dir is required for semantic methods")

    # S1 coarse (shared)
    if not os.path.exists(f"{root}/coarse_gs/point_cloud/iteration_{a.coarse_iters}/point_cloud.ply"):
        run("coarse", [py, "train_coarse.py", *common, "--model_path", f"{root}/coarse_gs", "--resolution", "2",
                       "--iterations", str(a.coarse_iters), "--opacity_reg_weight", "0.1", "--quiet"], tsv)

    # S2 lift (one per K source used by the methods; the default one also feeds the 2D loss of 'gt' methods)
    ksrc_of = lambda m: METHODS[m].get("ksrc", a.k_source)
    lift_of = lambda m: f"{root}/lift_{ksrc_of(m)}"
    for ks in sorted({ksrc_of(m) for m in methods} | {a.k_source}) if needs_sem else []:
        ld = f"{root}/lift_{ks}"
        if os.path.exists(f"{ld}/slots.json"):
            continue
        cmd = [py, "-m", "semart.lift", "--source_path", f"data/{a.dataset}/{a.subset}/{a.scene}",
               "--coarse_dir", f"{root}/coarse_gs", "--semseg_dir", a.semseg_dir, "--out_dir", ld,
               "--k_source", ks, "--seed", str(a.seed)]
        if ks == "gt":
            gt_k = json.load(open("arguments/num_slots.json"))[a.dataset][a.subset][a.scene]
            cmd += ["--gt_k", str(gt_k)]
        run(f"lift_{ks}", cmd, tsv)

    if a.only_shared:
        print("shared stages done")
        return

    # S3 predict (one per slot init that needs it)
    pred_done = set()
    for m in methods:
        cfg = METHODS[m]
        key = "gt" if cfg["k"] == "gt" else lift_of(m)
        if cfg["joints"] not in ("pred", "fuse") or key in pred_done:
            continue
        lift_dir = lift_of(m)
        if cfg["k"] == "gt":
            run("predict_gt", [py, "train_predict.py", *common, "--model_path", f"{root}/pred", "--eval",
                               "--resolution", "8", "--iterations", str(p_iters), "--densify_grad_threshold",
                               "0.001", "--coarse_name", "coarse_gs", "--seed", str(a.seed), "--quiet",
                               "--joint_types_out", f"{root}/pred/joint_types_pred.txt"], tsv)
        else:
            run("predict_sem", [py, "train_predict.py", *common, "--model_path", f"{lift_dir}/pred", "--eval",
                                "--resolution", "8", "--iterations", str(p_iters), "--densify_grad_threshold",
                                "0.001", "--coarse_name", "coarse_gs", "--seed", str(a.seed), "--quiet",
                                "--slot_init", lift_dir, "--joint_types_out", f"{lift_dir}/joint_types_pred.txt"], tsv)
        pred_done.add(key)

    # S4 train + S5 render per method
    for m in methods:
        cfg = METHODS[m]
        mp = f"{root}/{m}_s{a.seed}"
        lam = a.lambda_sem if cfg["sem_loss"] is None else cfg["sem_loss"]
        lift_dir = lift_of(m) if cfg["k"] == "sem" else f"{root}/lift_{a.k_source}"
        mask_dir = f"{lift_dir}/masks_filtered"      # motion-filtered masks for the 2D loss
        cmd = [py, "train.py", *common, "--model_path", mp, "--eval", "--resolution", str(a.res),
               "--iterations", str(a.iters), "--coarse_name", "coarse_gs", "--seed", str(a.seed),
               "--use_art_type_prior", "--densify_grad_threshold", "0.001", "--no_gt_select", "--quiet"]
        if cfg["k"] == "gt":
            cmd += ["--joint_types_file", f"{root}/pred/joint_types_pred.txt"]
        if cfg["k"] == "sem":
            cmd += ["--slot_init", lift_dir]
            if cfg["joints"] == "vlm":
                cmd += ["--joint_source", "vlm"]
            elif cfg["joints"] == "fuse":
                cmd += ["--joint_types_file", fuse_types(lift_dir)]
            else:
                cmd += ["--joint_types_file", f"{lift_dir}/joint_types_pred.txt"]
        if lam > 0:
            cmd += ["--lambda_sem", str(lam), "--sem_mask_dir", mask_dir, "--sem_until_iter", str(a.sem_until_iter)]
            cmd += cfg.get("sem_args", [])        # later flags override earlier ones
        cmd += a.extra_train_args.split()
        run(f"train_{m}", cmd, tsv)
        if not a.skip_render:
            # render.py's own CSV eval fails when K or types differ from GT; meshes are written first
            run(f"render_{m}", [py, "render.py", "--dataset", a.dataset, "--subset", a.subset, "--scene_name",
                                a.scene, "--model_path", mp, "--resolution", str(a.res), "--iteration",
                                str(a.iters), "--skip_test"], tsv, fatal=False)
            run(f"eval_{m}", [py, "-m", "semart.eval_robust", "--source_path",
                              f"data/{a.dataset}/{a.subset}/{a.scene}",
                              "--result_dir", f"{mp}/train/ours_{a.iters}"], tsv, fatal=False)
    print(f"\nall methods done: {methods}")


if __name__ == "__main__":
    main()
