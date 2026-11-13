"""Figure: input | GT parts | SAM3 (filtered) | part segmentation of every method, same views.

Predicted slots are coloured by the GT part they were matched to in semart_eval.json, so colours are
comparable across methods (grey = static base, magenta = slot matched to no GT part).

    python -m semart.vis_parts --run_dir runs/<RUN_ID> --scene dta/sapien/storage_47254 \
        --methods artgs,sem_k,sem_vlm,sem_mask,sem_full --views 0000,0030,0060 --out fig.jpg
Needs each <run_dir>/<scene>/<method>_s0/{cfg_args, deform/, point_cloud/iteration_<it>/point_cloud.ply}.
"""
import argparse
import json
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from PIL import Image, ImageDraw

STATES = ("start", "end")
PALETTE = np.array([(150, 150, 150), (255, 140, 0), (30, 144, 255), (50, 205, 50), (220, 20, 60),
                    (148, 103, 189), (255, 215, 0), (0, 206, 209), (240, 128, 128)], np.uint8)
UNMATCHED = np.array((255, 0, 255), np.uint8)


def overlay(rgb, lab, colors):
    o = rgb.astype(float).copy()
    m = lab > 0
    o[m] = 0.3 * o[m] + 0.7 * colors[lab[m]]
    s = lab == 0
    o[s] = 0.75 * o[s] + 0.25 * PALETTE[0]
    return o.astype(np.uint8)


def load_model(model_dir, iteration):
    from arguments import ModelParams, PipelineParams
    from scene import GaussianModel, DeformModel
    cfg = eval(open(os.path.join(model_dir, "cfg_args")).read(), {"Namespace": Namespace})
    cfg.model_path = model_dir
    parser = ArgumentParser()
    lp, pp = ModelParams(parser, sentinel=True), PipelineParams(parser)
    g = GaussianModel(cfg.sh_degree)
    g.load_ply(os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    d = DeformModel(lp.extract(cfg))
    assert d.load_weights(model_dir, iteration=iteration), model_dir
    d.update(20000)
    return cfg, g, d, pp.extract(cfg)


def render_parts(cam, g, d_values, pipe):
    """Per-pixel slot id = argmax of alpha-composited one-hot slot colours (3 slots per pass)."""
    from semart.mask_loss import SemMaskSupervisor
    K = int(d_values["mask"].max()) + 1
    onehot = torch.nn.functional.one_hot(d_values["mask"], K).float()
    P, alpha = SemMaskSupervisor.render_probs(cam, g, d_values["d_xyz"], d_values["d_rotation"], onehot)
    lab = P.argmax(0).cpu().numpy().astype(np.uint8)
    return np.where(alpha.cpu().numpy() > 0.5, lab, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--methods", default="artgs,sem_k,sem_vlm,sem_mask,sem_full")
    ap.add_argument("--views", default="0000,0030,0060")
    ap.add_argument("--iteration", default="20000")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=240)
    a = ap.parse_args()
    from gaussian_renderer import render
    from semart.gt_masks import render_labels
    from semart.lift import load_cameras  # noqa: F401  (camera order check)
    from scene.dataset_readers import readInfo_2states
    from utils.camera_utils import cameraList_from_camInfos

    views = a.views.split(",")
    src = os.path.abspath(f"data/{a.scene}")   # the reader joins paths and needs an absolute root
    root = os.path.join(a.run_dir, a.scene)
    methods = [m for m in a.methods.split(",") if os.path.isdir(f"{root}/{m}_s{a.seed}")]
    cfg0 = eval(open(f"{root}/{methods[0]}_s{a.seed}/cfg_args").read(), {"Namespace": Namespace})
    info = readInfo_2states(src, cfg0.white_background, True)
    cam_args = Namespace(resolution=cfg0.resolution, data_device="cuda", load2gpu_on_the_fly=False)
    cams = {}
    for s in (0, 1):
        sel = [ci for ci in info.train_cameras_2s[s] if ci.image_name in views]
        for c in cameraList_from_camInfos(sel, 1.0, cam_args):
            cams[(s, c.image_name)] = c
    gt = {s: render_labels(src, STATES[s], set(views)) for s in (0, 1)}

    cols = {"input": {}, "GT parts": {}}
    sam_dirs = [p for p in (f"{root}/lift_vlm_motion/masks_filtered", f"{root}/lift_vlm/masks_filtered")
                if os.path.isdir(p)][:1]
    if sam_dirs:
        cols["SAM3 (filtered)"] = {}
    for (s, v), cam in cams.items():
        rgb = (cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        H, W = rgb.shape[:2]
        cols["input"][(s, v)] = rgb
        g = np.array(Image.fromarray(gt[s][v]).resize((W, H), Image.NEAREST))
        cols["GT parts"][(s, v)] = overlay(rgb, g, PALETTE)
        if sam_dirs:
            p = f"{sam_dirs[0]}/{STATES[s]}/{v}.npz"
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                lab = np.array(Image.fromarray(z["labels"]).resize((W, H), Image.NEAREST))
                cats = [str(c) for c in z["cats"]]
                cc = np.array([PALETTE[0]] + [(255, 140, 0) if c in ("door", "lid", "flap") else (30, 144, 255)
                                              for c in cats] + [UNMATCHED] * 255, np.uint8)
                cols["SAM3 (filtered)"][(s, v)] = overlay(rgb, lab, cc)

    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        for m in methods:
            md = f"{root}/{m}_s{a.seed}"
            ev = json.load(open(f"{md}/train/ours_{a.iteration}/semart_eval.json"))
            colors = np.tile(UNMATCHED, (256, 1))
            colors[0] = PALETTE[0]
            for p in ev["parts"]:
                if "pred_part" in p:
                    colors[p["pred_part"] + 1] = PALETTE[1 + p["gt_part"]]
            cfg, g, d, pipe = load_model(md, a.iteration)
            dv = d.step(g, is_training=False)
            name = f"{m} (K={ev['K_pred']}, type {ev['type_acc']:.0%})"
            cols[name] = {}
            for (s, v), cam in cams.items():
                rgb = cols["input"][(s, v)]
                cols[name][(s, v)] = overlay(rgb, render_parts(cam, g, dv[s], pipe), colors)
            del g, d
            torch.cuda.empty_cache()

    T, head = a.tile, 22
    keys = sorted(cams)
    fig = Image.new("RGB", (len(cols) * T, head + len(keys) * (T + head)), "white")
    dr = ImageDraw.Draw(fig)
    for j, c in enumerate(cols):
        dr.text((j * T + 4, 5), c, fill="black")
    for i, k in enumerate(keys):
        y = head + i * (T + head)
        dr.text((4, y + 4), f"{a.scene}  {STATES[k[0]]}/{k[1]}", fill="black")
        for j, c in enumerate(cols):
            if k in cols[c]:
                fig.paste(Image.fromarray(cols[c][k]).resize((T, T)), (j * T, y + head - 4))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.save(a.out, quality=90)
    print("figure ->", a.out)


if __name__ == "__main__":
    main()
