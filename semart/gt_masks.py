"""Exact GT part label maps by ray casting the GT part meshes (open3d; correct occlusion).

Label 0 = static base, j+1 = GT dynamic part j (order of gt/trans.json). Used for mask metrics and figures.
    python -m semart.gt_masks --source_path data/dta/sapien/storage_47254 --out_dir <dir> [--views 0000,0030]
Writes <out_dir>/<state>/<frame>.npz with `labels` (uint8 HxW) and `types` (joint type per part).
"""
import argparse
import json
import os

import numpy as np

from semart.lift import load_cameras

STATES = ("start", "end")


def part_meshes(source_path, state):
    gt = f"{source_path}/gt/{state}"
    n = len(json.load(open(f"{source_path}/gt/trans.json"))["trans_info"]) \
        if isinstance(json.load(open(f"{source_path}/gt/trans.json"))["trans_info"], list) else 1
    files = [f"{gt}/{state}_static_rotate.ply"]
    files += [f"{gt}/{state}_dynamic_rotate.ply"] if n == 1 else [f"{gt}/{state}_dynamic_{k}_rotate.ply" for k in range(n)]
    return files


def render_labels(source_path, state, names=None, W=800, H=800):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    lut = {}
    for lab, f in enumerate(part_meshes(source_path, state)):
        m = o3d.io.read_triangle_mesh(f)
        lut[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))] = lab
    table = np.zeros(max(lut) + 2, np.uint8)
    for g, l in lut.items():
        table[g] = l
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    out = {}
    for cam in load_cameras(source_path, state):
        if names is not None and cam["name"] not in names:
            continue
        fx, fy = W / (2 * np.tan(cam["fovx"] / 2)), H / (2 * np.tan(cam["fovy"] / 2))
        d = np.stack([(jj - W / 2) / fx, (ii - H / 2) / fy, np.ones_like(jj)], -1).reshape(-1, 3)
        c2w = np.linalg.inv(cam["w2c"])
        dw = d @ c2w[:3, :3].T
        ow = np.broadcast_to(c2w[:3, 3], dw.shape)
        hit = scene.cast_rays(o3d.core.Tensor(np.concatenate([ow, dw], 1).astype(np.float32)))
        gid = hit["geometry_ids"].numpy().astype(np.int64)
        ok = np.isfinite(hit["t_hit"].numpy())
        out[cam["name"]] = np.where(ok, table[np.clip(gid, 0, len(table) - 1)], 0).reshape(H, W).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--views", default="", help="comma list of frame names (default: all)")
    a = ap.parse_args()
    names = set(a.views.split(",")) if a.views else None
    from utils.metrics import read_gt
    types = ["r" if g["type"] == "revolute" else "p" for g in read_gt(f"{a.source_path}/gt/trans.json")]
    for s in STATES:
        os.makedirs(f"{a.out_dir}/{s}", exist_ok=True)
        for name, lab in render_labels(a.source_path, s, names).items():
            np.savez_compressed(f"{a.out_dir}/{s}/{name}.npz", labels=lab, types=np.array(types))
    print("GT masks ->", a.out_dir)


if __name__ == "__main__":
    main()
