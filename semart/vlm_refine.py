"""SemArt VLM refinement after the coarse stage (runs in the preprocessing env: Qwen3-VL + SAM3).

Two modes, both producing a new semseg entry that semart.lift can consume:
  views     (B) re-ask the VLM with better views: angles chosen so that every moving cluster is seen and
                the angles differ by >= --min_sep deg; each angle is shown as a start|end pair with the
                moving region tinted. Real captures: the end image is the nearest end camera.
  feedback  (C) as `views`, plus a VLM <-> SAM3 <-> motion loop: the moving clusters of a previous lift
                are drawn in distinct colours and numbered, and the VLM is told which regions its part list
                leaves unexplained / where its count disagrees with the motion, then asked for a revised list.
After the VLM answers, SAM3 is re-run on every view with the (new) part names.

    python -m semart.vlm_refine --scene_dir data/artgs/sapien/oven_101908 --object_id artgs/sapien/oven_101908 \
        --coarse_dir outputs/.../coarse_gs --lift_dir outputs/.../lift_vlm_motion \
        --semseg_dir <round-0 entry> --mode feedback
Prints SEMSEG_DIR=<new entry>.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.cache import CacheEntry  # noqa: E402
from semart.lift import load_cameras, project, state_points, moving_mask  # noqa: E402
from semart import preprocess as pp  # noqa: E402

STATES = ("start", "end")
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60)]
REFINE_VERSION = "r1"

VIEWS_PROMPT = (
    "Each row shows ONE object from one viewpoint in articulation state A (left) and state B (right); "
    "{n} viewpoints are shown. Regions tinted red are the parts that move between the two states.\n"
    "List the object's movable parts: rigid parts that move relative to the main body. Do not list the main "
    "body, and do not list small parts rigidly attached to a movable part (handles, knobs). Group identical "
    "parts under one name with a count. Every red region must belong to some listed part.\n"
    "joint: \"revolute\" if the part rotates about a hinge (doors, lids, flaps, even if they fold down), "
    "\"prismatic\" if it slides along a straight line (drawers, sliding trays).\n"
    "Answer with JSON only, no prose:\n"
    '{{"object": "<category>", "movable_parts": [{{"name": "<short noun usable as a segmentation prompt>", '
    '"count": <int>, "joint": "revolute" or "prismatic"}}]}}')

FEEDBACK_PROMPT = (
    "Each row shows ONE object from one viewpoint in articulation state A (left) and state B (right); "
    "{n} viewpoints are shown. A motion analysis of the two states found {k} separately moving regions, "
    "drawn in colour and numbered: {regions}.\n"
    "An earlier answer listed these movable parts: {prev}.\n"
    "Problems found: {problems}\n"
    "Look at the numbered regions and give a corrected list of movable parts, so that every numbered region "
    "is one part (a region is one rigid part) and every listed part is one of the regions. Use names a "
    "segmentation model would understand (e.g. door, drawer, lid, cap, blade, seat).\n"
    "joint: \"revolute\" if the part rotates about a hinge, \"prismatic\" if it slides in a straight line.\n"
    "Answer with JSON only, no prose:\n"
    '{{"object": "<category>", "movable_parts": [{{"name": "<short noun>", "count": <int>, '
    '"joint": "revolute" or "prismatic"}}]}}')


def cam_center(cam):
    return np.linalg.inv(cam["w2c"])[:3, 3]


def visible_counts(cam, X, labels, n_lab, W, H, tol, grid=8):
    """Number of visible points per label (z-buffer on X)."""
    u, v, z = project(cam, X, W, H)
    ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    gw, gh = W // grid, H // grid
    cell = np.clip((v / grid).astype(int), 0, gh - 1) * gw + np.clip((u / grid).astype(int), 0, gw - 1)
    zb = np.full(gw * gh, np.inf)
    np.minimum.at(zb, cell[ok], z[ok])
    vis = ok & (z <= zb[cell] + tol)
    return np.bincount(labels[vis & (labels >= 0)], minlength=n_lab), u, v, vis


def select_views(cams, P, lab_full, n_lab, W, H, tol, n_views, min_sep):
    """Greedy: first the best view of every moving cluster, then the most-coverage views; all >= min_sep apart."""
    counts = []
    for cam in cams:
        c, *_ = visible_counts(cam, P, lab_full, n_lab, W, H, tol)
        counts.append(c)
    counts = np.array(counts, float)                                   # [n_cams, n_lab]
    norm = counts / np.maximum(counts.max(0, keepdims=True), 1)
    dirs = np.array([cam_center(c) for c in cams])
    dirs = dirs - P.mean(0)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    chosen = []

    def ok(i):
        return all(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1))) >= min_sep for j in chosen)
    for k in np.argsort(-counts.sum(0)):                               # big clusters first
        for i in np.argsort(-norm[:, k]):
            if ok(i):
                if i not in chosen:
                    chosen.append(int(i))
                break
        if len(chosen) >= n_views:
            break
    for i in np.argsort(-norm.sum(1)):
        if len(chosen) >= n_views:
            break
        if i not in chosen and ok(i):
            chosen.append(int(i))
    return chosen[:n_views]


def region_masks(u, v, vis, labels, n_lab, W, H, fg, radius=4, min_area_frac=0.002):
    """Splat visible moving points into per-label masks, keep only solid regions (drops stray points that
    see through sparse surfaces or are static points marked as moving)."""
    from scipy import ndimage
    out = {}
    for lab in range(n_lab):
        sel = vis & (labels == lab)
        m = np.zeros((H, W), bool)
        ui, vi = np.clip(u[sel].astype(int), 0, W - 1), np.clip(v[sel].astype(int), 0, H - 1)
        m[vi, ui] = True
        m = ndimage.binary_dilation(m, iterations=radius) & fg
        m = ndimage.binary_opening(m, iterations=radius)
        cc, n = ndimage.label(m)
        if n:
            area = ndimage.sum(m, cc, range(1, n + 1))
            keep = [i + 1 for i, a_ in enumerate(area) if a_ >= min_area_frac * W * H]
            m = np.isin(cc, keep)
            m = ndimage.binary_closing(m, iterations=radius)
        out[lab] = m
    return out


def tint(img, masks, colors, numbers=False):
    from scipy import ndimage
    im = np.array(img).astype(float)
    for lab, m in masks.items():
        if m.any():
            im[m] = 0.45 * im[m] + 0.55 * np.array(colors[lab])
    img = Image.fromarray(im.astype(np.uint8))
    if numbers:
        d = ImageDraw.Draw(img)
        for lab, m in masks.items():
            if not m.any():
                continue
            cc, n = ndimage.label(m)
            big = 1 + int(np.argmax(ndimage.sum(m, cc, range(1, n + 1))))
            y, x = ndimage.center_of_mass(cc == big)
            d.rectangle([x - 14, y - 14, x + 14, y + 14], fill=(255, 255, 255), outline=colors[lab], width=3)
            d.text((x - 4, y - 7), str(lab + 1), fill=(0, 0, 0))
    return img


def pair_image(a, b, size):
    a, b = a.resize((size, size)), b.resize((size, size))
    out = Image.new("RGB", (2 * size + 8, size), "white")
    out.paste(a, (0, 0))
    out.paste(b, (size + 8, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--object_id", required=True)
    ap.add_argument("--coarse_dir", required=True)
    ap.add_argument("--lift_dir", required=True, help="previous lift (moving_labels.npz, lift_report.json, slots.json)")
    ap.add_argument("--semseg_dir", required=True, help="previous semseg entry (its vlm.json for the feedback)")
    ap.add_argument("--mode", choices=["views", "feedback"], default="feedback")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--sam3", default="facebook/sam3")
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--min_sep", type=float, default=45.0)
    ap.add_argument("--img_size", type=int, default=336, help="side of each image in a start|end pair")
    ap.add_argument("--view_stride", type=int, default=1)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--dyn_ratio", type=float, default=0.02)
    ap.add_argument("--dry_run", default="", help="only write the VLM input images + prompt to this dir")
    a = ap.parse_args()

    prev = json.load(open(os.path.join(a.semseg_dir, "vlm.json")))["parsed"]
    lift_rep = json.load(open(os.path.join(a.lift_dir, "lift_report.json")))
    ml = np.load(os.path.join(a.lift_dir, "moving_labels.npz"))
    config = {"code": f"semart-refine-{REFINE_VERSION}", "mode": a.mode, "vlm": a.vlm, "seg_model": a.sam3,
              "seg_precision": "fp16", "n_views": a.n_views, "min_sep": a.min_sep, "img_size": a.img_size,
              "view_stride": a.view_stride, "score_thresh": a.score_thresh, "dyn_ratio": a.dyn_ratio,
              "prev_parts": prev, "lift_k": lift_rep.get("num_slots"),
              "prompt": FEEDBACK_PROMPT if a.mode == "feedback" else VIEWS_PROMPT}
    e = CacheEntry("semseg_refine", a.object_id, config)
    if e.complete:
        print(f"SEMSEG_DIR={e.path}")
        return
    out = e.prepare()
    t0 = time.time()

    # ---- motion: moving points per state; clusters of the previous lift (in state s), mapped to the other state
    pc = os.path.join(a.coarse_dir, "point_cloud", "iteration_10000")
    P, d2, Vd = state_points(pc)
    s = int(ml["state"])
    moving = [moving_mask(P[t], d2[t], a.dyn_ratio, "extent") for t in (0, 1)]
    from scipy.spatial import cKDTree
    n_lab = int(ml["label"].max()) + 1 if len(ml["label"]) else 0
    lab_s = np.full(len(P[s]), -1)
    lab_s[moving[s]] = ml["label"][cKDTree(ml["xyz"]).query(P[s][moving[s]])[1]] if n_lab else -1
    # the other state: label each moving point by its nearest moved counterpart (point + displacement)
    o = 1 - s
    lab_o = np.full(len(P[o]), -1)
    moved_s = P[s] + Vd[s]
    sel = moving[s]
    if sel.any():
        lab_o[moving[o]] = lab_s[sel][cKDTree(moved_s[sel]).query(P[o][moving[o]])[1]]
    labs = {s: lab_s, o: lab_o}

    cams = {t: load_cameras(a.scene_dir, STATES[t]) for t in (0, 1)}
    rgba = os.path.join(a.scene_dir, STATES[s], "train", "rgba")
    W, H = Image.open(os.path.join(rgba, sorted(os.listdir(rgba))[0])).size
    tol = 0.01 * np.linalg.norm(P[s].max(0) - P[s].min(0))
    chosen = select_views(cams[s], P[s], labs[s], max(n_lab, 1), W, H, tol, a.n_views, a.min_sep)
    names = {t: [c["name"] for c in cams[t]] for t in (0, 1)}
    cent_o = np.array([cam_center(c) for c in cams[o]])

    def rgb_path(t, name):
        return os.path.join(a.scene_dir, STATES[t], "train", "rgba", f"{name}.png")

    colors = ({k: COLORS[k % len(COLORS)] for k in range(n_lab)} if a.mode == "feedback"
              else {k: (230, 25, 75) for k in range(n_lab)})
    rows, pairs = [], []
    for i in chosen:
        cs = cams[s][i]
        j = names[o].index(cs["name"]) if cs["name"] in names[o] else \
            int(np.argmin(np.linalg.norm(cent_o - cam_center(cs), axis=1)))        # nearest camera (real data)
        imgs = {}
        for t, cam in ((s, cs), (o, cams[o][j])):
            im, fg = pp.load_rgb(rgb_path(t, cam["name"]))
            _, u, v, vis = visible_counts(cam, P[t], labs[t], max(n_lab, 1), W, H, tol)
            masks = region_masks(u, v, vis, labs[t], n_lab, W, H, fg)
            imgs[t] = tint(im, masks, colors, numbers=a.mode == "feedback")
        rows.append(pair_image(imgs[0], imgs[1], a.img_size))
        pairs.append([cs["name"], cams[o][j]["name"]])
    os.makedirs(out / "vis", exist_ok=True)
    for k, r in enumerate(rows):
        r.save(out / "vis" / f"vlm_input_{k}.jpg", quality=90)

    # ---- ask the VLM
    if a.mode == "views":
        prompt = VIEWS_PROMPT.format(n=len(rows))
    else:
        parts = lift_rep.get("parts", [])
        regions = "; ".join(f"region {k + 1} ({['red','green','blue','orange','purple','cyan','pink','lime'][k % 8]})"
                            + (f", labelled '{parts[k]['category']}'" if k < len(parts) and parts[k]['category'] != 'unknown'
                               else ", NOT covered by any listed part") for k in range(n_lab))
        problems = []
        n_prev = sum(p["count"] for p in prev["movable_parts"])
        if n_prev != n_lab:
            problems.append(f"the list has {n_prev} movable parts but {n_lab} regions move")
        if lift_rep.get("fallback_motion_only"):
            problems.append("the segmentation model could not find any of the listed parts in the images, so "
                            "the names are probably wrong")
        unlab = [k + 1 for k in range(n_lab) if k >= len(parts) or parts[k]["category"] == "unknown"]
        if unlab:
            problems.append(f"regions {unlab} match no listed part")
        if lift_rep.get("merged_clusters"):
            problems.append("two listed parts were found to move together as one rigid part")
        dropped = {c for c, n in lift_rep.get("inst_dropped", {}).items() if n > lift_rep.get("inst_kept", {}).get(c, 0)}
        if dropped:
            problems.append(f"{sorted(dropped)} were mostly detected on parts that do not move")
        prompt = FEEDBACK_PROMPT.format(n=len(rows), k=n_lab, regions=regions,
                                        prev=json.dumps(prev["movable_parts"]),
                                        problems=" ".join(p + "." for p in problems) or "none.")
    if a.dry_run:
        os.makedirs(a.dry_run, exist_ok=True)
        for k, r in enumerate(rows):
            r.save(os.path.join(a.dry_run, f"vlm_input_{k}.jpg"), quality=90)
        open(os.path.join(a.dry_run, "prompt.txt"), "w").write(prompt + "\n\npairs: " + json.dumps(pairs))
        print("dry run ->", a.dry_run)
        return
    vlm = pp.VLM(a.vlm)
    raw = vlm.ask(rows, prompt)
    try:
        parsed = pp.normalise_parts(pp.extract_json(raw))
    except ValueError:
        parsed = prev
    if not parsed["movable_parts"]:
        parsed = prev
    vlm.close()
    pp.log("refined VLM:", json.dumps(parsed))
    json.dump({"raw": raw, "parsed": parsed, "prompt": prompt, "view_pairs": pairs, "prev": prev,
               "mode": a.mode}, open(out / "vlm.json", "w"), indent=2)

    # ---- SAM3 on every view with the (new) names
    seg = pp.Sam3Segmenter(a.sam3)
    counts = {p["name"]: [] for p in parsed["movable_parts"]}
    n_done = 0
    for t in (0, 1):
        os.makedirs(out / "masks" / STATES[t], exist_ok=True)
        for vi, path in enumerate(pp.list_views(a.scene_dir, STATES[t])):
            if vi % a.view_stride:
                continue
            image, fg = pp.load_rgb(path)
            dets = seg.segment_all(image, [p["name"] for p in parsed["movable_parts"]], a.score_thresh)
            lab, cats, scores = pp.build_label_map(dets, fg)
            for c in counts:
                counts[c].append(sum(1 for x in cats if x == c))
            name = os.path.splitext(os.path.basename(path))[0]
            np.savez_compressed(out / "masks" / STATES[t] / f"{name}.npz", labels=lab,
                                cats=np.array(cats, dtype=object), scores=np.array(scores, np.float32))
            n_done += 1
    json.dump({"parts": parsed["movable_parts"], "counts_per_view": counts, "n_views": n_done,
               "timings": {"total_s": round(time.time() - t0, 1)}}, open(out / "summary.json", "w"), indent=2)
    e.mark_complete(n_views=n_done, parts=parsed["movable_parts"])
    print(f"SEMSEG_DIR={e.path}")


if __name__ == "__main__":
    main()
