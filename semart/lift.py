"""SemArt S2: lift per-view instance masks onto the moving coarse Gaussians -> K, centres, types.

Runs in the ArtGS venv after train_coarse.py. Uses no GT: the moving set is recomputed with a
rule that does not depend on the part count, and K comes from the semantic evidence.

Motion verifies semantics at two levels:
  2D  a mask instance whose visible coarse points are mostly static is dropped (segmenter false
      positives on the base), written as masks_filtered/ for the 2D loss;
  3D  a part cluster with too few moving points is dropped.
Categories the VLM counts once are merged per view (SAM often splits a drawer into front + box).

    python -m semart.lift --source_path data/artgs/sapien/storage_47648 \
        --coarse_dir outputs/artgs/sapien/storage_47648/coarse_gs \
        --semseg_dir <preprocess entry> --out_dir outputs/.../lift_vlm [--k_source vlm]

Writes out_dir/{center_info.npy, slots.json, lift_report.json, moving_labels.npz, masks_filtered/}.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

STATES = ("start", "end")
JOINT = {"prismatic": "p", "revolute": "r", "p": "p", "r": "r"}


def load_cameras(source_path, state):
    with open(os.path.join(source_path, f"transforms_train_{state}.json")) as f:
        c = json.load(f)
    cams = []
    for fr in sorted(c["frames"], key=lambda x: x["file_path"]):
        name = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1                      # OpenGL -> COLMAP, as scene/dataset_readers.py
        w2c = np.linalg.inv(c2w)
        cams.append({"name": name, "w2c": w2c, "fovx": c["camera_angle_x"], "fovy": c["camera_angle_y"]})
    return cams


def project(cam, X, W, H):
    Xc = X @ cam["w2c"][:3, :3].T + cam["w2c"][:3, 3]
    fx = W / (2 * np.tan(cam["fovx"] / 2))
    fy = H / (2 * np.tan(cam["fovy"] / 2))
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Xc[:, 0] / z + W / 2
        v = fy * Xc[:, 1] / z + H / 2
    return u, v, z


def state_points(coarse_pc):
    """All coarse points per state, squared displacement and displacement vector to the nearest
    point of the other state (a crude per-point motion estimate)."""
    P = [np.concatenate([np.load(f"{coarse_pc}/xyz_static_{s}.npy"), np.load(f"{coarse_pc}/xyz_dynamic_{s}.npy")])
         for s in (0, 1)]
    d2, V = [], []
    for s in (0, 1):
        d, nn = cKDTree(P[1 - s]).query(P[s])
        d2.append(d ** 2)                                            # like pytorch3d chamfer
        V.append(P[1 - s][nn] - P[s])
    return P, d2, V


def same_motion(u, v, cos_min=0.9, ratio=1.67):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return False
    return float(u @ v) / (nu * nv) > cos_min and max(nu, nv) / min(nu, nv) < ratio


def moving_mask(P, d2, ratio, rule):
    """rule 'max': sq. displacement > ratio * max (ArtGS; drops parts that move little next to a large
    swing, e.g. a 5 cm drawer next to a 120 deg door). rule 'extent': displacement > ratio * bbox diag."""
    if rule == "max":
        return d2 > ratio * d2.max()
    return np.sqrt(d2) > ratio * np.linalg.norm(P.max(0) - P.min(0))


def load_labels(path, img_wh, merge_cats):
    """Label map + categories; instances of a category in merge_cats are fused into one id."""
    d = np.load(path, allow_pickle=True)
    lab, cats = d["labels"], [str(c) for c in d["cats"]]
    W, H = img_wh
    if lab.shape != (H, W):
        lab = np.array(Image.fromarray(lab).resize((W, H), Image.NEAREST))
    if not merge_cats or not cats:
        return lab, cats
    new_id, new_cats, first = np.zeros(len(cats) + 1, np.int64), [], {}
    for k, c in enumerate(cats, 1):
        if c in merge_cats and c in first:
            new_id[k] = first[c]
        else:
            new_cats.append(c)
            new_id[k] = len(new_cats)
            if c in merge_cats:
                first[c] = new_id[k]
    return new_id[lab].astype(np.uint8), new_cats


def visible_labels(cam, lab, X, n_query, img_wh, tol, grid=4):
    """z-buffer over all of X; returns (visible, label) for the first n_query points."""
    W, H = img_wh
    u, v, z = project(cam, X, W, H)
    ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    gw, gh = W // grid, H // grid
    cell = np.clip((v / grid).astype(int), 0, gh - 1) * gw + np.clip((u / grid).astype(int), 0, gw - 1)
    zbuf = np.full(gw * gh, np.inf)
    np.minimum.at(zbuf, cell[ok], z[ok])
    q = slice(0, n_query)
    vis = ok[q] & (z[q] <= zbuf[cell[q]] + tol)
    uq = np.clip(np.nan_to_num(u[q]).astype(int), 0, W - 1)
    vq = np.clip(np.nan_to_num(v[q]).astype(int), 0, H - 1)
    return vis, np.where(vis, lab[vq, uq].astype(np.int64), 0)


def affinity(ids, n_inst, min_views):
    import scipy.sparse as sp
    lab = ids > 0
    rows, cols = np.nonzero(lab)
    S = sp.csr_matrix((np.ones(len(rows)), (rows, ids[rows, cols] - 1)), shape=(ids.shape[0], n_inst))
    same = (S @ S.T).toarray()
    vis = (ids >= 0).astype(np.float64)
    vis_unl = (ids == 0).astype(np.float64)
    both = vis @ vis.T - vis_unl @ vis_unl.T          # both visible and at least one labelled
    W = np.where(both >= min_views, same / np.maximum(both, 1), 0.0)
    np.fill_diagonal(W, 1.0)
    return W


def eigengap_k(W, k_max=8):
    d = W.sum(1)
    Dm = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L = np.eye(len(W)) - Dm[:, None] * W * Dm[None]
    ev = np.linalg.eigvalsh(L)[: k_max + 1]
    return int(np.argmax(np.diff(ev)) + 1), ev.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--coarse_dir", required=True)
    ap.add_argument("--semseg_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_source", default="vlm", choices=["vlm", "vlm_motion", "sam", "gt", "eigengap"])
    ap.add_argument("--gt_k", type=int, default=None, help="only for --k_source gt (oracle ablation)")
    ap.add_argument("--dyn_rule", default="extent", choices=["extent", "max"])
    ap.add_argument("--dyn_ratio", type=float, default=None, help="default 0.02 (extent) / 0.05 (max)")
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--min_views", type=int, default=3)
    ap.add_argument("--min_part_frac", type=float, default=0.02)
    ap.add_argument("--inst_min_moving_frac", type=float, default=0.4,
                    help="2D filter: keep an instance if this fraction of its visible points move")
    ap.add_argument("--no_merge", action="store_true", help="do not fuse instances of count-1 categories")
    ap.add_argument("--min_cover", type=float, default=0.3,
                    help="v3: below this fraction of mask-covered moving points, cluster by motion only")
    ap.add_argument("--merge_ratio", type=float, default=0.5,
                    help="v3: merge two clusters if their cross mask-affinity >= this x their own (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dyn_ratio is None:
        a.dyn_ratio = 0.02 if a.dyn_rule == "extent" else 0.05
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    pc = os.path.join(a.coarse_dir, "point_cloud", "iteration_10000")

    vlm = json.load(open(os.path.join(a.semseg_dir, "vlm.json")))
    parts = vlm["parsed"]["movable_parts"]
    n_vlm = {p["name"]: int(p.get("count", 1)) for p in parts}
    merge = set() if a.no_merge else {c for c, n in n_vlm.items() if n == 1}

    P, d2, V = state_points(pc)
    s = 0 if d2[0].mean() > d2[1].mean() else 1               # same rule as match_gaussians
    moving = [moving_mask(P[t], d2[t], a.dyn_ratio, a.dyn_rule) for t in (0, 1)]
    tol = 0.01 * np.linalg.norm(P[s].max(0) - P[s].min(0))
    mask_dir = os.path.join(a.semseg_dir, "masks")
    rgba = os.path.join(a.source_path, STATES[s], "train", "rgba")
    img_wh = Image.open(os.path.join(rgba, sorted(os.listdir(rgba))[0])).size

    # ---- 2D motion filter for both states -> masks_filtered/
    kept, dropped = Counter(), Counter()
    sam_counts = {c: [] for c in n_vlm}
    for t in (0, 1):
        out_t = os.path.join(a.out_dir, "masks_filtered", STATES[t])
        os.makedirs(out_t, exist_ok=True)
        for cam in load_cameras(a.source_path, STATES[t]):
            p = os.path.join(mask_dir, STATES[t], f"{cam['name']}.npz")
            if not os.path.exists(p):
                continue
            # filter each raw instance first, then fuse count-1 categories (fusing first would let a
            # false positive on the base drag a true instance of the same category down with it)
            lab, cats = load_labels(p, img_wh, set())
            vis, loc = visible_labels(cam, lab, P[t], len(P[t]), img_wh, tol)
            new_lab, new_cats, groups = np.zeros_like(lab), [], {}   # groups: cat -> [(id, mean motion)]
            for k, c in enumerate(cats, 1):
                inside = vis & (loc == k)
                n_mov, n_all = int((inside & moving[t]).sum()), int(inside.sum())
                if n_all >= 5 and n_mov >= a.inst_min_moving_frac * n_all:
                    mv = V[t][inside & moving[t]].mean(0)
                    # v3: fuse with an earlier instance of a count-1 category only if both MOVE alike
                    # (drawer front + box: yes; two oven doors on different hinges: no)
                    tgt = next((g for g, m in groups.get(c, []) if same_motion(mv, m)), None) if c in merge else None
                    if tgt is not None:
                        new_lab[lab == k] = tgt
                    else:
                        new_cats.append(c)
                        new_lab[lab == k] = len(new_cats)
                        groups.setdefault(c, []).append((len(new_cats), mv))
                    kept[c] += 1
                else:
                    dropped[c] += 1
            for c in sam_counts:
                sam_counts[c].append(new_cats.count(c))
            np.savez_compressed(os.path.join(out_t, f"{cam['name']}.npz"), labels=new_lab,
                                cats=np.array(new_cats, dtype=object))

    # ---- 3D: moving points of the larger-motion state, ids from the filtered masks
    dyn, stat = P[s][moving[s]], P[s][~moving[s]]
    sub = rng.choice(len(dyn), min(a.max_points, len(dyn)), replace=False) if len(dyn) else np.zeros(0, int)
    D = dyn[sub]
    Xall = np.concatenate([D, stat])
    ids, inst_cat = [], []
    for cam in load_cameras(a.source_path, STATES[s]):
        p = os.path.join(a.out_dir, "masks_filtered", STATES[s], f"{cam['name']}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        vis, loc = visible_labels(cam, z["labels"], Xall, len(D), img_wh, tol)
        cur = np.where(vis, 0, -1)
        lv = vis & (loc > 0)
        cur[lv] = len(inst_cat) + loc[lv]                    # global id <-> inst_cat[id - 1]
        inst_cat.extend(str(c) for c in z["cats"])
        ids.append(cur)
    ids = np.stack(ids, 1) if ids else np.zeros((len(D), 0), np.int64)
    covered = (ids > 0).sum(1) >= a.min_views

    n_sam = {c: int(round(np.percentile(v, 90))) if v else 0 for c, v in sam_counts.items()}
    # v3: where SAM3 keeps seeing >= 2 differently-moving instances of a category the VLM counted once,
    # trust that count (oven_101908: "1 oven door" for two doors)
    n_eff = {c: max(n_vlm[c], n_sam.get(c, 0)) if n_vlm[c] == 1 else n_vlm[c] for c in n_vlm}
    k_req = {"vlm": sum(n_vlm.values()), "vlm_motion": sum(n_eff.values()), "sam": sum(n_sam.values()),
             "gt": (a.gt_k or 0) - 1, "eigengap": None}[a.k_source]
    report = {"larger_motion_state": s, "n_moving": int(len(dyn)), "n_sampled": int(len(D)),
              "n_views_with_masks": int(ids.shape[1]), "covered_frac": float(covered.mean()) if len(D) else 0.0,
              "n_vlm": n_vlm, "n_sam_filtered": n_sam, "n_effective": n_eff, "merged_categories": sorted(merge),
              "inst_kept": dict(kept), "inst_dropped": dict(dropped)}
    Dc, idc = D[covered], ids[covered]
    fallback = len(Dc) < max(10, a.min_cover * len(D))
    report["fallback_motion_only"] = bool(fallback)
    if fallback:
        # v3: masks are silent (segmenter missed the part): cluster moving points by position, as ArtGS
        if len(D) < 10:
            json.dump(report, open(os.path.join(a.out_dir, "lift_report.json"), "w"), indent=2)
            raise SystemExit(f"[lift] only {len(D)} moving points")
        print(f"[lift] masks cover {len(Dc)}/{len(D)} moving points -> motion-only fallback")
        Dc, idc = D, np.zeros((len(D), 0), np.int64)
        Wsem = np.eye(len(Dc))
    else:
        Wsem = affinity(idc, max(len(inst_cat), 1), a.min_views)
    W = Wsem.copy()
    # weak spatial term keeps the graph connected where masks are silent
    dist, nn = cKDTree(Dc).query(Dc, k=min(9, len(Dc)))
    sigma = np.median(dist[:, -1]) + 1e-9
    r = np.arange(len(Dc))
    for j in range(1, nn.shape[1]):
        W[r, nn[:, j]] = np.maximum(W[r, nn[:, j]], 0.05 * np.exp(-(dist[:, j] / sigma) ** 2))
    W = np.maximum(W, W.T)
    k_eig, ev = eigengap_k(W)
    k = k_eig if (a.k_source == "eigengap" or fallback) else k_req
    from sklearn.cluster import SpectralClustering
    if a.k_source == "vlm_motion" and not fallback and k_eig != k_req and k_eig >= 1:
        # VLM count vs motion/mask eigengap disagree: keep the K whose partition is more consistent
        from sklearn.metrics import silhouette_score
        Dist = 1.0 - W / W.max()
        np.fill_diagonal(Dist, 0.0)
        sil = {}
        for kk in sorted({k_req, k_eig}):
            if kk > 1:
                l = SpectralClustering(kk, affinity="precomputed", assign_labels="discretize",
                                       random_state=a.seed).fit_predict(W)
                sil[kk] = float(silhouette_score(Dist, l, metric="precomputed"))
        if sil:
            k = max(sil, key=sil.get)
        report["k_silhouette"] = sil
    report.update(k_requested=k_req, k_eigengap=k_eig, eigenvalues=ev)
    if not k or k < 1:
        json.dump(report, open(os.path.join(a.out_dir, "lift_report.json"), "w"), indent=2)
        raise SystemExit(f"[lift] k={k}")

    lab = (SpectralClustering(k, affinity="precomputed", assign_labels="discretize", random_state=a.seed)
           .fit_predict(W) if k > 1 else np.zeros(len(Dc), int))
    if not fallback and a.merge_ratio > 0:
        # v3: two clusters whose points keep falling into the SAME masks are one part split in two
        # (VLM over-count: "2 handles" of a scissor, "backrest + seat"); merge them, K may drop to 1
        merged = []
        while len(np.unique(lab)) > 1:
            cs = list(np.unique(lab))
            m = {c: lab == c for c in cs}
            within = {c: Wsem[np.ix_(m[c], m[c])].mean() for c in cs}
            best, pair = 0.0, None
            for i, ca in enumerate(cs):
                for cb in cs[i + 1:]:
                    r_ = Wsem[np.ix_(m[ca], m[cb])].mean() / max(min(within[ca], within[cb]), 1e-9)
                    if r_ > best:
                        best, pair = r_, (ca, cb)
            if best < a.merge_ratio:
                break
            lab[lab == pair[1]] = pair[0]
            merged.append([int(pair[0]), int(pair[1]), round(float(best), 3)])
        lab = np.unique(lab, return_inverse=True)[1]
        k = int(lab.max()) + 1
        report["merged_clusters"] = merged
    frac = np.bincount(lab, minlength=k) / len(lab)
    keep = [c for c in range(k) if frac[c] >= a.min_part_frac]      # 3D motion verification
    report.update(cluster_frac=frac.tolist(), dropped_clusters=[c for c in range(k) if c not in keep])
    km = np.isin(lab, keep)
    remap = {c: i for i, c in enumerate(keep)}
    lab_k = np.array([remap[c] for c in lab[km]])
    Dk, idk = Dc[km], idc[km]
    full = lab_k[cKDTree(Dk).query(dyn)[1]]                         # diagnostics only

    joint_of = {p["name"]: JOINT.get(str(p.get("joint", "revolute")).lower(), "r") for p in parts}
    cats_out, jt = [], ["s"]
    for i in range(len(keep)):
        votes = Counter(inst_cat[g - 1] for g in idk[lab_k == i].ravel() if g > 0)
        cat = votes.most_common(1)[0][0] if votes else "unknown"
        cats_out.append({"category": cat, "votes": dict(votes), "n_points": int((lab_k == i).sum())})
        # unlabelled cluster (fallback): use the VLM's joint type if it named a single kind of part
        default = next(iter(joint_of.values())) if len(joint_of) == 1 else "r"
        jt.append(joint_of.get(cat, default))

    # centres from mask-covered moving points only (uncovered moving points include static noise)
    centers = [stat.mean(0)] + [Dk[lab_k == i].mean(0) for i in range(len(keep))]
    xs = [stat] + [Dk[lab_k == i] for i in range(len(keep))]
    radius = [np.linalg.norm(x - c, axis=1).max() / 2 for x, c in zip(xs, centers)]
    center_info = np.concatenate([np.stack(centers), np.array(radius)[:, None]], -1).astype(np.float32)
    np.save(os.path.join(a.out_dir, "center_info.npy"), center_info)
    np.savez_compressed(os.path.join(a.out_dir, "moving_labels.npz"), xyz=dyn, label=full, state=s)
    slots = {"num_slots": len(keep) + 1, "joint_types": ",".join(jt), "parts": cats_out,
             "k_source": a.k_source, "dyn_rule": a.dyn_rule, "dyn_ratio": a.dyn_ratio}
    json.dump(slots, open(os.path.join(a.out_dir, "slots.json"), "w"), indent=2)
    report.update(slots)
    json.dump(report, open(os.path.join(a.out_dir, "lift_report.json"), "w"), indent=2, default=float)
    print(f"[lift] K={slots['num_slots']} types={slots['joint_types']} (requested {k_req}, eigengap {k_eig}, "
          f"covered {report['covered_frac']:.2f}, inst kept {dict(kept)} dropped {dict(dropped)}) -> {a.out_dir}")


if __name__ == "__main__":
    main()
