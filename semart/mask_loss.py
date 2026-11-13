"""2D part-consistency loss from per-view instance masks (SemArt S4).

For a training view we render the per-Gaussian slot probabilities (K channels) with the
geometry detached, so the loss only shapes the segmentation field (centres + MLP). For each 2D
instance j with mean slot distribution p_j:
    intra  = mean over its pixels of ||P - p_j||^2      one instance -> one slot
    inter  = mean over pairs j<l of <p_j, p_l>         different instances -> different slots
    static = mean_j p_j[0]                             a movable instance is not the base
The loss is invariant to instance ids, so masks need no cross-view association.
"""
import math
import os

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.dual_quaternion import quaternion_mul

STATES = ("start", "end")


def load_label_map(path, size_wh):
    """npz written by semart/preprocess.py -> uint8 HxW label map at the given (W, H)."""
    d = np.load(path)
    lab = d["labels"]
    if (lab.shape[1], lab.shape[0]) != tuple(size_wh):
        lab = np.array(Image.fromarray(lab).resize(size_wh, Image.NEAREST))
    return lab


class SemMaskSupervisor:
    def __init__(self, mask_dir, viewpoint_stacks, sp, num_slots, device="cuda"):
        self.sp = sp
        self.K = num_slots
        self.labels = {}
        n_missing = 0
        for s, cams in enumerate(viewpoint_stacks):
            for cam in cams:
                p = os.path.join(mask_dir, STATES[s], f"{cam.image_name}.npz")
                if not os.path.exists(p):
                    n_missing += 1
                    continue
                lab = load_label_map(p, (int(cam.image_width), int(cam.image_height)))
                if lab.max() == 0:
                    continue
                self.labels[(s, cam.image_name)] = torch.from_numpy(lab).to(device)
        print(f"[semart] mask supervision on {len(self.labels)} views ({n_missing} without a mask file)")

    def has(self, state, cam):
        return (state, cam.image_name) in self.labels

    @staticmethod
    def render_probs(cam, gaussians, d_xyz, d_rot, probs):
        """Alpha-composite per-Gaussian probabilities [N, K] -> ([K, H, W], alpha [H, W])."""
        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=torch.zeros(3, device=probs.device), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=0, campos=cam.camera_center, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = (gaussians.get_xyz + d_xyz).detach()
        rot = gaussians.get_rotation
        rot = (quaternion_mul(d_rot, rot) if d_rot is not None else rot).detach()
        common = dict(opacities=gaussians.get_opacity.detach(), scales=gaussians.get_scaling.detach(),
                      rotations=rot, cov3D_precomp=None, shs=None)
        K = probs.shape[1]
        pad = (-K) % 3
        if pad:
            probs = torch.cat([probs, probs.new_zeros(probs.shape[0], pad)], 1)
        chans, alpha = [], None
        for c in range(0, probs.shape[1], 3):
            means2D = torch.zeros_like(means3D)
            img, _, _, a = rasterizer(means3D=means3D, means2D=means2D,
                                      colors_precomp=probs[:, c:c + 3].contiguous(), **common)
            chans.append(img)
            alpha = a
        return torch.cat(chans, 0)[:K], alpha.squeeze(0)

    def loss(self, cam, state, gaussians, d_xyz, d_rot, probs):
        lab = self.labels[(state, cam.image_name)]
        P, alpha = self.render_probs(cam, gaussians, d_xyz, d_rot, probs)
        valid = (alpha > 0.5) & (lab > 0)
        zero = probs.sum() * 0.0
        if valid.sum() < self.sp.sem_min_pixels:
            return zero, {}
        idx = lab[valid].long()                                 # [n] instance ids (1..)
        Pv = (P[:, valid] / alpha[valid].clamp(min=1e-3)).T     # [n, K]
        n_inst = int(idx.max()) + 1
        cnt = torch.zeros(n_inst, device=Pv.device).index_add_(0, idx, torch.ones_like(idx, dtype=Pv.dtype))
        keep = cnt >= self.sp.sem_min_pixels
        keep[0] = False
        if keep.sum() == 0:
            return zero, {}
        sums = torch.zeros(n_inst, Pv.shape[1], device=Pv.device).index_add_(0, idx, Pv)
        means = sums / cnt.clamp(min=1)[:, None]               # [n_inst, K]
        px = keep[idx]
        intra = ((Pv[px] - means[idx[px]]) ** 2).sum(-1).mean()
        m = means[keep]
        static = m[:, 0].mean()
        if m.shape[0] > 1:
            G = m @ m.T
            iu = torch.triu_indices(m.shape[0], m.shape[0], 1, device=m.device)
            inter = G[iu[0], iu[1]].mean()
        else:
            inter = zero
        sp = self.sp
        total = sp.sem_w_intra * intra + sp.sem_w_inter * inter + sp.sem_w_static * static
        return total, {"intra": intra.item(), "inter": float(inter), "static": static.item(),
                       "n_inst": int(keep.sum())}
