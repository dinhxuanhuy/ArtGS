"""Functional smoke test for the ArtGS environment (all compiled CUDA extensions)."""
import sys


def check(name, fn):
    try:
        out = fn()
        print(f"  OK   {name}" + (f" -> {out}" if out else ""))
        return True
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        return False


def main():
    import torch
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}")

    ok = True

    # --- tiny-cuda-nn ---
    def tcnn_test():
        import tinycudann as tcnn
        enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 8, "n_features_per_level": 2,
                                "log2_hashmap_size": 15, "base_resolution": 16,
                                "per_level_scale": 1.5}, dtype=torch.float32)
        x = torch.rand(1024, 3, device=dev) * 2 - 1
        y = enc(x)
        z = y.sum(); z.backward()
        return tuple(y.shape)
    ok &= check("tinycudann (HashGrid encoding + backward)", tcnn_test)

    # --- diff-gaussian-rasterization (ArtGS fork) ---
    def rasterizer_test():
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        n, H, W = 256, 64, 64
        means = torch.randn(n, 3, device=dev) * 0.3
        means[:, 2] += 3.0
        means.requires_grad_(True)
        scales = torch.rand(n, 3, device=dev) * 0.05 + 0.01
        rots = torch.zeros(n, 4, device=dev); rots[:, 0] = 1.0
        opac = torch.rand(n, 1, device=dev) * 0.5 + 0.4
        # NOTE: fork expects SH layout (N, K, 3)  (K = (deg+1)^2), not (N, 3, K)
        shs = torch.zeros(n, 16, 3, device=dev)
        shs[:, 0, :] = 1.0
        shs.requires_grad_(True)
        viewmat = torch.eye(4, device=dev)[None]
        projmat = torch.tensor([[1.299, 0, 0, 0], [0, 1.732, 0, 0],
                                [0, 0, 1.001, 0.2], [0, 0, 1, 0]], device=dev)[None]
        campos = torch.zeros(3, device=dev)
        bg = torch.zeros(3, device=dev)
        settings = GaussianRasterizationSettings(
            image_height=H, image_width=W,
            tanfovx=0.6, tanfovy=0.8, bg=bg,
            scale_modifier=1.0, viewmatrix=viewmat, projmatrix=projmat,
            sh_degree=3, campos=campos, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=settings)
        rendering, radii, depth, alpha = rasterizer(
            means3D=means, means2D=torch.zeros(n, 2, device=dev),
            shs=shs, colors_precomp=None, opacities=opac,
            scales=scales, rotations=rots, cov3D_precomp=None)
        (rendering.mean() + depth.mean() + alpha.mean()).backward()
        return f"image {tuple(rendering.shape)}, depth {tuple(depth.shape)}, alpha {tuple(alpha.shape)}, grad ok={means.grad is not None}"
    ok &= check("diff_gaussian_rasterization (render + depth + alpha + backward)", rasterizer_test)

    # --- simple-knn ---
    def knn_test():
        from simple_knn._C import distCUDA2
        pts = torch.rand(5000, 3, device=dev)
        d = distCUDA2(pts)
        return f"dist mean {d.mean().item():.5f}"
    ok &= check("simple_knn.distCUDA2", knn_test)

    # --- pointnet2 (NFPS) ---
    def pointnet_test():
        from utils.pointnet_lib import pointnet2_utils
        xyz = torch.rand(2, 1024, 3, device=dev)
        idx = pointnet2_utils.furthest_point_sample(xyz, 128)
        return f"idx shape {tuple(idx.shape)}"
    ok &= check("pointnet2 furthest_point_sample (NFPS)", pointnet_test)

    # --- pytorch3d ---
    def p3d_test():
        from pytorch3d.loss import chamfer_distance
        a = torch.rand(2, 512, 3, device=dev)
        b = torch.rand(2, 512, 3, device=dev)
        loss, _ = chamfer_distance(a, b)
        return f"chamfer {loss.item():.5f}"
    ok &= check("pytorch3d chamfer_distance", p3d_test)

    # --- repo modules ---
    def repo_test():
        import iopath, fvcore  # pytorch3d runtime deps
        import scene, gaussian_renderer
        from utils import metrics, loss_utils, geo_utils
        from utils.pointnet_lib import pointnet2_utils
        return "scene/gaussian_renderer/utils imported"
    ok &= check("ArtGS repo modules", repo_test)

    print("\nRESULT:", "ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
