"""M4: back-projection vs mj_ray, mesh projection vs segmentation, normals vs analytic."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import mujoco
    import numpy as np
    from cpick.sim.camera import Camera
    from cpick.sim.geometry import backproject, estimate_normal, pixel_ray, project
    from cpick.sim.reset import CanDropper
    from cpick.sim.scene_builder import build_scene_xml

    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
    data = mujoco.MjData(model)
    dropper = CanDropper(model, data, cfg)
    c, radius = cfg["camera"], cfg["suction"]["cup_radius"]
    body, geom = model.body("can").id, model.geom("can").id
    mid = model.geom_dataid[geom]
    V = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    rng = np.random.default_rng(0)
    gid = np.full(1, -1, dtype=np.int32)

    bp_err, bp_hit, off_centre = [], [], 0.0
    n_vis = n_ok = 0
    ang_up, ang_side = [], []
    outliers = []
    with Camera(model, c["name"], c["width"], c["height_px"]) as cam:
        for seed in range(args.n):
            dropper.reset(seed)
            depth, seg = cam.depth(data), cam.segmentation(data)
            K, T = cam.intrinsics(), cam.cam_to_world(data)
            P, cam_pos, (h, w) = backproject(depth, K, T), T[:3, 3].copy(), depth.shape
            W = V @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            ys, xs = np.nonzero(seg == geom)

            # (a) back-projection vs an independent ray cast through the same pixel
            for i in rng.choice(len(xs), size=min(200, len(xs)), replace=False):
                u, v = xs[i], ys[i]
                o, d, _ = pixel_ray(u, v, K, T)
                dist = mujoco.mj_ray(model, data, o, d, None, 1, -1, gid)
                bp_err.append(np.linalg.norm(o + dist * d - P[v, u]))
                bp_hit.append(gid[0] == geom)
                off_centre = max(off_centre, np.hypot(u - K[0, 2], v - K[1, 2]))

            # (b) mesh vertices -> pixels; visibility decided by mj_ray, not by the renderer
            sub = W[rng.choice(len(W), size=min(2000, len(W)), replace=False)]
            uv, _ = project(sub, K, T)
            for (uf, vf), pw in zip(uv, sub):
                vec = pw - cam_pos
                L = np.linalg.norm(vec)
                dist = mujoco.mj_ray(model, data, cam_pos, vec / L, None, 1, -1, gid)
                if gid[0] != geom or abs(dist - L) > 1e-3:
                    continue                                    # hidden vertex
                ui, vi = int(round(uf)), int(round(vf))
                if not (1 <= ui < w - 1 and 1 <= vi < h - 1):
                    continue
                n_vis += 1
                n_ok += bool((seg[vi - 1:vi + 2, ui - 1:ui + 2] == geom).any())   # within +-1 px

            # (c) normals vs analytic, only where the mesh actually matches the analytic shape
            R = data.xmat[body].reshape(3, 3)
            a, com = R[:, 2], data.xipos[body]
            relW = W - com
            axW = relW @ a
            rW = np.linalg.norm(relW - np.outer(axW, a), axis=1)
            r_full = rW.max()
            barrel = axW[rW > r_full - 2e-4]                   # vertices on the full-radius body
            band_lo, band_hi = barrel.min(), barrel.max()
            lid_r = rW[axW > axW.max() - 5e-4].max()           # radius of the flat top
            if seed == 0:
                print(f"mesh profile: full radius {r_full * 1e3:.2f} mm on axial "
                      f"[{band_lo * 1e3:.1f}, {band_hi * 1e3:.1f}] mm (ends {axW.min() * 1e3:.1f}, "
                      f"{axW.max() * 1e3:.1f}); lid radius {lid_r * 1e3:.2f} mm")
            for i in rng.choice(len(xs), size=min(150, len(xs)), replace=False):
                u, v = xs[i], ys[i]
                rel = P[v, u] - com
                axial = rel @ a
                radial = rel - axial * a
                if a[2] > 0.9:                                  # upright: ball fully on the lid
                    if axial < axW.max() - 2e-3 or np.linalg.norm(radial) > lid_r - radius - 1e-3:
                        continue
                    n_gt, bucket = np.array([0.0, 0.0, 1.0]), ang_up
                elif abs(a[2]) < 0.1:                           # side: ball fully on the barrel
                    n_gt = radial / np.linalg.norm(radial)
                    if n_gt[2] < 0.5 or not (band_lo + radius < axial < band_hi - radius):
                        continue
                    bucket = ang_side
                else:
                    continue
                n, _ = estimate_normal(P, depth, u, v, radius, K, cam_pos)
                if n is not None:
                    err = np.degrees(np.arccos(np.clip(n @ n_gt, -1, 1)))
                    bucket.append(err)
                    if err > 3.0:
                        outliers.append((seed, "up" if bucket is ang_up else "side",
                                         round(axial * 1e3, 1), round(float(n_gt[2]), 2), round(err, 2)))

    bp_err, ang_up, ang_side = map(np.asarray, (bp_err, ang_up, ang_side))
    pct = lambda x: np.round(np.percentile(x, [50, 95, 100]), 3) if len(x) else "none"
    print(f"coverage: furthest tested can pixel {off_centre:.0f} px from the image centre")
    print(f"[a] backproject vs ray, mm p50/p95/max: {pct(bp_err * 1000)}, ray hit can {np.mean(bp_hit):.1%}")
    print(f"[b] visible vertices {n_vis}, on can seg within 1 px: {n_ok / max(n_vis, 1):.2%}")
    print(f"[c] normal error deg p50/p95/max  upright n={len(ang_up)}: {pct(ang_up)}  "
          f"side n={len(ang_side)}: {pct(ang_side)}")
    for o in sorted(outliers, key=lambda t: -t[-1])[:8]:
            print(f"  outlier seed {o[0]} {o[1]}: axial {o[2]} mm, n_gt_z {o[3]}, err {o[4]} deg")
    ok = (bp_err.max() < 1e-3 and np.mean(bp_hit) == 1.0 and n_ok / max(n_vis, 1) >= 0.99
          and len(ang_up) and len(ang_side) and ang_up.max() < 2.0 and ang_side.max() < 5.0)
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED ")


if __name__ == "__main__":
    main()