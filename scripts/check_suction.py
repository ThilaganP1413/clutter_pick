"""M5: suction oracle on the 128x128 action grid -> reason codes, timing, disc size, side spreads."""
import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "m5"))
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import mujoco
    import numpy as np
    from cpick.sim.camera import Camera
    from cpick.sim.geometry import backproject
    from cpick.sim.reset import CanDropper
    from cpick.sim.scene_builder import build_scene_xml
    from cpick.sim.suction import REASONS, SuctionChecker

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
    data = mujoco.MjData(model)
    dropper, checker = CanDropper(model, data, cfg), SuctionChecker(model, cfg)
    c, body = cfg["camera"], model.body("can").id
    objects = {model.geom("can").id}
    cmap = ListedColormap(["tab:green", "lightgrey", "black", "white", "tab:orange", "tab:red"])

    counts = dict.fromkeys(REASONS, 0)
    t_in, scene_s, disc_px, side_spreads, side_ok = [], [], [], [], 0
    clean = []
    with Camera(model, c["name"], c["width"], c["height_px"]) as cam:
        for seed in range(args.n):
            dropper.reset(seed)
            depth = cam.depth(data)
            K, T = cam.intrinsics(), cam.cam_to_world(data)
            P, cam_pos = backproject(depth, K, T), T[:3, 3].copy()
            R = data.xmat[body].reshape(3, 3)
            a, com, az = R[:, 2], data.xipos[body].copy(), R[2, 2]
            pose = "upright" if az > 0.9 else "inverted" if az < -0.9 else "side" if abs(az) < 0.1 else "leaning"
            if pose == "side":                                    # barrel band from the mesh, as in M4
                g = model.geom("can").id
                mid = model.geom_dataid[g]
                Vm = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                rel = Vm @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - com
                axv = rel @ a
                rr = np.linalg.norm(rel - np.outer(axv, a), axis=1)
                barrel = axv[rr > rr.max() - 2e-4]
                band = (barrel.min() + checker.r, barrel.max() - checker.r)
            code = np.zeros((128, 128), dtype=int)
            spreads = []
            ts = time.perf_counter()
            for i in range(128):
                for j in range(128):
                    t0 = time.perf_counter()
                    r = checker.check(data, P, depth, 2 * j, 2 * i, K, cam_pos, objects)  # cell -> pixel x2
                    if r.reason != "outside_mask":
                        t_in.append(time.perf_counter() - t0)
                    code[i, j] = REASONS.index(r.reason)
                    counts[r.reason] += 1
                    if r.reason in ("ok", "not_planar"):
                        spreads.append(r.spread)
                        if pose == "side" and not r.clipped:
                            rel_p = r.point - com
                            axial = rel_p @ a
                            radial = rel_p - axial * a
                            if band[0] < axial < band[1] and radial[2] / np.linalg.norm(radial) > np.cos(np.radians(50)):
                                clean.append(r.spread)

            scene_s.append(time.perf_counter() - ts)

            n_ok = int((code == 0).sum())
            if pose == "upright":
                disc_px.append(2 * np.sqrt(4 * n_ok / np.pi))     # each cell = 2x2 px
            elif pose == "side":
                side_spreads += spreads
                side_ok += n_ok
            if seed < 6:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].imshow(depth, cmap="viridis"); ax[0].set_title(f"seed {seed} depth ({pose})")
                im = ax[1].imshow(code, cmap=cmap, vmin=-0.5, vmax=len(REASONS) - 0.5)
                ax[1].set_title(f"oracle, ok cells {n_ok}")
                cb = fig.colorbar(im, ax=ax[1], ticks=range(len(REASONS)))
                cb.ax.set_yticklabels(REASONS)
                fig.savefig(out / f"oracle_seed{seed:03d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

    t_in, ss = np.asarray(t_in) * 1e3, np.asarray(side_spreads) * 1e3
    pct = lambda x: np.round(np.percentile(x, [50, 90, 100]), 3) if len(x) else "none"
    print(f"reason counts: {counts}")
    print(f"in-mask check ms p50/p90/max: {pct(t_in)}; oracle s/scene p50/max: {pct(np.asarray(scene_s))}")
    print(f"upright ok-disc diameter px (expect ~15-17): {np.round(disc_px, 1)}")
    if len(ss):
        print(f"side scenes: ok cells {side_ok}; ring-hit spread mm p50/p90/max: {pct(ss)}; "
              f"fraction over budget {np.mean(ss > cfg['suction']['max_spread'] * 1e3):.1%}")
    cl = np.asarray(clean) * 1e3
    budget = cfg["suction"]["max_spread"] * 1e3
    print(f"clean barrel n={len(cl)}: spread mm p50/p90/max {pct(cl)}, "
       f"over budget {np.mean(cl > budget):.1%}" if len(cl) else "clean barrel: no cells")    
    ok = (all(13 <= d <= 19 for d in disc_px) and len(disc_px) > 0 and side_ok > 0
          and np.median(t_in) < 1.0 and max(scene_s) < 5.0)
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED -- paste this output")


if __name__ == "__main__":
    main()