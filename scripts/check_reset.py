"""M3: 100 seeded drops -> settle stats, determinism, post-settle drift, final-pose mix."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "m3"))
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from cpick.sim.camera import Camera
    from cpick.sim.reset import CanDropper
    from cpick.sim.scene_builder import build_scene_xml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
    data = mujoco.MjData(model)
    dropper = CanDropper(model, data, cfg)
    rc, body = cfg["reset"], model.body("can").id
    verify_steps = int(round(rc["verify_s"] / model.opt.timestep))

    steps, walls, attempts, timeouts, drifts, poses, failed = [], [], [], [], [], [], []
    with Camera(model, cfg["debug_camera"]["name"], 640, 480) as dbg:
        drawn_all, accepted_draw = [], []
        for seed in range(args.n):
            try:
                _, info = dropper.reset(seed)
                drawn_all += info["drawn"]; accepted_draw.append(info["drawn"][-1])
            except RuntimeError as e:
                failed.append(seed)
                print(e)
                continue
            steps.append(info["steps"]); walls.append(info["wall_s"])
            attempts.append(info["attempts"]); timeouts.append(info["timeouts"])
            axis_z = data.xmat[body].reshape(3, 3)[2, 2]    # can axis . world up
            poses.append("upright" if axis_z > 0.9 else "inverted" if axis_z < -0.9
                         else "side" if abs(axis_z) < 0.1 else "leaning")
            if seed < 6:
                plt.imsave(out / f"seed{seed:03d}.png", dbg.rgb(data))
            c0 = data.xipos[body].copy()
            for _ in range(verify_steps):
                mujoco.mj_step(model, data)
            drifts.append(np.linalg.norm(data.xipos[body] - c0))

    dets = []
    for seed in range(5):
        q1, _ = dropper.reset(seed)
        q2, _ = dropper.reset(seed)
        dets.append(np.abs(q1 - q2).max())

    steps, walls, drifts = map(np.asarray, (steps, walls, drifts))
    pct = lambda x: np.round(np.percentile(x, [50, 90, 100]), 4)
    print(f"failed seeds: {failed}")
    print(f"settle steps p50/p90/max: {pct(steps)}  (sim s: {pct(steps * model.opt.timestep)})")
    print(f"wall-clock s p50/p90/max: {pct(walls)}")
    print(f"resets needing >1 attempt: {sum(a > 1 for a in attempts)}, total timeouts: {sum(timeouts)}")
    print(f"post-settle drift over {rc['verify_s']} s, mm p50/p90/max: {pct(drifts * 1000)}")
    print(f"final pose mix: { {k: poses.count(k) for k in sorted(set(poses))} }")
    print(f"determinism max |dq| over 5 seeds: {max(dets):.3e}")
    print(f"drawn mix (all attempts): { {k: drawn_all.count(k) for k in sorted(set(drawn_all))} }")
    print(f"accepted draw type:       { {k: accepted_draw.count(k) for k in sorted(set(accepted_draw))} }")

    drifts_by = {}
    for pose, d in zip(poses, drifts):
        drifts_by.setdefault(pose, []).append(d * 1000)
    for k, v in sorted(drifts_by.items()):
        print(f"  drift {k:9s} n={len(v):3d} max {max(v):.3f} mm")

    ok = (not failed and sum(timeouts) == 0 and drifts.max() < rc["verify_drift"]
          and max(dets) < 1e-9)
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED ")


if __name__ == "__main__":
    main()