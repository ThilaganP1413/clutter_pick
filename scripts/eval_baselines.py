"""M6: oracle coverage + random-in-mask + flattest-point baselines on a seed split, saved to JSON."""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def git_state():
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default=str(ROOT / "outputs" / "m6"))
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import mujoco
    import numpy as np
    from cpick.policies.baselines import flattest_above_floor, random_in_mask
    from cpick.seeds import split_seeds
    from cpick.sim.camera import Camera
    from cpick.sim.geometry import backproject, surface_variation
    from cpick.sim.reset import CanDropper
    from cpick.sim.scene_builder import build_scene_xml
    from cpick.sim.suction import SuctionChecker

    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
    data = mujoco.MjData(model)
    dropper, checker = CanDropper(model, data, cfg), SuctionChecker(model, cfg)
    c, body = cfg["camera"], model.body("can").id
    objects = {model.geom("can").id}
    floor_z, radius = cfg["bin"]["floor_thickness"], cfg["suction"]["cup_radius"]
    rng = np.random.default_rng(12345)          # baseline sampling RNG, separate from scene seeds

    rows = []
    with Camera(model, c["name"], c["width"], c["height_px"]) as cam:
        for seed in split_seeds(args.split):
            dropper.reset(seed)
            depth = cam.depth(data)
            K, T = cam.intrinsics(), cam.cam_to_world(data)
            P, cam_pos = backproject(depth, K, T), T[:3, 3].copy()
            az = data.xmat[body].reshape(3, 3)[2, 2]
            pose = "upright" if az > 0.9 else "inverted" if az < -0.9 else "side" if abs(az) < 0.1 else "leaning"
            valid = checker.valid_mask(P)[::2, ::2]
            okmap = np.zeros_like(valid)
            for i, j in zip(*np.nonzero(valid)):
                okmap[i, j] = checker.check(data, P, depth, 2 * j, 2 * i, K, cam_pos, objects).ok
            ri, rj = random_in_mask(valid, rng)
            h = flattest_above_floor(P, depth, K, valid, floor_z, radius)
            if h is not None and not okmap[h]:
                i, j = h
                r = checker.check(data, P, depth, 2 * j, 2 * i, K, cam_pos, objects)
                a, com = data.xmat[body].reshape(3, 3)[:, 2], data.xipos[body]
                rel = P[2 * i, 2 * j] - com
                radial = np.linalg.norm(rel - (rel @ a) * a)
                var, n = surface_variation(P, depth, 2 * j, 2 * i, radius, K)
                ok_rad = []
                for oi, oj in zip(*np.nonzero(okmap)):
                    q = P[2 * oi, 2 * oj] - com
                    ok_rad.append(np.linalg.norm(q - (q @ a) * a))
                print(f"seed {seed} {pose}: {r.reason}, spread {r.spread * 1e3:.2f} mm, pick radial "
                      f"{radial * 1e3:.1f} mm (ok cells span {min(ok_rad) * 1e3:.1f}-{max(ok_rad) * 1e3:.1f}), "
                      f"z {P[2 * i, 2 * j, 2] * 1e3:.2f} mm, var {var:.1e}, n {n}")
            rows.append(dict(
                seed=seed,
                pose="upright" if az > 0.9 else "inverted" if az < -0.9 else "side" if abs(az) < 0.1 else "leaning",
                any_ok=bool(okmap.any()), ok_frac=float(okmap.sum() / valid.sum()),
                random=bool(okmap[ri, rj]), heuristic=bool(okmap[h]) if h else False,
                heuristic_none=h is None))

    def rate(key, sub):
        k = sum(r[key] for r in sub)
        lo, hi = wilson(k, len(sub))
        return {"rate": k / len(sub), "ci95": [lo, hi], "n": len(sub)}

    summary = {"all": {k: rate(k, rows) for k in ("any_ok", "random", "heuristic")}}
    summary["all"]["random_expected"] = float(np.mean([r["ok_frac"] for r in rows]))
    summary["all"]["heuristic_none"] = sum(r["heuristic_none"] for r in rows)
    for pose in sorted({r["pose"] for r in rows}):
        sub = [r for r in rows if r["pose"] == pose]
        summary[pose] = {k: rate(k, sub) for k in ("any_ok", "random", "heuristic")}

    fmt = lambda s: f"{s['rate']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}] n={s['n']}"
    print(f"split {args.split}, {len(rows)} scenes, {git_state()}")
    for group, s in summary.items():
        print(f"{group:9s} oracle {fmt(s['any_ok'])} | random {fmt(s['random'])} | heuristic {fmt(s['heuristic'])}")
    print(f"random expected (mean ok fraction): {summary['all']['random_expected']:.1%}; "
          f"heuristic found no candidate in {summary['all']['heuristic_none']} scenes")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"baselines_{args.split}.json").write_text(json.dumps(
        {"split": args.split, "git": git_state(), "config": cfg, "summary": summary, "rows": rows},
        indent=1, default=float))


if __name__ == "__main__":
    main()