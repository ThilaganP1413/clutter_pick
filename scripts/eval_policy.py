"""M7 gate: evaluate each run's best.pt once on a split; mean / worst seed vs the pre-registered gate."""
import argparse
import glob
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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


def val_history(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run / "tb"))
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars("val/all/argmax")]


def rising(hist):
    """Mean val argmax over the last 20% of training minus the preceding 20%."""
    steps = np.array([s for s, _ in hist if s > 0], float)
    vals = np.array([v for s, v in hist if s > 0])
    if len(steps) < 4:
        return float("nan")
    top = steps.max()
    last, prev = vals[steps >= 0.8 * top], vals[(steps >= 0.6 * top) & (steps < 0.8 * top)]
    return float(last.mean() - prev.mean()) if len(last) and len(prev) else float("nan")


@torch.no_grad()
def eval_net(net, env, seeds, dev, rl, gen):
    from cpick.models.picknet import masked_dist
    rows = []
    for s in seeds:
        obs, info = env.reset(options={"scene_seed": s})
        depth = torch.as_tensor(obs["depth"], device=dev)[None]
        mask = torch.as_tensor(obs["mask"], device=dev).bool()[None]
        logits, _ = net(depth)
        d = masked_dist(logits, mask)
        row = {"seed": s, "pose": info["pose"]}
        for key, a in (("argmax", int(d.logits.argmax(-1))),
                       ("sampled", int(torch.multinomial(d.probs, 1, generator=gen)))):
            _, r, _, _, si = env.step(a)
            row[key], row[key + "_reason"] = bool(r == rl["reward_success"]), si["reason"]
        rows.append(row)
    return rows


def summarize(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r[key])
        groups[r["pose"]].append(r[key])
    out = {}
    for g, v in groups.items():
        lo, hi = wilson(sum(v), len(v))
        out[g] = {"rate": sum(v) / len(v), "ci95": [lo, hi], "n": len(v)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "outputs" / "runs" / "1a_depth_s*"), help="glob of run dirs")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--gate-overall", type=float, default=0.80)
    p.add_argument("--gate-upright", type=float, default=0.80)
    p.add_argument("--out", default=str(ROOT / "outputs" / "m7"))
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        raise SystemExit("working tree is dirty: commit first (or --allow-dirty)")
    runs = sorted(Path(r) for r in glob.glob(args.runs))
    if not runs:
        raise SystemExit(f"no runs match {args.runs}")

    from cpick.config import load_config
    cfgs = [load_config(str(r / "config.yaml")) for r in runs]
    scene = lambda c: {k: v for k, v in c.items() if k != "rl"}
    if any(scene(c) != scene(cfgs[0]) for c in cfgs[1:]):
        raise SystemExit("runs were trained on different scene configs; refusing to compare")
    cfg = cfgs[0]
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    from cpick.envs.pick_env import PickEnv
    from cpick.models.picknet import PickNet
    from cpick.seeds import split_seeds

    dev, rl = cfg["rl"]["device"], cfg["rl"]
    env = PickEnv(cfg, split="train")                   # scenes are chosen explicitly below
    seeds = list(split_seeds(args.split))
    fmt = lambda s: f"{s['rate']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}]"
    results = {}
    try:
        for run in runs:
            ck = torch.load(run / "ckpt" / "best.pt", map_location=dev)
            last = torch.load(run / "ckpt" / "last.pt", map_location="cpu")["step"]
            meta = json.loads((run / "meta.json").read_text())
            net = PickNet(pretrained=False).to(dev).eval()
            net.load_state_dict(ck["net"])
            gen = torch.Generator(device=dev).manual_seed(20_000)
            rows = eval_net(net, env, seeds, dev, rl, gen)
            hist = val_history(run)
            res = {"best_step": ck["step"], "last_step": last, "budget": meta["total_steps"],
                   "train_commit": meta["git"]["commit"], "val_rise_last20": rising(hist),
                   "argmax": summarize(rows, "argmax"), "sampled": summarize(rows, "sampled"),
                   "val_history": hist, "rows": rows}
            results[run.name] = res
            a = res["argmax"]
            print(f"{run.name}: best step {ck['step']} (trained to {last}/{meta['total_steps']}), "
                  f"val rise over last 20% {res['val_rise_last20']:+.1%}")
            print(f"   argmax  all {fmt(a['all'])}  upright {fmt(a.get('upright', {'rate': 0, 'ci95': [0, 0]}))}  "
                  f"| sampled all {fmt(res['sampled']['all'])}")
            print("   per pose argmax: " + "  ".join(f"{k} {v['rate']:.1%} (n={v['n']})"
                                                  for k, v in sorted(a.items()) if k != "all"))
    finally:
        env.close()

    alls = [r["argmax"]["all"]["rate"] for r in results.values()]
    ups = [r["argmax"].get("upright", {"rate": 0.0})["rate"] for r in results.values()]
    passed = np.mean(alls) >= args.gate_overall and np.mean(ups) >= args.gate_upright
    print(f"\n{len(runs)} seeds on {args.split}: argmax all mean {np.mean(alls):.1%} (worst {min(alls):.1%}), "
          f"upright mean {np.mean(ups):.1%} (worst {min(ups):.1%})")
    ref = ROOT / "outputs" / "m6" / f"baselines_{args.split}.json"
    if ref.exists():
        h = json.loads(ref.read_text())["summary"]
        print(f"reference heuristic ({ref.name}): all {h['all']['heuristic']['rate']:.1%}, "
              f"upright {h['upright']['heuristic']['rate']:.1%}")
    print(f"GATE (all >= {args.gate_overall:.0%}, upright >= {args.gate_upright:.0%}, mean over seeds): "
          f"{'PASS' if passed else 'FAIL'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gate_{args.split}.json").write_text(json.dumps(
        {"split": args.split, "git": git, "gate": {"overall": args.gate_overall, "upright": args.gate_upright},
         "passed": bool(passed), "results": results}, indent=1, default=float))


if __name__ == "__main__":
    main()