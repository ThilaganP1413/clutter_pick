"""M7: PPO training for 1a (one can). Refuses a dirty tree; TensorBoard logs; val eval + checkpoints."""
import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def git_state():
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


@torch.no_grad()
def evaluate(net, env, seeds, device, rl, gen):
    """Argmax and sampled success on fixed scenes, per pose. Sampling uses its own generator."""
    from cpick.models.picknet import masked_dist
    res = defaultdict(lambda: [0, 0, 0])                  # pose -> [n, argmax wins, sampled wins]
    for s in seeds:
        obs, info = env.reset(options={"scene_seed": s})
        depth = torch.as_tensor(obs["depth"], device=device)[None]
        mask = torch.as_tensor(obs["mask"], device=device).bool()[None]
        logits, _ = net(depth)
        d = masked_dist(logits, mask)
        picks = (int(d.logits.argmax(-1)), int(torch.multinomial(d.probs, 1, generator=gen)))
        res[info["pose"]][0] += 1
        for k, a in enumerate(picks, start=1):
            _, r, *_ = env.step(a)                         # step() has no side effects: same scene twice
            res[info["pose"]][k] += r == rl["reward_success"]
    n = sum(v[0] for v in res.values())
    out = {"all/argmax": sum(v[1] for v in res.values()) / n, "all/sampled": sum(v[2] for v in res.values()) / n}
    for pose, (m, g, sa) in res.items():
        out[f"{pose}/argmax"], out[f"{pose}/sampled"], out[f"{pose}/n"] = g / m, sa / m, m
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--name", default="1a_depth")
    p.add_argument("--seed", type=int, default=0, help="training seed (torch + scene-draw RNG)")
    p.add_argument("--steps", type=int, default=None, help="override rl.total_steps (smoke tests)")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        raise SystemExit("working tree is dirty: commit first (or --allow-dirty for local experiments)")

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    from torch.utils.tensorboard import SummaryWriter
    from cpick.envs.pick_env import PickEnv
    from cpick.models.picknet import PickNet
    from cpick.rl.ppo import Rollout, act, ppo_update
    from cpick.seeds import split_seeds

    rl = cfg["rl"]
    total, dev = args.steps or rl["total_steps"], rl["device"]
    run = ROOT / "outputs" / "runs" / f"{args.name}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    (run / "ckpt").mkdir(parents=True)
    shutil.copy(args.config, run / "config.yaml")
    (run / "meta.json").write_text(json.dumps({"git": git, "seed": args.seed, "total_steps": total}, indent=1))
    print(f"run dir {run}")

    torch.manual_seed(args.seed)
    env = PickEnv(cfg, split="train")
    val_seeds = list(split_seeds("val"))
    net = PickNet().to(dev).eval()                        # BN frozen (see M7 piece 3)
    opt = torch.optim.Adam(net.parameters(), lr=rl["lr"])
    gen = torch.Generator(device=dev).manual_seed(10_000 + args.seed)
    tb = SummaryWriter(str(run / "tb"))
    obs, info = env.reset(seed=args.seed)
    buf = Rollout(rl["rollout"], *obs["depth"].shape, obs["mask"].shape[0], dev)
    step, next_eval, best, n_upd = 0, 0, -1.0, 0
    try:
        while True:
            if step >= next_eval:
                t_ev = time.perf_counter()
                ev = evaluate(net, env, val_seeds, dev, rl, gen)
                for k, v in ev.items():
                    if not k.endswith("/n"):
                        tb.add_scalar(f"val/{k}", v, step)
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "val": ev},
                           run / "ckpt" / "last.pt")
                tag = ""
                if step > 0 and ev["all/argmax"] > best:
                    best = ev["all/argmax"]
                    torch.save({"net": net.state_dict(), "step": step, "val": ev}, run / "ckpt" / "best.pt")
                    tag = "  * best"
                poses = "  ".join(f"{k.split('/')[0]} {ev[k]:.1%}" for k in sorted(ev)
                                  if k.endswith("/argmax") and not k.startswith("all"))
                print(f"[eval {step:6d}] argmax {ev['all/argmax']:.1%}  sampled {ev['all/sampled']:.1%}  "
                      f"| {poses}  ({time.perf_counter() - t_ev:.1f}s){tag}")
                next_eval += rl["eval_every"]
            if step >= total:
                break

            buf.reset()
            reasons, pose_n, pose_w = Counter(), Counter(), Counter()
            t0 = time.perf_counter()
            for _ in range(rl["rollout"]):
                a, logp, v, d, m = act(net, obs["depth"], obs["mask"], dev)
                _, r, _, _, si = env.step(a)
                buf.add(d, m, a, logp, v, r)
                reasons[si["reason"]] += 1
                pose_n[info["pose"]] += 1
                pose_w[info["pose"]] += r == rl["reward_success"]
                obs, info = env.reset()                    # new training scene every step
            t1 = time.perf_counter()
            st = ppo_update(net, opt, buf, rl)
            t2 = time.perf_counter()
            step += rl["rollout"]
            n_upd += 1

            tb.add_scalar("train/success", buf.reward.mean().item(), step)
            tb.add_scalar("train/V", buf.value.mean().item(), step)
            for pose in pose_n:
                tb.add_scalar(f"train_pose/{pose}", pose_w[pose] / pose_n[pose], step)
            for k, c in reasons.items():
                tb.add_scalar(f"train_reason/{k}", c / rl["rollout"], step)
            for k in ("pg", "vf", "ent", "kl", "stop_kl", "grad_norm", "clipfrac", "minibatches"):
                if k in st:
                    tb.add_scalar(f"ppo/{k}", st[k], step)
            sps = rl["rollout"] / (t2 - t0)
            tb.add_scalar("time/steps_per_s", sps, step)
            tb.add_scalar("time/rollout_s", t1 - t0, step)
            tb.add_scalar("time/update_s", t2 - t1, step)
            if n_upd % 10 == 0:
                print(f"step {step:6d}  succ {buf.reward.mean().item():5.1%}  ent {st['ent']:.2f}  "
                      f"V {buf.value.mean().item():.3f}  mb {st['minibatches']:2d}  stop_kl {st.get('stop_kl', 0):.3f}  "
                      f"{sps:5.1f} steps/s")
    finally:
        tb.close()
        env.close()


if __name__ == "__main__":
    main()