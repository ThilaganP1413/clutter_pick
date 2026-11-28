"""PPO must overfit one fixed scene (sampled success -> ~100%, argmax succeeds)."""
import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", default="side", choices=["side", "upright"])
    p.add_argument("--max-updates", type=int, default=150)
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(str(ROOT / "configs" / "scene.yaml"))
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import torch
    from cpick.envs.pick_env import PickEnv
    from cpick.models.picknet import PickNet
    from cpick.rl.ppo import Rollout, act, ppo_update

    rl, torch_seed = cfg["rl"], 0
    torch.manual_seed(torch_seed)
    env = PickEnv(cfg, split="train")
    try:
        for s in range(1000):
            obs, info = env.reset(options={"scene_seed": s})
            if info["pose"] == args.pose:
                break
        print(f"scene seed {s} ({args.pose}), {int(obs['mask'].sum())} valid cells")
        dev = rl["device"]
        net = PickNet().to(dev).eval()                   # BN frozen: same normalisation in rollout and update
        opt = torch.optim.Adam(net.parameters(), lr=rl["lr"])
        buf = Rollout(rl["rollout"], *obs["depth"].shape, obs["mask"].shape[0], dev)
        streak, t_start = 0, time.perf_counter()
        for u in range(1, args.max_updates + 1):
            buf.reset()
            for _ in range(rl["rollout"]):
                a, logp, v, d, m = act(net, obs["depth"], obs["mask"], dev)
                _, r, *_ = env.step(a)      # same scene every time: step() changes nothing, so no reset
                buf.add(d, m, a, logp, v, r)
            succ = buf.reward.mean().item()
            st = ppo_update(net, opt, buf, rl)
            extra = f", first ratio dev {st['first_ratio_dev']:.1e}" if u == 1 else ""
            ga, *_ = act(net, obs["depth"], obs["mask"], dev, greedy=True)
            _, gr, _, _, _ = env.step(ga)
            g_ok = gr == rl["reward_success"]
            print(f"upd {u:3d}  success {succ:5.1%}  argmax {'ok ' if g_ok else 'BAD'}  "
                  f"V {buf.value.mean().item():.3f}  ent {st['ent']:.2f}  kl {st.get('kl', 0):.4f}  "
                  f"stop_kl {st.get('stop_kl', 0):.3f}  mb {st['minibatches']:2d}  gnorm {st['grad_norm']:.1f}  "
                  f"t {time.perf_counter() - t_start:5.0f}s{extra}")
            streak = streak + 1 if (succ >= 0.9 and g_ok) else 0
            if streak >= 3:
                break
        a, *_ = act(net, obs["depth"], obs["mask"], dev, greedy=True)
        _, r, _, _, si = env.step(a)
        print(f"argmax pick {divmod(a, 128)}: reward {r}, reason {si['reason']}")
        print("OVERFIT PASS" if streak >= 3 else "OVERFIT FAIL")
    finally:
        env.close()


if __name__ == "__main__":
    main()