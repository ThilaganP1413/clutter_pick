"""M7 piece 1: env sanity -> random-in-mask success ~ M6, determinism, timing."""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from cpick.config import load_config
    cfg = load_config(str(ROOT / "configs" / "scene.yaml"))
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import numpy as np
    from cpick.envs.pick_env import GRID, PickEnv

    env = PickEnv(cfg, split="train")
    rng = np.random.default_rng(0)
    try:
        obs, info = env.reset(seed=0)
        assert obs["depth"].shape == (cfg["camera"]["height_px"], cfg["camera"]["width"])
        assert obs["depth"].dtype == np.float32 and obs["mask"].shape == (GRID, GRID)
        d1 = env.reset(options={"scene_seed": 123})[0]["depth"].copy()
        d2 = env.reset(options={"scene_seed": 123})[0]["depth"]
        print(f"determinism: max |d depth| {np.abs(d1 - d2).max():.2e}")

        n, wins, t_reset, t_step, reasons = 1000, 0, 0.0, 0.0, {}
        for _ in range(n):
            t0 = time.perf_counter()
            obs, info = env.reset()
            t1 = time.perf_counter()
            ii, jj = np.nonzero(obs["mask"])
            k = rng.integers(len(ii))
            _, r, term, trunc, si = env.step(ii[k] * GRID + jj[k])
            t_step += time.perf_counter() - t1
            t_reset += t1 - t0
            wins += r == cfg["rl"]["reward_success"]
            reasons[si["reason"]] = reasons.get(si["reason"], 0) + 1
            assert term and not trunc
        print(f"random-in-mask success {wins / n:.1%} (M6 ~4.5%); reasons {reasons}")
        print(f"ms/reset {t_reset / n * 1e3:.2f}, ms/step {t_step / n * 1e3:.3f}, "
              f"env-only steps/s {n / (t_reset + t_step):.0f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()