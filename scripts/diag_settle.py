"""M3 diagnosis: what do tilted drops do during the settle timeout?"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from cpick.config import load_config
    cfg = load_config(str(ROOT / "configs" / "scene.yaml"))
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from cpick.sim.reset import CanDropper
    from cpick.sim.scene_builder import build_scene_xml

    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
    data = mujoco.MjData(model)
    dropper = CanDropper(model, data, cfg)
    s, dt, body, va = cfg["reset"]["settle"], model.opt.timestep, model.body("can").id, dropper.va
    n, last = int(round(s["timeout_s"] / dt)), int(round(1.0 / dt))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    found = 0
    for seed in range(100):
        dropper._spawn(np.random.default_rng(seed))        # same first draw as reset(seed)
        if data.xmat[body].reshape(3, 3)[2, 2] > 0.99:
            continue                                        # skip upright draws
        v, w, nc = np.zeros(n), np.zeros(n), np.zeros(n)
        for k in range(n):
            mujoco.mj_step(model, data)
            q = data.qvel[va:va + 6]
            v[k], w[k], nc[k] = np.linalg.norm(q[:3]), np.linalg.norm(q[3:]), data.ncon
        t = np.arange(1, n + 1) * dt
        axes[0].semilogy(t, v + 1e-9, label=f"seed {seed}")
        axes[1].semilogy(t, w + 1e-9)
        axes[2].plot(t, nc)
        az = data.xmat[body].reshape(3, 3)[2, 2]
        print(f"seed {seed}: final |v| {v[-1] * 1e3:.3f} mm/s, |w| {w[-1]:.4f} rad/s | "
              f"last 1 s mean |v| {v[-last:].mean() * 1e3:.3f} mm/s, |w| {w[-last:].mean():.4f}, "
              f"ncon min/max {nc[-last:].min():.0f}/{nc[-last:].max():.0f} | axis_z {az:+.3f}, "
              f"COM {np.round(data.xipos[body] * 1e3, 1)} mm")
        found += 1
        if found == 4:
            break

    axes[0].axhline(s["lin_vel"], color="k", ls="--"); axes[0].set_ylabel("|v| [m/s]")
    axes[1].axhline(s["ang_vel"], color="k", ls="--"); axes[1].set_ylabel("|w| [rad/s]")
    axes[2].set_ylabel("ncon"); axes[2].set_xlabel("time [s]"); axes[0].legend()
    out = ROOT / "outputs" / "m3"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diag_settle.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()