# scripts/dump_scenes.py
import os, numpy as np
from pathlib import Path
from cpick.config import load_config
ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(str(ROOT / "configs" / "scene.yaml"))
os.environ.setdefault("MUJOCO_GL", "egl")
from cpick.envs.pick_env import PickEnv
env = PickEnv(cfg)
qs, ds = [], []
for s in range(20):
    obs, _ = env.reset(options={"scene_seed": s})
    qs.append(env.data.qpos.copy()); ds.append(obs["depth"].copy())
env.close()
np.savez(ROOT / "outputs" / f"scenes_{os.uname().nodename}.npz", q=np.stack(qs), d=np.stack(ds))