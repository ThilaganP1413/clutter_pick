"""Interactive viewer (head mode). Needs a display; never used by training."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    args = p.parse_args()

    os.environ["MUJOCO_GL"] = "glfw"     # the viewer needs a windowing backend, not egl
    import mujoco
    import mujoco.viewer
    from cpick.config import load_config
    from cpick.sim.scene_builder import build_scene_xml

    model = mujoco.MjModel.from_xml_string(build_scene_xml(load_config(args.config)))
    mujoco.viewer.launch(model)
