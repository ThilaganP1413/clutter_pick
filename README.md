# clutter_pick_ur5e_mujoco

Sim-only reproduction of CODs / SRDONs suction bin-picking in MuJoCo.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt && pip install -e .
    python scripts/check_render.py          # headless, writes outputs/stage1/
    python scripts/view_scene.py            # needs a display

`cpick/sim` and `cpick/envs` must never import torch or rclpy.
