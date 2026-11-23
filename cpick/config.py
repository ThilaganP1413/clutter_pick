"""Config loading. Deliberately does not import mujoco (MUJOCO_GL must be set first)."""
from pathlib import Path

import yaml


def load_config(path):
    with open(Path(path)) as f:
        return yaml.safe_load(f)
