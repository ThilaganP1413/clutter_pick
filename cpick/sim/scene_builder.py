"""Compose the bin-scene MJCF from a config dict.

Plain XML composition (not MjSpec) on purpose: it is stable across MuJoCo
versions and the generated file can be dumped and inspected.
"""
import xml.etree.ElementTree as ET

import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def _v(values):
    return " ".join(f"{float(x):.6g}" for x in values)


def _look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera axes: x right, y up, looking along -z."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return np.concatenate([x, y])


def build_scene_xml(cfg, with_can=False):
    t, b, c, dc = cfg["table"], cfg["bin"], cfg["camera"], cfg["debug_camera"]

    root = ET.Element("mujoco", model="clutter_bin")
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(root, "option", timestep=str(cfg["physics"]["timestep"]))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")

    if with_can:
        cc = cfg["can"]
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="can_mesh", file=str(ROOT / cc["mesh"]), inertia="exact")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")

    # Table: top surface at z = 0.
    tx, ty, tz = t["size"]
    ET.SubElement(world, "geom", name="table", type="box",
                  size=_v([tx / 2, ty / 2, tz / 2]), pos=_v([0, 0, -tz / 2]),
                  rgba="0.85 0.82 0.70 1")

    # Bin: floor + 4 walls. x-walls span the full outer y, y-walls span inner x only.
    ix, iy = b["inner_size"]
    wt, ft, h = b["wall_thickness"], b["floor_thickness"], b["wall_height"]
    rgba = "0.95 0.85 0.10 1"
    bin_body = ET.SubElement(world, "body", name="bin", pos="0 0 0")
    ET.SubElement(bin_body, "geom", name="bin_floor", type="box",
                  size=_v([ix / 2 + wt, iy / 2 + wt, ft / 2]), pos=_v([0, 0, ft / 2]), rgba=rgba)
    for sign, tag in ((1, "px"), (-1, "nx")):
        ET.SubElement(bin_body, "geom", name=f"bin_wall_{tag}", type="box",
                      size=_v([wt / 2, iy / 2 + wt, h / 2]),
                      pos=_v([sign * (ix / 2 + wt / 2), 0, h / 2]), rgba=rgba)
    for sign, tag in ((1, "py"), (-1, "ny")):
        ET.SubElement(bin_body, "geom", name=f"bin_wall_{tag}", type="box",
                      size=_v([ix / 2, wt / 2, h / 2]),
                      pos=_v([0, sign * (iy / 2 + wt / 2), h / 2]), rgba=rgba)

    # Policy camera: straight down (MuJoCo default orientation looks along -z).
    ET.SubElement(world, "camera", name=c["name"], pos=_v([0, 0, c["height"]]),
                  fovy=str(c["fovy"]))
    # Debug camera: angled, only for human-readable images/videos.
    ET.SubElement(world, "camera", name=dc["name"], pos=_v(dc["pos"]), fovy="45",
                  xyaxes=_v(_look_at_xyaxes(dc["pos"], [0, 0, h / 2])))

    if with_can:
        x, y = cc["init_xy"]
        can = ET.SubElement(world, "body", name="can", pos=_v([x, y, b["floor_thickness"]]))
        ET.SubElement(can, "freejoint", name="can_free")
        ET.SubElement(can, "geom", name="can", type="mesh", mesh="can_mesh",
                      mass=str(cc["mass"]), rgba=_v(cc["rgba"]),
                      condim=str(cc["condim"]), friction=_v(cc["friction"]))

    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
