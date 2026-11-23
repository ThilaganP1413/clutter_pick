"""Stage 1, piece 1: build the empty bin scene, render, verify, save images. Headless-safe."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "stage1"))
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])   # before importing mujoco

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from cpick.sim.camera import Camera
    from cpick.sim.scene_builder import build_scene_xml

    print(f"mujoco {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xml = build_scene_xml(cfg)
    (out / "scene.xml").write_text(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    c, b = cfg["camera"], cfg["bin"]
    with Camera(model, c["name"], c["width"], c["height_px"]) as cam:
        rgb, depth, seg = cam.rgb(data), cam.depth(data), cam.segmentation(data)
        K, T = cam.intrinsics(), cam.cam_to_world(data)
    with Camera(model, cfg["debug_camera"]["name"], 640, 480) as dbg:
        rgb_debug = dbg.rgb(data)

    ok = True
    # 1) Is depth already metric? Image centre sees the bin floor top.
    h, w = depth.shape
    measured = float(depth[h // 2, w // 2])
    expected = c["height"] - b["floor_thickness"]
    err = abs(measured - expected)
    print(f"[1] centre depth {measured:.5f} m, expected {expected:.5f} m, |err| {err*1000:.2f} mm")
    ok &= err < 1e-3

    # 2) Camera pose: OpenCV z-axis (optical axis) should point straight down.
    z_axis = T[:3, 2]
    print(f"[2] optical axis in world {np.round(z_axis, 4)}, camera at {np.round(T[:3, 3], 4)}")
    ok &= np.allclose(z_axis, [0, 0, -1], atol=1e-6)

    # 3) Segmentation: which geoms are visible, and how much of the frame is bin.
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    bin_geoms = np.flatnonzero(model.geom_bodyid == bin_id)
    visible = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in np.unique(seg) if g >= 0]
    fill = float(np.isin(seg, bin_geoms).mean())
    print(f"[3] visible geoms {visible}; bin covers {fill:.1%} of the frame")
    ok &= "bin_floor" in visible

    print("K =\n", np.round(K, 3))
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED -- paste this output")

    plt.imsave(out / "rgb.png", rgb)
    plt.imsave(out / "rgb_debug.png", rgb_debug)
    plt.imsave(out / "seg.png", seg, cmap="tab20")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(depth, cmap="viridis")
    fig.colorbar(im, ax=ax, label="depth [m]")
    ax.set_title("top_cam depth")
    fig.savefig(out / "depth.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved images + scene.xml to {out}")


if __name__ == "__main__":
    main()
