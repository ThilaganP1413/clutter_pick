"""M2: compile the can into the bin scene; verify extents, frame, inertia, rest, render."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "scene.yaml"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "m2"))
    args = p.parse_args()

    from cpick.config import load_config
    cfg = load_config(args.config)
    if cfg["render"]["gl_backend"]:
        os.environ.setdefault("MUJOCO_GL", cfg["render"]["gl_backend"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from cpick.sim.camera import Camera
    from cpick.sim.scene_builder import build_scene_xml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xml = build_scene_xml(cfg, with_can=True)
    (out / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cc, c = cfg["can"], cfg["camera"]
    body, geom = model.body("can").id, model.geom("can").id
    mid = model.geom_dataid[geom]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    V = model.mesh_vert[a:a + n]                       # mesh frame (MuJoCo-recentred)

    def world_verts():
        return V @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]

    nominal = np.asarray(cc["nominal_dims"], float)
    ok = True

    # 1) Extents after compilation, in world frame.
    W = world_verts()
    ext = W.max(0) - W.min(0)
    err = np.abs(ext - nominal).max()
    print(f"[1] extents {np.round(ext * 1000, 2)} mm, nominal {nominal * 1000} mm, max err {err * 1000:.2f} mm")
    ok &= err < 1e-3

    # 2) Frame: body origin at the base, COM on the axis.
    base_err = abs(W[:, 2].min() - data.xpos[body][2])
    com_off = np.linalg.norm(data.xipos[body][:2] - data.xpos[body][:2])
    print(f"[2] base vs body origin {base_err * 1000:.3f} mm, COM off-axis {com_off * 1000:.3f} mm")
    ok &= base_err < 1e-3 and com_off < 5e-4

    # 3) Mass and inertia vs a solid cylinder (principal values, sorted).
    m, I = model.body_mass[body], np.sort(model.body_inertia[body])
    r, h = nominal[0] / 2, nominal[2]
    I_cyl = np.sort([m * (3 * r * r + h * h) / 12] * 2 + [m * r * r / 2])
    rel = np.abs(I / I_cyl - 1).max()
    tri = I[0] + I[1] >= I[2]
    print(f"[3] mass {m:.4f} kg, I {I}, cylinder {I_cyl}, max rel diff {rel:.1%}, triangle ok {tri}")
    ok &= abs(m - cc["mass"]) < 1e-6 and rel < 0.2 and tri and (I > 0).all()

    # 4) Rest test: 2 s upright on the floor.
    p0, pen = data.xpos[body].copy(), 0.0
    for _ in range(int(round(2.0 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            pen = min(pen, data.contact[i].dist)
    drift = np.linalg.norm(data.xpos[body] - p0)
    tilt = np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))
    print(f"[4] drift {drift * 1000:.3f} mm, tilt {tilt:.3f} deg, max penetration {-pen * 1000:.3f} mm")
    ok &= drift < 1e-3 and tilt < 1.0 and -pen < 1e-3

    # 5) Render vs ray: cast mj_ray through the same pixel the depth image samples.
    with Camera(model, c["name"], c["width"], c["height_px"]) as cam:
        depth, seg = cam.depth(data), cam.segmentation(data)
        K, T = cam.intrinsics(), cam.cam_to_world(data)
    with Camera(model, cfg["debug_camera"]["name"], 640, 480) as dbg:
        rgb_debug = dbg.rgb(data)
    hh, ww = depth.shape
    u, v = ww // 2, hh // 2
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])          # z-component = 1
    d_world = T[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
    gid = np.full(1, -1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, T[:3, 3].copy(), d_world, None, 1, -1, gid)
    expected = dist / np.linalg.norm(d_cam)                   # ray length -> z-depth
    measured = float(depth[v, u])
    hit = seg[v, u] == geom and gid[0] == geom
    rim_z = world_verts()[:, 2].max()
    surf_z = c["height"] - measured
    print(f"[5] depth {measured:.5f} m, ray {expected:.5f} m, |err| {abs(measured - expected) * 1000:.3f} mm, "
          f"seg+ray hit can {hit}; surface {(rim_z - surf_z) * 1000:.2f} mm below rim top")
    ok &= abs(measured - expected) < 1e-3 and hit

    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED -- paste this output")
    plt.imsave(out / "rgb_debug.png", rgb_debug)
    plt.imsave(out / "seg.png", seg, cmap="tab20")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(depth, cmap="viridis")
    fig.colorbar(im, ax=ax, label="depth [m]")
    fig.savefig(out / "depth.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()