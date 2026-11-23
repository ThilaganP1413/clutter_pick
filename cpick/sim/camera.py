"""Offscreen rendering for one named MuJoCo camera, plus OpenCV-convention K and pose."""
import mujoco
import numpy as np


class Camera:
    """Always close() it (or use `with Camera(...) as cam:`) to free the GL context."""

    def __init__(self, model, name, width, height):
        self.model = model
        self.name = name
        self.width, self.height = width, height
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cam_id < 0:
            raise ValueError(f"camera '{name}' not found in model")
        self._r = mujoco.Renderer(model, height=height, width=width)

    def _render(self, data, depth=False, seg=False):
        r = self._r
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        if depth:
            r.enable_depth_rendering()
        if seg:
            r.enable_segmentation_rendering()
        r.update_scene(data, camera=self.name)
        out = r.render().copy()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        return out

    def rgb(self, data):
        return self._render(data)                      # (H, W, 3) uint8

    def depth(self, data):
        return self._render(data, depth=True)          # (H, W) float32, see check_render.py

    def segmentation(self, data):
        """(H, W) int geom ids; -1 where no geom (background or non-geom object)."""
        s = self._render(data, seg=True)
        return np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)

    def intrinsics(self):
        """Pinhole K in OpenCV pixel convention (pixel (0,0) centre at 0). fovy is vertical."""
        fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])
        f = 0.5 * self.height / np.tan(0.5 * fovy)
        return np.array([[f, 0.0, (self.width - 1) / 2],
                         [0.0, f, (self.height - 1) / 2],
                         [0.0, 0.0, 1.0]])

    def cam_to_world(self, data):
        """4x4 pose, OpenCV axes (x right, y down, z forward). Needs mj_forward first."""
        R_mj = data.cam_xmat[self.cam_id].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = data.cam_xpos[self.cam_id]
        return T

    def close(self):
        if self._r is not None:
            self._r.close()
            self._r = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
