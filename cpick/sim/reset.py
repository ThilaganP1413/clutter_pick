"""Single-object drop reset: seeded spawn, settle detection, in-bin check. No torch."""
import time

import mujoco
import numpy as np


def _random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)                 # uniform over all 3D rotations, (w, x, y, z)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class CanDropper:
    def __init__(self, model, data, cfg):
        self.m, self.d = model, data
        self.rc, self.b = cfg["reset"], cfg["bin"]
        self.body = model.body("can").id
        j = model.joint("can_free").id
        self.qa, self.va = model.jnt_qposadr[j], model.jnt_dofadr[j]
        self.com_local = model.body_ipos[self.body].copy()   # COM in body frame

    def _spawn(self, rng):
        xr, yr = self.rc["xy_range"]
        com = np.array([rng.uniform(-xr, xr), rng.uniform(-yr, yr), self.rc["drop_z"]])
        upright = rng.random() < self.rc["p_upright"]
        q = _yaw_quat(rng.uniform(0.0, 2 * np.pi)) if upright else _random_quat(rng)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qa:self.qa + 3] = com - _quat_to_mat(q) @ self.com_local
        self.d.qpos[self.qa + 3:self.qa + 7] = q
        mujoco.mj_forward(self.m, self.d)
        return "upright" if upright else "tilted"

    def _settle(self):
        """Settled = pose change below thresholds for `calm_windows` consecutive windows."""
        s, dt = self.rc["settle"], self.m.opt.timestep
        win, limit = int(round(s["window_s"] / dt)), int(round(s["timeout_s"] / dt))
        p_ref = R_ref = None
        calm = 0
        for step in range(1, limit + 1):
            mujoco.mj_step(self.m, self.d)
            if step % win:
                continue
            p = self.d.xipos[self.body].copy()
            R = self.d.xmat[self.body].reshape(3, 3).copy()
            if p_ref is not None:
                dp = np.linalg.norm(p - p_ref)
                da = np.arccos(np.clip((np.trace(R_ref.T @ R) - 1) / 2, -1.0, 1.0))
                calm = calm + 1 if (dp < s["max_disp"] and da < s["max_rot"]) else 0
                if calm >= s["calm_windows"]:
                    return True, step
            p_ref, R_ref = p, R
        return False, limit

    def _in_bin(self):
        ix, iy = self.b["inner_size"]
        p = self.d.xipos[self.body]                         # COM in world
        return abs(p[0]) < ix / 2 and abs(p[1]) < iy / 2 and p[2] < self.b["wall_height"]

    def reset(self, seed):
        """Returns (can qpos[7], info). Same seed -> same draws -> same result."""
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        timeouts = 0
        drawn = []
        for attempt in range(1, self.rc["max_attempts"] + 1):
            drawn.append(self._spawn(rng))
            settled, steps = self._settle()
            timeouts += not settled
            if settled and self._in_bin():
                return self.d.qpos[self.qa:self.qa + 7].copy(), dict(
                    steps=steps, attempts=attempt, timeouts=timeouts, drawn=drawn,
                    wall_s=time.perf_counter() - t0)
        raise RuntimeError(f"seed {seed}: no valid settle in {self.rc['max_attempts']} attempts")