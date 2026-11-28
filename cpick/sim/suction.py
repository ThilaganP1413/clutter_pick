"""Ray-based suction check: CODs-style centre + ring rays, plus a bellows-stroke budget. No torch."""
from dataclasses import dataclass

import mujoco
import numpy as np

from cpick.sim.geometry import estimate_normal

REASONS = ("ok", "outside_mask", "no_normal", "not_object", "ring_miss", "not_planar")
_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class SuctionResult:
    ok: bool
    reason: str
    point: np.ndarray
    approach: np.ndarray = None   # unit direction the cup moves along (-clipped normal)
    geom: int = -1                # geom hit by the centre ray
    spread: float = float("nan")  # max - min hit distance along the approach [m]
    clipped: bool = False


def clip_normal(n, max_tilt_deg):
    """CODs: limit the normal to <= max_tilt from vertical, keeping its horizontal direction."""
    c, s = np.cos(np.radians(max_tilt_deg)), np.sin(np.radians(max_tilt_deg))
    if n @ _UP >= c:
        return n, False
    h = n - (n @ _UP) * _UP
    hn = np.linalg.norm(h)
    return (_UP.copy() if hn < 1e-9 else c * _UP + s * h / hn), True


class SuctionChecker:
    def __init__(self, model, cfg):
        s, (ix, iy) = cfg["suction"], cfg["bin"]["inner_size"]
        self.m = model
        self.r, self.n_ring, self.standoff = s["cup_radius"], s["n_ring"], s["standoff"]
        self.max_spread, self.max_tilt, self.min_pts = s["max_spread"], s["max_tilt_deg"], s["min_normal_pts"]
        self.half = np.array([ix / 2 - self.r, iy / 2 - self.r])   # interior shrunk by the cup radius
        self._gid = np.full(1, -1, dtype=np.int32)

    def valid_mask(self, points_map):
        """Depth-based mask: back-projected x/y inside the shrunk interior."""
        return (np.abs(points_map[..., 0]) < self.half[0]) & (np.abs(points_map[..., 1]) < self.half[1])

    def _ray(self, data, origin, direction):
        self._gid[0] = -1
        dist = mujoco.mj_ray(self.m, data, origin, direction, None, 1, -1, self._gid)
        return dist, int(self._gid[0])

    def check(self, data, points_map, depth, u, v, K, cam_pos, object_geoms):
        p = points_map[v, u]
        if not (abs(p[0]) < self.half[0] and abs(p[1]) < self.half[1]):
            return SuctionResult(False, "outside_mask", p)
        n, _ = estimate_normal(points_map, depth, u, v, self.r, K, cam_pos, self.min_pts)
        if n is None:
            return SuctionResult(False, "no_normal", p)
        n, clipped = clip_normal(n, self.max_tilt)
        d = -n
        e1 = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        centre = p + n * self.standoff
        dc, gc = self._ray(data, centre, d)
        if gc not in object_geoms:
            return SuctionResult(False, "not_object", p, d, gc, clipped=clipped)
        dists = [dc]
        for k in range(self.n_ring):
            phi = 2 * np.pi * k / self.n_ring
            dk, gk = self._ray(data, centre + self.r * (np.cos(phi) * e1 + np.sin(phi) * e2), d)
            if gk != gc:
                return SuctionResult(False, "ring_miss", p, d, gc, clipped=clipped)
            dists.append(dk)
        spread = max(dists) - min(dists)
        ok = spread <= self.max_spread
        return SuctionResult(ok, "ok" if ok else "not_planar", p, d, gc, spread, clipped)