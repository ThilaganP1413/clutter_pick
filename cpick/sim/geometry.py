"""Pixel <-> 3D geometry for a pinhole depth camera (OpenCV convention)."""
import numpy as np


def backproject(depth, K, T):
    """depth (H, W) z-depth [m] -> (H, W, 3) world points."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], axis=-1)
    return pc @ T[:3, :3].T + T[:3, 3]


def project(points, K, T):
    """(N, 3) world points -> (N, 2) float pixel coords (u, v) and (N,) camera-frame z."""
    pc = (points - T[:3, 3]) @ T[:3, :3]                 # row-wise R^T (p - t)
    uv = pc[:, :2] / pc[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
    return uv, pc[:, 2]


def pixel_ray(u, v, K, T):
    """World ray through pixel (u, v): origin, unit direction, and |K^-1 [u v 1]|
    (ray length = z-depth * that factor)."""
    d = np.linalg.solve(K, np.array([u, v, 1.0]))
    s = np.linalg.norm(d)
    return T[:3, 3].copy(), T[:3, :3] @ (d / s), s


def estimate_normal(points_map, depth, u, v, radius, K, cam_pos, min_pts=6):
    """PCA plane fit over back-projected points inside a 3D ball of `radius` around pixel (u, v).
    Returns (unit normal toward the camera, n_points), or (None, n_points) if too few points."""
    z = depth[v, u]
    r = int(np.ceil(K[0, 0] * radius / max(z - radius, 1e-3))) + 1   # window that covers the ball
    win = points_map[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].reshape(-1, 3)
    p0 = points_map[v, u]
    pts = win[np.linalg.norm(win - p0, axis=1) < radius]
    if len(pts) < min_pts:
        return None, len(pts)
    q = pts - pts.mean(0)
    _, vecs = np.linalg.eigh(q.T @ q)
    n = vecs[:, 0]                                        # smallest-eigenvalue direction
    return (n if n @ (cam_pos - p0) > 0 else -n), len(pts)