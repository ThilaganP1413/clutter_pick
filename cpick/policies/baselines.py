"""Non-learned pick baselines on the 128x128 action grid (cell (i, j) -> pixel (2j, 2i)). No torch."""
import numpy as np

from cpick.sim.geometry import surface_variation
from scipy.ndimage import distance_transform_edt


def random_in_mask(valid, rng):
    ii, jj = np.nonzero(valid)
    k = rng.integers(len(ii))
    return ii[k], jj[k]

def flattest_above_floor(P, depth, K, valid, floor_z, radius,
                         height_tol=0.005, min_frac=0.6, tie=1e-5):
    """Flattest cells above the known floor; among ties, the most interior one
    (farthest from any non-tied cell), so the pick keeps a margin from the flat region's edge.
    Balls with < min_frac of a full disc's points are skipped (truncated balls look falsely flat).
    Returns (i, j) or None if no candidate."""
    f = K[0, 0]
    var = np.full(valid.shape, np.nan)
    for i, j in zip(*np.nonzero(valid)):
        v, u = 2 * i, 2 * j
        if P[v, u, 2] < floor_z + height_tol:
            continue
        s, n = surface_variation(P, depth, u, v, radius, K)
        if n >= min_frac * np.pi * (radius * f / depth[v, u]) ** 2:
            var[i, j] = s
    if np.all(np.isnan(var)):
        return None
    cand = var <= np.nanmin(var) + tie                 # NaN compares False -> not a candidate
    dist = distance_transform_edt(cand)                # cells to the nearest non-candidate
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    return int(i), int(j)