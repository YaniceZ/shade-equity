"""Direct-sun shading by ray marching over gridded building and tree heights.

Surfaces are on one north-up grid (row 0 at the top):
- ``ground``: terrain elevation (m)
- ``bld``: building height above ground (m, 0 where no building)
- ``veg``: tree height above ground (m, 0 where no tree)

A ray leaves the ground at each point towards the sun. Terrain and buildings are opaque.
A tree blocks the ray only between its crown base (``trunk_ratio`` x height) and its top,
and lets ``transmissivity`` of the direct beam through. The result is the sunlit fraction
of the direct beam: 1 = full sun, ``transmissivity`` = under tree shade only, 0 = building
or terrain shade.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _sunlit_points(rows, cols, ground, bld, veg, az_deg, el_deg, cell, trans, trunk_ratio, zmax, step):
    n = rows.shape[0]
    out = np.empty(n, dtype=np.float32)
    nr, nc = ground.shape
    az = math.radians(az_deg)
    tanel = math.tan(math.radians(el_deg))
    dr = -math.cos(az) * step / cell  # row change per step (north = up)
    dc = math.sin(az) * step / cell
    for i in prange(n):
        r0 = rows[i]
        c0 = cols[i]
        ir0 = int(r0)
        ic0 = int(c0)
        if ir0 < 0 or ir0 >= nr or ic0 < 0 or ic0 >= nc or el_deg <= 0:
            out[i] = 0.0
            continue
        z0 = ground[ir0, ic0]
        in_tree = False
        blocked = False
        r = r0
        c = c0
        k = 0
        while True:
            k += 1
            r += dr
            c += dc
            ir = int(r)
            ic = int(c)
            if ir < 0 or ir >= nr or ic < 0 or ic >= nc:
                break
            zray = z0 + k * step * tanel
            if zray > zmax:
                break
            g = ground[ir, ic]
            if g >= zray:
                blocked = True
                break
            hb = bld[ir, ic]
            if hb > 0 and g + hb >= zray:
                blocked = True
                break
            hv = veg[ir, ic]
            if hv > 0 and not in_tree:
                if g + hv >= zray and g + trunk_ratio * hv <= zray:
                    in_tree = True
        if blocked:
            out[i] = 0.0
        elif in_tree:
            out[i] = trans
        else:
            out[i] = 1.0
    return out


def sunlit_points(rows, cols, ground, bld, veg, az, el, cell=1.0, transmissivity=0.1, trunk_ratio=0.25, step=None):
    """Sunlit fraction of the direct beam at continuous grid positions (row = (top - y) / cell)."""
    ground = np.ascontiguousarray(ground, dtype=np.float32)
    bld = np.ascontiguousarray(bld, dtype=np.float32)
    veg = np.ascontiguousarray(veg, dtype=np.float32)
    zmax = float(np.nanmax(ground + np.maximum(bld, veg)))
    step = cell if step is None else step
    return _sunlit_points(
        np.asarray(rows, dtype=np.float64),
        np.asarray(cols, dtype=np.float64),
        ground,
        bld,
        veg,
        float(az),
        float(el),
        float(cell),
        float(transmissivity),
        float(trunk_ratio),
        zmax,
        float(step),
    )


def sunlit_raster(ground, bld, veg, az, el, cell=1.0, transmissivity=0.1, trunk_ratio=0.25, window=None):
    """Sunlit fraction for every cell of the grid (or of a row/col window ``(r0, r1, c0, c1)``)."""
    nr, nc = ground.shape
    r0, r1, c0, c1 = window if window else (0, nr, 0, nc)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    vals = sunlit_points(rr.ravel() + 0.5, cc.ravel() + 0.5, ground, bld, veg, az, el, cell, transmissivity, trunk_ratio)
    return vals.reshape(rr.shape)
