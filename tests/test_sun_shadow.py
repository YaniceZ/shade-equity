from datetime import datetime, timezone

import numpy as np
import pytest

from shadeq.shadow import sunlit_points, sunlit_raster
from shadeq.sun import solar_position, sun_series


def test_solar_noon_vancouver_solstice():
    # Solar noon in Vancouver (123.1 W) on 21 June is about 20:14 UTC; elevation = 90 - 49.25 + 23.44.
    best = max((solar_position(datetime(2021, 6, 21, 20, m, tzinfo=timezone.utc), 49.25, -123.1) for m in range(0, 30)), key=lambda x: x[1])
    az, el = best
    assert el == pytest.approx(64.2, abs=0.3)
    assert az == pytest.approx(180, abs=3)


def test_morning_sun_in_east():
    (h, az, el), = sun_series((2021, 6, 28), [8.0], -7.0, 49.25, -123.1)
    assert 70 < az < 110 and 20 < el < 40


def _wall(height=10.0):
    n = 60
    ground = np.zeros((n, n), np.float32)
    bld = np.zeros_like(ground)
    veg = np.zeros_like(ground)
    bld[40, :] = height  # an east-west wall in row 40 (south of rows < 40)
    return ground, bld, veg


def test_wall_shadow_length_matches_geometry():
    ground, bld, veg = _wall(10.0)
    # Sun due south at 45 degrees: shadow falls north (smaller rows), length = 10 m.
    rows = np.array([39.5 - d for d in range(1, 16)])
    cols = np.full(rows.shape, 30.5)
    sun = sunlit_points(rows, cols, ground, bld, veg, az=180, el=45)
    shaded = rows[sun == 0]
    lengths = 40 - shaded
    assert lengths.max() == pytest.approx(10.5, abs=1.01)
    assert (sun[(40 - rows) > 11.5] == 1).all()


def test_sun_from_north_no_shadow_on_north_side():
    ground, bld, veg = _wall(10.0)
    sun = sunlit_points(np.array([35.5]), np.array([30.5]), ground, bld, veg, az=0, el=45)
    assert sun[0] == 1


def test_tree_transmissivity_and_crown_base():
    n = 60
    ground = np.zeros((n, n), np.float32)
    bld = np.zeros_like(ground)
    veg = np.zeros_like(ground)
    veg[38:42, 28:33] = 12.0  # crown 3 m to 12 m (trunk ratio 0.25)
    # Point 2 m north of the crown edge, sun from the south at 60 degrees: ray passes the crown at 3.5-12 m.
    sun = sunlit_points(np.array([36.5]), np.array([30.5]), ground, bld, veg, az=180, el=60, transmissivity=0.1)
    assert sun[0] == pytest.approx(0.1)
    # Directly below the crown at very high sun the ray leaves through the crown too.
    sun_under = sunlit_points(np.array([39.5]), np.array([30.5]), ground, bld, veg, az=180, el=80)
    assert sun_under[0] == pytest.approx(0.1)
    # Far away the tree casts no shade.
    sun_far = sunlit_points(np.array([10.5]), np.array([30.5]), ground, bld, veg, az=180, el=60)
    assert sun_far[0] == 1


def test_building_beats_tree():
    ground, bld, veg = _wall(10.0)
    veg[42, :] = 15.0
    sun = sunlit_points(np.array([36.5]), np.array([30.5]), ground, bld, veg, az=180, el=45)
    assert sun[0] == 0


def test_raster_matches_points():
    ground, bld, veg = _wall(8.0)
    ras = sunlit_raster(ground, bld, veg, az=200, el=40)
    pts = sunlit_points(np.array([30.5]), np.array([20.5]), ground, bld, veg, az=200, el=40)
    assert ras[30, 20] == pts[0]


def test_sun_below_horizon_is_dark():
    ground, bld, veg = _wall(1.0)
    assert sunlit_points(np.array([5.5]), np.array([5.5]), ground, bld, veg, az=180, el=-1)[0] == 0
