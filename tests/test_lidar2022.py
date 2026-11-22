import zipfile

import numpy as np
import pytest

from shadeq.lidar2022 import tile_classes

laspy = pytest.importorskip("laspy")


def test_highest_point_class(tmp_path):
    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.scales = [0.01, 0.01, 0.01]
    hdr.offsets = [489000, 5456000, 0]
    las = laspy.LasData(hdr)
    # cell (row 0, col 5): ground at 10 m, wire at 14 m, tree top at 16 m; cell (row 999, col 0): ground only
    las.x = np.array([489005.5, 489005.5, 489005.5, 489000.5])
    las.y = np.array([5456999.5, 5456999.5, 5456999.5, 5456000.5])
    las.z = np.array([16.0, 14.0, 10.0, 11.0])
    las.classification = np.array([5, 14, 2, 2], dtype=np.uint8)
    las.write(tmp_path / "t.las")
    with zipfile.ZipFile(tmp_path / "t.zip", "w") as z:
        z.write(tmp_path / "t.las", "t.las")
    cls = tile_classes(tmp_path / "t.zip", 489000, 5456000)
    assert cls[0, 5] == 5
    assert cls[999, 0] == 0
    assert (cls > 0).sum() == 1
