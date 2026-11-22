"""Check the 2016 LiDAR tall-object layer against the classified 2022 City of Vancouver point cloud.

The 2022 tiles sit behind a browser check, so they are downloaded by hand into data/raw/lidar2022/
(one 1 km zip per tile, named like 491000_5458000.zip). For each 1 m cell with a tall non-building
object in 2016, the class of the highest 2022 point at least 2.5 m above the 2022 ground is recorded and cross-tabulated
against the 2016 categories (green canopy, and non-green thin / smooth / rough).
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import rasterio

from . import config as C
from . import data as D
from . import surfaces as S

TILES = ["491000_5458000", "493000_5459000", "490000_5449000", "489000_5456000"]
URL = "https://webtransfer.vancouver.ca/opendata/2022LiDAR/{}.zip"
ASPRS = {1: "unclassified", 2: "ground", 3: "vegetation", 4: "vegetation", 5: "vegetation", 6: "building",
         7: "noise", 9: "water", 13: "wire", 14: "wire", 15: "tower", 16: "wire", 17: "bridge", 18: "noise"}
CATS_2016 = {0: "green canopy", 1: "non-green thin", 2: "non-green smooth", 3: "non-green rough"}


def _points(zpath, chunk=5_000_000):
    """Stream (x, y, z, class) arrays from the LAS/LAZ files in a tile zip without unpacking it."""
    import laspy  # optional dependency (extra "data")

    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.lower().endswith((".las", ".laz")):
                with z.open(name) as f, laspy.open(f) as r:
                    for pts in r.chunk_iterator(chunk):
                        yield np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z), np.asarray(pts.classification)


def _keep_extreme(grid, idx, vals, highest):
    """Write vals into grid.ravel()[idx], keeping the highest (or lowest) value per cell."""
    order = np.argsort(vals if highest else -vals)  # with repeated indices the last write wins
    idx, vals = idx[order], vals[order]
    flat = grid.ravel()
    better = vals > flat[idx] if highest else vals < flat[idx]
    flat[idx[better]] = vals[better]
    return idx[better], order[better]


def tile_classes(zpath, e0, n0):
    """1000 x 1000 array of the ASPRS class of the highest point at least MIN_TREE above ground (0 = none).

    Heights are taken from the tile's own ground points (class 2), so the 2022 vertical datum does not matter.
    Cells without ground points take the nearest ground value.
    """
    from scipy import ndimage

    ground = np.full((1000, 1000), np.inf, np.float64)
    top = np.full((1000, 1000), -np.inf, np.float64)
    cls = np.zeros((1000, 1000), np.uint8)
    for x, y, z, c in _points(zpath):
        col = np.floor(x - e0).astype(np.int64)
        row = np.floor(n0 + 1000 - y).astype(np.int64)
        ok = (col >= 0) & (col < 1000) & (row >= 0) & (row < 1000) & ~np.isin(c, [7, 9, 18])
        idx, z, c = (row * 1000 + col)[ok], z[ok], c[ok]
        g = c == 2
        _keep_extreme(ground, idx[g], z[g], highest=False)
        cells, order = _keep_extreme(top, idx[~g], z[~g], highest=True)
        cls.ravel()[cells] = c[~g][order]
    if np.isinf(ground).all():
        return np.zeros((1000, 1000), np.uint8)
    missing = np.isinf(ground)
    if missing.any():
        near = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        ground = ground[tuple(near)]
    cls[(top - ground) < C.MIN_TREE] = 0
    return cls


def run(city=C.VANCOUVER, tiles=TILES):
    d = D.city_dir(city)
    raw = C.RAW / "lidar2022"
    L = S.load(city)
    nongreen = np.load(d / "nongreen_classes.npy")
    ndvi = D.read_raster(d / "s2_ndvi.tif")
    with rasterio.open(d / "lidar_ndsm.tif") as r:
        T = r.transform
    rows = []
    for name in tiles:
        zpath = raw / f"{name}.zip"
        if not zpath.exists():
            print(f"missing {zpath} (download {URL.format(name)})")
            continue
        e0, n0 = map(int, name.split("_"))
        c0, r0 = int(round((e0 - T.c) / T.a)), int(round((T.f - (n0 + 1000)) / -T.e))
        win = np.s_[r0:r0 + 1000, c0:c0 + 1000]
        tall = L["ref_veg"][win] > 0
        cat = np.where(nongreen[win] > 0, nongreen[win], np.where(tall & (ndvi[win] >= 0.3), 0, 255))
        cat[~tall] = 255
        new = tile_classes(zpath, e0, n0)
        for k, label in CATS_2016.items():
            m = cat == k
            if not m.any():
                continue
            names = pd.Series(new[m]).map(lambda v: "nothing tall" if v == 0 else ASPRS.get(int(v), f"class {v}"))
            for cls_name, n in names.value_counts().items():
                rows.append({"tile": name, "class_2016": label, "class_2022": cls_name, "cells": int(n)})
        print(name, "done")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = C.RESULTS / "lidar2022_check.csv"
    df.to_csv(out, index=False)
    tab = df.groupby(["class_2016", "class_2022"]).cells.sum().unstack(fill_value=0)
    share = tab.div(tab.sum(axis=1), axis=0).round(3)
    print(share.to_string())
    ng = tab.loc[[i for i in tab.index if i.startswith("non-green")]].sum()
    summary = {"tiles": sorted(df.tile.unique().tolist()),
               "share_by_2016_class": share.to_dict(orient="index"),
               "non_green_all": (ng / ng.sum()).round(3).to_dict()}
    (C.RESULTS / "lidar2022_check.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
