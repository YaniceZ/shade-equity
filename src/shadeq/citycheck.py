"""Check the non-green tall LiDAR objects against City of Vancouver open data.

Two independent city layers are used: the 2015 building footprints (which include garages and
sheds that Overture may miss) and the public tree inventory (city-owned street and park trees,
with height classes). Each 1 m tall-object cell in the city is labelled green canopy (Sentinel-2
NDVI >= 0.3) or non-green thin / smooth / rough (height texture), and we report how often each
label falls inside a city footprint or under the crown of a public tree. Private trees are not in
the inventory, so tree proximity is compared between labels rather than read as an absolute rate.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features

from . import analysis as A
from . import config as C
from . import data as D
from . import surfaces as S

SRC = C.RAW / "vancouver_city"
URLS = {
    "public-trees": "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/public-trees/exports/geojson",
    "building-footprints-2015": "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/building-footprints-2015/exports/geojson",
}
LABELS = {0: "green canopy", 1: "non-green thin", 2: "non-green smooth", 3: "non-green rough"}


def fetch():
    import urllib.request

    SRC.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        out = SRC / f"{name}.geojson"
        if not out.exists():
            urllib.request.urlretrieve(url, out)


def crown_radius(height_m):
    """Rough crown radius from the inventory height class, plus 1 m for point location error."""
    return np.clip(0.3 * np.nan_to_num(height_m, nan=6.0), 2.0, 7.0) + 1.0


def run(city=C.VANCOUVER):
    fetch()
    d = D.city_dir(city)
    g = D.grid(city)
    shape, T = (g["height"], g["width"]), g["transform"]
    L = S.load(city)
    ndvi = D.read_raster(d / "s2_ndvi.tif")
    nongreen = np.load(d / "nongreen_classes.npy")
    tall = (L["ref_veg"] > 0) & A._city_mask(city)
    label = np.full(shape, 255, np.uint8)
    label[tall & (ndvi >= 0.3)] = 0
    m = tall & (nongreen > 0)
    label[m] = nongreen[m]
    del ndvi, nongreen, m

    bld = gpd.read_file(SRC / "building-footprints-2015.geojson").to_crs(city.crs)
    in_bld = features.rasterize(((geom, 1) for geom in bld.geometry if geom is not None), out_shape=shape, transform=T, dtype="uint8").astype(bool)

    trees = gpd.read_file(SRC / "public-trees.geojson").to_crs(city.crs)
    trees = trees[trees.geometry.notna() & ~trees.geometry.is_empty]
    crowns = trees.geometry.buffer(crown_radius(trees["height_m"].astype(float).to_numpy()), resolution=6)
    in_tree = features.rasterize(((geom, 1) for geom in crowns), out_shape=shape, transform=T, dtype="uint8").astype(bool)

    rows = []
    for k, name in LABELS.items():
        sel = label == k
        n = int(sel.sum())
        rows.append({
            "class_2016": name,
            "cells": n,
            "share_of_tall": n / int((label != 255).sum()),
            "in_city_footprint_2015": float((sel & in_bld).sum() / max(n, 1)),
            "under_public_tree_crown": float((sel & in_tree).sum() / max(n, 1)),
            "neither": float((sel & ~in_bld & ~in_tree).sum() / max(n, 1)),
        })
    df = pd.DataFrame(rows)
    ng = df[df.class_2016.str.startswith("non-green")]
    w = ng.cells / ng.cells.sum()
    summary = {
        "by_class": df.round(4).to_dict(orient="records"),
        "non_green_all": {c: round(float((ng[c] * w).sum()), 4) for c in ["in_city_footprint_2015", "under_public_tree_crown", "neither"]},
        "public_trees": int(len(trees)),
        "city_footprints_2015": int(len(bld)),
        "note": "Public inventory excludes private trees; compare rates between classes.",
    }
    df.to_csv(C.RESULTS / "city_data_check.csv", index=False)
    (C.RESULTS / "city_data_check.json").write_text(json.dumps(summary, indent=2))
    print(df.round(3).to_string(index=False))
    print(summary["non_green_all"])
    return summary


if __name__ == "__main__":
    run()
