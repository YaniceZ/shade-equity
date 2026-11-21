"""Build the two obstacle models on the 1 m grid.

Reference (LiDAR, 2016): normalised DSM = DSM - DTM. Cells inside building footprints
(Overture, all sources) are buildings; cells outside a 1 m dilation of the footprints and
taller than ``MIN_TREE`` are trees.

Global open data: building heights from GlobalBuildingAtlas LoD1 rasterised on GBA
footprints; trees from the Meta/WRI 1 m canopy height map outside those footprints.

Both models share the LiDAR terrain so that differences come from the obstacles only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio import features
from scipy import ndimage

from . import config as C
from . import data as D

LAYERS = ("ground", "ref_bld", "ref_veg", "ref_veg_green", "glb_bld", "glb_veg", "bld_mask")


def _rasterize(gdf, g, values=None, dtype="uint8", fill=0):
    shapes = zip(gdf.geometry, values if values is not None else np.ones(len(gdf)), strict=True)
    return features.rasterize(shapes, out_shape=(g["height"], g["width"]), transform=g["transform"], fill=fill, dtype=dtype)


def build(city: C.City) -> None:
    d = D.city_dir(city)
    if all((d / f"{n}.tif").exists() for n in LAYERS if n != "ref_veg_green"):
        D.ndvi(city)
        green_layer(city)
        return
    g = D.grid(city)
    dtm = D.read_raster(d / "lidar_dtm.tif")
    dsm = D.read_raster(d / "lidar_dsm.tif")
    nodata = ~np.isfinite(dtm)
    ground = np.where(nodata, 0.0, dtm).astype(np.float32)  # water and gaps at sea level
    ndsm = np.where(np.isfinite(dsm) & ~nodata, dsm - dtm, 0.0)
    ndsm = np.clip(ndsm, 0, 400).astype(np.float32)
    del dsm

    ov = D.overture_buildings(city)
    D.log(f"rasterising {len(ov)} Overture footprints")
    ov_mask = _rasterize(ov, g).astype(bool)
    near_bld = ndimage.binary_dilation(ov_mask, iterations=1)
    ref_bld = np.where(ov_mask & (ndsm >= 2.0), ndsm, 0).astype(np.float32)
    ref_veg = np.where(~near_bld & (ndsm >= C.MIN_TREE), ndsm, 0).astype(np.float32)

    gba = D.gba_buildings(city)
    h = gba["height"].to_numpy(dtype=float)
    fallback = float(np.nanmedian(h)) if np.isfinite(h).any() else 6.0
    h = np.where(np.isfinite(h) & (h > 0), h, fallback)
    order = np.argsort(h)  # taller buildings drawn last so they win overlaps
    D.log(f"rasterising {len(gba)} GBA footprints (fallback height {fallback:.1f} m for {np.mean(~np.isfinite(gba['height'])):.1%})")
    glb_bld = _rasterize(gba.iloc[order], g, values=h[order], dtype="float32", fill=0.0)
    gba_mask = glb_bld > 0
    chm = D.read_raster(d / "meta_chm.tif")
    chm = np.where(np.isfinite(chm), chm, 0)
    glb_veg = np.where(~ndimage.binary_dilation(gba_mask, iterations=1) & (chm >= C.MIN_TREE_META), chm, 0).astype(np.float32)

    # Union of footprints from both sources: sample points that fall inside are indoors and dropped.
    bld_mask = (ov_mask | gba_mask).astype(np.uint8)
    out = {"ground": ground, "ref_bld": ref_bld, "ref_veg": ref_veg, "glb_bld": glb_bld, "glb_veg": glb_veg, "bld_mask": bld_mask}
    for k, v in out.items():
        D.save_raster(d / f"{k}.tif", v, g, nodata=None)
    D.save_raster(d / "lidar_ndsm.tif", ndsm, g)
    D.ndvi(city)
    green_layer(city)


def green_layer(city: C.City) -> None:
    """LiDAR tall objects in 10 m cells with Sentinel-2 NDVI >= 0.3 only: a conservative (lower-bound) tree layer."""
    d = D.city_dir(city)
    out = d / "ref_veg_green.tif"
    if out.exists():
        return
    veg = D.read_raster(d / "ref_veg.tif")
    nd = D.read_raster(d / "s2_ndvi.tif")
    D.save_raster(out, np.where(nd >= 0.3, veg, 0).astype(np.float32), D.grid(city))


def clean_layers(city: C.City) -> None:
    """Best-estimate LiDAR reference: non-green tall objects split by shape and texture.

    Outside footprints, LiDAR objects in 10 m cells with NDVI < 0.3 are (i) thin or isolated (fewer than 6 connected
    cells or under 30% tall cover in a 5 x 5 window: wires, poles; removed), (ii) smooth (3 x 3 height s.d. < 0.5 m and
    at least 10 connected cells: sheds, garages and other structures missing from footprints; made opaque), or
    (iii) rough (tree crowns whose 10 m pixel also covers pavement; kept as trees). The class of each
    non-green cell (1 thin, 2 smooth, 3 rough) is saved to ``nongreen_classes.npy``.
    """
    from scipy import ndimage

    d = D.city_dir(city)
    if (d / "ref_bld_clean.tif").exists():
        return
    bld = D.read_raster(d / "ref_bld.tif")
    veg = D.read_raster(d / "ref_veg.tif")
    nd = D.read_raster(d / "s2_ndvi.tif")
    ndsm = D.read_raster(d / "lidar_ndsm.tif")
    tall = veg > 0
    ng = tall & ~(nd >= 0.3)
    m = ndimage.uniform_filter(ndsm, 3)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(ndsm**2, 3) - m * m, 0))
    dens = ndimage.uniform_filter(tall.astype(np.float32), 5)
    lab, _ = ndimage.label(ng)
    comp = np.bincount(lab.ravel())[lab]
    thin = ng & ((dens < 0.3) | (comp < 6))
    smooth = ng & ~thin & (sd < 0.5) & (comp >= 10)
    rough = ng & ~thin & ~smooth
    np.save(d / "nongreen_classes.npy", (thin * 1 + smooth * 2 + rough * 3).astype(np.uint8))
    g = D.grid(city)
    D.save_raster(d / "ref_bld_clean.tif", np.where(smooth, veg, bld).astype(np.float32), g)
    D.save_raster(d / "ref_veg_clean.tif", np.where(thin | smooth, 0, veg).astype(np.float32), g)


def load(city: C.City) -> dict[str, np.ndarray]:
    d = D.city_dir(city)
    out = {n: D.read_raster(d / f"{n}.tif") for n in LAYERS}
    for n in ("ref_bld_clean", "ref_veg_clean"):
        if (d / f"{n}.tif").exists():
            out[n] = D.read_raster(d / f"{n}.tif")
    return out


SCENARIOS = {
    # name: (building layer, tree layer)
    "lidar": ("ref_bld", "ref_veg"),
    "global": ("glb_bld", "glb_veg"),
    "lidar_bld+global_veg": ("ref_bld", "glb_veg"),
    "global_bld+lidar_veg": ("glb_bld", "ref_veg"),
}
# Sensitivity only (not part of the default run).
EXTRA_SCENARIOS = {"lidar_green": ("ref_bld", "ref_veg_green"), "lidar_clean": ("ref_bld_clean", "ref_veg_clean")}


def building_heights(city: C.City) -> pd.DataFrame:
    """Per GBA footprint: GBA height vs LiDAR height (median and 90th percentile of nDSM inside)."""
    d = D.city_dir(city)
    path = d / "building_heights.parquet"
    if path.exists():
        return pd.read_parquet(path)
    g = D.grid(city)
    gba = D.gba_buildings(city).reset_index(drop=True)
    gba = gba[gba.area >= 20]
    lab = _rasterize(gba, g, values=np.arange(1, len(gba) + 1), dtype="int32")
    ndsm = D.read_raster(d / "lidar_ndsm.tif")
    m = lab > 0
    df = pd.DataFrame({"lab": lab[m], "h": ndsm[m]})
    q = df.groupby("lab")["h"].agg(lidar_median="median", lidar_p90=lambda s: s.quantile(0.9), n="size")
    out = gba.iloc[q.index - 1][["source", "height"]].copy()
    out = out.rename(columns={"height": "gba_height"})
    out["area"] = gba.iloc[q.index - 1].area.to_numpy()
    out[["lidar_median", "lidar_p90", "n"]] = q.to_numpy()
    x, y = gba.iloc[q.index - 1].centroid.x.to_numpy(), gba.iloc[q.index - 1].centroid.y.to_numpy()
    out["x"], out["y"] = x, y
    out = out.reset_index(drop=True)
    out.to_parquet(path)
    return out
