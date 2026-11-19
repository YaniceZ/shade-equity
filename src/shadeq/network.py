"""Walk network, sidewalk sample points and per-edge shade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from . import config as C
from . import data as D
from . import shadow, surfaces
from .sun import sun_series


def _first(v):
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    if isinstance(v, str) and v.startswith("["):
        return v.strip("[]").split(",")[0].strip(" '\"")
    return v


def edges(city: C.City):
    """Undirected walk edges (one row per street segment) and node coordinates."""
    import osmnx as ox

    d = D.city_dir(city)
    path = d / "edges.parquet"
    npath = d / "nodes.parquet"
    if path.exists() and npath.exists():
        import geopandas as gpd

        return gpd.read_parquet(path), pd.read_parquet(npath)
    G = ox.load_graphml(d / "walk.graphml")
    nodes, e = ox.graph_to_gdfs(G, nodes=True, edges=True)
    e = e.reset_index()
    a = np.minimum(e["u"], e["v"])
    b = np.maximum(e["u"], e["v"])
    e = e.assign(a=a, b=b).drop_duplicates(subset=["a", "b", "key"]).reset_index(drop=True)
    e["highway"] = e["highway"].map(_first).astype(str)
    e["length"] = e.geometry.length
    e = e[e["length"] > 0]
    keep = ["a", "b", "key", "highway", "length", "geometry"]
    e = e[keep].rename(columns={"a": "u", "b": "v"}).reset_index(drop=True)
    e["eid"] = np.arange(len(e))
    nodes = nodes[["x", "y"]].reset_index().rename(columns={"osmid": "node"})
    e.to_parquet(path)
    nodes.to_parquet(npath)
    return e, nodes


def sample_points(e, step: float = C.SAMPLE_STEP) -> pd.DataFrame:
    """Points every ``step`` m along each edge; streets are sampled on both sidewalks."""
    half = e["highway"].map(C.HALF_WIDTH).where(~e["highway"].isin(C.NO_SIDEWALK))
    off = (half + C.SIDEWALK_SETBACK).fillna(0.0).to_numpy()
    geoms = e.geometry.to_numpy()
    lengths = e["length"].to_numpy()
    out = []
    for side in (-1, 0, 1):
        if side == 0:
            sel = off == 0
            lines = geoms[sel]
        else:
            sel = off > 0
            lines = shapely.offset_curve(geoms[sel], side * off[sel], quad_segs=2, join_style="mitre", mitre_limit=2.0)
        eid = e["eid"].to_numpy()[sel]
        L = shapely.length(lines)
        # Offset curves can be empty or multi-part around sharp bends; fall back to the centreline then.
        bad = ~np.isfinite(L) | (L <= 0) | (shapely.get_type_id(lines) != 1)
        if bad.any():
            lines = lines.copy()
            lines[bad] = geoms[sel][bad]
            L = shapely.length(lines)
        n = np.maximum(1, np.round(L / step).astype(int))
        idx = np.repeat(np.arange(len(lines)), n)
        within = np.concatenate([(np.arange(k) + 0.5) / k for k in n]) if len(n) else np.array([])
        pts = shapely.line_interpolate_point(lines[idx], within, normalized=True)
        out.append(pd.DataFrame({"eid": eid[idx], "side": side, "x": shapely.get_x(pts), "y": shapely.get_y(pts)}))
    del lengths
    return pd.concat(out, ignore_index=True)


def suns(city: C.City):
    b = D.boundary(city).to_crs(4326).geometry.union_all().centroid
    return sun_series(city.day, city.hours, city.utc_offset, b.y, b.x)


def edge_shade(city: C.City, transmissivity: float = C.TREE_TRANSMISSIVITY, trunk_ratio: float = C.TRUNK_RATIO,
               scenarios=None, tag: str = "", day=None, sidewalks: bool = True) -> pd.DataFrame:
    """Mean sunlit fraction per edge, scenario and time. Saved to ``edge_sun{tag}.parquet``."""
    d = D.city_dir(city)
    path = d / f"edge_sun{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    e, _ = edges(city)
    g = D.grid(city)
    S = surfaces.load(city)
    if sidewalks:
        pts = sample_points(e)
    else:
        pts = sample_points(e.assign(highway="footway"))
    x0, _, _, y1 = g["bounds"]
    rows = (y1 - pts["y"].to_numpy()) / C.CELL
    cols = (pts["x"].to_numpy() - x0) / C.CELL
    ir, ic = rows.astype(int), cols.astype(int)
    inside = (ir >= 0) & (ir < g["height"]) & (ic >= 0) & (ic < g["width"])
    valid = inside.copy()
    valid[inside] = S["bld_mask"][ir[inside], ic[inside]] == 0
    pts = pts[valid].reset_index(drop=True)
    rows, cols = rows[valid], cols[valid]
    D.log(f"{len(pts):,} sample points on {pts['eid'].nunique():,} edges ({(~valid).mean():.1%} dropped: outside grid or indoors)")
    sun_pos = suns(city) if day is None else sun_series(day, city.hours, city.utc_offset, *_latlon(city))
    frames = []
    for name in scenarios or surfaces.SCENARIOS:
        bk, vk = {**surfaces.SCENARIOS, **surfaces.EXTRA_SCENARIOS}[name]
        for h, az, el in sun_pos:
            s = shadow.sunlit_points(rows, cols, S["ground"], S[bk], S[vk], az, el, C.CELL, transmissivity, trunk_ratio)
            m = pd.Series(s).groupby(pts["eid"].to_numpy()).mean()
            frames.append(pd.DataFrame({"eid": m.index, "scenario": name, "hour": h, "sun": m.to_numpy(np.float32)}))
        D.log(f"  {name}: done")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path)
    if not tag:
        pts.to_parquet(d / "sample_points.parquet")
    return out


def _latlon(city):
    b = D.boundary(city).to_crs(4326).geometry.union_all().centroid
    return b.y, b.x


def point_shade(city: C.City, hour: float, scenarios=("lidar", "global")) -> pd.DataFrame:
    """Sunlit fraction at every sample point at one hour (for pixel-level agreement)."""
    d = D.city_dir(city)
    pts = pd.read_parquet(d / "sample_points.parquet")
    g = D.grid(city)
    S = surfaces.load(city)
    x0, _, _, y1 = g["bounds"]
    rows = (y1 - pts["y"].to_numpy()) / C.CELL
    cols = (pts["x"].to_numpy() - x0) / C.CELL
    (h, az, el), = sun_series(city.day, [hour], city.utc_offset, *_latlon(city))
    for name in scenarios:
        bk, vk = surfaces.SCENARIOS[name]
        pts[name] = shadow.sunlit_points(rows, cols, S["ground"], S[bk], S[vk], az, el, C.CELL, C.TREE_TRANSMISSIVITY, C.TRUNK_RATIO)
    return pts
