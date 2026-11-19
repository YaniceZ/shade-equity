"""Validation of the open global model against LiDAR, and route-based equity metrics."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import data as D
from . import network, surfaces

DAY_HOURS = None  # all modelled hours (10:00-17:00) form the "daytime mean"


# --- generic helpers ------------------------------------------------------------------


def agreement(ref: np.ndarray, est: np.ndarray, w: np.ndarray | None = None) -> dict:
    ref = np.asarray(ref, float)
    est = np.asarray(est, float)
    m = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[m], est[m]
    w = np.ones_like(ref) if w is None else np.asarray(w, float)[m]
    w = w / w.sum()
    err = est - ref
    ss_res = np.sum(w * err**2)
    ss_tot = np.sum(w * (ref - np.sum(w * ref)) ** 2)
    return {
        "n": int(m.sum()),
        "ref_mean": float(np.sum(w * ref)),
        "est_mean": float(np.sum(w * est)),
        "bias": float(np.sum(w * err)),
        "mae": float(np.sum(w * np.abs(err))),
        "rmse": float(np.sqrt(ss_res)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": float(np.corrcoef(ref, est)[0, 1]) if len(ref) > 2 else float("nan"),
        "spearman": float(stats.spearmanr(ref, est).statistic) if len(ref) > 2 else float("nan"),
    }


def wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def block_ids(x, y, size=1000.0):
    return (np.floor(np.asarray(x) / size).astype(np.int64) * 100000 + np.floor(np.asarray(y) / size).astype(np.int64))


def block_bootstrap(df: pd.DataFrame, fn, blocks: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% interval of ``fn(df)`` resampling whole spatial blocks with replacement."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    groups = {b: np.nonzero(blocks == b)[0] for b in ub}
    vals = []
    for _ in range(n):
        pick = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([groups[b] for b in pick])
        vals.append(fn(df.iloc[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --- input data validation ---------------------------------------------------------------


def building_validation(city: C.City) -> dict:
    bh = surfaces.building_heights(city)
    bh = bh[(bh["lidar_median"] >= 2.5) & (bh["n"] >= 20) & bh["gba_height"].notna()]
    out = {"all": agreement(bh["lidar_median"], bh["gba_height"])}
    for lo, hi in ((0, 10), (10, 30), (30, 400)):
        s = bh[(bh["lidar_median"] >= lo) & (bh["lidar_median"] < hi)]
        out[f"{lo}-{hi} m"] = agreement(s["lidar_median"], s["gba_height"])
    return out


def canopy_validation(city: C.City, n_sample: int = 3_000_000, seed: int = 3) -> dict:
    d = D.city_dir(city)
    city_mask = _city_mask(city)
    S = surfaces.load(city)
    ndsm = D.read_raster(d / "lidar_ndsm.tif")
    chm = D.read_raster(d / "meta_chm.tif")
    chm = np.where(np.isfinite(chm), chm, 0)
    outside = (S["bld_mask"] == 0) & city_mask
    idx = np.flatnonzero(outside)
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
    lid = ndsm.ravel()[idx]
    met = chm.ravel()[idx]
    lt = lid >= C.MIN_TREE
    mt = met >= C.MIN_TREE_META
    tp = np.sum(lt & mt)
    out = {
        "pixels": int(len(idx)),
        "lidar_cover": float(lt.mean()),
        "meta_cover": float(mt.mean()),
        "precision": float(tp / mt.sum()),
        "recall": float(tp / lt.sum()),
        "iou": float(tp / np.sum(lt | mt)),
        "height_both_tree": agreement(lid[lt & mt], met[lt & mt]),
    }
    # 10 m cover agreement (less sensitive to registration error)
    agg = []
    for arr, thr in ((ndsm, C.MIN_TREE), (chm, C.MIN_TREE_META)):
        t = ((arr >= thr) & outside).astype(np.float32)
        h, w = (t.shape[0] // 10) * 10, (t.shape[1] // 10) * 10
        agg.append(t[:h, :w].reshape(h // 10, 10, w // 10, 10).mean(axis=(1, 3)))
    o10 = outside[:h, :w].reshape(h // 10, 10, w // 10, 10).mean(axis=(1, 3)) > 0.5
    out["cover_10m"] = agreement(agg[0][o10], agg[1][o10])
    # Green canopy only: LiDAR objects in 10 m cells with Sentinel-2 NDVI >= 0.3 (removes sheds, garages, bridges,
    # old houses missing from current footprints; also removes some street-tree crowns over pavement).
    nd = D.read_raster(d / "s2_ndvi.tif").ravel()[idx]
    green = lt & (nd >= 0.3)
    out["green"] = {
        "lidar_cover": float(green.mean()),
        "share_of_lidar_tall_objects": float(green.sum() / lt.sum()),
        "recall": float(np.sum(green & mt) / green.sum()),
    }
    for thr in (3.0, 5.0, 8.0):
        big = (lid >= thr) & (nd >= 0.3)
        out["green"][f"recall_ge{thr:g}m"] = float(np.sum(big & mt) / big.sum())
    return out


def _city_mask(city: C.City) -> np.ndarray:
    from rasterio import features

    g = D.grid(city)
    poly = D.boundary(city).geometry.union_all()
    return features.rasterize([(poly, 1)], out_shape=(g["height"], g["width"]), transform=g["transform"], dtype="uint8").astype(bool)


# --- shade on the network ---------------------------------------------------------------------


def edge_table(city: C.City, tag: str = "") -> pd.DataFrame:
    """Edges inside the city with daytime-mean and peak-hour shade for every scenario."""
    e, _ = network.edges(city)
    es = network.edge_shade(city, tag=tag)
    day = es.groupby(["eid", "scenario"])["sun"].mean().unstack()
    peak = es[es.hour == C.PEAK_HOUR].set_index(["eid", "scenario"])["sun"].unstack()
    t = e[["eid", "highway", "length", "geometry"]].set_index("eid")
    for s in day.columns:
        t[f"shade_{s}"] = 1 - day[s]
        t[f"shade15_{s}"] = 1 - peak[s]
    t = t.dropna(subset=[f"shade_{day.columns[0]}"])
    poly = D.boundary(city).geometry.union_all()
    t = gpd.GeoDataFrame(t, geometry="geometry", crs=city.crs)
    t = t[t.geometry.centroid.within(poly)]
    return t


def network_validation(city: C.City) -> dict:
    t = edge_table(city)
    w = t["length"].to_numpy()
    es = network.edge_shade(city)
    es = es[es.eid.isin(t.index)]
    L = t["length"]
    diurnal = {}
    for s, df in es.groupby("scenario"):
        df = df.assign(length=L.reindex(df.eid).to_numpy())
        diurnal[s] = {f"{h:g}": float(1 - np.average(g["sun"], weights=g["length"])) for h, g in df.groupby("hour")}
    city_mean = {s: wmean(t[f"shade_{s}"], w) for s in surfaces.SCENARIOS}
    lid = city_mean["lidar"]
    decomposition = {
        "lidar": lid,
        "global": city_mean["global"],
        "total_error": city_mean["global"] - lid,
        "building_error": city_mean["global_bld+lidar_veg"] - lid,
        "tree_error": city_mean["lidar_bld+global_veg"] - lid,
    }
    decomposition["interaction"] = decomposition["total_error"] - decomposition["building_error"] - decomposition["tree_error"]
    return {
        "edges": int(len(t)),
        "km": float(w.sum() / 1000),
        "edge_daytime": agreement(t["shade_lidar"], t["shade_global"], w),
        "edge_peak": agreement(t["shade15_lidar"], t["shade15_global"], w),
        "city_mean_shade": city_mean,
        "decomposition": decomposition,
        "diurnal": diurnal,
    }


def point_validation(city: C.City, hour: float = C.PEAK_HOUR) -> dict:
    pts = network.point_shade(city, hour)
    poly = D.boundary(city).geometry.union_all()
    inside = gpd.GeoSeries(gpd.points_from_xy(pts.x, pts.y), crs=city.crs).within(poly).to_numpy()
    pts = pts[inside]
    r = pts["lidar"].to_numpy() < 0.5
    g = pts["global"].to_numpy() < 0.5
    po = np.mean(r == g)
    pe = r.mean() * g.mean() + (1 - r.mean()) * (1 - g.mean())
    return {
        "hour": hour,
        "points": int(len(pts)),
        "shaded_lidar": float(r.mean()),
        "shaded_global": float(g.mean()),
        "accuracy": float(po),
        "kappa": float((po - pe) / (1 - pe)),
        "shade_recall": float(np.sum(r & g) / r.sum()),
        "shade_precision": float(np.sum(r & g) / g.sum()),
    }


def area_validation(city: C.City) -> dict:
    """Neighbourhood (local area) and 500 m hexagon means: does the global model rank areas the same way?"""
    t = edge_table(city)
    nb = D.neighbourhoods(city)
    c = t.geometry.centroid
    pts = gpd.GeoDataFrame(t[["length", "shade_lidar", "shade_global"]], geometry=c, crs=city.crs)
    j = gpd.sjoin(pts, nb, predicate="within")
    agg = j.groupby("name").apply(lambda g: pd.Series({k: np.average(g[k], weights=g["length"]) for k in ("shade_lidar", "shade_global")}),
                                  include_groups=False)
    hexes = hex_grid(city, 500.0)
    jh = gpd.sjoin(pts, hexes, predicate="within")
    aggh = jh.groupby("hex").apply(
        lambda g: pd.Series({**{k: np.average(g[k], weights=g["length"]) for k in ("shade_lidar", "shade_global")}, "km": g["length"].sum() / 1000}),
        include_groups=False,
    )
    aggh = aggh[aggh["km"] >= 1.0]
    agg.to_csv(C.RESULTS / "neighbourhood_shade.csv")
    return {
        "neighbourhoods": agreement(agg["shade_lidar"], agg["shade_global"]),
        "hex500": agreement(aggh["shade_lidar"], aggh["shade_global"]),
        "top5_lidar_least_shaded": agg["shade_lidar"].nsmallest(5).index.tolist(),
        "top5_global_least_shaded": agg["shade_global"].nsmallest(5).index.tolist(),
    }


def hex_grid(city: C.City, size: float) -> gpd.GeoDataFrame:
    import shapely

    xmin, ymin, xmax, ymax = D.boundary(city).total_bounds
    w = np.sqrt(3) * size
    h = 1.5 * size
    polys = []
    row = 0
    y = ymin - size
    while y < ymax + size:
        x = xmin - w + (w / 2 if row % 2 else 0)
        while x < xmax + w:
            ang = np.radians(np.arange(6) * 60 + 30)
            polys.append(shapely.Polygon(np.c_[x + size * np.cos(ang), y + size * np.sin(ang)]))
            x += w
        y += h
        row += 1
    g = gpd.GeoDataFrame({"hex": np.arange(len(polys))}, geometry=polys, crs=city.crs)
    return g[g.intersects(D.boundary(city).geometry.union_all())]


# --- route equity -------------------------------------------------------------------------------------


def route_summary(routes: pd.DataFrame, hours: list[float]) -> pd.DataFrame:
    """Population-weighted route metrics per destination type, scenario and population group."""
    sun_cols = [f"sun_{h:g}" for h in hours]
    r = routes.copy()
    r["sun_day"] = r[sun_cols].mean(axis=1)
    r["shade_share_day"] = np.where(r["dist"] > 0, 1 - r["sun_day"] / r["dist"], np.nan)
    peak = f"sun_{C.PEAK_HOUR:g}"
    r["shade_share_peak"] = np.where(r["dist"] > 0, 1 - r[peak] / r["dist"], np.nan)
    r["sun_min_older"] = r[peak] / C.SPEED_OLDER / 60
    r["sun_min_adult"] = r[peak] / C.SPEED_ADULT / 60
    r["walk_min_older"] = r["dist"] / C.SPEED_OLDER / 60
    r["near"] = r["walk_min_older"] <= 10
    r["near_shaded"] = r["near"] & (r["shade_share_peak"] >= 0.5)
    rows = []
    for (kind, sc), g in r.groupby(["kind", "scenario"]):
        for grp, w in (("65+", g["pop65"]), ("all", g["pop"])):
            rows.append({
                "kind": kind, "scenario": sc, "group": grp,
                "dist_m": wmean(g["dist"], w),
                "shade_share_day": wmean(g["shade_share_day"], w),
                "shade_share_peak": wmean(g["shade_share_peak"], w),
                "sun_min_older_peak": wmean(g["sun_min_older"], w),
                "near_10min": wmean(g["near"], w),
                "near_and_shaded": wmean(g["near_shaded"], w),
            })
    return pd.DataFrame(rows), r


def write_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=conv)


def _route_frame(per: pd.DataFrame, scenario: str) -> pd.DataFrame:
    r = per[per.scenario == scenario].copy()
    r["block"] = block_ids(r["x"], r["y"])
    return r


def equity(city: C.City, per: pd.DataFrame, sa: pd.DataFrame) -> dict:
    out: dict = {"speed_older_mps": C.SPEED_OLDER, "peak_hour": C.PEAK_HOUR, "day": list(city.day)}
    pop = D.population(city)
    out["population"] = {"total": float(pop["pop"].sum()), "age65": float(pop["pop65"].sum()), "cells": int(len(pop))}
    for sc in ("lidar", "global"):
        r = _route_frame(per, sc)
        res = {}
        for kind, g in r.groupby("kind"):
            d65 = wmean(g["sun_min_older"], g["pop65"])
            gap = wmean(g["shade_share_peak"], g["pop65"]) - wmean(g["shade_share_peak"], g["pop"])
            res[kind] = {
                "sun_min_older_peak_65": d65,
                "sun_min_older_peak_65_ci": block_bootstrap(g, lambda x: wmean(x["sun_min_older"], x["pop65"]), g["block"].to_numpy(), n=300),
                "shade_share_peak_65": wmean(g["shade_share_peak"], g["pop65"]),
                "shade_share_peak_all": wmean(g["shade_share_peak"], g["pop"]),
                "gap_65_minus_all": gap,
                "near_10min_65": wmean(g["near"], g["pop65"]),
                "near_and_shaded_65": wmean(g["near_shaded"], g["pop65"]),
                "near_and_shaded_65_ci": block_bootstrap(g, lambda x: wmean(x["near_shaded"], x["pop65"]), g["block"].to_numpy(), n=300),
                "distance_access_but_sunny_65": wmean(g["near"] & ~g["near_shaded"], g["pop65"]) / max(wmean(g["near"], g["pop65"]), 1e-9),
                # Decomposition: everyone at adult speed -> everyone at older speed -> where older adults live.
                "sun_min_all_adult_speed": wmean(g["sun_min_adult"], g["pop"]),
                "sun_min_all_older_speed": wmean(g["sun_min_older"], g["pop"]),
                "dist_65": wmean(g["dist"], g["pop65"]),
                "dist_all": wmean(g["dist"], g["pop"]),
            }
        out[sc] = res
    # Neighbourhood view: share of 65+ and their route shade
    r = _route_frame(per, "lidar")
    rg = _route_frame(per, "global")
    cell = r.groupby("cell").agg(x=("x", "first"), y=("y", "first"), pop=("pop", "first"), pop65=("pop65", "first"),
                                 shade=("shade_share_peak", "mean"), sunmin=("sun_min_older", "mean"))
    cell["shade_global"] = rg.groupby("cell")["shade_share_peak"].mean()
    cell["sunmin_global"] = rg.groupby("cell")["sun_min_older"].mean()
    pts = gpd.GeoDataFrame(cell, geometry=gpd.points_from_xy(cell.x, cell.y), crs=city.crs)
    nb = D.neighbourhoods(city)
    j = gpd.sjoin(pts, nb, predicate="within")
    agg = j.groupby("name").apply(lambda g: pd.Series({
        "pop": g["pop"].sum(), "pop65": g["pop65"].sum(),
        "share65": g["pop65"].sum() / g["pop"].sum(),
        "route_shade_65": np.average(g["shade"], weights=g["pop65"]),
        "route_shade_65_global": np.average(g["shade_global"], weights=g["pop65"]),
        "sun_min_65": np.average(g["sunmin"], weights=g["pop65"]),
        "sun_min_65_global": np.average(g["sunmin_global"], weights=g["pop65"]),
    }), include_groups=False)
    agg.to_csv(C.RESULTS / "neighbourhood_equity.csv")
    cell.drop(columns="geometry", errors="ignore").to_parquet(D.city_dir(city) / "cell_summary.parquet")
    out["neighbourhoods"] = {
        "spearman_share65_vs_route_shade": float(stats.spearmanr(agg["share65"], agg["route_shade_65"]).statistic),
        "spearman_route_shade_lidar_vs_global": float(stats.spearmanr(agg["route_shade_65"], agg["route_shade_65_global"]).statistic),
        "spearman_sun_min_lidar_vs_global": float(stats.spearmanr(agg["sun_min_65"], agg["sun_min_65_global"]).statistic),
        "least_shaded_lidar": agg["route_shade_65"].nsmallest(5).index.tolist(),
        "least_shaded_global": agg["route_shade_65_global"].nsmallest(5).index.tolist(),
    }
    # Cell level: are cells with more older adults on shadier or sunnier routes?
    c = cell[cell["pop"] > 5]
    share = c["pop65"] / c["pop"]
    out["cells"] = {
        "n": int(len(c)),
        "spearman_share65_vs_route_shade": float(stats.spearmanr(share, c["shade"]).statistic),
        "spearman_share65_vs_route_shade_global": float(stats.spearmanr(share, c["shade_global"]).statistic),
        "cell_route_shade_lidar_vs_global": agreement(c["shade"], c["shade_global"], c["pop65"]),
    }
    # Shade-seeking routes
    base = sa[sa.alpha == 0].set_index(["cell", "kind"])
    rows = {}
    for a, g in sa[sa.alpha > 0].groupby("alpha"):
        g = g.set_index(["cell", "kind"])
        b = base.loc[g.index]
        w = g["pop65"]
        rows[f"{a:g}"] = {
            "detour_pct_65": 100 * (wmean(g["length"], w) / wmean(b["length"], w) - 1),
            "sun_reduction_pct_65": 100 * (1 - wmean(g["sun"], w) / wmean(b["sun"], w)),
        }
    out["shade_seeking"] = rows
    return out


def sensitivity(city: C.City) -> dict:
    """Re-run network shade for LiDAR and global models with alternative assumptions."""
    variants = {
        "baseline": {},
        "transmissivity_0.3": {"transmissivity": 0.3},
        "solid_crowns_trunk_0": {"trunk_ratio": 0.0},
        "centreline_only": {"sidewalks": False},
        "day_2021-07-29": {"day": (2021, 7, 29)},
    }
    out = {}
    # Alternative LiDAR references: green canopy only (lower bound) and shape/texture-cleaned (best estimate).
    t = edge_table(city)
    for ref, tag in (("lidar_green", "_green"), ("lidar_clean", "_clean")):
        network.edge_shade(city, scenarios=[ref], tag=tag)
        tr = edge_table(city, tag=tag)
        tj = t.join(tr[[f"shade_{ref}", f"shade15_{ref}"]], how="inner")
        w = tj["length"].to_numpy()
        out[f"reference_{ref}"] = {
            "city_shade_reference": wmean(tj[f"shade_{ref}"], w),
            "city_shade_lidar": wmean(tj["shade_lidar"], w),
            "city_shade_global": wmean(tj["shade_global"], w),
            "edge_daytime_global_vs_reference": agreement(tj[f"shade_{ref}"], tj["shade_global"], w),
            "edge_daytime_lidar_vs_reference": agreement(tj[f"shade_{ref}"], tj["shade_lidar"], w),
        }
        D.log(f"  {ref}: {out[f'reference_{ref}']['city_shade_reference']:.3f}")
    for name, kw in variants.items():
        tag = "" if name == "baseline" else "_" + name.replace(".", "").replace("-", "")
        if tag:
            network.edge_shade(city, scenarios=["lidar", "global"], tag=tag, **kw)
        t = edge_table(city, tag=tag)
        w = t["length"].to_numpy()
        out[name] = {
            "city_shade_lidar": wmean(t["shade_lidar"], w),
            "city_shade_global": wmean(t["shade_global"], w),
            "edge_daytime": agreement(t["shade_lidar"], t["shade_global"], w),
        }
        D.log(f"  sensitivity {name}: lidar {out[name]['city_shade_lidar']:.3f} global {out[name]['city_shade_global']:.3f}")
    return out
