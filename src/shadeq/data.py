"""Download and clip every input to the study grid.

All rasters end up on one 1 m north-up grid in the city's metric CRS, saved under
``data/interim/<city>/``. Vector layers are saved as GeoParquet in the same CRS.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from . import config as C


def city_dir(city: C.City) -> Path:
    d = C.INTERIM / city.key
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    print(msg, flush=True)


# --- boundary and grid -------------------------------------------------------


def boundary(city: C.City) -> gpd.GeoDataFrame:
    path = city_dir(city) / "boundary.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    import osmnx as ox

    b = ox.geocode_to_gdf(city.place)[["geometry"]].to_crs(city.crs)
    b.to_parquet(path)
    return b


def grid(city: C.City) -> dict:
    """1 m grid covering the city plus ``BUFFER`` (cells snapped to whole metres)."""
    xmin, ymin, xmax, ymax = boundary(city).total_bounds
    b = C.BUFFER
    xmin, ymin = math.floor(xmin - b), math.floor(ymin - b)
    xmax, ymax = math.ceil(xmax + b), math.ceil(ymax + b)
    width = int((xmax - xmin) / C.CELL)
    height = int((ymax - ymin) / C.CELL)
    return {
        "crs": city.crs,
        "transform": from_origin(xmin, ymax, C.CELL, C.CELL),
        "width": width,
        "height": height,
        "bounds": (xmin, ymin, xmax, ymax),
    }


def bbox_lonlat(city: C.City, pad: float = 0.0) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = grid(city)["bounds"]
    box = gpd.GeoSeries([shapely.box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)], crs=city.crs).to_crs(4326)
    return tuple(box.total_bounds)


def save_raster(path: Path, arr: np.ndarray, g: dict, nodata=None) -> None:
    prof = {
        "driver": "GTiff",
        "dtype": arr.dtype,
        "count": 1,
        "crs": g["crs"],
        "transform": g["transform"],
        "width": g["width"],
        "height": g["height"],
        "compress": "deflate",
        "predictor": 2 if arr.dtype.kind == "f" else 1,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def warp_to_grid(src_path: str, g: dict, resampling=Resampling.bilinear, dtype="float32") -> np.ndarray:
    """Resample any raster (local or remote COG) onto the study grid."""
    env = {"GDAL_HTTP_MULTIRANGE": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5", "VSI_CACHE": "TRUE"}
    with rasterio.Env(**env), rasterio.open(src_path) as src:
        nd = src.nodata
        with WarpedVRT(
            src, crs=g["crs"], transform=g["transform"], width=g["width"], height=g["height"], resampling=resampling, src_nodata=nd, nodata=np.nan
        ) as vrt:
            arr = vrt.read(1, out_dtype=dtype)
    return arr


# --- LiDAR (reference) ---------------------------------------------------------


def lidar(city: C.City) -> None:
    d = city_dir(city)
    g = grid(city)
    for name, url in (("dtm", city.lidar_dtm), ("dsm", city.lidar_dsm)):
        out = d / f"lidar_{name}.tif"
        if out.exists():
            continue
        log(f"reading LiDAR {name} from {url}")
        arr = warp_to_grid("/vsicurl/" + url, g)
        save_raster(out, arr.astype("float32"), g, nodata=np.nan)


# --- building footprints ---------------------------------------------------------


def overture_buildings(city: C.City) -> gpd.GeoDataFrame:
    """Overture building footprints (all sources) for the grid area; used to split LiDAR heights into buildings and trees."""
    path = city_dir(city) / "overture_buildings.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    import duckdb

    x0, y0, x1, y1 = bbox_lonlat(city)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    q = f"""
    SELECT id, height, num_floors, sources[1].dataset AS source, ST_AsWKB(geometry) AS wkb
    FROM read_parquet('s3://overturemaps-us-west-2/release/{C.OVERTURE_RELEASE}/theme=buildings/type=building/*', hive_partitioning=1)
    WHERE bbox.xmin > {x0} AND bbox.xmax < {x1} AND bbox.ymin > {y0} AND bbox.ymax < {y1}
    """
    log("querying Overture buildings")
    df = con.execute(q).fetch_df()
    geom = shapely.from_wkb(df.pop("wkb").apply(bytes))
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326).to_crs(city.crs)
    gdf.to_parquet(path)
    return gdf


def gba_buildings(city: C.City) -> gpd.GeoDataFrame:
    """GlobalBuildingAtlas polygons (ODbL part + other part) joined to their LoD1 heights."""
    path = city_dir(city) / "gba_buildings.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    raw = C.RAW / "gba"
    x0, y0, x1, y1 = bbox_lonlat(city)
    b3857 = gpd.GeoSeries([shapely.box(x0, y0, x1, y1)], crs=4326).to_crs(3857).total_bounds
    parts = []
    for f in [f"odbl_{city.gba_tile.split('_')[0]}_{city.gba_tile.split('_')[1]}.geojson", f"poly_{city.gba_tile.split('_')[0]}_{city.gba_tile.split('_')[1]}.geojson"]:
        log(f"reading GBA polygons {f}")
        p = gpd.read_file(raw / f, bbox=tuple(b3857), engine="pyogrio")
        parts.append(p)
    poly = pd.concat(parts, ignore_index=True)
    poly["key"] = poly["source"].astype(str) + poly["id"].astype(str) + poly["region"].astype(str)
    keys = set(poly["key"])
    log(f"{len(poly)} GBA polygons in box; loading LoD1 heights")
    tile = city.gba_tile.split("_")
    with open(raw / f"lod1_{tile[0]}_{tile[1]}.json", encoding="utf-8") as fh:
        lod = json.load(fh)
    poly["height"] = [lod[k]["height"] if k in lod else np.nan for k in poly["key"]]
    del lod
    poly = poly.set_crs(3857, allow_override=True).to_crs(city.crs)
    log(f"GBA heights found for {poly['height'].notna().mean():.1%} of polygons ({len(keys)} keys)")
    poly.to_parquet(path)
    return poly


# --- Meta/WRI canopy height ----------------------------------------------------------


def meta_chm(city: C.City) -> None:
    out = city_dir(city) / "meta_chm.tif"
    if out.exists():
        return
    g = grid(city)
    tiles = gpd.read_file(C.RAW / "meta_tiles.geojson")
    box = shapely.box(*bbox_lonlat(city))
    sel = tiles[tiles.intersects(box)]["tile"].tolist()
    acc = np.full((g["height"], g["width"]), np.nan, np.float32)
    for t in sel:
        local = C.RAW / "meta" / f"{t}.tif"
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            log(f"downloading Meta CHM tile {t}")
            with requests.get(f"{C.META_CHM}/chm/{t}.tif", stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
        arr = warp_to_grid(str(local), g, resampling=Resampling.bilinear)
        acc = np.where(np.isnan(acc), arr, acc)
    save_raster(out, acc, g, nodata=np.nan)


# --- population -------------------------------------------------------------------------


WORLDPOP_AGES = (65, 70, 75, 80, 85, 90)


def worldpop(city: C.City) -> gpd.GeoDataFrame:
    """100 m WorldPop 2025 cells inside the city: total population and population aged 65+."""
    path = city_dir(city) / "population.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    raw = C.RAW / "worldpop"
    iso = city.worldpop_iso.lower()
    x0, y0, x1, y1 = bbox_lonlat(city)
    files = {"pop": raw / f"{iso}_pop.tif", **{f"a{a}": raw / f"{iso}_t_{a}.tif" for a in WORLDPOP_AGES}}
    arrays = {}
    for k, f in files.items():
        with rasterio.open(f) as src:
            win = rasterio.windows.from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            a = src.read(1, window=win).astype("float64")
            nd = src.nodata
            if nd is not None:
                a[a == nd] = 0
            a[~np.isfinite(a) | (a < 0)] = 0
            arrays[k] = a
            tr = src.window_transform(win)
    rows, cols = np.nonzero(arrays["pop"] > 0)
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    df = pd.DataFrame({"pop": arrays["pop"][rows, cols], "pop65": sum(arrays[f"a{a}"][rows, cols] for a in WORLDPOP_AGES)})
    cell = gpd.GeoSeries(
        [shapely.box(x - tr.a / 2, y + tr.e / 2, x + tr.a / 2, y - tr.e / 2) for x, y in zip(xs, ys, strict=True)], crs=4326
    )
    gdf = gpd.GeoDataFrame(df, geometry=cell.to_crs(city.crs))
    city_poly = boundary(city).geometry.union_all()
    gdf = gdf[gdf.centroid.within(city_poly)].reset_index(drop=True)
    gdf.to_parquet(path)
    return gdf


STATCAN_DA = "https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer/12/query"
STATCAN_SDMX = "https://api.statcan.gc.ca/census-recensement/profile/sdmx/rest/data/STC_CP,DF_DA"
# 2021 Census Profile characteristic ids: 1 = population, 8 = total age groups, 24 = 65 years and over.
CENSUS_CHARS = {"1": "pop_census", "8": "age_total", "24": "age65"}


def census_da(city: C.City) -> gpd.GeoDataFrame:
    """2021 Census dissemination areas (cartographic boundaries) with population and population aged 65+."""
    path = city_dir(city) / "census_da.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    x0, y0, x1, y1 = bbox_lonlat(city)
    feats, offset = [], 0
    while True:
        r = requests.get(STATCAN_DA, params={"geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                                             "spatialRel": "esriSpatialRelIntersects", "outFields": "DAUID,DGUID", "outSR": 4326,
                                             "f": "geojson", "resultOffset": offset, "resultRecordCount": 2000}, timeout=300)
        r.raise_for_status()
        batch = r.json()["features"]
        feats += batch
        if len(batch) < 2000:
            break
        offset += 2000
    da = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(city.crs)
    poly = boundary(city).geometry.union_all()
    da = da[da.representative_point().within(poly)].reset_index(drop=True)
    log(f"{len(da)} dissemination areas in the city; querying the 2021 Census Profile")
    rows = []
    ids = da["DGUID"].tolist()
    for i in range(0, len(ids), 80):
        key = "A5." + "+".join(ids[i : i + 80]) + ".1." + "+".join(CENSUS_CHARS) + ".1"
        r = requests.get(f"{STATCAN_SDMX}/{key}", headers={"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}, timeout=300)
        r.raise_for_status()
        rows.append(pd.read_csv(io.StringIO(r.text), usecols=["REF_AREA", "CHARACTERISTIC", "OBS_VALUE"]))
    t = pd.concat(rows).pivot_table(index="REF_AREA", columns="CHARACTERISTIC", values="OBS_VALUE", aggfunc="first")
    t.columns = [CENSUS_CHARS[str(c)] for c in t.columns]
    da = da.join(t, on="DGUID")
    da.to_parquet(path)
    return da


def population(city: C.City) -> gpd.GeoDataFrame:
    """WorldPop 100 m cells with 65+ counts re-allocated from 2021 Census dissemination areas.

    WorldPop's age structure is uniform inside Canadian cities (one 65+ share for all of Vancouver), so each
    cell keeps its WorldPop total and takes the 65+ share of the dissemination area containing its centre.
    """
    path = city_dir(city) / "population_census.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    wp = worldpop(city).rename(columns={"pop65": "pop65_worldpop"})
    da = census_da(city)
    da["share65"] = (da["age65"] / da["age_total"]).where(da["age_total"] > 0)
    c = gpd.GeoDataFrame(wp[[]], geometry=wp.geometry.centroid, crs=city.crs)
    j = gpd.sjoin(c, da[["DGUID", "share65", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    wp["DGUID"] = j["DGUID"]
    share = j["share65"].fillna(float(da["age65"].sum() / da["age_total"].sum()))
    wp["pop65"] = wp["pop"] * share
    wp.to_parquet(path)
    return wp


def neighbourhoods(city: C.City) -> gpd.GeoDataFrame | None:
    if not city.neighbourhoods_url:
        return None
    path = city_dir(city) / "neighbourhoods.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    r = requests.get(city.neighbourhoods_url, timeout=120)
    r.raise_for_status()
    gdf = gpd.read_file(io.StringIO(r.text)).to_crs(city.crs)
    name = next(c for c in ("name", "mapid", "NAME") if c in gdf.columns)
    gdf = gdf.rename(columns={name: "name"})[["name", "geometry"]]
    gdf.to_parquet(path)
    return gdf


# --- OpenStreetMap: walk network and destinations -----------------------------------------


def osm(city: C.City):
    import osmnx as ox

    d = city_dir(city)
    gpath = d / "walk.graphml"
    dpath = d / "destinations.parquet"
    poly = boundary(city).geometry.union_all().buffer(C.NETWORK_BUFFER)
    poly_ll = gpd.GeoSeries([poly], crs=city.crs).to_crs(4326).iloc[0]
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(C.DATA / "cache" / "osmnx")
    ox.settings.requests_timeout = 600
    ox.settings.overpass_url = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api")
    if not gpath.exists():
        log("downloading OSM walk network")
        G = ox.graph_from_polygon(poly_ll, network_type="walk", simplify=True, retain_all=False)
        G = ox.project_graph(G, to_crs=city.crs)
        ox.save_graphml(G, gpath)
    if not dpath.exists():
        frames = []
        for kind, tags in C.DESTINATIONS.items():
            log(f"downloading OSM destinations: {kind}")
            try:
                f = ox.features_from_polygon(poly_ll, tags)
            except Exception as exc:  # noqa: BLE001 - an empty query raises
                log(f"  none ({exc.__class__.__name__})")
                continue
            f = f.reset_index()
            keep = np.zeros(len(f), bool)
            for k, vals in tags.items():
                if k in f.columns:
                    keep |= f[k].isin(vals).to_numpy()
            f = f[keep]
            f = gpd.GeoDataFrame({"kind": kind, "osm_type": f["element"].astype(str), "osm_id": f["id"].astype(str),
                                  "name": f.get("name", pd.Series(index=f.index, dtype=object))}, geometry=f.geometry, crs=4326)
            frames.append(f)
        dest = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326).to_crs(city.crs)
        dest.to_parquet(dpath)


def osm_from_pbf(city: C.City, pbf: Path) -> None:
    """Same outputs as :func:`osm`, built from a Geofabrik extract with pyrosm (when Overpass is unavailable)."""
    import networkx as nx
    import osmnx as ox
    from pyrosm import OSM

    d = city_dir(city)
    gpath = d / "walk.graphml"
    dpath = d / "destinations.parquet"
    poly = boundary(city).geometry.union_all().buffer(C.NETWORK_BUFFER)
    poly_ll = gpd.GeoSeries([poly], crs=city.crs).to_crs(4326).iloc[0]
    osm_ = OSM(str(pbf), bounding_box=poly_ll)
    if not gpath.exists():
        log("building walk network from PBF")
        nodes, edges = osm_.get_network(network_type="walking", nodes=True)
        G = osm_.to_graph(nodes, edges, graph_type="networkx", osmnx_compatible=True, retain_all=False)
        # Left unsimplified: pyrosm edges carry list attributes that osmnx cannot merge. Routing is unaffected.
        G = ox.project_graph(G, to_crs=city.crs)
        if not nx.is_strongly_connected(G):
            big = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(big).copy()
        ox.save_graphml(G, gpath)
    if not dpath.exists():
        frames = []
        for kind, tags in C.DESTINATIONS.items():
            log(f"destinations from PBF: {kind}")
            f = osm_.get_data_by_custom_criteria(custom_filter=tags, keep_nodes=True, keep_ways=True, keep_relations=True)
            if f is None or len(f) == 0:
                continue
            keep = np.zeros(len(f), bool)
            for k, vals in tags.items():
                if k in f.columns:
                    keep |= f[k].isin(vals).to_numpy()
            f = f[keep]
            frames.append(gpd.GeoDataFrame({"kind": kind, "osm_type": f["osm_type"].astype(str), "osm_id": f["id"].astype(str),
                                            "name": f["name"] if "name" in f.columns else None}, geometry=f.geometry, crs=4326))
        dest = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326).to_crs(city.crs)
        dest = dest[dest.geometry.notna() & ~dest.geometry.is_empty]
        dest.to_parquet(dpath)


def ndvi(city: C.City) -> None:
    """Sentinel-2 L2A NDVI (10 m, cloud-free summer scene) on the 1 m grid, to separate green canopy from other tall objects."""
    out = city_dir(city) / "s2_ndvi.tif"
    if out.exists():
        return
    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    item = pc.sign(cat.get_collection("sentinel-2-l2a").get_item(city.extra["s2_item"]))
    g = grid(city)
    red = warp_to_grid(item.assets["B04"].href, g, Resampling.nearest)
    nir = warp_to_grid(item.assets["B08"].href, g, Resampling.nearest)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ((nir - red) / (nir + red)).astype("float32")
    save_raster(out, v, g, nodata=np.nan)


def fetch_all(city: C.City) -> None:
    boundary(city)
    g = grid(city)
    log(f"grid {g['width']} x {g['height']} cells at {C.CELL} m")
    neighbourhoods(city)
    worldpop(city)
    population(city)
    pbf = C.RAW / "osm" / "bc.osm.pbf"
    osm_from_pbf(city, pbf) if pbf.exists() else osm(city)
    overture_buildings(city)
    gba_buildings(city)
    meta_chm(city)
    lidar(city)
    ndvi(city)
