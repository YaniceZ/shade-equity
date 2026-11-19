"""Export compact GeoJSON layers for the interactive project page (docs/index.html, MapLibre)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from . import config as C
from . import data as D
from .analysis import edge_table


def _write(gdf: gpd.GeoDataFrame, path, grid=1e-5):
    """Compact GeoJSON: WGS84, coordinates snapped to about 1 m, no feature ids."""
    import shapely

    gdf = gdf.to_crs(4326)
    gdf = gdf.set_geometry(shapely.set_precision(gdf.geometry.to_numpy(), grid))
    gdf = gdf[~gdf.geometry.is_empty]
    text = gdf.to_json(drop_id=True, na="null")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def export(city: C.City) -> None:
    out = C.ROOT / "docs" / "data"
    out.mkdir(parents=True, exist_ok=True)
    t = edge_table(city)
    flow = pd.read_parquet(D.city_dir(city) / "older_flow.parquet")["older_flow"] if (D.city_dir(city) / "older_flow.parquet").exists() else None
    e = gpd.GeoDataFrame({
        "sl": (t["shade_lidar"] * 100).round().astype(int),
        "sg": (t["shade_global"] * 100).round().astype(int),
        "pl": (t["shade15_lidar"] * 100).round().astype(int),
        "pg": (t["shade15_global"] * 100).round().astype(int),
    }, geometry=t.geometry.simplify(1.0), crs=city.crs)
    if flow is not None:
        e["f"] = flow.reindex(t.index).fillna(0).round().astype(int).to_numpy()
    _write(e, out / "walkways.geojson")
    cell = pd.read_parquet(D.city_dir(city) / "cell_summary.parquet")
    pop = D.population(city)
    c = gpd.GeoDataFrame(cell.join(pop[["geometry"]]), geometry="geometry", crs=city.crs)
    c = gpd.GeoDataFrame({
        "p65": c["pop65"].round(1), "pop": c["pop"].round(1),
        "sh": (c["shade"] * 100).round(1), "shg": (c["shade_global"] * 100).round(1),
        "sm": c["sunmin"].round(2), "smg": c["sunmin_global"].round(2),
    }, geometry=c.geometry, crs=city.crs)
    _write(c, out / "cells.geojson")
    nb = D.neighbourhoods(city)
    nbe = pd.read_csv(C.RESULTS / "neighbourhood_equity.csv", index_col=0)
    nb = nb.join(nbe, on="name")
    _write(nb[["name", "share65", "route_shade_65", "route_shade_65_global", "sun_min_65", "geometry"]], out / "neighbourhoods.geojson")
    D.log(f"website layers written to {out}")
