"""Route-based shade access from home (100 m population cells) to the nearest essential destination.

For each destination type a multi-source Dijkstra on walking length gives every network node
its nearest destination and the shortest path to it. Sun-exposed length is then accumulated
along each node's path. Exposure is reported for a slow older pedestrian (``SPEED_OLDER``) and
weighted by population aged 65+ or by total population.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config as C
from . import data as D
from . import network


class Net:
    """Undirected simple graph in index form (parallel edges collapsed to the shortest)."""

    def __init__(self, e: pd.DataFrame, nodes: pd.DataFrame):
        ids = nodes["node"].to_numpy()
        self.index = pd.Series(np.arange(len(ids)), index=ids)
        self.xy = nodes[["x", "y"]].to_numpy()
        u = self.index.loc[e["u"].to_numpy()].to_numpy()
        v = self.index.loc[e["v"].to_numpy()].to_numpy()
        df = pd.DataFrame({"u": u, "v": v, "length": e["length"].to_numpy(), "eid": e["eid"].to_numpy()})
        df = df[df.u != df.v].sort_values("length").drop_duplicates(["u", "v"]).reset_index(drop=True)
        self.e = df
        self.n = len(ids)

    def matrix(self, weight: np.ndarray) -> sparse.csr_matrix:
        w = np.maximum(weight, 1e-6)
        return sparse.csr_matrix((w, (self.e.u, self.e.v)), shape=(self.n, self.n))

    def edge_lookup(self) -> sparse.csr_matrix:
        k = np.arange(1, len(self.e) + 1)
        m = sparse.coo_matrix((k, (self.e.u, self.e.v)), shape=(self.n, self.n))
        return (m + m.T).tocsr()


@njit(cache=True)
def _accumulate(order, pred, eidx, values):
    out = np.zeros((pred.shape[0], values.shape[1]), dtype=np.float64)
    for n in order:
        p = pred[n]
        if p >= 0:
            k = eidx[n]
            for t in range(values.shape[1]):
                out[n, t] = out[p, t] + values[k, t]
    return out


def tree_sums(net: Net, sources: np.ndarray, weight: np.ndarray, values: np.ndarray):
    """Nearest-source shortest-path tree on ``weight``; returns (cost, sums of ``values`` columns along each path, reached)."""
    A = net.matrix(weight)
    cost, pred, _ = dijkstra(A, directed=False, indices=np.unique(sources), min_only=True, return_predecessors=True)
    reached = np.isfinite(cost)
    lk = net.edge_lookup()
    has = pred >= 0
    eidx = np.zeros(net.n, dtype=np.int64)
    nodes = np.nonzero(has)[0]
    eidx[nodes] = np.asarray(lk[pred[nodes], nodes]).ravel() - 1
    order = np.argsort(np.where(reached, cost, np.inf)).astype(np.int64)
    order = order[reached[order]]
    sums = _accumulate(order, pred.astype(np.int64), eidx, np.ascontiguousarray(values, dtype=np.float64))
    return cost, sums, reached


def destination_nodes(city: C.City, net: Net) -> dict[str, np.ndarray]:
    dest = gpd.read_parquet(D.city_dir(city) / "destinations.parquet")
    tree = cKDTree(net.xy)
    out = {}
    for kind, df in dest.groupby("kind"):
        if kind == "park":
            # Any network node within 20 m of the park counts as an entrance.
            polys = df[df.geom_type.isin(["Polygon", "MultiPolygon"])]
            polys = polys[polys.area >= 2000]  # drop pocket greens and traffic islands
            nodes_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(net.xy[:, 0], net.xy[:, 1]), crs=city.crs)
            hit = gpd.sjoin(nodes_gdf, gpd.GeoDataFrame(geometry=polys.buffer(20), crs=city.crs), predicate="within")
            out[kind] = np.unique(hit.index.to_numpy())
        else:
            pts = df.geometry.representative_point()
            _, idx = tree.query(np.c_[pts.x, pts.y])
            out[kind] = np.unique(idx)
    return out


def origins(city: C.City, net: Net) -> pd.DataFrame:
    pop = D.population(city)
    c = pop.geometry.centroid
    dist, idx = cKDTree(net.xy).query(np.c_[c.x, c.y])
    return pd.DataFrame({"cell": np.arange(len(pop)), "node": idx, "snap": dist, "pop": pop["pop"].to_numpy(),
                         "pop65": pop["pop65"].to_numpy(), "x": c.x.to_numpy(), "y": c.y.to_numpy()})


def edge_sun_table(city: C.City, net: Net, scenario: str, tag: str = "") -> tuple[np.ndarray, list[float]]:
    """Sunlit length (m) of each graph edge at each hour for one scenario."""
    es = network.edge_shade(city, tag=tag) if tag else network.edge_shade(city)
    es = es[es.scenario == scenario]
    wide = es.pivot(index="eid", columns="hour", values="sun")
    hours = list(wide.columns)
    frac = wide.reindex(net.e["eid"].to_numpy()).to_numpy()
    # Edges without a valid sample: indoor links (corridors, arcades) count as shaded; edges beyond the
    # surface grid (outside the city buffer) get the hourly network mean so they neither help nor hurt.
    xmin, ymin, xmax, ymax = D.grid(city)["bounds"]
    mid = net.xy[net.e["u"].to_numpy()] * 0.5 + net.xy[net.e["v"].to_numpy()] * 0.5
    outside = (mid[:, 0] < xmin) | (mid[:, 0] > xmax) | (mid[:, 1] < ymin) | (mid[:, 1] > ymax)
    mean = np.nanmean(frac, axis=0)
    frac = np.where(np.isnan(frac) & outside[:, None], mean[None, :], frac)
    frac = np.nan_to_num(frac, nan=0.0)
    return frac * net.e["length"].to_numpy()[:, None], hours


def route_exposure(city: C.City, scenario: str, tag: str = "") -> pd.DataFrame:
    """Per origin cell and destination type: distance and sun-exposed length at each hour (shortest path)."""
    e, nodes = network.edges(city)
    net = Net(e, nodes)
    sun_len, hours = edge_sun_table(city, net, scenario, tag)
    o = origins(city, net)
    dests = destination_nodes(city, net)
    frames = []
    for kind, src in dests.items():
        dist, sums, reached = tree_sums(net, src, net.e["length"].to_numpy(), sun_len)
        df = o[["cell", "pop", "pop65", "x", "y", "snap"]].copy()
        df["kind"] = kind
        df["dist"] = dist[o["node"]]
        for j, h in enumerate(hours):
            df[f"sun_{h:g}"] = sums[o["node"], j]
        frames.append(df[reached[o["node"]]])
    out = pd.concat(frames, ignore_index=True)
    out["scenario"] = scenario
    return out


def shade_aware(city: C.City, scenario: str = "lidar", alphas=(0.5, 1, 2, 4), hour: float = C.PEAK_HOUR) -> pd.DataFrame:
    """Shade-seeking routes (cost = length + alpha x sunlit length) versus shortest routes at one hour."""
    e, nodes = network.edges(city)
    net = Net(e, nodes)
    sun_len, hours = edge_sun_table(city, net, scenario)
    j = hours.index(hour)
    sl = sun_len[:, j]
    L = net.e["length"].to_numpy()
    vals = np.c_[L, sl]
    o = origins(city, net)
    rows = []
    for kind, src in destination_nodes(city, net).items():
        _, base, reached = tree_sums(net, src, L, vals)
        for a in (0.0, *alphas):
            _, s, _ = tree_sums(net, src, L + a * sl, vals) if a else (None, base, None)
            df = pd.DataFrame({"cell": o["cell"], "kind": kind, "alpha": a, "length": s[o["node"], 0], "sun": s[o["node"], 1],
                               "pop": o["pop"], "pop65": o["pop65"]})
            rows.append(df[reached[o["node"]]])
    return pd.concat(rows, ignore_index=True)
