"""Street-tree planting scenarios (simulation, not a causal prediction).

Older adults' shortest routes to essential destinations are loaded onto the network. Two
planting rules with the same sidewalk budget are compared:

- ``targeted``: sunny sidewalk length weighted by the number of older adults (65+) routed over it
- ``sunniest``: the sunniest sidewalks regardless of who walks there

Trees are added to the LiDAR tree layer as 8 m crowns (6 m wide) every 8 m on both sidewalks of
the selected streets where the sidewalk is not already under canopy, and shade is recomputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from numba import njit
from rasterio import features
from scipy.sparse.csgraph import dijkstra

from . import config as C
from . import data as D
from . import network, shadow, surfaces
from .access import Net, destination_nodes, edge_sun_table, origins

TREE_H = 8.0
CROWN_R = 3.0
SPACING = 8.0


@njit(cache=True)
def _flow(order_desc, pred, start):
    flow = start.copy()
    for n in order_desc:
        p = pred[n]
        if p >= 0:
            flow[p] += flow[n]
    return flow


def older_flow(city: C.City) -> pd.Series:
    """Older adults (65+) routed over each edge, summed over destination types (shortest paths)."""
    e, nodes = network.edges(city)
    net = Net(e, nodes)
    o = origins(city, net)
    start = np.bincount(o["node"], weights=o["pop65"], minlength=net.n)
    lk = net.edge_lookup()
    total = np.zeros(len(net.e))
    for _, src in destination_nodes(city, net).items():
        A = net.matrix(net.e["length"].to_numpy())
        cost, pred, _ = dijkstra(A, directed=False, indices=np.unique(src), min_only=True, return_predecessors=True)
        reached = np.isfinite(cost)
        order = np.argsort(-np.where(reached, cost, -1)).astype(np.int64)
        order = order[reached[order]]
        flow = _flow(order, pred.astype(np.int64), start * reached)
        has = np.nonzero(pred >= 0)[0]
        k = np.asarray(lk[pred[has], has]).ravel() - 1
        np.add.at(total, k, flow[has])
    return pd.Series(total, index=net.e["eid"].to_numpy())


def plant(city: C.City, eids: np.ndarray, veg: np.ndarray, e) -> tuple[np.ndarray, int]:
    """Return a copy of ``veg`` with street trees on both sidewalks of the given edges."""
    g = D.grid(city)
    sel = e.set_index("eid").loc[eids]
    # Crowns centred on the sidewalk line (both sidewalks for streets sampled with offsets; footways on their line).
    pts = network.sample_points(sel.reset_index(), step=SPACING)
    crowns = shapely.buffer(shapely.points(pts["x"], pts["y"]), CROWN_R)
    m = features.rasterize(((c, 1) for c in crowns), out_shape=veg.shape, transform=g["transform"], dtype="uint8").astype(bool)
    out = veg.copy()
    add = m & (out < TREE_H)
    out[add] = TREE_H
    return out, int(len(pts))


def run(city: C.City, budget_km=(10, 25, 50), hour: float = C.PEAK_HOUR) -> pd.DataFrame:
    e, nodes = network.edges(city)
    net = Net(e, nodes)
    S = surfaces.load(city)
    flow = older_flow(city)
    sun_len, hours = edge_sun_table(city, net, "lidar")
    j = hours.index(hour)
    sunny = pd.Series(sun_len[:, j], index=net.e["eid"].to_numpy())
    street = ~e.set_index("eid")["highway"].isin(["corridor", "steps", "elevator", "platform"])
    poly = D.boundary(city).geometry.union_all()
    in_city = e.set_index("eid").geometry.centroid.within(poly)
    ok = street & in_city
    L = e.set_index("eid")["length"]
    ranks = {
        "targeted": (sunny * flow)[ok[ok].index.intersection(sunny.index)].sort_values(ascending=False),
        "sunniest": (sunny / L.reindex(sunny.index))[ok[ok].index.intersection(sunny.index)].sort_values(ascending=False),
    }
    pts = pd.read_parquet(D.city_dir(city) / "sample_points.parquet")
    g = D.grid(city)
    x0, _, _, y1 = g["bounds"]
    rows = (y1 - pts["y"].to_numpy()) / C.CELL
    cols = (pts["x"].to_numpy() - x0) / C.CELL
    (h, az, el), = [s for s in network.suns(city) if s[0] == hour]
    base_sun = pd.Series(
        shadow.sunlit_points(rows, cols, S["ground"], S["ref_bld"], S["ref_veg"], az, el, C.CELL, C.TREE_TRANSMISSIVITY, C.TRUNK_RATIO)
    ).groupby(pts["eid"].to_numpy()).mean()
    o = origins(city, net)
    dests = destination_nodes(city, net)

    def exposure(edge_sun: pd.Series) -> float:
        """Older-adult weighted sun-exposed minutes per trip (mean over destination types)."""
        from .access import tree_sums

        frac = edge_sun.reindex(net.e["eid"].to_numpy()).fillna(0).to_numpy()
        sl = frac * net.e["length"].to_numpy()
        vals = []
        for _, src in dests.items():
            _, s, reached = tree_sums(net, src, net.e["length"].to_numpy(), sl[:, None])
            m = reached[o["node"]]
            vals.append(np.average(s[o["node"], 0][m], weights=o["pop65"][m]) / C.SPEED_OLDER / 60)
        return float(np.mean(vals))

    rows_out = [{"rule": "baseline", "budget_km": 0, "trees": 0, "sun_min_older": exposure(base_sun)}]
    for rule, r in ranks.items():
        for km in budget_km:
            cum = L.reindex(r.index).cumsum() / 1000
            eids = r.index[cum <= km].to_numpy()
            veg2, ntrees = plant(city, eids, S["ref_veg"], e)
            s2 = pd.Series(
                shadow.sunlit_points(rows, cols, S["ground"], S["ref_bld"], veg2, az, el, C.CELL, C.TREE_TRANSMISSIVITY, C.TRUNK_RATIO)
            ).groupby(pts["eid"].to_numpy()).mean()
            rows_out.append({"rule": rule, "budget_km": km, "trees": ntrees, "sun_min_older": exposure(s2), "edges": len(eids)})
            if km == max(budget_km):
                pd.DataFrame({"eid": eids}).to_parquet(D.city_dir(city) / f"planted_{rule}.parquet")
            D.log(f"  scenario {rule} {km} km: {rows_out[-1]['sun_min_older']:.2f} min")
    flow.to_frame("older_flow").to_parquet(D.city_dir(city) / "older_flow.parquet")
    return pd.DataFrame(rows_out)
