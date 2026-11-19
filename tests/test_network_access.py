import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from shadeq.access import Net, tree_sums
from shadeq.network import _first, sample_points


def _line_net():
    # 0 - 1 - 2 - 3 in a line, plus a long sunny shortcut-free detour 0 - 4 - 3.
    nodes = pd.DataFrame({"node": [10, 11, 12, 13, 14], "x": [0, 100, 200, 300, 150], "y": [0, 0, 0, 0, 100]})
    e = pd.DataFrame({"u": [10, 11, 12, 10, 14], "v": [11, 12, 13, 14, 13], "length": [100.0, 100, 100, 180, 180], "eid": [0, 1, 2, 3, 4]})
    return Net(e, nodes)


def test_tree_sums_shortest_and_accumulated_values():
    net = _line_net()
    L = net.e["length"].to_numpy()
    sun = np.where(net.e["eid"].isin([0, 1, 2]), 1.0, 0.0) * L  # straight route fully sunny, detour fully shaded
    src = np.array([net.index[13]])
    cost, sums, reached = tree_sums(net, src, L, np.c_[L, sun])
    i0 = net.index[10]
    assert reached.all()
    assert cost[i0] == pytest.approx(300)
    assert sums[i0, 0] == pytest.approx(300)
    assert sums[i0, 1] == pytest.approx(300)
    # Shade-seeking cost (alpha = 1): straight = 600, detour = 360, so the detour wins.
    cost2, sums2, _ = tree_sums(net, src, L + 1.0 * sun, np.c_[L, sun])
    assert cost2[i0] == pytest.approx(360)
    assert sums2[i0, 0] == pytest.approx(360)
    assert sums2[i0, 1] == pytest.approx(0)


def test_nearest_of_several_sources():
    net = _line_net()
    L = net.e["length"].to_numpy()
    src = np.array([net.index[10], net.index[13]])
    cost, sums, _ = tree_sums(net, src, L, np.c_[L])
    assert cost[net.index[11]] == pytest.approx(100)
    assert cost[net.index[12]] == pytest.approx(100)
    assert sums[net.index[12], 0] == pytest.approx(100)


def test_parallel_edges_keep_shortest():
    nodes = pd.DataFrame({"node": [1, 2], "x": [0, 50], "y": [0, 0]})
    e = pd.DataFrame({"u": [1, 2], "v": [2, 1], "length": [80.0, 50.0], "eid": [0, 1]})
    net = Net(e, nodes)
    assert len(net.e) == 2 or net.e["length"].min() == 50  # both directions may stay, each as its own (u, v)
    cost, _, _ = tree_sums(net, np.array([0]), net.e["length"].to_numpy(), np.c_[net.e["length"].to_numpy()])
    assert cost[1] == pytest.approx(50)


def test_sample_points_streets_get_two_sidewalks():
    e = gpd.GeoDataFrame(
        {"eid": [0, 1], "highway": ["residential", "footway"], "length": [100.0, 100.0]},
        geometry=[shapely.LineString([(0, 0), (100, 0)]), shapely.LineString([(0, 50), (100, 50)])],
    )
    pts = sample_points(e, step=2.0)
    street = pts[pts.eid == 0]
    assert set(street.side) == {-1, 1}
    assert len(street) == 100
    assert np.allclose(np.abs(street.y), 6.5)  # 5 m half width + 1.5 m setback
    foot = pts[pts.eid == 1]
    assert (foot.side == 0).all() and len(foot) == 50 and np.allclose(foot.y, 50)


def test_first_highway_value():
    assert _first(["residential", "service"]) == "residential"
    assert _first("['footway', 'path']") == "footway"
    assert _first("primary") == "primary"
