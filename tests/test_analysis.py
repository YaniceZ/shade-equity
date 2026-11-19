import numpy as np
import pandas as pd
import pytest

from shadeq.analysis import agreement, block_bootstrap, block_ids, route_summary, wmean


def test_agreement_perfect_and_biased():
    x = np.linspace(0, 1, 50)
    a = agreement(x, x)
    assert a["bias"] == 0 and a["r2"] == pytest.approx(1) and a["spearman"] == pytest.approx(1)
    b = agreement(x, x + 0.1)
    assert b["bias"] == pytest.approx(0.1) and b["mae"] == pytest.approx(0.1)


def test_weighted_agreement_uses_weights():
    ref = np.array([0.0, 1.0])
    est = np.array([1.0, 1.0])
    assert agreement(ref, est, w=np.array([1, 3]))["bias"] == pytest.approx(0.25)


def test_wmean_ignores_nan():
    assert wmean([1, np.nan, 3], [1, 1, 1]) == pytest.approx(2)


def test_block_bootstrap_brackets_mean():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.normal(5, 1, 400), "x": rng.uniform(0, 5000, 400), "y": rng.uniform(0, 5000, 400)})
    lo, hi = block_bootstrap(df, lambda d: d["v"].mean(), block_ids(df.x, df.y), n=200)
    assert lo < df["v"].mean() < hi


def test_route_summary_metrics():
    routes = pd.DataFrame({
        "cell": [0, 1], "pop": [10.0, 10.0], "pop65": [1.0, 3.0], "x": [0, 0], "y": [0, 0], "snap": [0, 0],
        "kind": "grocery", "dist": [300.0, 1000.0], "sun_10": [300.0, 0.0], "sun_15": [300.0, 0.0], "scenario": "lidar",
    })
    s, per = route_summary(routes, [10.0, 15.0])
    old = s[s.group == "65+"].iloc[0]
    # cell 0: 300 m in sun at 15:00 -> 300 / 0.9 / 60 min; cell 1: none. Weights 1 and 3.
    assert old["sun_min_older_peak"] == pytest.approx(300 / 0.9 / 60 / 4)
    assert per.loc[0, "near"] and not per.loc[1, "near"]  # 1000 m at 0.9 m/s is 18.5 min
    assert old["near_and_shaded"] == 0
