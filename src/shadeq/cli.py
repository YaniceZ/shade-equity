"""Command line: ``shadeq [--city vancouver] <step>``."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from . import analysis as A
from . import config as C
from . import data as D
from . import network, surfaces

STEPS = ["fetch", "surfaces", "shade", "validate", "citycheck", "routes", "scenario", "sensitivity", "figures", "website"]


def step_fetch(city):
    D.fetch_all(city)


def step_surfaces(city):
    surfaces.build(city)


def step_shade(city):
    network.edge_shade(city)


def step_validate(city):
    out = {
        "buildings": A.building_validation(city),
        "canopy": A.canopy_validation(city),
        "network": A.network_validation(city),
        "points_15h": A.point_validation(city),
        "areas": A.area_validation(city),
    }
    C.RESULTS.mkdir(exist_ok=True)
    A.write_json(out, C.RESULTS / "validation.json")
    D.log(f"edge daytime shade: {out['network']['edge_daytime']}")


def step_routes(city):
    from . import access

    hours = list(city.hours)
    frames = [access.route_exposure(city, s) for s in ("lidar", "global")]
    routes = pd.concat(frames, ignore_index=True)
    summary, per = A.route_summary(routes, hours)
    per.to_parquet(D.city_dir(city) / "routes.parquet")
    summary.to_csv(C.RESULTS / "route_summary.csv", index=False)
    sa = access.shade_aware(city)
    sa.to_parquet(D.city_dir(city) / "shade_aware.parquet")
    A.write_json(A.equity(city, per, sa), C.RESULTS / "equity.json")


def step_scenario(city):
    from . import scenario

    df = scenario.run(city)
    df.to_csv(C.RESULTS / "planting_scenarios.csv", index=False)


def step_sensitivity(city):
    A.write_json(A.sensitivity(city), C.RESULTS / "sensitivity.json")


def step_figures(city):
    from . import figures

    figures.all_figures(city)


def step_website(city):
    from . import website

    website.export(city)


def step_citycheck(city):
    from . import citycheck

    citycheck.run(city)


def step_lidar2022(city):
    """Optional: needs 2022 City of Vancouver tiles downloaded by hand into data/raw/lidar2022/."""
    from . import lidar2022

    lidar2022.run(city)


def main(argv=None):
    p = argparse.ArgumentParser(prog="shadeq")
    p.add_argument("--city", default="vancouver", choices=sorted(C.CITIES))
    p.add_argument("step", choices=STEPS + ["lidar2022", "all"])
    a = p.parse_args(argv)
    city = C.CITIES[a.city]
    for s in STEPS if a.step == "all" else [a.step]:
        t = time.time()
        D.log(f"== {s}")
        globals()[f"step_{s}"](city)
        D.log(f"== {s} done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
