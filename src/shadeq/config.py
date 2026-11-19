"""Study settings: city, dates, sun times, data sources and model parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

HRDEM = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-lidar"
META_CHM = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
WORLDPOP_AGE = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/2025/{iso}/v1/100m/constrained"
WORLDPOP_POP = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/{iso}/v1/100m/constrained"
OVERTURE_RELEASE = "2026-11-18.0"


@dataclass(frozen=True)
class City:
    key: str
    place: str  # OSM name used to fetch the city boundary
    crs: str  # metric CRS for all processing
    utc_offset: float  # local clock offset on the study day (hours)
    lidar_dsm: str
    lidar_dtm: str
    lidar_year: int
    worldpop_iso: str
    gba_tile: str  # GlobalBuildingAtlas 5-degree tile name
    gba_region: str  # GBA continent folder
    # Study day: peak of the 2021 Pacific Northwest heat dome in Vancouver.
    day: tuple[int, int, int] = (2021, 6, 28)
    # Local clock times (hours) for shade, every 30 min from 10:00 to 17:00.
    hours: tuple[float, ...] = tuple(10 + 0.5 * i for i in range(15))
    neighbourhoods_url: str | None = None
    extra: dict = field(default_factory=dict)


VANCOUVER = City(
    key="vancouver",
    place="Vancouver, British Columbia, Canada",
    crs="EPSG:26910",
    utc_offset=-7.0,
    lidar_dsm=f"{HRDEM}/BC-Lower_Mainland_2016-1m-dsm.tif",
    lidar_dtm=f"{HRDEM}/BC-Lower_Mainland_2016-1m-dtm.tif",
    lidar_year=2016,
    worldpop_iso="CAN",
    gba_tile="w125_n50_w120_n45",
    gba_region="northamerica",
    neighbourhoods_url="https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/local-area-boundary/exports/geojson",
    extra={"s2_item": "S2B_MSIL2A_20180728T191909_R099_T10UDV_20201011T211249"},
)

CITIES = {c.key: c for c in [VANCOUVER]}

# Model parameters -----------------------------------------------------------
CELL = 1.0  # m, surface grid resolution
BUFFER = 600.0  # m, surfaces and network extend this far beyond the city
NETWORK_BUFFER = 1500.0  # m, network and destinations beyond the city (edge effects)
SAMPLE_STEP = 2.0  # m, spacing of shade sample points along walkways
SIDEWALK_SETBACK = 1.5  # m, sidewalk centre beyond the carriageway edge
TREE_TRANSMISSIVITY = 0.10  # share of direct sun passing through a crown (Gu et al. 2026)
TRUNK_RATIO = 0.25  # crown base as a share of tree height (UMEP default)
MIN_TREE = 2.5  # m, LiDAR vegetation threshold outside building footprints
MIN_TREE_META = 3.0  # m, Meta/WRI canopy threshold (the product's own canopy definition)
SPEED_OLDER = 0.9  # m/s, older pedestrian (65+)
SPEED_ADULT = 1.3  # m/s, general adult
PEAK_HOUR = 15.0  # local hour used for the single-time route analyses

# Half carriageway width (m) by OSM highway class; sidewalks sit beyond it.
HALF_WIDTH = {
    "motorway": 12.0, "trunk": 10.0, "primary": 9.0, "secondary": 8.0, "tertiary": 7.0,
    "unclassified": 5.0, "residential": 5.0, "living_street": 3.5, "service": 3.0,
    "motorway_link": 5.0, "trunk_link": 5.0, "primary_link": 5.0, "secondary_link": 5.0, "tertiary_link": 5.0,
}

# Lanes and driveways: no sidewalks, people walk on the centreline.
NO_SIDEWALK = {"service", "living_street"}

# Essential destinations (OSM tags) for older adults.
DESTINATIONS = {
    "grocery": {"shop": ["supermarket", "greengrocer", "grocery"]},
    "pharmacy": {"amenity": ["pharmacy"], "healthcare": ["pharmacy"]},
    "clinic": {"amenity": ["clinic", "doctors", "hospital"], "healthcare": ["clinic", "doctor", "hospital", "centre"]},
    "transit": {"highway": ["bus_stop"], "railway": ["station"], "public_transport": ["station"]},
    "park": {"leisure": ["park"]},
    "community": {"amenity": ["community_centre", "library", "social_centre"]},
}
