"""Solar position (NOAA general solar position equations, accurate to ~0.1 degree)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


def solar_position(when_utc: datetime, lat: float, lon: float) -> tuple[float, float]:
    """Return (azimuth, elevation) in degrees. Azimuth is clockwise from true north."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    when_utc = when_utc.astimezone(timezone.utc)
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    days = 366 if (when_utc.year % 4 == 0 and (when_utc.year % 100 != 0 or when_utc.year % 400 == 0)) else 365
    g = 2 * math.pi / days * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    tst = hour * 60 + eqtime + 4 * lon  # true solar time (minutes)
    ha = math.radians(tst / 4 - 180)
    phi = math.radians(lat)
    cosz = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    cosz = max(-1.0, min(1.0, cosz))
    zen = math.acos(cosz)
    elev = 90 - math.degrees(zen)
    # Azimuth from north, clockwise.
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi))) + 180
    return az % 360, elev


def sun_series(day: tuple[int, int, int], hours, utc_offset: float, lat: float, lon: float):
    """Sun positions for local clock hours on one day: list of (hour, azimuth, elevation)."""
    out = []
    base = datetime(*day, tzinfo=timezone.utc)
    for h in hours:
        t = base + timedelta(hours=h - utc_offset)
        az, el = solar_position(t, lat, lon)
        out.append((float(h), az, el))
    return out
