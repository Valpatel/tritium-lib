# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Real, public U.S. government GIS layers for the tactical map.

This package turns four free government data sources into the two normalized
shapes the rest of Tritium consumes — a GeoJSON ``FeatureCollection`` dict for
vector layers, and an :class:`ElevationGrid` raster for terrain — plus a disk
cache and packaged demo-AO fixtures so everything renders fully offline.

Sources:
    * USGS 3DEP elevation  (``UsgsElevationFetcher`` -> ``ElevationGrid``)
    * US Census TIGERweb roads  (``TigerRoadsFetcher``)
    * FEMA National Flood Hazard Layer  (``FemaFloodFetcher``)
    * NOAA / NWS active weather alerts  (``NoaaAlertsFetcher``)

Stdlib only (``urllib.request``) — no new hard dependencies.  See ``README.md``
for the raster (row 0 = north) and vector (``source``/``kind`` properties)
conventions the costmap lane and SC frontend depend on.
"""

from __future__ import annotations

from .cache import GISCache
from .fetchers import (
    USER_AGENT,
    USGS_HILLSHADE_TILE_URL,
    FemaFloodFetcher,
    NoaaAlertsFetcher,
    TigerRoadsFetcher,
    UsgsElevationFetcher,
)
from .models import ElevationGrid, GeoBBox

__all__ = [
    "GeoBBox",
    "ElevationGrid",
    "GISCache",
    "UsgsElevationFetcher",
    "TigerRoadsFetcher",
    "FemaFloodFetcher",
    "NoaaAlertsFetcher",
    "USGS_HILLSHADE_TILE_URL",
    "USER_AGENT",
]
