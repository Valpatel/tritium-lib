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

from .ao_packs import AOPack, active_ao_id, get_ao_pack, list_ao_packs
from .cache import GISCache
from .capture import capture_ao_pack
from .contours import auto_levels, contour_lines
from .fetchers import (
    USER_AGENT,
    USGS_HILLSHADE_TILE_URL,
    FemaFloodFetcher,
    NlcdLandCoverFetcher,
    NoaaAlertsFetcher,
    OverpassBuildingsFetcher,
    TigerRoadsFetcher,
    UsgsElevationFetcher,
    filter_features_bbox,
)
from .landcover import (
    NLCD_CLASSES,
    LandCoverClass,
    LandCoverGrid,
    classify_rgb,
    tactical_profile,
)
from .models import ElevationGrid, GeoBBox

__all__ = [
    "AOPack",
    "list_ao_packs",
    "get_ao_pack",
    "active_ao_id",
    "GeoBBox",
    "ElevationGrid",
    "GISCache",
    "UsgsElevationFetcher",
    "TigerRoadsFetcher",
    "FemaFloodFetcher",
    "NoaaAlertsFetcher",
    "OverpassBuildingsFetcher",
    "NlcdLandCoverFetcher",
    "LandCoverClass",
    "LandCoverGrid",
    "NLCD_CLASSES",
    "classify_rgb",
    "tactical_profile",
    "USGS_HILLSHADE_TILE_URL",
    "USER_AGENT",
    "auto_levels",
    "contour_lines",
    "filter_features_bbox",
    "capture_ao_pack",
]
