# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fetchers for real, public U.S. government GIS layers.

One class per source.  Every fetcher follows the same shape:

    * a pure ``parse_*`` staticmethod that turns a raw provider payload into the
      normalized contract (a GeoJSON ``FeatureCollection`` dict, or an
      :class:`ElevationGrid`) — no IO, fully unit-testable.
    * a ``fetch`` (``fetch_grid`` for elevation) method with a three-stage
      degradation chain so the demo AO *always* renders:

          live HTTP  --success-->  parse, cache.put, return
                     --failure-->  cache.get (no age limit)
                                      --miss-->  packaged demo fixture
                                                    --miss-->  empty result

Only the stdlib ``urllib.request`` is used — zero new hard dependencies.

Normalized vector convention (see README): every feature's ``properties`` carry
``source`` and ``kind`` plus a small set of layer-specific fields.  Style props
(``fill_color`` etc.) are added by the SC provider, never here.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from importlib import resources

from .cache import GISCache
from .models import ElevationGrid, GeoBBox

__all__ = [
    "USER_AGENT",
    "USGS_HILLSHADE_TILE_URL",
    "UsgsElevationFetcher",
    "TigerRoadsFetcher",
    "FemaFloodFetcher",
    "NoaaAlertsFetcher",
]

logger = logging.getLogger(__name__)

USER_AGENT = "Tritium/1.0 (+https://github.com/Valpatel/tritium)"

#: Public USGS shaded-relief basemap tile template (XYZ, TMS-style ``{z}/{y}/{x}``).
#: No fetcher needed — the SC provider hands this straight to the map client.
USGS_HILLSHADE_TILE_URL = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}"
)

_FIXTURE_PKG = "tritium_lib.geo.gis.fixtures"


def _empty_fc() -> dict:
    """An empty (but valid) GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": []}


def _http_json(url: str, *, data: bytes | None = None, timeout: float = 20.0) -> dict:
    """GET/POST ``url`` and decode a JSON body. Raises on any transport error."""
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed hosts)
        raw = resp.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _load_fixture(name: str):
    """Load a packaged fixture JSON by filename, or ``None`` if absent."""
    try:
        resource = resources.files(_FIXTURE_PKG).joinpath(name)
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError) as exc:
        logger.debug("GIS fixture %s unavailable: %s", name, exc)
        return None


def _as_bbox(bbox) -> GeoBBox:
    if isinstance(bbox, GeoBBox):
        return bbox
    if isinstance(bbox, str):
        return GeoBBox.from_string(bbox)
    w, s, e, n = bbox
    return GeoBBox(west=float(w), south=float(s), east=float(e), north=float(n))


class _VectorFetcher:
    """Shared degradation machinery for GeoJSON-emitting fetchers."""

    SOURCE = ""          # cache-key source tag
    FIXTURE_NAME = ""    # packaged demo fixture filename

    def __init__(self, cache: GISCache | None = None, timeout_s: float = 20.0):
        self.cache = cache
        self.timeout_s = timeout_s

    # -- subclass hooks -----------------------------------------------------
    def _build_url(self, bbox: GeoBBox) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _parse(raw: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def fetch(self, bbox) -> dict:
        """Return a normalized FeatureCollection via the degradation chain."""
        box = _as_bbox(bbox)
        key = self.cache.key(self.SOURCE, box) if self.cache else None

        # 1. Live.
        try:
            raw = _http_json(self._build_url(box), timeout=self.timeout_s)
            result = self._parse(raw)
            if self.cache and key is not None:
                self.cache.put(key, result)
            return result
        except Exception as exc:  # noqa: BLE001 - any failure => degrade
            logger.info("%s live fetch failed, degrading: %s", self.SOURCE, exc)

        # 2. Cache (no age limit).
        if self.cache and key is not None:
            cached = self.cache.get(key, max_age_s=None)
            if cached is not None:
                return cached

        # 3. Packaged fixture.
        fixture = _load_fixture(self.FIXTURE_NAME)
        if fixture is not None:
            return fixture

        # 4. Empty but valid.
        return _empty_fc()


class TigerRoadsFetcher(_VectorFetcher):
    """US Census TIGERweb local roads (Transportation layer 8)."""

    SOURCE = "tiger"
    FIXTURE_NAME = "tiger_roads_ao.json"
    URL = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "Transportation/MapServer/8/query"
    )

    def _build_url(self, bbox: GeoBBox) -> str:
        params = {
            "where": "1=1",
            "geometry": bbox.to_string(),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "BASENAME,NAME,MTFCC",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        return f"{self.URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def parse_roads(raw: dict) -> dict:
        """Normalize TIGER roads: ``kind`` = MTFCC code, plus ``name``."""
        features = []
        for feat in (raw or {}).get("features", []) or []:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties") or {}
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "source": "tiger",
                        "kind": props.get("MTFCC") or "",
                        "name": props.get("NAME") or "",
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    _parse = parse_roads


class FemaFloodFetcher(_VectorFetcher):
    """FEMA National Flood Hazard Layer flood zones (NFHL layer 28)."""

    SOURCE = "fema"
    FIXTURE_NAME = "fema_flood_ao.json"
    URL = (
        "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/"
        "MapServer/28/query"
    )

    def _build_url(self, bbox: GeoBBox) -> str:
        params = {
            "where": "1=1",
            "geometry": bbox.to_string(),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        return f"{self.URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def parse_flood(raw: dict) -> dict:
        """Normalize flood zones: ``kind`` = FLD_ZONE, plus ``subtype``/``sfha``."""
        features = []
        for feat in (raw or {}).get("features", []) or []:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties") or {}
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "source": "fema",
                        "kind": props.get("FLD_ZONE") or "",
                        "subtype": props.get("ZONE_SUBTY") or "",
                        "sfha": str(props.get("SFHA_TF") or "").upper() == "T",
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    _parse = parse_flood


class NoaaAlertsFetcher(_VectorFetcher):
    """NOAA / NWS active weather alerts for the bbox centre point."""

    SOURCE = "noaa"
    FIXTURE_NAME = "noaa_alerts_ao.json"
    URL = "https://api.weather.gov/alerts/active"

    def _build_url(self, bbox: GeoBBox) -> str:
        lon, lat = bbox.center()
        params = {"point": f"{lat},{lon}"}
        return f"{self.URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def parse_alerts(raw: dict) -> dict:
        """Normalize NWS alerts: ``kind`` = event name; plus severity/headline/expires.

        Features WITHOUT geometry are DROPPED — the normalized layer must be
        renderable.  An empty input yields an empty (but valid) collection.
        """
        features = []
        for feat in (raw or {}).get("features", []) or []:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties") or {}
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "source": "noaa",
                        "kind": props.get("event") or "",
                        "severity": props.get("severity") or "",
                        "headline": props.get("headline") or "",
                        "expires": props.get("expires") or "",
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    _parse = parse_alerts


class UsgsElevationFetcher:
    """USGS 3DEP elevation, sampled as a regular grid via ImageServer getSamples."""

    SOURCE = "usgs"
    FIXTURE_NAME = "usgs_dem_ao.json"
    URL = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
        "ImageServer/getSamples"
    )

    def __init__(self, cache: GISCache | None = None, timeout_s: float = 20.0):
        self.cache = cache
        self.timeout_s = timeout_s

    @staticmethod
    def build_points(bbox: GeoBBox, ncols: int, nrows: int) -> list:
        """Row-major sample points, **row 0 = north**, west→east within a row."""
        points = []
        for iy in range(nrows):
            lat = bbox.north if nrows == 1 else (
                bbox.north - (bbox.north - bbox.south) * iy / (nrows - 1)
            )
            for ix in range(ncols):
                lon = bbox.west if ncols == 1 else (
                    bbox.west + (bbox.east - bbox.west) * ix / (ncols - 1)
                )
                points.append([lon, lat])
        return points

    def _build_request(self, bbox: GeoBBox, ncols: int, nrows: int) -> bytes:
        geometry = {
            "points": self.build_points(bbox, ncols, nrows),
            "spatialReference": {"wkid": 4326},
        }
        params = {
            "geometry": json.dumps(geometry),
            "geometryType": "esriGeometryMultipoint",
            "returnFirstValueOnly": "true",
            "f": "json",
        }
        return urllib.parse.urlencode(params).encode("utf-8")

    @staticmethod
    def parse_grid(
        raw: dict, bbox: GeoBBox, ncols: int, nrows: int, source: str = "usgs"
    ) -> ElevationGrid:
        """Turn a getSamples payload into an :class:`ElevationGrid` (row 0 = north).

        A missing value or the string ``"NoData"`` becomes ``None``.  Samples
        are placed by ``locationId`` (falling back to encounter order) so the
        grid stays correct even if the server reorders them.
        """
        values: list = [None] * (ncols * nrows)
        resolution = None
        samples = (raw or {}).get("samples", []) or []
        for order, sample in enumerate(samples):
            lid = sample.get("locationId")
            if lid is None:
                lid = order
            if not (0 <= lid < len(values)):
                continue
            raw_val = sample.get("value")
            if raw_val in (None, "", "NoData", "null"):
                fval = None
            else:
                try:
                    fval = float(raw_val)
                except (TypeError, ValueError):
                    fval = None
            values[lid] = fval
            if resolution is None:
                resolution = sample.get("resolution")
        return ElevationGrid(
            west=bbox.west,
            south=bbox.south,
            east=bbox.east,
            north=bbox.north,
            ncols=ncols,
            nrows=nrows,
            values=values,
            source=source,
            resolution_m=resolution,
        )

    def fetch_grid(self, bbox, ncols: int = 16, nrows: int = 16) -> ElevationGrid:
        """Return an :class:`ElevationGrid` via the degradation chain."""
        box = _as_bbox(bbox)
        key = (
            self.cache.key(self.SOURCE, box, ncols=ncols, nrows=nrows)
            if self.cache
            else None
        )

        # 1. Live.
        try:
            data = self._build_request(box, ncols, nrows)
            raw = _http_json(self.URL, data=data, timeout=self.timeout_s)
            grid = self.parse_grid(raw, box, ncols, nrows)
            if self.cache and key is not None:
                self.cache.put(key, grid.to_dict())
            return grid
        except Exception as exc:  # noqa: BLE001 - any failure => degrade
            logger.info("usgs live fetch failed, degrading: %s", exc)

        # 2. Cache.
        if self.cache and key is not None:
            cached = self.cache.get(key, max_age_s=None)
            if cached is not None:
                return ElevationGrid.from_dict(cached)

        # 3. Fixture.
        fixture = _load_fixture(self.FIXTURE_NAME)
        if fixture is not None:
            return ElevationGrid.from_dict(fixture)

        # 4. Empty grid (all NoData over the requested window).
        return ElevationGrid(
            west=box.west,
            south=box.south,
            east=box.east,
            north=box.north,
            ncols=ncols,
            nrows=nrows,
            values=[None] * (ncols * nrows),
            source="usgs-empty",
            resolution_m=None,
        )
