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
    "OverpassBuildingsFetcher",
    "filter_features_bbox",
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


def _iter_positions(coords):
    """Yield ``(lon, lat)`` for every position in a GeoJSON coordinate tree."""
    if isinstance(coords, (list, tuple)):
        if (
            len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
        ):
            yield (float(coords[0]), float(coords[1]))
        else:
            for child in coords:
                yield from _iter_positions(child)


def _geometry_bbox(geometry):
    """Bounding box ``(w, s, e, n)`` of any GeoJSON geometry, or ``None``.

    Handles every geometry type (Point through MultiPolygon) plus
    ``GeometryCollection``.  A geometry with no usable coordinate positions
    returns ``None`` so the caller can drop it.
    """
    if not isinstance(geometry, dict):
        return None
    lons: list = []
    lats: list = []
    coords = geometry.get("coordinates")
    if coords is not None:
        for lon, lat in _iter_positions(coords):
            lons.append(lon)
            lats.append(lat)
    else:
        for sub in geometry.get("geometries") or []:
            box = _geometry_bbox(sub)
            if box is not None:
                lons.extend((box[0], box[2]))
                lats.extend((box[1], box[3]))
    if not lons:
        return None
    return (min(lons), min(lats), max(lons), max(lats))


def _bbox_intersects(a, b) -> bool:
    """True when the two ``(w, s, e, n)`` boxes overlap (edge-touch counts)."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def filter_features_bbox(fc: dict, bbox) -> dict:
    """Keep only features whose geometry bounding box intersects ``bbox``.

    A pure helper for the *packaged-fixture* branch of the degradation chain:
    packaged fixtures cover the whole demo AO, so a query for a distant window
    would otherwise get the entire AO back.  Live results are already
    bbox-scoped by the upstream service and cache keys are bbox-rounded, so this
    is applied *only* to fixtures.

    Features whose geometry yields no coordinates are dropped.  Any top-level
    keys on ``fc`` other than ``features`` (notably the ``"fixture": true``
    marker) are preserved so downstream fixture-detection still works.
    """
    box = _as_bbox(bbox)
    target = (box.west, box.south, box.east, box.north)
    kept = []
    for feat in (fc or {}).get("features", []) or []:
        gbox = _geometry_bbox((feat or {}).get("geometry"))
        if gbox is None:
            continue
        if _bbox_intersects(gbox, target):
            kept.append(feat)
    result = {k: v for k, v in (fc or {}).items() if k != "features"}
    result["type"] = "FeatureCollection"
    result["features"] = kept
    return result


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

    def _build_body(self, bbox: GeoBBox) -> bytes | None:
        """POST body for the live request, or ``None`` for a plain GET.

        Default is ``None`` (GET) — the ArcGIS/NWS fetchers put everything in the
        query string.  ``OverpassBuildingsFetcher`` overrides this to POST an
        Overpass QL body.
        """
        return None

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
            raw = _http_json(
                self._build_url(box), data=self._build_body(box), timeout=self.timeout_s
            )
            result = self._parse(raw)
            if self.cache and key is not None:
                self.cache.put(key, result)
            return result
        except Exception as exc:  # noqa: BLE001 - any failure => degrade
            logger.info("%s live fetch failed, degrading: %s", self.SOURCE, exc)

        # 2. Cache (no age limit).  Cache keys are bbox-rounded, so cached
        #    entries are already scoped to the requested window.
        if self.cache and key is not None:
            cached = self.cache.get(key, max_age_s=None)
            if cached is not None:
                return cached

        # 3. Packaged fixture — clipped to the requested bbox (fixtures cover
        #    the whole demo AO, so a distant window must not get the lot).
        fixture = _load_fixture(self.FIXTURE_NAME)
        if fixture is not None:
            return filter_features_bbox(fixture, box)

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


def _building_height(tags: dict) -> float:
    """Best-effort building height in metres from OSM tags.

    Order: an explicit ``height`` tag (``"12 m"`` / ``"12m"`` / ``"12"``), else
    ``building:levels`` at 3 m per storey plus 1 m, else a plain ``8.0`` default.
    """
    raw_h = tags.get("height")
    if raw_h not in (None, ""):
        try:
            return float(str(raw_h).lower().replace("m", "").strip())
        except (TypeError, ValueError):
            pass
    raw_l = tags.get("building:levels")
    if raw_l not in (None, ""):
        try:
            return float(str(raw_l).split(";")[0].strip()) * 3.0 + 1.0
        except (TypeError, ValueError):
            pass
    return 8.0


class OverpassBuildingsFetcher(_VectorFetcher):
    """OpenStreetMap building footprints via the Overpass API (``out geom``)."""

    SOURCE = "osm"
    FIXTURE_NAME = "osm_buildings_ao.json"
    URL = "https://overpass-api.de/api/interpreter"

    def _build_url(self, bbox: GeoBBox) -> str:
        return self.URL

    def _build_body(self, bbox: GeoBBox) -> bytes:
        # Overpass bbox order is (south, west, north, east) — NOT w,s,e,n.
        query = (
            "[out:json][timeout:30];"
            f'way["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
            "out geom;"
        )
        return urllib.parse.urlencode({"data": query}).encode("utf-8")

    @staticmethod
    def parse_buildings(raw: dict) -> dict:
        """Normalize Overpass ways into closed-ring ``Polygon`` features.

        Ways with fewer than three geometry points are dropped.  ``kind`` is the
        ``building`` tag value (``"yes"`` when untyped); ``height_m`` follows
        :func:`_building_height`; ``levels`` is ``max(1, int(height / 3))``.
        """
        features = []
        for el in (raw or {}).get("elements", []) or []:
            if el.get("type") != "way":
                continue
            geom = el.get("geometry") or []
            ring = [
                [pt["lon"], pt["lat"]]
                for pt in geom
                if isinstance(pt, dict) and "lon" in pt and "lat" in pt
            ]
            if len(ring) < 3:
                continue
            if ring[0] != ring[-1]:
                ring.append(list(ring[0]))
            tags = el.get("tags") or {}
            height = _building_height(tags)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "source": "osm",
                        "kind": tags.get("building") or "yes",
                        "name": tags.get("name") or "",
                        "height_m": height,
                        "levels": max(1, int(height / 3)),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    _parse = parse_buildings


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
