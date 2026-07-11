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
import math
import urllib.parse
import urllib.request
from importlib import resources

from .cache import GISCache
from .landcover import LandCoverGrid, classify_rgb
from .models import ElevationGrid, GeoBBox

__all__ = [
    "USER_AGENT",
    "USGS_HILLSHADE_TILE_URL",
    "UsgsElevationFetcher",
    "TigerRoadsFetcher",
    "FemaFloodFetcher",
    "NoaaAlertsFetcher",
    "NhdHydrographyFetcher",
    "OverpassBuildingsFetcher",
    "NlcdLandCoverFetcher",
    "filter_features_bbox",
]

#: Web-Mercator (EPSG:3857) sphere radius — WGS-84 semi-major axis.
_WEB_MERCATOR_R = 6378137.0
#: Web-Mercator latitude clamp (the projection is undefined at the poles).
_WEB_MERCATOR_LAT_LIMIT = 85.05112878


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple:
    """Project WGS-84 ``(lon, lat)`` degrees to EPSG:3857 ``(x, y)`` metres.

    Latitude is clamped to +/-85.0511 deg (the standard Web-Mercator limit) so a
    caller cannot blow up the ``log(tan(...))`` at the poles.
    """
    lat = max(min(lat, _WEB_MERCATOR_LAT_LIMIT), -_WEB_MERCATOR_LAT_LIMIT)
    x = _WEB_MERCATOR_R * math.radians(lon)
    y = _WEB_MERCATOR_R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return (x, y)

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


def _http_bytes(url: str, *, timeout: float = 20.0) -> bytes:
    """GET ``url`` and return the raw response body. Raises on any transport error.

    Used for binary payloads (the NLCD WMS GetMap PNG) — the JSON helper cannot
    decode an image.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed hosts)
        return resp.read()


def _load_fixture(name: str):
    """Load a packaged fixture JSON by filename, or ``None`` if absent."""
    try:
        resource = resources.files(_FIXTURE_PKG).joinpath(name)
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError) as exc:
        logger.debug("GIS fixture %s unavailable: %s", name, exc)
        return None


def _resolve_fixture_names(fixture_names, fixture_name: str) -> tuple:
    """Ordered fixture filenames for a fetcher (multi-AO, back-compatible).

    A fetcher declares one or more packaged Area-of-Operations fixture packs in
    the class attribute ``FIXTURE_NAMES`` (a tuple, tried in order).  For
    backward compatibility a fetcher that only sets the legacy single
    ``FIXTURE_NAME`` still works: resolution order is ``FIXTURE_NAMES`` when it
    is non-empty, else ``(FIXTURE_NAME,)``.  Empty strings are dropped so a
    fetcher with neither configured degrades cleanly to no fixtures.
    """
    names = tuple(fixture_names) if fixture_names else (fixture_name,)
    return tuple(n for n in names if n)


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

    SOURCE = ""               # cache-key source tag
    FIXTURE_NAME = ""         # legacy single packaged fixture filename
    FIXTURE_NAMES: tuple = () # multi-AO packaged fixtures, tried in order

    def __init__(self, cache: GISCache | None = None, timeout_s: float = 20.0):
        self.cache = cache
        self.timeout_s = timeout_s

    def _fixture_names(self) -> tuple:
        """Ordered packaged fixtures for this fetcher (see :func:`_resolve_fixture_names`)."""
        return _resolve_fixture_names(self.FIXTURE_NAMES, self.FIXTURE_NAME)

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

        # 3. Packaged fixtures — clipped to the requested bbox (a pack covers a
        #    whole AO, so a distant window must not get the lot).  Multi-AO:
        #    try each packaged pack in order and return the FIRST whose clipped
        #    features are non-empty.  If a fixture loaded but every pack clips to
        #    empty (the bbox falls outside every packaged AO), return that empty
        #    result — still ``fixture``-marked — preserving the pre-multi-AO
        #    contract that an offline far-away fetch yields a fixture-marked
        #    (empty) collection.
        last_empty = None
        for fixture_name in self._fixture_names():
            fixture = _load_fixture(fixture_name)
            if fixture is None:
                continue
            clipped = filter_features_bbox(fixture, box)
            if clipped.get("features"):
                return clipped
            last_empty = clipped
        if last_empty is not None:
            return last_empty

        # 4. Empty but valid (no packaged fixtures at all).
        return _empty_fc()


class TigerRoadsFetcher(_VectorFetcher):
    """US Census TIGERweb local roads (Transportation layer 8)."""

    SOURCE = "tiger"
    FIXTURE_NAME = "tiger_roads_ao.json"
    FIXTURE_NAMES = ("tiger_roads_ao.json", "tiger_roads_boulder.json")
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
    FIXTURE_NAMES = ("fema_flood_ao.json", "fema_flood_boulder.json")
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
    # Only the Dublin pack: the Boulder AO had no active NWS alerts at capture
    # time (a legitimately-empty layer), so no boulder fixture was written.
    FIXTURE_NAMES = ("noaa_alerts_ao.json",)
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


#: NHDFlowline FType (integer) -> normalized ``kind`` string.  The residual
#: (any code not listed, incl. StreamRiver 460) is a plain ``"stream"``; the
#: named kinds let the SC style weight rivers/canals/artificial paths apart.
_NHD_FLOWLINE_FTYPE_KIND: dict[int, str] = {
    460: "river",       # StreamRiver — a named/perennial watercourse
    336: "canal",       # CanalDitch
    558: "artificial",  # ArtificialPath (through a waterbody)
    334: "connector",   # Connector
    420: "conduit",     # UndergroundConduit / pipeline
    566: "coastline",   # Coastline
}


def _ci_get(props: dict, *names, default=None):
    """Case-insensitive lookup of the first present key in ``names``.

    ArcGIS field casing varies across service revisions (``GNIS_NAME`` vs
    ``GNIS_Name``, ``FTYPE`` vs ``FType``); this reads them without pinning a
    single casing.  Returns ``default`` when none of ``names`` is present.
    """
    if not isinstance(props, dict):
        return default
    lower = {str(k).lower(): v for k, v in props.items()}
    for name in names:
        val = lower.get(str(name).lower())
        if val is not None:
            return val
    return default


def _to_int(value, default=None):
    """Best-effort int coercion (NHD FType arrives as int, str, or float)."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class NhdHydrographyFetcher(_VectorFetcher):
    """USGS National Hydrography Dataset — the real surface-water NETWORK.

    Complements NLCD's water *class* and FEMA's flood *zones* with the actual
    hydrography: stream/river centerlines (NHDFlowline, LineStrings) plus lake /
    pond / reservoir footprints (NHDWaterbody, Polygons), from the USGS
    ``NHDPlus_HR`` ArcGIS MapServer.

    Unlike the single-layer vector fetchers, the live step queries **two** layers
    (flowline + waterbody) and merges them into one normalized
    ``FeatureCollection`` — so the layer toggles as a single "hydrography" layer.
    Everything else (disk cache, multi-AO packaged fixtures with bbox clipping,
    empty fallback) reuses the shared ``_VectorFetcher`` machinery.

    Normalized ``properties`` (style props are added by the SC provider, never
    here):
        * ``source`` = ``"nhd"``
        * ``kind``   = ``"river"``/``"canal"``/``"artificial"``/``"connector"``/
          ``"conduit"``/``"coastline"``/``"stream"`` (flowlines, by FType) or
          ``"waterbody"`` (polygons)
        * ``name``   = GNIS name (may be empty)
        * ``ftype``  = raw NHD FType integer (or ``None``)
        * ``area_sqkm`` = waterbody area (waterbodies only)
    """

    SOURCE = "nhd"
    FIXTURE_NAME = "nhd_hydro_ao.json"
    FIXTURE_NAMES = ("nhd_hydro_ao.json", "nhd_hydro_boulder.json")
    BASE_URL = (
        "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"
    )
    FLOWLINE_LAYER = 3   # NetworkNHDFlowline (polyline)
    WATERBODY_LAYER = 9  # NHDWaterbody (polygon)

    #: Cap live records per layer so a dense metro AO cannot pull thousands of
    #: features (politeness + a renderable overlay). Fixtures are pre-trimmed.
    MAX_RECORDS = 800

    def _layer_url(self, bbox: GeoBBox, layer_id: int, out_fields: str) -> str:
        params = {
            "where": "1=1",
            "geometry": bbox.to_string(),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": str(self.MAX_RECORDS),
            "f": "geojson",
        }
        return f"{self.BASE_URL}/{layer_id}/query?{urllib.parse.urlencode(params)}"

    # _VectorFetcher declares _build_url abstract; NHD queries two layers so it
    # overrides fetch() directly. Keep a concrete stub so the class is not
    # accidentally used through the single-URL base path.
    def _build_url(self, bbox: GeoBBox) -> str:  # pragma: no cover - unused
        return self._layer_url(bbox, self.FLOWLINE_LAYER, "GNIS_NAME,FType,FCode")

    @staticmethod
    def parse_flowlines(raw: dict) -> list:
        """Normalize NHDFlowline features (LineStrings). Geometryless dropped."""
        features = []
        for feat in (raw or {}).get("features", []) or []:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties") or {}
            ftype = _to_int(_ci_get(props, "FType", "FTYPE"))
            kind = _NHD_FLOWLINE_FTYPE_KIND.get(ftype or -1, "stream")
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "source": "nhd",
                        "kind": kind,
                        "name": _ci_get(props, "GNIS_NAME", "GNIS_Name", default="") or "",
                        "ftype": ftype,
                    },
                }
            )
        return features

    @staticmethod
    def parse_waterbodies(raw: dict) -> list:
        """Normalize NHDWaterbody features (Polygons). Geometryless dropped."""
        features = []
        for feat in (raw or {}).get("features", []) or []:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties") or {}
            ftype = _to_int(_ci_get(props, "FType", "FTYPE"))
            area = _ci_get(props, "AreaSqKm", "AREASQKM")
            try:
                area = round(float(area), 4) if area is not None else None
            except (TypeError, ValueError):
                area = None
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "source": "nhd",
                        "kind": "waterbody",
                        "name": _ci_get(props, "GNIS_NAME", "GNIS_Name", default="") or "",
                        "ftype": ftype,
                        "area_sqkm": area,
                    },
                }
            )
        return features

    def _fetch_live(self, box: GeoBBox) -> dict:
        """Query both NHD layers and merge into one normalized FeatureCollection.

        Raises on any transport error so ``fetch`` degrades to cache/fixture.
        """
        flow_raw = _http_json(
            self._layer_url(box, self.FLOWLINE_LAYER, "GNIS_NAME,FType,FCode"),
            timeout=self.timeout_s,
        )
        wb_raw = _http_json(
            self._layer_url(box, self.WATERBODY_LAYER, "GNIS_NAME,FType,AreaSqKm"),
            timeout=self.timeout_s,
        )
        features = self.parse_flowlines(flow_raw) + self.parse_waterbodies(wb_raw)
        return {"type": "FeatureCollection", "features": features}

    def fetch(self, bbox) -> dict:
        """Return merged NHD hydrography via the standard degradation chain.

        Mirrors ``_VectorFetcher.fetch`` (live -> cache -> packaged fixture ->
        empty) but the live step merges two NHD layers (see :meth:`_fetch_live`).
        """
        box = _as_bbox(bbox)
        key = self.cache.key(self.SOURCE, box) if self.cache else None

        # 1. Live (two-layer merge).
        try:
            result = self._fetch_live(box)
            if self.cache and key is not None:
                self.cache.put(key, result)
            return result
        except Exception as exc:  # noqa: BLE001 - any failure => degrade
            logger.info("%s live fetch failed, degrading: %s", self.SOURCE, exc)

        # 2. Cache (no age limit; bbox-rounded key).
        if self.cache and key is not None:
            cached = self.cache.get(key, max_age_s=None)
            if cached is not None:
                return cached

        # 3. Packaged fixtures — clipped to the requested bbox, first non-empty.
        last_empty = None
        for fixture_name in self._fixture_names():
            fixture = _load_fixture(fixture_name)
            if fixture is None:
                continue
            clipped = filter_features_bbox(fixture, box)
            if clipped.get("features"):
                return clipped
            last_empty = clipped
        if last_empty is not None:
            return last_empty

        # 4. Empty but valid.
        return _empty_fc()


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
    FIXTURE_NAMES = ("osm_buildings_ao.json", "osm_buildings_boulder.json")
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
    FIXTURE_NAME = "usgs_dem_ao.json"      # legacy single packaged fixture
    FIXTURE_NAMES = ("usgs_dem_ao.json", "usgs_dem_boulder.json")  # multi-AO, in order
    URL = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
        "ImageServer/getSamples"
    )

    def __init__(self, cache: GISCache | None = None, timeout_s: float = 20.0):
        self.cache = cache
        self.timeout_s = timeout_s

    def _fixture_names(self) -> tuple:
        """Ordered packaged DEM fixtures (see :func:`_resolve_fixture_names`)."""
        return _resolve_fixture_names(self.FIXTURE_NAMES, self.FIXTURE_NAME)

    @staticmethod
    def _fixture_bbox(fixture: dict):
        """AO bounding box ``(w, s, e, n)`` of a packaged DEM fixture, or ``None``.

        Prefers a cheap top-level ``"bbox": [w, s, e, n]`` marker (written by the
        AO capture tool); falls back to the grid's own
        ``west``/``south``/``east``/``north`` fields when ``"bbox"`` is absent
        (the original Dublin DEM fixture predates the marker).
        """
        marker = fixture.get("bbox")
        if isinstance(marker, (list, tuple)) and len(marker) == 4:
            try:
                return (float(marker[0]), float(marker[1]),
                        float(marker[2]), float(marker[3]))
            except (TypeError, ValueError):
                pass
        try:
            return (
                float(fixture["west"]), float(fixture["south"]),
                float(fixture["east"]), float(fixture["north"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

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
        """Return an :class:`ElevationGrid` via the degradation chain.

        ⚠️ **Fixture gating (contract change).** The packaged-DEM step now returns
        a fixture only when its AO bounding box *intersects* the requested bbox.
        Previously the single Dublin DEM was returned for **any** offline bbox;
        an offline request for a window outside *every* packaged AO now falls
        through to the empty all-NoData grid (``source="usgs-empty"``) instead of
        silently getting Dublin terrain.  Consumers (the costmap lane) must treat
        an all-NoData grid as "no terrain data here", not as flat ground.
        """
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

        # 3. Fixtures.  Multi-AO: return the first packaged DEM whose AO bbox
        #    INTERSECTS the requested bbox.  A request outside every packaged AO
        #    falls through to the empty grid below (see the method docstring for
        #    the contract change this represents).
        target = (box.west, box.south, box.east, box.north)
        for fixture_name in self._fixture_names():
            fixture = _load_fixture(fixture_name)
            if fixture is None:
                continue
            fbox = self._fixture_bbox(fixture)
            if fbox is not None and _bbox_intersects(fbox, target):
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


class NlcdLandCoverFetcher:
    """USGS / MRLC NLCD 2021 land cover, sampled as a classified grid via WMS.

    Mirrors :class:`UsgsElevationFetcher` shape-for-shape (same 4-stage
    degradation chain, same fixture-intersect gating, same cache usage) but the
    live payload is an image, not JSON:

        1. **Live** — a WMS ``GetMap`` PNG at ``ncols x nrows`` (EPSG:3857),
           decoded with Pillow and classified nearest-colour per cell.  The
           Pillow import is deferred to call time and wrapped: on
           ``ImportError`` (Pillow is *not* a lib dependency) the live step is
           skipped and the chain falls through to cache/fixture — the pure-lib
           install still serves the packaged AO offline.
        2. **Cache** — last good grid dict (no age limit).
        3. **Packaged fixture** whose AO bbox *intersects* the requested bbox.
        4. **Empty** all-NoData grid (``source="nlcd-empty"``) — a request
           outside every packaged AO offline.  Consumers must treat this as
           "no land-cover data here", not as open ground.
    """

    SOURCE = "nlcd"
    FIXTURE_NAME = "nlcd_ao.json"                       # legacy single fixture
    FIXTURE_NAMES = ("nlcd_ao.json", "nlcd_boulder.json")  # multi-AO, in order
    WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
    LAYER = "NLCD_2021_Land_Cover_L48"

    def __init__(self, cache: GISCache | None = None, timeout_s: float = 20.0):
        self.cache = cache
        self.timeout_s = timeout_s

    def _fixture_names(self) -> tuple:
        """Ordered packaged NLCD fixtures (see :func:`_resolve_fixture_names`)."""
        return _resolve_fixture_names(self.FIXTURE_NAMES, self.FIXTURE_NAME)

    @staticmethod
    def _fixture_bbox(fixture: dict):
        """AO bbox ``(w, s, e, n)`` of a packaged NLCD fixture, or ``None``.

        Same marker convention as the DEM fixtures — delegates to
        :meth:`UsgsElevationFetcher._fixture_bbox` so the ``"bbox"`` /
        ``west/south/east/north`` fallback logic lives in exactly one place.
        """
        return UsgsElevationFetcher._fixture_bbox(fixture)

    def _build_url(self, bbox: GeoBBox, ncols: int, nrows: int) -> str:
        """Build the MRLC WMS GetMap URL (v1.1.1, EPSG:3857, x/y mercator bbox)."""
        minx, miny = lonlat_to_web_mercator(bbox.west, bbox.south)
        maxx, maxy = lonlat_to_web_mercator(bbox.east, bbox.north)
        params = {
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": self.LAYER,
            "srs": "EPSG:3857",
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "width": str(int(ncols)),
            "height": str(int(nrows)),
            "format": "image/png",
        }
        return f"{self.WMS_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def parse_png_grid(
        png_bytes: bytes, bbox: GeoBBox, ncols: int, nrows: int, source: str = "nlcd"
    ) -> LandCoverGrid:
        """Decode a WMS PNG into a :class:`LandCoverGrid` (row 0 = north).

        Resamples the image to ``ncols x nrows`` with **nearest-neighbour** (so a
        class colour is never blended into a spurious intermediate) and runs
        :func:`~tritium_lib.geo.gis.landcover.classify_rgb` on each cell.  A
        fully-transparent pixel (the WMS's NoData/background) becomes ``None``.

        Requires Pillow — the caller (``fetch_grid``) guards the import; this
        raises ``ImportError`` if Pillow is absent so the guard can degrade.
        """
        import io

        from PIL import Image  # noqa: F401 - optional dep, guarded by caller

        with Image.open(io.BytesIO(png_bytes)) as im:
            rgba = im.convert("RGBA")
            if rgba.size != (ncols, nrows):
                rgba = rgba.resize((ncols, nrows), Image.NEAREST)
            pixels = rgba.load()
            codes: list = []
            for iy in range(nrows):
                for ix in range(ncols):
                    r, g, b, a = pixels[ix, iy]
                    codes.append(None if a == 0 else classify_rgb(r, g, b))
        return LandCoverGrid(
            west=bbox.west,
            south=bbox.south,
            east=bbox.east,
            north=bbox.north,
            ncols=ncols,
            nrows=nrows,
            codes=codes,
            source=source,
        )

    def fetch_grid(self, bbox, ncols: int = 32, nrows: int = 32) -> LandCoverGrid:
        """Return a :class:`LandCoverGrid` via the 4-stage degradation chain.

        See the class docstring.  The live step is skipped cleanly when Pillow is
        unavailable (``ImportError`` is caught by the broad guard), so a pure-lib
        install degrades straight to cache/fixture.
        """
        box = _as_bbox(bbox)
        key = (
            self.cache.key(self.SOURCE, box, ncols=ncols, nrows=nrows)
            if self.cache
            else None
        )

        # 1. Live (needs Pillow — import is deferred + guarded).
        try:
            png = _http_bytes(self._build_url(box, ncols, nrows), timeout=self.timeout_s)
            grid = self.parse_png_grid(png, box, ncols, nrows)
            if self.cache and key is not None:
                self.cache.put(key, grid.to_dict())
            return grid
        except Exception as exc:  # noqa: BLE001 - ImportError / network => degrade
            logger.info("nlcd live fetch failed, degrading: %s", exc)

        # 2. Cache.
        if self.cache and key is not None:
            cached = self.cache.get(key, max_age_s=None)
            if cached is not None:
                return LandCoverGrid.from_dict(cached)

        # 3. Fixtures — first packaged pack whose AO bbox INTERSECTS the request.
        target = (box.west, box.south, box.east, box.north)
        for fixture_name in self._fixture_names():
            fixture = _load_fixture(fixture_name)
            if fixture is None:
                continue
            fbox = self._fixture_bbox(fixture)
            if fbox is not None and _bbox_intersects(fbox, target):
                return LandCoverGrid.from_dict(fixture)

        # 4. Empty grid (all NoData over the requested window).
        return LandCoverGrid(
            west=box.west,
            south=box.south,
            east=box.east,
            north=box.north,
            ncols=ncols,
            nrows=nrows,
            codes=[None] * (ncols * nrows),
            source="nlcd-empty",
        )
