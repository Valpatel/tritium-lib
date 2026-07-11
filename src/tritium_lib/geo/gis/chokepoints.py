# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Water-crossing chokepoints — where roads cross the hydrography network.

Pure, deterministic 2-D geometry: intersect TIGER road centrelines (LineStrings)
with the NHD hydrography network (flowline LineStrings + waterbody Polygon
edges) to find **bridges** — the tactical chokepoints a defender holds, an
attacker is funnelled through, and a blown span severs.

No IO, no network, no third-party deps (``math`` + stdlib only) — feed it the
GeoJSON FeatureCollections the GIS layer already fetches (roads + hydro) and it
returns a FeatureCollection of chokepoint ``Point`` features. Reusable by the
costmap / riot lanes as tactical objects (a bridge = hold-point / severable
link); see :func:`chokepoint_tactical_object` for the tactical projection.

Normalized chokepoint ``properties``:
    * ``source``     = ``"chokepoint"``
    * ``kind``       = ``"bridge"`` | ``"ford"`` | ``"culvert"`` (inferred from
      the crossing road class + water class — see :func:`infer_crossing_kind`)
    * ``road_name``  / ``road_kind`` (MTFCC) — the crossing road
    * ``water_name`` / ``water_kind`` (NHD kind) — the crossed water feature
    * ``name``       = human label, e.g. ``"9th St @ Boulder Creek"``
    * ``id``         = stable deterministic id (``chk_<hash>``)

Coordinates are WGS-84 lon/lat. Whether two lines cross is a topological
property preserved under the lon/lat parametrisation, so the planar segment test
is exact for *detecting* a crossing; the returned point carries sub-metre error
at street scale (fine for a chokepoint marker).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Iterator

#: Denominator floor below which two segments are treated as parallel/collinear.
_EPS = 1e-12

#: MTFCC road classes that are trails / unpaved tracks / paths / stairways —
#: these *ford* a natural watercourse rather than bridging it.
_TRAIL_MTFCC = frozenset({"S1500", "S1710", "S1720", "S1820", "S1830"})

#: NHD kinds that are small *managed* channels (ditch / canal / notional path):
#: a real road culverts them; a trail fords them.
_MANAGED_WATER = frozenset({"canal", "artificial", "connector", "conduit"})


# ---------------------------------------------------------------------------
# Kind inference (pure, table-driven, deterministic)
# ---------------------------------------------------------------------------

def infer_crossing_kind(road_kind: str | None, water_kind: str | None) -> str:
    """Classify a road/water crossing as ``bridge`` / ``ford`` / ``culvert``.

    Deterministic, order-independent:

        * A **managed channel** (canal / artificial path / ditch / conduit) is
          ``culvert`` under a real road, ``ford`` under a trail.
        * A **natural watercourse** (river / stream / waterbody / coastline) is
          ``ford`` under a trail (unpaved tracks cross through the water),
          ``bridge`` under any real vehicular road.

    ``road_kind`` is a TIGER MTFCC code (e.g. ``"S1400"``); ``water_kind`` is a
    normalized NHD kind (e.g. ``"river"``).
    """
    rk = (road_kind or "").strip().upper()
    wk = (water_kind or "").strip().lower()
    if wk in _MANAGED_WATER:
        return "ford" if rk in _TRAIL_MTFCC else "culvert"
    if rk in _TRAIL_MTFCC:
        return "ford"
    return "bridge"


# ---------------------------------------------------------------------------
# Core geometry (pure)
# ---------------------------------------------------------------------------

def segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection point of segment ``p1p2`` with ``p3p4``, or ``None``.

    Planar (Wikipedia) line-segment intersection. Returns the ``(x, y)``
    crossing point when the two *segments* properly intersect (both parameters
    in ``[0, 1]``); ``None`` for parallel, collinear, or non-overlapping
    segments. Deterministic — no RNG, no tolerance beyond the parallel floor.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _EPS:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _is_xy(pos: Any) -> bool:
    """True if ``pos`` is a usable ``[lon, lat]`` (or richer) coordinate."""
    return (
        isinstance(pos, (list, tuple))
        and len(pos) >= 2
        and isinstance(pos[0], (int, float))
        and isinstance(pos[1], (int, float))
    )


def _iter_segments(
    coords: list,
) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
    """Yield consecutive ``(a, b)`` segments of a coordinate list."""
    prev: tuple[float, float] | None = None
    for pos in coords or []:
        if not _is_xy(pos):
            prev = None
            continue
        cur = (float(pos[0]), float(pos[1]))
        if prev is not None:
            yield prev, cur
        prev = cur


def _linear_rings(geometry: Any) -> Iterator[list]:
    """Yield every linear coordinate ring/line of a GeoJSON geometry.

    LineString -> its coords; MultiLineString -> each line; Polygon -> each ring
    (exterior + holes); MultiPolygon -> every ring. Points / null yield nothing.
    A polygon's *edges* are the shoreline a road bridges, so rings are linear
    water for the crossing test.
    """
    if not isinstance(geometry, dict):
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "LineString":
        if isinstance(coords, list):
            yield coords
    elif gtype == "MultiLineString":
        for line in coords or []:
            if isinstance(line, list):
                yield line
    elif gtype == "Polygon":
        for ring in coords or []:
            if isinstance(ring, list):
                yield ring
    elif gtype == "MultiPolygon":
        for poly in coords or []:
            for ring in poly or []:
                if isinstance(ring, list):
                    yield ring


def _ring_bbox(coords: list) -> tuple[float, float, float, float] | None:
    """Axis-aligned ``(w, s, e, n)`` bbox of a coordinate list, or ``None``."""
    xs: list[float] = []
    ys: list[float] = []
    for pos in coords or []:
        if _is_xy(pos):
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True if two ``(w, s, e, n)`` boxes overlap (edge-touching counts)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def meters_between(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Approximate ground distance (m) between two lon/lat points.

    Equirectangular approximation — accurate to well under a percent at the
    sub-kilometre scale of a dedupe radius. Deterministic.
    """
    lon1, lat1 = a
    lon2, lat2 = b
    mlat = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * math.cos(mlat) * 111320.0
    dy = (lat2 - lat1) * 110540.0
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _features(fc: Any) -> list[dict]:
    """Extract the feature list of a GeoJSON FeatureCollection (defensive)."""
    if not isinstance(fc, dict):
        return []
    feats = fc.get("features")
    if not isinstance(feats, list):
        return []
    return [f for f in feats if isinstance(f, dict)]


def _prop(props: dict, *names: str, default: str = "") -> str:
    """First non-empty string property among ``names``, else ``default``."""
    for name in names:
        val = props.get(name)
        if val:
            return str(val)
    return default


def _chokepoint_id(lon: float, lat: float, road_name: str, water_name: str) -> str:
    """Stable, deterministic chokepoint id from location + crossing identity."""
    key = f"{round(lon, 6)},{round(lat, 6)}|{road_name}|{water_name}"
    return "chk_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _build_feature(
    lon: float, lat: float, road_props: dict, water_props: dict
) -> dict:
    """Assemble one chokepoint Point feature with normalized properties."""
    road_name = _prop(road_props, "name", "road_name")
    road_kind = _prop(road_props, "kind", "mtfcc")
    water_name = _prop(water_props, "name", "water_name")
    water_kind = _prop(water_props, "kind")
    kind = infer_crossing_kind(road_kind, water_kind)
    water_label = water_name or (water_kind or "water")
    name = f"{road_name or 'road'} @ {water_label}"
    cid = _chokepoint_id(lon, lat, road_name, water_name)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "source": "chokepoint",
            "id": cid,
            "kind": kind,
            "name": name,
            "road_name": road_name,
            "road_kind": road_kind,
            "water_name": water_name,
            "water_kind": water_kind,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_water_crossings(
    roads: Any,
    hydro: Any,
    *,
    dedupe_m: float = 25.0,
    bbox: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Return chokepoint Points where ``roads`` cross the ``hydro`` network.

    Args:
        roads: GeoJSON FeatureCollection of road centrelines (LineStrings /
            MultiLineStrings) — e.g. the TIGER roads layer. ``properties.name``
            and ``properties.kind`` (MTFCC) drive the crossing label + kind.
        hydro: GeoJSON FeatureCollection of NHD hydrography — flowline
            LineStrings plus waterbody Polygons (their ring edges are treated as
            linear water). ``properties.name`` / ``properties.kind`` drive the
            crossed-feature label + kind.
        dedupe_m: crossings of the *same* road/water pair within this many
            metres collapse to one marker (a wiggly road weaving across a stream
            should not stamp a cluster of bridges). Genuinely distinct crossings
            (different roads, or the same road at far-apart bends) are kept.
        bbox: optional ``(west, south, east, north)`` clip — only crossings
            inside are returned.

    Returns:
        A GeoJSON FeatureCollection of chokepoint ``Point`` features, sorted
        deterministically by ``(lon, lat, id)``. Empty (never raises) for empty
        / malformed input.
    """
    road_feats = _features(roads)
    hydro_feats = _features(hydro)

    # Pre-extract water linear rings once, each with a bbox for fast rejection.
    water_lines: list[tuple[list, dict, tuple[float, float, float, float]]] = []
    for wf in hydro_feats:
        wprops = wf.get("properties") or {}
        for ring in _linear_rings(wf.get("geometry")):
            box = _ring_bbox(ring)
            if box is not None:
                water_lines.append((ring, wprops, box))

    clip = tuple(float(v) for v in bbox) if bbox is not None else None

    # (lon, lat, road_props, water_props) for every raw segment crossing.
    raw: list[tuple[float, float, dict, dict]] = []
    for rf in road_feats:
        rprops = rf.get("properties") or {}
        for rline in _linear_rings(rf.get("geometry")):
            rbox = _ring_bbox(rline)
            if rbox is None:
                continue
            # Candidate water lines whose bbox overlaps this road line.
            candidates = [
                (wline, wprops)
                for (wline, wprops, wbox) in water_lines
                if _bbox_overlap(rbox, wbox)
            ]
            if not candidates:
                continue
            for ra, rb in _iter_segments(rline):
                seg_box = (
                    min(ra[0], rb[0]), min(ra[1], rb[1]),
                    max(ra[0], rb[0]), max(ra[1], rb[1]),
                )
                for wline, wprops in candidates:
                    for wa, wb in _iter_segments(wline):
                        wseg_box = (
                            min(wa[0], wb[0]), min(wa[1], wb[1]),
                            max(wa[0], wb[0]), max(wa[1], wb[1]),
                        )
                        if not _bbox_overlap(seg_box, wseg_box):
                            continue
                        pt = segment_intersection(ra, rb, wa, wb)
                        if pt is None:
                            continue
                        lon, lat = pt
                        if clip is not None and not (
                            clip[0] <= lon <= clip[2]
                            and clip[1] <= lat <= clip[3]
                        ):
                            continue
                        raw.append((lon, lat, rprops, wprops))

    features = _dedupe(raw, dedupe_m)
    features.sort(
        key=lambda f: (
            f["geometry"]["coordinates"][0],
            f["geometry"]["coordinates"][1],
            f["properties"]["id"],
        )
    )
    return {"type": "FeatureCollection", "features": features}


def _dedupe(
    raw: list[tuple[float, float, dict, dict]], dedupe_m: float
) -> list[dict]:
    """Collapse same-road/same-water crossings within ``dedupe_m`` to one.

    Grouped by ``(road_name, water_name, road_kind, water_kind)`` so two
    different roads (or the same road crossing two different streams) are never
    merged; within a group a point is dropped only if it sits within
    ``dedupe_m`` of an already-kept point in that same group.
    """
    radius = max(0.0, float(dedupe_m))
    kept_by_group: dict[tuple, list[tuple[float, float]]] = {}
    features: list[dict] = []
    # Deterministic processing order so which representative point survives a
    # cluster does not depend on input ordering noise.
    raw_sorted = sorted(raw, key=lambda r: (r[0], r[1]))
    for lon, lat, rprops, wprops in raw_sorted:
        group = (
            _prop(rprops, "name", "road_name"),
            _prop(wprops, "name", "water_name"),
            _prop(rprops, "kind", "mtfcc"),
            _prop(wprops, "kind"),
        )
        kept = kept_by_group.setdefault(group, [])
        if radius > 0.0 and any(
            meters_between((lon, lat), (klon, klat)) <= radius
            for klon, klat in kept
        ):
            continue
        kept.append((lon, lat))
        features.append(_build_feature(lon, lat, rprops, wprops))
    return features


# ---------------------------------------------------------------------------
# Production half — chokepoint as a tactical object (costmap / riot consumers)
# ---------------------------------------------------------------------------

def chokepoint_tactical_object(feature: dict) -> dict[str, Any]:
    """Project a chokepoint feature into a lane-agnostic tactical object.

    The open contract other lanes consume WITHOUT importing this module's
    geometry: a bridge/ford/culvert as a hold-point (defender advantage) and a
    severable link (blowing it removes a route). ``sever`` says whether cutting
    the crossing is a meaningful route denial; ``hold_value`` ranks it as a
    defensive position (a primary-road river bridge outranks a ditch culvert).

    Returns a plain dict (no lib types) so callers stay decoupled:
        ``{id, kind, position:{lon,lat}, road, water, hold_value, sever,
           passable, tags:[...]}``
    """
    props = feature.get("properties") or {}
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [0.0, 0.0]
    kind = props.get("kind", "bridge")
    road_kind = (props.get("road_kind") or "").upper()
    water_kind = (props.get("water_kind") or "").lower()

    # Hold value: primary/secondary roads and real rivers make the most
    # contested crossings; culverts / ditch crossings the least.
    hold = 3
    if road_kind in ("S1100", "S1200"):
        hold += 2
    elif road_kind == "S1400":
        hold += 1
    if water_kind == "river":
        hold += 2
    elif water_kind in ("waterbody", "stream"):
        hold += 1
    if kind == "culvert":
        hold -= 1
    hold = max(1, min(10, hold))

    # A blown bridge/ford over a real watercourse severs the route; a culvert
    # over a ditch does not meaningfully deny movement.
    sever = kind in ("bridge", "ford") and water_kind not in _MANAGED_WATER

    tags = ["chokepoint", kind]
    if sever:
        tags.append("severable")
    if hold >= 6:
        tags.append("key_terrain")

    return {
        "id": props.get("id"),
        "kind": kind,
        "position": {"lon": float(coords[0]), "lat": float(coords[1])},
        "road": {"name": props.get("road_name", ""), "class": road_kind},
        "water": {"name": props.get("water_name", ""), "class": water_kind},
        "hold_value": hold,
        "sever": sever,
        "passable": True,
        "tags": tags,
    }
