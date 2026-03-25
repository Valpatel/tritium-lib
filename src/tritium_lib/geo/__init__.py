# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Coordinate transforms between lat/lng and local meters.

A geo-reference point (map center) grounds all local coordinates to
real-world lat/lng.  Physics, tracking, and simulation run in local meters
for speed; lat/lng is computed on serialization so every API response
carries real coordinates.

Convention:
    - Local origin (0, 0, 0) = geo-reference point (lat, lng, alt)
    - 1 local unit = 1 meter
    - +X = East, +Y = North, +Z = Up
    - Heading 0 = North, clockwise in degrees

Shared between tritium-sc (tactical engine) and tritium-edge (fleet server).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

METERS_PER_DEG_LAT = 111_320.0


@dataclass
class GeoReference:
    """A real-world reference point that anchors local coordinates."""

    lat: float = 0.0
    lng: float = 0.0
    alt: float = 0.0  # meters above sea level
    initialized: bool = False

    @property
    def meters_per_deg_lng(self) -> float:
        return METERS_PER_DEG_LAT * math.cos(math.radians(self.lat))


@dataclass
class CameraCalibration:
    """Calibration data for projecting camera pixel coords to ground plane.

    Simple ground-plane projection model: camera at known position + heading + FOV,
    assume flat ground. Gives +/-5m accuracy for objects 10-30m from camera.
    """

    position: tuple[float, float]   # (x, y) in local meters
    heading: float                  # degrees, 0=North, clockwise
    fov_h: float = 60.0            # horizontal FOV in degrees
    mount_height: float = 2.5      # meters above ground
    max_range: float = 30.0        # max detection range in meters


# ---------------------------------------------------------------------------
# Module-level singleton — set once at startup, read from any thread.
# ---------------------------------------------------------------------------

_ref = GeoReference()
_lock = threading.Lock()


def init_reference(lat: float, lng: float, alt: float = 0.0) -> GeoReference:
    """Set the geo-reference point (map center).

    Call once at startup (from config or geocoding result).
    Thread-safe; subsequent calls update the reference.
    """
    global _ref
    with _lock:
        _ref = GeoReference(lat=lat, lng=lng, alt=alt, initialized=True)
    return _ref


def get_reference() -> GeoReference:
    """Return the current geo-reference point."""
    return _ref


def is_initialized() -> bool:
    """True if a real reference point has been set."""
    return _ref.initialized


def reset() -> None:
    """Reset the geo-reference to uninitialized state (for testing)."""
    global _ref
    with _lock:
        _ref = GeoReference()


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def local_to_latlng(x: float, y: float, z: float = 0.0) -> dict:
    """Convert local meters (x=East, y=North, z=Up) to lat/lng/alt.

    Returns {"lat": float, "lng": float, "alt": float}.
    """
    ref = _ref
    if not ref.initialized:
        return {"lat": 0.0, "lng": 0.0, "alt": z}
    lat = ref.lat + y / METERS_PER_DEG_LAT
    lng = ref.lng + x / ref.meters_per_deg_lng
    alt = ref.alt + z
    return {"lat": lat, "lng": lng, "alt": alt}


def latlng_to_local(lat: float, lng: float, alt: float = 0.0) -> tuple[float, float, float]:
    """Convert lat/lng/alt to local meters (x=East, y=North, z=Up).

    Returns (x, y, z) tuple.
    """
    ref = _ref
    if not ref.initialized:
        return (0.0, 0.0, alt)
    y = (lat - ref.lat) * METERS_PER_DEG_LAT
    x = (lng - ref.lng) * ref.meters_per_deg_lng
    z = alt - ref.alt
    return (x, y, z)


def local_to_latlng_2d(x: float, y: float) -> tuple[float, float]:
    """Convert 2D local meters to (lat, lng). Convenience for flat targets."""
    result = local_to_latlng(x, y, 0.0)
    return (result["lat"], result["lng"])


# ---------------------------------------------------------------------------
# Camera ground-plane projection
# ---------------------------------------------------------------------------

def camera_pixel_to_ground(
    cx: float, cy: float, calib: CameraCalibration
) -> tuple[float, float] | None:
    """Project normalized image coordinates to ground plane position.

    Args:
        cx: Horizontal position in image (0.0=left, 1.0=right)
        cy: Vertical position in image (0.0=top, 1.0=bottom)
        calib: Camera calibration data

    Returns:
        (x, y) in local meters, or None if projection fails
        (e.g., looking at sky, object above horizon)
    """
    # Horizontal angle offset from center of FOV
    angle_h = (cx - 0.5) * calib.fov_h
    bearing = calib.heading + angle_h

    # Range estimate from vertical position
    # cy=0.0 = top (far), cy=1.0 = bottom (close)
    # Objects above horizon (cy < ~0.1) can't be projected
    if cy < 0.1:
        return None

    range_factor = 1.0 - cy  # 0=close, 1=far
    range_m = 2.0 + range_factor * calib.max_range

    # Project to ground
    bearing_rad = math.radians(bearing)
    dx = range_m * math.sin(bearing_rad)
    dy = range_m * math.cos(bearing_rad)

    return (calib.position[0] + dx, calib.position[1] + dy)


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def point_in_polygon(
    px: float, py: float, polygon: list[tuple[float, float]]
) -> bool:
    """Ray-casting point-in-polygon test.

    Args:
        px: X coordinate of the point to test.
        py: Y coordinate of the point to test.
        polygon: List of (x, y) vertices defining a closed polygon.

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_polygon_latlng(
    lat: float, lng: float, polygon: list
) -> bool:
    """Point-in-polygon test using (lat, lng) coordinates.

    Accepts polygon vertices as either tuples ``(lat, lng)`` or dicts
    ``{"lat": float, "lon": float}`` (as used by floorplan room polygons).

    Args:
        lat: Latitude of the point to test.
        lng: Longitude of the point to test.
        polygon: List of vertices — tuples or dicts with "lat"/"lon" keys.

    Returns:
        True if the point is inside the polygon.
    """
    if not polygon:
        return False
    # Normalize dicts to tuples if needed
    first = polygon[0]
    if isinstance(first, dict):
        polygon = [(v.get("lat", 0), v.get("lon", v.get("lng", 0))) for v in polygon]
    return point_in_polygon(lat, lng, polygon)


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters between two WGS84 points.

    Uses the Haversine formula. Accurate for any distance.
    """
    R = 6_371_000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# WGS84 ellipsoid constants and precise projection
# ---------------------------------------------------------------------------

# WGS84 semi-major and semi-minor axes
WGS84_A = 6_378_137.0  # equatorial radius in meters
WGS84_B = 6_356_752.314245  # polar radius in meters
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # first eccentricity squared


def _wgs84_N(lat_rad: float) -> float:
    """Radius of curvature in the prime vertical (N) at a given latitude.

    This is the distance from the surface to the polar axis along the
    ellipsoid normal.  Used for precise lat/lng <-> ECEF conversions.
    """
    sin_lat = math.sin(lat_rad)
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)


def _wgs84_M(lat_rad: float) -> float:
    """Radius of curvature in the meridian (M) at a given latitude.

    This is the radius of the north-south arc at the given latitude.
    """
    sin_lat = math.sin(lat_rad)
    denom = (1.0 - WGS84_E2 * sin_lat * sin_lat) ** 1.5
    return WGS84_A * (1.0 - WGS84_E2) / denom


def meters_per_degree_lat(lat: float) -> float:
    """Meters per degree of latitude at a given latitude (WGS84 ellipsoid).

    More precise than the constant ``METERS_PER_DEG_LAT`` which assumes a
    sphere.  At the equator this returns ~110,574 m; at the poles ~111,694 m.

    Args:
        lat: Latitude in degrees.

    Returns:
        Meters per degree of latitude.
    """
    lat_rad = math.radians(lat)
    return _wgs84_M(lat_rad) * math.radians(1.0)


def meters_per_degree_lng(lat: float) -> float:
    """Meters per degree of longitude at a given latitude (WGS84 ellipsoid).

    At the equator this returns ~111,320 m; at the poles it approaches 0.

    Args:
        lat: Latitude in degrees.

    Returns:
        Meters per degree of longitude.
    """
    lat_rad = math.radians(lat)
    return _wgs84_N(lat_rad) * math.cos(lat_rad) * math.radians(1.0)


def latlng_to_ecef(lat: float, lng: float, alt: float = 0.0) -> tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates to Earth-Centered Earth-Fixed (ECEF).

    Args:
        lat: Latitude in degrees.
        lng: Longitude in degrees.
        alt: Altitude above the WGS84 ellipsoid in meters.

    Returns:
        (x, y, z) tuple in meters.  X points to 0N/0E, Y to 0N/90E,
        Z to the North Pole.
    """
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    N = _wgs84_N(lat_rad)
    x = (N + alt) * math.cos(lat_rad) * math.cos(lng_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lng_rad)
    z = (N * (1.0 - WGS84_E2) + alt) * math.sin(lat_rad)
    return (x, y, z)


def ecef_to_latlng(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF coordinates back to WGS84 geodetic (lat, lng, alt).

    Uses Bowring's iterative method (converges in 2-3 iterations for
    any point on or near Earth's surface).

    Args:
        x: ECEF X in meters.
        y: ECEF Y in meters.
        z: ECEF Z in meters.

    Returns:
        (lat, lng, alt) with lat/lng in degrees and alt in meters above
        the WGS84 ellipsoid.
    """
    lng = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)

    # Initial estimate: spherical approximation
    lat_rad = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        N = _wgs84_N(lat_rad)
        lat_rad_new = math.atan2(z + WGS84_E2 * N * math.sin(lat_rad), p)
        if abs(lat_rad_new - lat_rad) < 1e-12:
            break
        lat_rad = lat_rad_new

    lat = math.degrees(lat_rad)
    N = _wgs84_N(lat_rad)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - WGS84_B
    return (lat, lng, alt)


# ---------------------------------------------------------------------------
# Bearing and midpoint
# ---------------------------------------------------------------------------

def initial_bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial bearing (forward azimuth) from point 1 to point 2.

    Uses the spherical law of sines / cosines.  The bearing is measured
    clockwise from true north in degrees [0, 360).

    Args:
        lat1: Latitude of point 1 in degrees.
        lng1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lng2: Longitude of point 2 in degrees.

    Returns:
        Bearing in degrees, [0, 360).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lng2 - lng1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360.0) % 360.0


def midpoint(lat1: float, lng1: float, lat2: float, lng2: float) -> tuple[float, float]:
    """Geographic midpoint between two WGS84 points on a great circle.

    Args:
        lat1: Latitude of point 1 in degrees.
        lng1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lng2: Longitude of point 2 in degrees.

    Returns:
        (lat, lng) tuple of the midpoint in degrees.
    """
    phi1 = math.radians(lat1)
    lam1 = math.radians(lng1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lng2 - lng1)

    bx = math.cos(phi2) * math.cos(dlam)
    by = math.cos(phi2) * math.sin(dlam)
    phi_m = math.atan2(
        math.sin(phi1) + math.sin(phi2),
        math.sqrt((math.cos(phi1) + bx) ** 2 + by ** 2),
    )
    lam_m = lam1 + math.atan2(by, math.cos(phi1) + bx)
    return (math.degrees(phi_m), math.degrees(lam_m))


def destination_point(
    lat: float, lng: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Compute the destination point given start, bearing, and distance.

    Uses the spherical Earth model (mean radius 6,371,000 m).

    Args:
        lat: Starting latitude in degrees.
        lng: Starting longitude in degrees.
        bearing_deg: Bearing in degrees (0=North, clockwise).
        distance_m: Distance in meters.

    Returns:
        (lat, lng) of the destination point in degrees.
    """
    R = 6_371_000.0
    phi1 = math.radians(lat)
    lam1 = math.radians(lng)
    theta = math.radians(bearing_deg)
    delta = distance_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return (math.degrees(phi2), math.degrees(lam2))


# ---------------------------------------------------------------------------
# Area computation
# ---------------------------------------------------------------------------

def compute_area(polygon: list[tuple[float, float]]) -> float:
    """Compute the area of a polygon defined in local-meter coordinates.

    Uses the shoelace formula.  The polygon does **not** need to be
    explicitly closed (i.e. first vertex == last vertex).

    Args:
        polygon: List of (x, y) vertices in local meters.

    Returns:
        Unsigned area in square meters.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    j = n - 1
    for i in range(n):
        area += (polygon[j][0] + polygon[i][0]) * (polygon[j][1] - polygon[i][1])
        j = i
    return abs(area) / 2.0


def compute_area_latlng(polygon: list[tuple[float, float]]) -> float:
    """Compute the approximate area of a lat/lng polygon in square meters.

    Projects the polygon to a local tangent plane centred on the polygon's
    centroid, then uses the shoelace formula.  Good to <0.1 % for polygons
    smaller than ~100 km across.

    Args:
        polygon: List of (lat, lng) vertices in degrees.

    Returns:
        Unsigned area in square meters.
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    # Centroid for projection reference
    clat = sum(v[0] for v in polygon) / n
    clng = sum(v[1] for v in polygon) / n

    m_lat = meters_per_degree_lat(clat)
    m_lng = meters_per_degree_lng(clat)

    local = [
        ((v[1] - clng) * m_lng, (v[0] - clat) * m_lat)
        for v in polygon
    ]
    return compute_area(local)


# ---------------------------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------------------------

def bounding_box(
    lat: float, lng: float, radius_m: float
) -> tuple[float, float, float, float]:
    """Compute an axis-aligned bounding box around a point.

    Args:
        lat: Centre latitude in degrees.
        lng: Centre longitude in degrees.
        radius_m: Radius in meters.

    Returns:
        (min_lat, min_lng, max_lat, max_lng) tuple in degrees.
    """
    d_lat = radius_m / meters_per_degree_lat(lat)
    d_lng = radius_m / meters_per_degree_lng(lat) if abs(lat) < 89.99 else 180.0
    return (lat - d_lat, lng - d_lng, lat + d_lat, lng + d_lng)


# ---------------------------------------------------------------------------
# Reverse geocoding (offline, using OSM Overpass)
# ---------------------------------------------------------------------------

def reverse_geocode(lat: float, lng: float, radius_m: float = 50.0) -> dict:
    """Return the nearest named OSM feature to a lat/lng coordinate.

    Queries the Overpass API for named features within ``radius_m`` and
    returns the closest one.  Results are returned as a dict with keys:

    - ``name``: Feature name (e.g. "Congress Avenue") or ``None``.
    - ``osm_type``: ``"node"``, ``"way"``, or ``"relation"``.
    - ``osm_id``: Numeric OSM element ID.
    - ``tags``: Full dict of OSM tags.
    - ``distance_m``: Approximate distance from the query point.
    - ``lat``/``lng``: Coordinates of the feature.

    If no feature is found or the network is unavailable the dict will
    have ``name=None`` and empty tags.

    This function is intentionally simple and not cached — for bulk
    lookups use :class:`tritium_lib.intelligence.geospatial.osm_enrichment.OSMEnrichment`.

    Args:
        lat: Latitude in degrees.
        lng: Longitude in degrees.
        radius_m: Search radius in meters (default 50).

    Returns:
        Dict describing the nearest named feature.
    """
    empty: dict = {
        "name": None,
        "osm_type": None,
        "osm_id": None,
        "tags": {},
        "distance_m": None,
        "lat": lat,
        "lng": lng,
    }

    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return empty

    query = (
        f"[out:json][timeout:10];"
        f"is_in({lat},{lng})->.a;"
        f"("
        f"  node(around:{radius_m},{lat},{lng})[\"name\"];"
        f"  way(around:{radius_m},{lat},{lng})[\"name\"];"
        f");"
        f"out center 1;"
    )

    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=10,
            headers={"User-Agent": "Tritium/1.0 (reverse_geocode)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return empty

    elements = data.get("elements", [])
    if not elements:
        return empty

    # Find the closest element by Euclidean approximation
    best = None
    best_dist = float("inf")
    for el in elements:
        el_lat = el.get("lat") or el.get("center", {}).get("lat", lat)
        el_lng = el.get("lon") or el.get("center", {}).get("lon", lng)
        d = haversine_distance(lat, lng, el_lat, el_lng)
        if d < best_dist:
            best_dist = d
            best = el

    if best is None:
        return empty

    best_lat = best.get("lat") or best.get("center", {}).get("lat", lat)
    best_lng = best.get("lon") or best.get("center", {}).get("lon", lng)
    return {
        "name": best.get("tags", {}).get("name"),
        "osm_type": best.get("type"),
        "osm_id": best.get("id"),
        "tags": best.get("tags", {}),
        "distance_m": best_dist,
        "lat": best_lat,
        "lng": best_lng,
    }
