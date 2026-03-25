# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo — coordinate transforms, projection, area, and geocoding."""

import math
from unittest.mock import patch, MagicMock

import pytest

from tritium_lib.geo import (
    METERS_PER_DEG_LAT,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    GeoReference,
    CameraCalibration,
    init_reference,
    get_reference,
    is_initialized,
    reset,
    local_to_latlng,
    latlng_to_local,
    local_to_latlng_2d,
    camera_pixel_to_ground,
    haversine_distance,
    meters_per_degree_lat,
    meters_per_degree_lng,
    latlng_to_ecef,
    ecef_to_latlng,
    initial_bearing,
    midpoint,
    destination_point,
    compute_area,
    compute_area_latlng,
    bounding_box,
    reverse_geocode,
)


@pytest.fixture(autouse=True)
def _reset_geo():
    """Reset geo-reference before each test."""
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# GeoReference dataclass
# ---------------------------------------------------------------------------

class TestGeoReference:
    def test_defaults(self):
        ref = GeoReference()
        assert ref.lat == 0.0
        assert ref.lng == 0.0
        assert ref.alt == 0.0
        assert ref.initialized is False

    def test_meters_per_deg_lng_at_equator(self):
        ref = GeoReference(lat=0.0)
        assert abs(ref.meters_per_deg_lng - METERS_PER_DEG_LAT) < 1.0

    def test_meters_per_deg_lng_at_pole(self):
        ref = GeoReference(lat=90.0)
        assert abs(ref.meters_per_deg_lng) < 0.1

    def test_meters_per_deg_lng_at_45(self):
        ref = GeoReference(lat=45.0)
        expected = METERS_PER_DEG_LAT * math.cos(math.radians(45.0))
        assert abs(ref.meters_per_deg_lng - expected) < 0.01


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_not_initialized_by_default(self):
        assert is_initialized() is False

    def test_init_reference(self):
        ref = init_reference(37.7749, -122.4194, 10.0)
        assert ref.lat == 37.7749
        assert ref.lng == -122.4194
        assert ref.alt == 10.0
        assert ref.initialized is True
        assert is_initialized() is True

    def test_get_reference_returns_current(self):
        init_reference(34.0, -118.0)
        ref = get_reference()
        assert ref.lat == 34.0

    def test_reset_clears(self):
        init_reference(34.0, -118.0)
        reset()
        assert is_initialized() is False

    def test_reinitialize(self):
        init_reference(34.0, -118.0)
        init_reference(40.0, -74.0)
        ref = get_reference()
        assert ref.lat == 40.0


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

class TestCoordinateTransforms:
    def test_uninitialized_returns_zero(self):
        result = local_to_latlng(100, 200, 5)
        assert result == {"lat": 0.0, "lng": 0.0, "alt": 5.0}

    def test_uninitialized_latlng_to_local(self):
        x, y, z = latlng_to_local(37.0, -122.0, 50.0)
        assert x == 0.0
        assert y == 0.0
        assert z == 50.0

    def test_origin_is_reference(self):
        init_reference(37.7749, -122.4194, 10.0)
        result = local_to_latlng(0, 0, 0)
        assert abs(result["lat"] - 37.7749) < 1e-10
        assert abs(result["lng"] - (-122.4194)) < 1e-10
        assert abs(result["alt"] - 10.0) < 1e-10

    def test_roundtrip(self):
        init_reference(37.7749, -122.4194, 10.0)
        lat, lng, alt = 37.780, -122.415, 15.0
        x, y, z = latlng_to_local(lat, lng, alt)
        back = local_to_latlng(x, y, z)
        assert abs(back["lat"] - lat) < 1e-6
        assert abs(back["lng"] - lng) < 1e-6
        assert abs(back["alt"] - alt) < 1e-6

    def test_north_offset(self):
        init_reference(0.0, 0.0, 0.0)
        # 111320m north = 1 degree of latitude
        result = local_to_latlng(0, METERS_PER_DEG_LAT, 0)
        assert abs(result["lat"] - 1.0) < 1e-6
        assert abs(result["lng"]) < 1e-10

    def test_east_offset_at_equator(self):
        init_reference(0.0, 0.0, 0.0)
        result = local_to_latlng(METERS_PER_DEG_LAT, 0, 0)
        assert abs(result["lng"] - 1.0) < 1e-6

    def test_2d_convenience(self):
        init_reference(37.7749, -122.4194, 0.0)
        lat, lng = local_to_latlng_2d(100, 200)
        full = local_to_latlng(100, 200, 0)
        assert lat == full["lat"]
        assert lng == full["lng"]


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------

class TestCameraProjection:
    def test_sky_returns_none(self):
        calib = CameraCalibration(position=(0, 0), heading=0)
        result = camera_pixel_to_ground(0.5, 0.05, calib)
        assert result is None

    def test_center_bottom(self):
        calib = CameraCalibration(position=(0, 0), heading=0)
        result = camera_pixel_to_ground(0.5, 1.0, calib)
        assert result is not None
        x, y = result
        # Looking north, bottom of frame = closest point
        assert abs(x) < 0.1  # centered horizontally
        assert y > 0  # ahead (north)

    def test_heading_affects_direction(self):
        calib_n = CameraCalibration(position=(0, 0), heading=0)
        calib_e = CameraCalibration(position=(0, 0), heading=90)
        rn = camera_pixel_to_ground(0.5, 0.5, calib_n)
        re = camera_pixel_to_ground(0.5, 0.5, calib_e)
        assert rn is not None and re is not None
        # North-facing: y >> x. East-facing: x >> y
        assert rn[1] > rn[0]
        assert re[0] > re[1]

    def test_position_offset(self):
        calib = CameraCalibration(position=(10, 20), heading=0)
        result = camera_pixel_to_ground(0.5, 0.5, calib)
        assert result is not None
        assert result[0] > 9  # offset by position.x
        assert result[1] > 20  # offset by position.y + projection


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point(self):
        d = haversine_distance(37.0, -122.0, 37.0, -122.0)
        assert d == 0.0

    def test_known_distance(self):
        # ~1 degree latitude = ~111.32 km
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert abs(d - 111_195) < 500  # within 500m

    def test_symmetric(self):
        d1 = haversine_distance(37.0, -122.0, 40.0, -74.0)
        d2 = haversine_distance(40.0, -74.0, 37.0, -122.0)
        assert abs(d1 - d2) < 0.01

    def test_antipodal_points(self):
        """Distance between antipodal points should be half Earth circumference."""
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        half_circ = math.pi * 6_371_000.0
        assert abs(d - half_circ) < 100.0

    def test_pole_to_pole(self):
        """North pole to south pole = half Earth circumference."""
        d = haversine_distance(90.0, 0.0, -90.0, 0.0)
        half_circ = math.pi * 6_371_000.0
        assert abs(d - half_circ) < 100.0

    def test_dateline_crossing(self):
        """Distance across the international date line."""
        d = haversine_distance(0.0, 179.0, 0.0, -179.0)
        # Should be ~222 km (2 degrees at equator), not ~39800 km
        assert d < 250_000


# ---------------------------------------------------------------------------
# WGS84 ellipsoid precision
# ---------------------------------------------------------------------------

class TestWGS84Precision:
    """Test WGS84 ellipsoid-based functions for geodetic precision."""

    def test_meters_per_deg_lat_at_equator(self):
        m = meters_per_degree_lat(0.0)
        # WGS84 value at equator is ~110,574 m/deg
        assert 110_500 < m < 110_700

    def test_meters_per_deg_lat_at_pole(self):
        m = meters_per_degree_lat(89.0)
        # Near poles, m/deg lat ~111,694
        assert 111_600 < m < 111_800

    def test_meters_per_deg_lat_increases_with_latitude(self):
        """Due to Earth's flattening, m/deg lat is larger near the poles."""
        m0 = meters_per_degree_lat(0.0)
        m60 = meters_per_degree_lat(60.0)
        m89 = meters_per_degree_lat(89.0)
        assert m0 < m60 < m89

    def test_meters_per_deg_lng_at_equator(self):
        m = meters_per_degree_lng(0.0)
        # ~111,320 m at equator
        assert 111_200 < m < 111_400

    def test_meters_per_deg_lng_shrinks_to_zero_at_pole(self):
        m = meters_per_degree_lng(89.5)
        # Very close to 0 near pole
        assert m < 1000.0

    def test_meters_per_deg_lng_at_45(self):
        m = meters_per_degree_lng(45.0)
        # ~78,847 m at 45 degrees
        assert 78_500 < m < 79_200

    def test_meters_per_deg_lat_symmetric(self):
        """Northern and southern hemispheres should give same result."""
        mn = meters_per_degree_lat(30.0)
        ms = meters_per_degree_lat(-30.0)
        assert abs(mn - ms) < 0.01


# ---------------------------------------------------------------------------
# ECEF conversions
# ---------------------------------------------------------------------------

class TestECEF:
    """Test lat/lng <-> ECEF round-trip conversions."""

    def test_equator_prime_meridian(self):
        """(0, 0, 0) should map to (WGS84_A, 0, 0) in ECEF."""
        x, y, z = latlng_to_ecef(0.0, 0.0, 0.0)
        assert abs(x - WGS84_A) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_north_pole(self):
        """North pole should be at (0, 0, WGS84_B)."""
        x, y, z = latlng_to_ecef(90.0, 0.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z - WGS84_B) < 0.5

    def test_south_pole(self):
        x, y, z = latlng_to_ecef(-90.0, 0.0, 0.0)
        assert abs(z + WGS84_B) < 0.5

    def test_roundtrip_san_francisco(self):
        lat0, lng0, alt0 = 37.7749, -122.4194, 16.0
        x, y, z = latlng_to_ecef(lat0, lng0, alt0)
        lat, lng, alt = ecef_to_latlng(x, y, z)
        assert abs(lat - lat0) < 1e-8
        assert abs(lng - lng0) < 1e-8
        assert abs(alt - alt0) < 0.01

    def test_roundtrip_southern_hemisphere(self):
        lat0, lng0, alt0 = -33.8688, 151.2093, 58.0  # Sydney
        x, y, z = latlng_to_ecef(lat0, lng0, alt0)
        lat, lng, alt = ecef_to_latlng(x, y, z)
        assert abs(lat - lat0) < 1e-8
        assert abs(lng - lng0) < 1e-8
        assert abs(alt - alt0) < 0.01

    def test_roundtrip_high_altitude(self):
        """Aircraft at 10km altitude."""
        lat0, lng0, alt0 = 51.4775, -0.4614, 10_000.0  # Heathrow
        x, y, z = latlng_to_ecef(lat0, lng0, alt0)
        lat, lng, alt = ecef_to_latlng(x, y, z)
        assert abs(lat - lat0) < 1e-7
        assert abs(lng - lng0) < 1e-7
        assert abs(alt - alt0) < 0.1

    def test_roundtrip_near_dateline(self):
        lat0, lng0, alt0 = 0.0, 179.999, 0.0
        x, y, z = latlng_to_ecef(lat0, lng0, alt0)
        lat, lng, alt = ecef_to_latlng(x, y, z)
        assert abs(lat - lat0) < 1e-8
        assert abs(lng - lng0) < 1e-8

    def test_ecef_altitude_effect(self):
        """Higher altitude should increase ECEF distance from origin."""
        _, _, z_low = latlng_to_ecef(0.0, 0.0, 0.0)
        x_high, _, _ = latlng_to_ecef(0.0, 0.0, 1000.0)
        assert x_high > WGS84_A


# ---------------------------------------------------------------------------
# Bearing and navigation
# ---------------------------------------------------------------------------

class TestBearing:
    def test_due_north(self):
        b = initial_bearing(0.0, 0.0, 1.0, 0.0)
        assert abs(b - 0.0) < 0.01

    def test_due_east(self):
        b = initial_bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(b - 90.0) < 0.01

    def test_due_south(self):
        b = initial_bearing(0.0, 0.0, -1.0, 0.0)
        assert abs(b - 180.0) < 0.01

    def test_due_west(self):
        b = initial_bearing(0.0, 0.0, 0.0, -1.0)
        assert abs(b - 270.0) < 0.01

    def test_northeast(self):
        b = initial_bearing(0.0, 0.0, 1.0, 1.0)
        # Should be between 0 and 90
        assert 0.0 < b < 90.0

    def test_bearing_always_positive(self):
        b = initial_bearing(40.0, -74.0, 37.0, -122.0)
        assert 0.0 <= b < 360.0


class TestMidpoint:
    def test_equator_midpoint(self):
        lat, lng = midpoint(0.0, 0.0, 0.0, 90.0)
        assert abs(lat - 0.0) < 0.01
        assert abs(lng - 45.0) < 0.01

    def test_same_point_midpoint(self):
        lat, lng = midpoint(37.0, -122.0, 37.0, -122.0)
        assert abs(lat - 37.0) < 1e-10
        assert abs(lng - (-122.0)) < 1e-10

    def test_symmetric(self):
        lat1, lng1 = midpoint(30.0, -90.0, 40.0, -80.0)
        lat2, lng2 = midpoint(40.0, -80.0, 30.0, -90.0)
        assert abs(lat1 - lat2) < 1e-10
        assert abs(lng1 - lng2) < 1e-10


class TestDestinationPoint:
    def test_north_one_degree(self):
        """Moving ~111 km north from equator should add ~1 degree latitude."""
        lat, lng = destination_point(0.0, 0.0, 0.0, 111_195.0)
        assert abs(lat - 1.0) < 0.01
        assert abs(lng) < 0.01

    def test_east_at_equator(self):
        lat, lng = destination_point(0.0, 0.0, 90.0, 111_195.0)
        assert abs(lat) < 0.01
        assert abs(lng - 1.0) < 0.01

    def test_zero_distance(self):
        lat, lng = destination_point(37.0, -122.0, 45.0, 0.0)
        assert abs(lat - 37.0) < 1e-10
        assert abs(lng - (-122.0)) < 1e-10

    def test_roundtrip_with_bearing_and_distance(self):
        """Go somewhere, compute bearing back, go back — should return to start."""
        start_lat, start_lng = 40.0, -74.0
        bearing = 45.0
        distance = 50_000.0  # 50 km

        dest_lat, dest_lng = destination_point(start_lat, start_lng, bearing, distance)

        # Bearing back should be approximately 225 (opposite of 45)
        back_bearing = initial_bearing(dest_lat, dest_lng, start_lat, start_lng)
        d_back = haversine_distance(dest_lat, dest_lng, start_lat, start_lng)

        # Go back
        final_lat, final_lng = destination_point(dest_lat, dest_lng, back_bearing, d_back)
        assert abs(final_lat - start_lat) < 0.001
        assert abs(final_lng - start_lng) < 0.001


# ---------------------------------------------------------------------------
# Area computation
# ---------------------------------------------------------------------------

class TestComputeArea:
    def test_unit_square(self):
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert abs(compute_area(polygon) - 1.0) < 1e-10

    def test_right_triangle(self):
        polygon = [(0, 0), (4, 0), (0, 3)]
        assert abs(compute_area(polygon) - 6.0) < 1e-10

    def test_closed_polygon_same_as_open(self):
        """Explicitly closing the polygon should not change the area."""
        open_poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        closed_poly = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        assert abs(compute_area(open_poly) - compute_area(closed_poly)) < 1e-10

    def test_degenerate_line(self):
        polygon = [(0, 0), (1, 1)]
        assert compute_area(polygon) == 0.0

    def test_degenerate_point(self):
        polygon = [(5, 5)]
        assert compute_area(polygon) == 0.0

    def test_empty_polygon(self):
        assert compute_area([]) == 0.0

    def test_clockwise_and_counterclockwise_same(self):
        """Area should be the same regardless of winding order."""
        ccw = [(0, 0), (10, 0), (10, 10), (0, 10)]
        cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
        assert abs(compute_area(ccw) - compute_area(cw)) < 1e-10

    def test_complex_polygon(self):
        """L-shaped polygon area check."""
        # An L-shape: 3x3 square minus 1x1 corner
        poly = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
        area = compute_area(poly)
        # Total: 3*3 - 2*2 = 5
        assert abs(area - 5.0) < 1e-10

    def test_very_small_area(self):
        """A tiny polygon (1mm x 1mm)."""
        s = 0.001
        polygon = [(0, 0), (s, 0), (s, s), (0, s)]
        area = compute_area(polygon)
        assert abs(area - s * s) < 1e-12


class TestComputeAreaLatLng:
    def test_one_degree_square_at_equator(self):
        """A 1-degree square at the equator should be roughly 12,300 km^2."""
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        area = compute_area_latlng(polygon)
        # ~111 km * ~111 km = ~12,321 km^2 = ~1.23e10 m^2
        assert 1.2e10 < area < 1.25e10

    def test_small_city_block(self):
        """A ~100m x 100m block in Austin, TX."""
        # Roughly 0.0009 deg lat x 0.0011 deg lng at 30.27N
        polygon = [
            (30.270, -97.750),
            (30.270, -97.749),
            (30.271, -97.749),
            (30.271, -97.750),
        ]
        area = compute_area_latlng(polygon)
        # Should be roughly 100m * 100m = 10,000 m^2, within a factor of 2
        assert 5_000 < area < 20_000

    def test_degenerate_latlng(self):
        polygon = [(30.0, -97.0), (30.0, -97.0)]
        assert compute_area_latlng(polygon) == 0.0

    def test_empty_latlng(self):
        assert compute_area_latlng([]) == 0.0

    def test_high_latitude_area_smaller(self):
        """Same angular polygon near pole should have less area than at equator."""
        eq_poly = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
        hi_poly = [(70.0, 0.0), (70.1, 0.0), (70.1, 0.1), (70.0, 0.1)]
        area_eq = compute_area_latlng(eq_poly)
        area_hi = compute_area_latlng(hi_poly)
        assert area_hi < area_eq


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_equator_100m(self):
        min_lat, min_lng, max_lat, max_lng = bounding_box(0.0, 0.0, 100.0)
        # Should be roughly symmetric and ~0.001 degrees
        assert min_lat < 0.0 < max_lat
        assert min_lng < 0.0 < max_lng
        width_m = haversine_distance(0.0, min_lng, 0.0, max_lng)
        assert abs(width_m - 200.0) < 5.0

    def test_high_latitude_wider_lng(self):
        """At high latitudes, longitude range should be wider for same radius."""
        _, min_lng_eq, _, max_lng_eq = bounding_box(0.0, 0.0, 1000.0)
        _, min_lng_hi, _, max_lng_hi = bounding_box(60.0, 0.0, 1000.0)
        range_eq = max_lng_eq - min_lng_eq
        range_hi = max_lng_hi - min_lng_hi
        assert range_hi > range_eq

    def test_zero_radius(self):
        min_lat, min_lng, max_lat, max_lng = bounding_box(30.0, -90.0, 0.0)
        assert min_lat == max_lat == 30.0
        assert min_lng == max_lng == -90.0


# ---------------------------------------------------------------------------
# Point-in-polygon edge cases
# ---------------------------------------------------------------------------

class TestPointInPolygonEdgeCases:
    """Additional edge-case tests for point_in_polygon."""

    def test_collinear_vertices(self):
        """Polygon with collinear vertices should still work."""
        from tritium_lib.geo import point_in_polygon
        poly = [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon(5, 5, poly) is True
        assert point_in_polygon(15, 5, poly) is False

    def test_very_large_polygon(self):
        """Polygon covering a huge area should still work."""
        from tritium_lib.geo import point_in_polygon
        big = 1_000_000.0
        poly = [(-big, -big), (big, -big), (big, big), (-big, big)]
        assert point_in_polygon(0, 0, poly) is True
        assert point_in_polygon(big + 1, 0, poly) is False

    def test_latlng_dict_format(self):
        """point_in_polygon_latlng should accept dicts with lon or lng key."""
        from tritium_lib.geo import point_in_polygon_latlng
        poly_lon = [
            {"lat": 0.0, "lon": 0.0},
            {"lat": 1.0, "lon": 0.0},
            {"lat": 1.0, "lon": 1.0},
            {"lat": 0.0, "lon": 1.0},
        ]
        assert point_in_polygon_latlng(0.5, 0.5, poly_lon) is True

        poly_lng = [
            {"lat": 0.0, "lng": 0.0},
            {"lat": 1.0, "lng": 0.0},
            {"lat": 1.0, "lng": 1.0},
            {"lat": 0.0, "lng": 1.0},
        ]
        assert point_in_polygon_latlng(0.5, 0.5, poly_lng) is True

    def test_empty_polygon_returns_false(self):
        from tritium_lib.geo import point_in_polygon_latlng
        assert point_in_polygon_latlng(0, 0, []) is False


# ---------------------------------------------------------------------------
# Reverse geocode (mocked)
# ---------------------------------------------------------------------------

class TestReverseGeocode:
    """Test reverse_geocode with mocked HTTP to avoid network calls."""

    def _mock_overpass_response(self, elements):
        """Create a mock requests response with given Overpass elements."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"elements": elements}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_nearest_named_feature(self):
        elements = [
            {
                "type": "way",
                "id": 12345,
                "center": {"lat": 30.267, "lon": -97.743},
                "tags": {"name": "Congress Avenue", "highway": "primary"},
            },
            {
                "type": "node",
                "id": 67890,
                "lat": 30.268,
                "lon": -97.740,
                "tags": {"name": "Some Shop", "amenity": "cafe"},
            },
        ]
        mock_resp = self._mock_overpass_response(elements)
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = reverse_geocode(30.267, -97.743)
            assert result["name"] == "Congress Avenue"
            assert result["osm_id"] == 12345
            assert result["osm_type"] == "way"
            assert result["distance_m"] is not None
            assert result["distance_m"] < 10.0
            mock_post.assert_called_once()

    def test_no_results_returns_empty(self):
        mock_resp = self._mock_overpass_response([])
        with patch("requests.post", return_value=mock_resp):
            result = reverse_geocode(0.0, 0.0)
            assert result["name"] is None
            assert result["tags"] == {}

    def test_network_error_returns_empty(self):
        with patch("requests.post", side_effect=Exception("timeout")):
            result = reverse_geocode(30.0, -97.0)
            assert result["name"] is None
            assert result["lat"] == 30.0
            assert result["lng"] == -97.0

    def test_picks_closer_feature(self):
        """When multiple results, should pick the one closest to query point."""
        elements = [
            {
                "type": "node",
                "id": 111,
                "lat": 30.300,
                "lon": -97.700,
                "tags": {"name": "Far Away Place"},
            },
            {
                "type": "way",
                "id": 222,
                "center": {"lat": 30.267, "lon": -97.743},
                "tags": {"name": "Nearby Street"},
            },
        ]
        mock_resp = self._mock_overpass_response(elements)
        with patch("requests.post", return_value=mock_resp):
            result = reverse_geocode(30.267, -97.743)
            assert result["name"] == "Nearby Street"
            assert result["osm_id"] == 222


# ---------------------------------------------------------------------------
# Coordinate transform edge cases (poles, dateline, negative coords)
# ---------------------------------------------------------------------------

class TestCoordinateEdgeCases:
    """Edge cases for coordinate transformations near poles and dateline."""

    def test_reference_at_north_pole(self):
        """Setting reference near north pole should not crash."""
        init_reference(89.999, 0.0, 0.0)
        result = local_to_latlng(100, 100, 0)
        assert result["lat"] > 89.0
        # At the pole, longitude is essentially meaningless but should not NaN
        assert math.isfinite(result["lng"])

    def test_reference_at_south_pole(self):
        init_reference(-89.999, 0.0, 0.0)
        result = local_to_latlng(0, 100, 0)
        assert result["lat"] > -90.0
        assert math.isfinite(result["lng"])

    def test_reference_at_dateline(self):
        init_reference(0.0, 180.0, 0.0)
        result = local_to_latlng(1000, 0, 0)
        assert math.isfinite(result["lng"])

    def test_reference_negative_longitude(self):
        init_reference(0.0, -179.99, 0.0)
        result = local_to_latlng(-1000, 0, 0)
        # Moving west from near -180 should wrap or stay finite
        assert math.isfinite(result["lng"])

    def test_large_local_offset(self):
        """Very large offsets (1000 km) should still produce valid coords."""
        init_reference(30.0, -97.0, 0.0)
        result = local_to_latlng(1_000_000, 0, 0)
        assert math.isfinite(result["lat"])
        assert math.isfinite(result["lng"])

    def test_negative_altitude(self):
        """Below sea level (e.g., Dead Sea at -430m) should work."""
        init_reference(31.5, 35.5, -430.0)
        result = local_to_latlng(0, 0, 0)
        assert result["alt"] == -430.0

    def test_latlng_to_local_roundtrip_near_pole(self):
        init_reference(85.0, 0.0, 0.0)
        lat, lng = 85.001, 0.001
        x, y, z = latlng_to_local(lat, lng, 0.0)
        back = local_to_latlng(x, y, z)
        assert abs(back["lat"] - lat) < 1e-5
        assert abs(back["lng"] - lng) < 1e-4
