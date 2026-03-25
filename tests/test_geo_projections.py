# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo projection and coordinate system enhancements.

Covers: ProjectedCoordinate, Vincenty distance, UTM projection, geodetic
polygon area, and MGRS grid references.
"""

import math

import pytest

from tritium_lib.geo import (
    ProjectedCoordinate,
    distance_vincenty,
    utm_zone_from_latlng,
    latlng_to_utm,
    utm_to_latlng,
    polygon_area_geodetic,
    grid_reference,
    haversine_distance,
    compute_area_latlng,
    reset,
)


@pytest.fixture(autouse=True)
def _reset_geo():
    """Reset geo-reference before each test."""
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# ProjectedCoordinate
# ---------------------------------------------------------------------------

class TestProjectedCoordinate:
    def test_from_latlng_at_origin(self):
        """Converting the origin itself should give (0, 0, 0)."""
        pc = ProjectedCoordinate.from_latlng(
            30.0, -97.0, 100.0,
            origin_lat=30.0, origin_lng=-97.0, origin_alt=100.0,
        )
        assert abs(pc.x) < 1e-6
        assert abs(pc.y) < 1e-6
        assert abs(pc.z) < 1e-6

    def test_from_latlng_north_offset(self):
        """A point 1 degree north should be ~110 km in +Y."""
        pc = ProjectedCoordinate.from_latlng(
            31.0, -97.0,
            origin_lat=30.0, origin_lng=-97.0,
        )
        assert abs(pc.x) < 1.0  # should be very close to 0 east
        assert 110_000 < pc.y < 112_000  # ~111 km north

    def test_from_latlng_east_offset(self):
        """A point 1 degree east at the equator should be ~111 km in +X."""
        pc = ProjectedCoordinate.from_latlng(
            0.0, 1.0,
            origin_lat=0.0, origin_lng=0.0,
        )
        assert 111_000 < pc.x < 112_000
        assert abs(pc.y) < 1.0

    def test_roundtrip(self):
        """from_latlng -> to_latlng should recover the original coordinates."""
        lat, lng, alt = 37.7749, -122.4194, 16.0
        pc = ProjectedCoordinate.from_latlng(
            lat, lng, alt,
            origin_lat=37.77, origin_lng=-122.42, origin_alt=10.0,
        )
        lat2, lng2, alt2 = pc.to_latlng()
        assert abs(lat2 - lat) < 1e-6
        assert abs(lng2 - lng) < 1e-6
        assert abs(alt2 - alt) < 1e-3

    def test_distance_to(self):
        """Euclidean distance between two projected points."""
        p1 = ProjectedCoordinate(x=0.0, y=0.0, z=0.0)
        p2 = ProjectedCoordinate(x=3.0, y=4.0, z=0.0)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-10

    def test_distance_to_3d(self):
        """3D Euclidean distance."""
        p1 = ProjectedCoordinate(x=0.0, y=0.0, z=0.0)
        p2 = ProjectedCoordinate(x=1.0, y=2.0, z=2.0)
        assert abs(p1.distance_to(p2) - 3.0) < 1e-10

    def test_bearing_to_north(self):
        """Bearing from origin to a point due north should be 0."""
        p1 = ProjectedCoordinate(x=0.0, y=0.0)
        p2 = ProjectedCoordinate(x=0.0, y=100.0)
        assert abs(p1.bearing_to(p2) - 0.0) < 0.01

    def test_bearing_to_east(self):
        """Bearing from origin to a point due east should be 90."""
        p1 = ProjectedCoordinate(x=0.0, y=0.0)
        p2 = ProjectedCoordinate(x=100.0, y=0.0)
        assert abs(p1.bearing_to(p2) - 90.0) < 0.01

    def test_bearing_to_southwest(self):
        """Bearing to southwest should be ~225 degrees."""
        p1 = ProjectedCoordinate(x=0.0, y=0.0)
        p2 = ProjectedCoordinate(x=-100.0, y=-100.0)
        assert abs(p1.bearing_to(p2) - 225.0) < 0.01

    def test_altitude_preserved(self):
        """Altitude offset should be preserved in round-trip."""
        pc = ProjectedCoordinate.from_latlng(
            30.0, -97.0, 500.0,
            origin_lat=30.0, origin_lng=-97.0, origin_alt=100.0,
        )
        assert abs(pc.z - 400.0) < 1e-6
        _, _, alt = pc.to_latlng()
        assert abs(alt - 500.0) < 1e-3


# ---------------------------------------------------------------------------
# Vincenty distance
# ---------------------------------------------------------------------------

class TestVincenty:
    def test_coincident_points(self):
        """Distance between identical points should be 0."""
        d = distance_vincenty(37.0, -122.0, 37.0, -122.0)
        assert d == 0.0

    def test_one_degree_latitude_at_equator(self):
        """1 degree of latitude at the equator is ~110,574 m on WGS84."""
        d = distance_vincenty(0.0, 0.0, 1.0, 0.0)
        # Known WGS84 value: 110,574.389 m
        assert abs(d - 110_574.389) < 1.0

    def test_one_degree_longitude_at_equator(self):
        """1 degree of longitude at the equator is ~111,319 m on WGS84."""
        d = distance_vincenty(0.0, 0.0, 0.0, 1.0)
        assert abs(d - 111_319.49) < 5.0

    def test_symmetric(self):
        """Vincenty distance should be symmetric."""
        d1 = distance_vincenty(37.7749, -122.4194, 40.7128, -74.0060)
        d2 = distance_vincenty(40.7128, -74.0060, 37.7749, -122.4194)
        assert abs(d1 - d2) < 0.001

    def test_sf_to_nyc(self):
        """San Francisco to New York: known geodesic distance ~4,133 km."""
        d = distance_vincenty(37.7749, -122.4194, 40.7128, -74.0060)
        assert abs(d - 4_139_000) < 5_000  # within 5 km

    def test_london_to_paris(self):
        """London to Paris: ~341 km geodesic."""
        d = distance_vincenty(51.5074, -0.1278, 48.8566, 2.3522)
        assert abs(d - 341_000) < 3_000

    def test_more_precise_than_haversine(self):
        """Vincenty should differ from haversine by up to ~0.3% (ellipsoid vs sphere)."""
        lat1, lng1 = 51.5074, -0.1278  # London
        lat2, lng2 = -33.8688, 151.2093  # Sydney
        dv = distance_vincenty(lat1, lng1, lat2, lng2)
        dh = haversine_distance(lat1, lng1, lat2, lng2)
        # They should be close but not identical
        assert abs(dv - dh) / dv < 0.005  # within 0.5%
        # The difference should be non-trivial for long distances
        assert abs(dv - dh) > 1000  # at least 1 km difference

    def test_nearly_antipodal_fallback(self):
        """Nearly antipodal points where Vincenty may not converge should still work."""
        d = distance_vincenty(0.0, 0.0, 0.0, 179.999)
        half_circ = math.pi * 6_371_000.0
        assert abs(d - half_circ) < 10_000  # within 10 km of half circumference

    def test_equatorial_line(self):
        """Two points on the equator should give accurate distance."""
        d = distance_vincenty(0.0, 10.0, 0.0, 20.0)
        # 10 degrees of longitude at equator ~ 1,113,195 m
        assert abs(d - 1_113_195) < 100


# ---------------------------------------------------------------------------
# UTM zone detection
# ---------------------------------------------------------------------------

class TestUTMZone:
    def test_austin_texas(self):
        """Austin, TX should be in zone 14N."""
        zone, hemi = utm_zone_from_latlng(30.27, -97.74)
        assert zone == 14
        assert hemi == "N"

    def test_london(self):
        """London should be in zone 30N."""
        zone, hemi = utm_zone_from_latlng(51.5074, -0.1278)
        assert zone == 30
        assert hemi == "N"

    def test_sydney(self):
        """Sydney should be in zone 56S."""
        zone, hemi = utm_zone_from_latlng(-33.8688, 151.2093)
        assert zone == 56
        assert hemi == "S"

    def test_norway_exception(self):
        """Norway special zone: 56-64N, 3-12E should be zone 32."""
        zone, _ = utm_zone_from_latlng(60.0, 5.0)
        assert zone == 32

    def test_svalbard_exception(self):
        """Svalbard special zones at high latitudes."""
        zone, _ = utm_zone_from_latlng(75.0, 10.0)
        assert zone == 33

    def test_southern_hemisphere(self):
        """Points below equator should be 'S'."""
        _, hemi = utm_zone_from_latlng(-10.0, 0.0)
        assert hemi == "S"

    def test_out_of_range_raises(self):
        """Latitudes outside [-80, 84] should raise ValueError."""
        with pytest.raises(ValueError):
            utm_zone_from_latlng(-81.0, 0.0)
        with pytest.raises(ValueError):
            utm_zone_from_latlng(85.0, 0.0)

    def test_dateline(self):
        """Points near the dateline should resolve correctly."""
        zone, _ = utm_zone_from_latlng(0.0, 179.0)
        assert zone == 60
        zone, _ = utm_zone_from_latlng(0.0, -179.0)
        assert zone == 1


# ---------------------------------------------------------------------------
# UTM forward/inverse projection
# ---------------------------------------------------------------------------

class TestUTMProjection:
    def test_roundtrip_equator(self):
        """UTM roundtrip at the equator."""
        lat0, lng0 = 0.0, 3.0
        e, n, z, h = latlng_to_utm(lat0, lng0)
        lat, lng = utm_to_latlng(e, n, z, h)
        assert abs(lat - lat0) < 1e-6
        assert abs(lng - lng0) < 1e-6

    def test_roundtrip_midlatitude(self):
        """UTM roundtrip at mid-latitudes (Austin, TX)."""
        lat0, lng0 = 30.2672, -97.7431
        e, n, z, h = latlng_to_utm(lat0, lng0)
        lat, lng = utm_to_latlng(e, n, z, h)
        assert abs(lat - lat0) < 1e-6
        assert abs(lng - lng0) < 1e-6

    def test_roundtrip_high_latitude(self):
        """UTM roundtrip at high latitudes (Tromso, Norway at 69.6N)."""
        lat0, lng0 = 69.6489, 18.9551
        e, n, z, h = latlng_to_utm(lat0, lng0)
        lat, lng = utm_to_latlng(e, n, z, h)
        assert abs(lat - lat0) < 1e-5
        assert abs(lng - lng0) < 1e-5

    def test_roundtrip_southern_hemisphere(self):
        """UTM roundtrip in the southern hemisphere (Cape Town)."""
        lat0, lng0 = -33.9249, 18.4241
        e, n, z, h = latlng_to_utm(lat0, lng0)
        lat, lng = utm_to_latlng(e, n, z, h)
        assert abs(lat - lat0) < 1e-6
        assert abs(lng - lng0) < 1e-6

    def test_easting_at_central_meridian(self):
        """Easting at the central meridian should be 500,000 m (false easting)."""
        # Zone 30 central meridian is -3 degrees
        e, n, z, h = latlng_to_utm(0.0, -3.0)
        assert abs(e - 500_000) < 1.0

    def test_southern_hemisphere_false_northing(self):
        """Southern hemisphere should have northing offset by 10,000,000."""
        e_n, n_n, _, _ = latlng_to_utm(0.001, 3.0)  # just north of equator
        e_s, n_s, _, _ = latlng_to_utm(-0.001, 3.0)  # just south of equator
        # The northing for the southern point should be close to 10,000,000
        assert n_s > 9_999_000
        # Northern point should be small
        assert n_n < 1_000

    def test_zone_returned_correctly(self):
        """The returned zone and hemisphere should match zone detection."""
        lat, lng = 48.8566, 2.3522  # Paris
        e, n, z, h = latlng_to_utm(lat, lng)
        z2, h2 = utm_zone_from_latlng(lat, lng)
        assert z == z2
        assert h == h2


# ---------------------------------------------------------------------------
# Geodetic polygon area
# ---------------------------------------------------------------------------

class TestPolygonAreaGeodetic:
    def test_one_degree_square_equator(self):
        """A 1-degree square at the equator should be ~12,309 km^2."""
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        area = polygon_area_geodetic(polygon)
        assert abs(area - 12_309e6) < 50e6  # within 50 km^2

    def test_agrees_with_compute_area_latlng(self):
        """Geodetic area should closely match the existing tangent-plane method."""
        polygon = [
            (30.270, -97.750),
            (30.270, -97.749),
            (30.271, -97.749),
            (30.271, -97.750),
        ]
        a_geodetic = polygon_area_geodetic(polygon)
        a_tangent = compute_area_latlng(polygon)
        # For small polygons, both methods should agree within 1%
        assert abs(a_geodetic - a_tangent) / a_tangent < 0.01

    def test_high_latitude_smaller_area(self):
        """Same angular polygon near pole should have less area than at equator."""
        eq = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
        hi = [(70.0, 0.0), (70.1, 0.0), (70.1, 0.1), (70.0, 0.1)]
        assert polygon_area_geodetic(hi) < polygon_area_geodetic(eq)

    def test_degenerate_returns_zero(self):
        """Degenerate inputs should return 0."""
        assert polygon_area_geodetic([]) == 0.0
        assert polygon_area_geodetic([(0, 0)]) == 0.0
        assert polygon_area_geodetic([(0, 0), (1, 1)]) == 0.0

    def test_winding_order_independent(self):
        """Both CW and CCW winding should give the same area."""
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cw = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert abs(polygon_area_geodetic(ccw) - polygon_area_geodetic(cw)) < 1.0

    def test_large_polygon(self):
        """A large 10-degree polygon at mid-latitude."""
        polygon = [(30.0, -100.0), (40.0, -100.0), (40.0, -90.0), (30.0, -90.0)]
        area = polygon_area_geodetic(polygon)
        # ~10 degrees lat * ~10 degrees lng at ~35N
        # ~1111 km * ~910 km = ~1,011,000 km^2
        assert 900_000e6 < area < 1_100_000e6


# ---------------------------------------------------------------------------
# MGRS grid reference
# ---------------------------------------------------------------------------

class TestGridReference:
    def test_format_structure(self):
        """MGRS string should start with 2-digit zone, band letter, 2 grid letters."""
        ref = grid_reference(30.27, -97.74)
        assert len(ref) >= 7  # zone(2) + band(1) + grid(2) + digits
        assert ref[:2].isdigit()
        assert ref[2].isalpha()
        assert ref[3].isalpha()
        assert ref[4].isalpha()

    def test_precision_1(self):
        """Precision 1 should give 2 digits (10 km resolution)."""
        ref = grid_reference(30.27, -97.74, precision=1)
        # Format: ZZBCCddddd -> zone(2) + band(1) + grid(2) + 2 digits
        digits_part = ref[5:]
        assert len(digits_part) == 2

    def test_precision_5(self):
        """Precision 5 should give 10 digits (1 m resolution)."""
        ref = grid_reference(30.27, -97.74, precision=5)
        digits_part = ref[5:]
        assert len(digits_part) == 10

    def test_different_locations_differ(self):
        """Different locations should produce different grid references."""
        ref1 = grid_reference(30.27, -97.74)
        ref2 = grid_reference(51.5074, -0.1278)
        assert ref1 != ref2

    def test_close_points_share_prefix(self):
        """Points very close together should share the MGRS prefix."""
        ref1 = grid_reference(30.270, -97.740, precision=3)
        ref2 = grid_reference(30.271, -97.740, precision=3)
        # They should share at least the zone/band/grid square
        assert ref1[:5] == ref2[:5]

    def test_invalid_precision_raises(self):
        """Precision outside 1-5 should raise ValueError."""
        with pytest.raises(ValueError):
            grid_reference(30.0, -97.0, precision=0)
        with pytest.raises(ValueError):
            grid_reference(30.0, -97.0, precision=6)

    def test_out_of_range_latitude_raises(self):
        """Latitudes outside UTM range should raise ValueError."""
        with pytest.raises(ValueError):
            grid_reference(-85.0, 0.0)
        with pytest.raises(ValueError):
            grid_reference(85.0, 0.0)

    def test_southern_hemisphere(self):
        """Should work for southern hemisphere locations."""
        ref = grid_reference(-33.87, 151.21)  # Sydney
        assert ref[:2].isdigit()
        zone_num = int(ref[:2])
        assert zone_num == 56

    def test_deterministic(self):
        """Same input should always produce the same output."""
        ref1 = grid_reference(30.27, -97.74)
        ref2 = grid_reference(30.27, -97.74)
        assert ref1 == ref2

    def test_zone_number_in_reference(self):
        """The zone number in the MGRS string should match UTM zone."""
        lat, lng = 48.8566, 2.3522  # Paris
        ref = grid_reference(lat, lng)
        zone, _ = utm_zone_from_latlng(lat, lng)
        assert int(ref[:2]) == zone
