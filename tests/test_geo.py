# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo — coordinate transforms and camera projection."""

import math
import pytest

from tritium_lib.geo import (
    METERS_PER_DEG_LAT,
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
