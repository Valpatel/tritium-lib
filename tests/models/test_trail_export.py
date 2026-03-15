# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TrailExport and TrailPoint models."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models.trail_export import (
    TrailExport,
    TrailFormat,
    TrailPoint,
)


class TestTrailPoint:
    def test_basic_point(self):
        pt = TrailPoint(lat=33.0, lng=-117.0)
        assert pt.lat == 33.0
        assert pt.lng == -117.0
        assert pt.alt is None

    def test_full_point(self):
        t = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        pt = TrailPoint(
            lat=33.0, lng=-117.0, alt=100.0,
            timestamp=t, speed=5.0, heading=90.0,
            confidence=0.85, source="ble",
        )
        assert pt.speed == 5.0
        assert pt.heading == 90.0
        assert pt.confidence == 0.85
        assert pt.source == "ble"

    def test_to_dict(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        pt = TrailPoint(lat=1.0, lng=2.0, alt=3.0, timestamp=t)
        d = pt.to_dict()
        assert d["lat"] == 1.0
        assert d["lng"] == 2.0
        assert d["alt"] == 3.0
        assert "2026" in d["timestamp"]

    def test_from_dict_basic(self):
        d = {"lat": 10.0, "lng": 20.0}
        pt = TrailPoint.from_dict(d)
        assert pt.lat == 10.0
        assert pt.lng == 20.0

    def test_from_dict_with_lon_alias(self):
        d = {"lat": 10.0, "lon": 20.0}
        pt = TrailPoint.from_dict(d)
        assert pt.lng == 20.0

    def test_from_dict_with_string_time(self):
        d = {"lat": 0.0, "lng": 0.0, "timestamp": "2026-03-14T12:00:00+00:00"}
        pt = TrailPoint.from_dict(d)
        assert pt.timestamp is not None
        assert pt.timestamp.year == 2026

    def test_from_dict_with_epoch_time(self):
        d = {"lat": 0.0, "lng": 0.0, "timestamp": 1773768000.0}
        pt = TrailPoint.from_dict(d)
        assert pt.timestamp is not None

    def test_from_dict_with_time_key(self):
        d = {"lat": 0.0, "lng": 0.0, "time": "2026-03-14T12:00:00+00:00"}
        pt = TrailPoint.from_dict(d)
        assert pt.timestamp is not None

    def test_from_dict_altitude_aliases(self):
        for key in ["alt", "altitude", "ele"]:
            d = {"lat": 0.0, "lng": 0.0, key: 150.0}
            pt = TrailPoint.from_dict(d)
            assert pt.alt == 150.0

    def test_roundtrip(self):
        t = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)
        pt = TrailPoint(lat=33.0, lng=-117.0, alt=50.0, timestamp=t, speed=3.5)
        d = pt.to_dict()
        pt2 = TrailPoint.from_dict(d)
        assert pt2.lat == pt.lat
        assert pt2.lng == pt.lng
        assert pt2.alt == pt.alt
        assert pt2.speed == pt.speed


class TestTrailFormat:
    def test_enum_values(self):
        assert TrailFormat.GPX.value == "gpx"
        assert TrailFormat.KML.value == "kml"
        assert TrailFormat.GEOJSON.value == "geojson"
        assert TrailFormat.CSV.value == "csv"
        assert TrailFormat.JSON.value == "json"


class TestTrailExport:
    def _make_trail(self, n=5):
        points = []
        for i in range(n):
            t = datetime(2026, 3, 14, 10, i, 0, tzinfo=timezone.utc)
            points.append(TrailPoint(
                lat=33.0 + i * 0.001,
                lng=-117.0 + i * 0.001,
                timestamp=t,
                speed=float(i),
            ))
        return TrailExport(
            target_id="ble_AA:BB:CC:DD:EE:FF",
            format=TrailFormat.GPX,
            points=points,
            metadata={"name": "Test Phone", "alliance": "unknown"},
        )

    def test_basic_properties(self):
        trail = self._make_trail()
        assert trail.target_id == "ble_AA:BB:CC:DD:EE:FF"
        assert trail.point_count == 5
        assert trail.format == TrailFormat.GPX

    def test_duration_seconds(self):
        trail = self._make_trail()
        dur = trail.duration_seconds
        assert dur is not None
        assert dur == 240.0  # 4 minutes

    def test_duration_no_timestamps(self):
        trail = TrailExport(
            target_id="test",
            points=[TrailPoint(lat=0, lng=0), TrailPoint(lat=1, lng=1)],
        )
        assert trail.duration_seconds is None

    def test_total_distance(self):
        trail = self._make_trail()
        dist = trail.total_distance_m
        assert dist > 0  # some positive distance

    def test_empty_distance(self):
        trail = TrailExport(target_id="test", points=[])
        assert trail.total_distance_m == 0.0

    def test_simplify(self):
        # Create a trail with many collinear points that can be simplified
        points = []
        for i in range(100):
            points.append(TrailPoint(lat=33.0 + i * 0.0001, lng=-117.0 + i * 0.0001))
        trail = TrailExport(target_id="test", points=points)
        simplified = trail.simplify(tolerance=0.001)
        # Collinear points should be massively reduced
        assert simplified.point_count < trail.point_count
        assert simplified.point_count >= 2  # at least start and end

    def test_simplify_short_trail(self):
        trail = TrailExport(
            target_id="test",
            points=[TrailPoint(lat=0, lng=0), TrailPoint(lat=1, lng=1)],
        )
        simplified = trail.simplify()
        assert simplified.point_count == 2

    def test_simplify_preserves_metadata(self):
        trail = self._make_trail()
        trail.metadata = {"alliance": "hostile"}
        simplified = trail.simplify()
        assert simplified.target_id == trail.target_id
        assert simplified.metadata == trail.metadata

    def test_to_dict(self):
        trail = self._make_trail(3)
        d = trail.to_dict()
        assert d["target_id"] == "ble_AA:BB:CC:DD:EE:FF"
        assert d["format"] == "gpx"
        assert d["point_count"] == 3
        assert len(d["points"]) == 3
        assert "duration_seconds" in d
        assert "total_distance_m" in d

    def test_from_dict_roundtrip(self):
        trail = self._make_trail(3)
        d = trail.to_dict()
        trail2 = TrailExport.from_dict(d)
        assert trail2.target_id == trail.target_id
        assert trail2.format == trail.format
        assert trail2.point_count == trail.point_count

    def test_from_dict_unknown_format(self):
        d = {"target_id": "test", "format": "invalid_format", "points": []}
        trail = TrailExport.from_dict(d)
        assert trail.format == TrailFormat.GPX  # defaults to GPX

    def test_kml_format(self):
        trail = TrailExport(
            target_id="test",
            format=TrailFormat.KML,
            points=[TrailPoint(lat=0, lng=0)],
        )
        assert trail.format == TrailFormat.KML
        d = trail.to_dict()
        assert d["format"] == "kml"
