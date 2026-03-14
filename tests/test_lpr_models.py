# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for LPR (License Plate Recognition) models."""

import time

from tritium_lib.models.lpr import (
    LPRStats,
    PlateAlert,
    PlateColor,
    PlateDetection,
    PlateRecord,
    PlateRegion,
    PlateWatchEntry,
    PlateWatchlist,
)


class TestPlateDetection:
    def test_basic_creation(self):
        d = PlateDetection(plate_text="ABC1234", confidence=0.95)
        assert d.plate_text == "ABC1234"
        assert d.confidence == 0.95
        assert d.region == PlateRegion.UNKNOWN

    def test_compute_target_id(self):
        d = PlateDetection(plate_text="AB-C 1234")
        assert d.compute_target_id() == "lpr_ABC1234"

    def test_to_target_dict(self):
        d = PlateDetection(
            plate_text="XYZ789",
            vehicle_type="truck",
            camera_id="cam01",
            confidence=0.88,
        )
        td = d.to_target_dict()
        assert td["target_id"] == "lpr_XYZ789"
        assert td["source"] == "lpr"
        assert td["classification"] == "vehicle"
        assert td["metadata"]["vehicle_type"] == "truck"

    def test_plate_regions(self):
        for region in PlateRegion:
            d = PlateDetection(plate_text="TEST", region=region)
            assert d.region == region

    def test_plate_colors(self):
        for color in PlateColor:
            d = PlateDetection(plate_text="TEST", plate_color=color)
            assert d.plate_color == color

    def test_vehicle_context(self):
        d = PlateDetection(
            plate_text="ABC",
            vehicle_type="car",
            vehicle_color="blue",
            vehicle_confidence=0.92,
            vehicle_bbox_x=100,
            vehicle_bbox_y=200,
            vehicle_bbox_w=300,
            vehicle_bbox_h=150,
        )
        assert d.vehicle_type == "car"
        assert d.vehicle_color == "blue"
        assert d.vehicle_confidence == 0.92


class TestPlateWatchlist:
    def test_empty_watchlist(self):
        wl = PlateWatchlist(name="test")
        assert wl.check_plate("ABC123") is None

    def test_check_plate_found(self):
        entry = PlateWatchEntry(
            plate_text="ABC123",
            alert_type=PlateAlert.STOLEN,
            description="stolen vehicle",
        )
        wl = PlateWatchlist(name="test", entries=[entry])
        result = wl.check_plate("ABC123")
        assert result is not None
        assert result.alert_type == PlateAlert.STOLEN

    def test_check_plate_normalized(self):
        entry = PlateWatchEntry(plate_text="AB-C 123")
        wl = PlateWatchlist(entries=[entry])
        assert wl.check_plate("ABC123") is not None
        assert wl.check_plate("abc123") is not None
        assert wl.check_plate("AB C123") is not None

    def test_entry_expiry(self):
        entry = PlateWatchEntry(
            plate_text="OLD",
            expires_at=time.time() - 3600,
        )
        assert entry.is_expired is True

        entry2 = PlateWatchEntry(
            plate_text="FUTURE",
            expires_at=time.time() + 3600,
        )
        assert entry2.is_expired is False

        entry3 = PlateWatchEntry(plate_text="FOREVER")
        assert entry3.is_expired is False


class TestPlateRecord:
    def test_basic(self):
        r = PlateRecord(
            plate_text="ABC123",
            camera_id="cam1",
            latitude=37.7749,
            longitude=-122.4194,
        )
        assert r.plate_text == "ABC123"
        assert r.latitude == 37.7749


class TestLPRStats:
    def test_defaults(self):
        s = LPRStats()
        assert s.total_detections == 0
        assert s.unique_plates == 0

    def test_populated(self):
        s = LPRStats(
            total_detections=100,
            unique_plates=45,
            watchlist_hits=3,
            avg_confidence=0.87,
            detections_per_camera={"cam1": 60, "cam2": 40},
        )
        assert s.total_detections == 100
        assert len(s.detections_per_camera) == 2
