# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.heatmap."""

import time
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.heatmap import (
    HeatmapEngine,
    HeatmapEvent,
    VALID_LAYERS,
    DEFAULT_RETENTION_SECONDS,
)


class TestHeatmapEvent:
    def test_fields(self):
        e = HeatmapEvent(layer="ble_activity", x=1.0, y=2.0, weight=0.5)
        assert e.layer == "ble_activity"
        assert e.x == 1.0
        assert e.y == 2.0
        assert e.weight == 0.5


class TestRecordEvent:
    def test_record_valid_layer(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 10.0, 20.0)
        assert eng.event_count("ble_activity") == 1

    def test_record_invalid_layer_raises(self):
        eng = HeatmapEngine()
        with pytest.raises(ValueError, match="Invalid layer"):
            eng.record_event("nonexistent_layer", 0.0, 0.0)

    def test_record_multiple_layers(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0)
        eng.record_event("camera_activity", 1.0, 1.0)
        eng.record_event("motion_activity", 2.0, 2.0)
        assert eng.event_count("all") == 3

    def test_record_with_custom_weight(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 5.0, 5.0, weight=3.0)
        result = eng.get_heatmap(time_window_minutes=9999, layer="ble_activity")
        assert result["max_value"] == 3.0

    def test_record_with_custom_timestamp(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=1000.0)
        assert eng.event_count("ble_activity") == 1


class TestGetHeatmap:
    def test_empty_heatmap(self):
        eng = HeatmapEngine()
        result = eng.get_heatmap()
        assert result["event_count"] == 0
        assert result["max_value"] == 0.0

    def test_heatmap_with_events(self):
        eng = HeatmapEngine()
        now = time.time()
        eng.record_event("ble_activity", 10.0, 20.0, timestamp=now)
        eng.record_event("ble_activity", 10.5, 20.5, timestamp=now)
        result = eng.get_heatmap(time_window_minutes=60, layer="ble_activity")
        assert result["event_count"] == 2
        assert result["max_value"] > 0

    def test_heatmap_resolution(self):
        eng = HeatmapEngine()
        now = time.time()
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=now)
        result = eng.get_heatmap(resolution=25, layer="ble_activity", time_window_minutes=9999)
        assert result["resolution"] == 25
        assert len(result["grid"]) == 25
        assert len(result["grid"][0]) == 25

    def test_heatmap_all_layers(self):
        eng = HeatmapEngine()
        now = time.time()
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=now)
        eng.record_event("camera_activity", 1.0, 1.0, timestamp=now)
        result = eng.get_heatmap(layer="all", time_window_minutes=9999)
        assert result["event_count"] == 2

    def test_heatmap_time_window_filters_old(self):
        eng = HeatmapEngine()
        old_time = time.time() - 7200  # 2 hours ago
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=old_time)
        result = eng.get_heatmap(time_window_minutes=60, layer="ble_activity")
        assert result["event_count"] == 0


class TestGetTimeline:
    def test_timeline_empty(self):
        eng = HeatmapEngine()
        result = eng.get_timeline(start=0, end=100, buckets=5)
        assert len(result["frames"]) == 5
        assert result["global_max"] == 0.0

    def test_timeline_with_events(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=50.0)
        result = eng.get_timeline(start=0, end=100, buckets=2)
        assert len(result["frames"]) == 2
        total_events = sum(f["event_count"] for f in result["frames"])
        assert total_events == 1


class TestPrune:
    def test_prune_old_events(self):
        eng = HeatmapEngine()
        old_time = time.time() - 100000
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=old_time)
        eng.record_event("ble_activity", 1.0, 1.0, timestamp=time.time())
        removed = eng.prune()
        assert removed == 1
        assert eng.event_count("ble_activity") == 1

    def test_prune_with_custom_cutoff(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0, timestamp=100.0)
        eng.record_event("ble_activity", 1.0, 1.0, timestamp=200.0)
        removed = eng.prune(before=150.0)
        assert removed == 1


class TestClear:
    def test_clear_all(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0)
        eng.record_event("camera_activity", 1.0, 1.0)
        eng.clear()
        assert eng.event_count("all") == 0

    def test_clear_specific_layer(self):
        eng = HeatmapEngine()
        eng.record_event("ble_activity", 0.0, 0.0)
        eng.record_event("camera_activity", 1.0, 1.0)
        eng.clear("ble_activity")
        assert eng.event_count("ble_activity") == 0
        assert eng.event_count("camera_activity") == 1


class TestEventCount:
    def test_count_per_layer(self):
        eng = HeatmapEngine()
        for i in range(5):
            eng.record_event("ble_activity", float(i), 0.0)
        for i in range(3):
            eng.record_event("camera_activity", float(i), 0.0)
        assert eng.event_count("ble_activity") == 5
        assert eng.event_count("camera_activity") == 3
        assert eng.event_count("all") == 8
        assert eng.event_count("motion_activity") == 0
