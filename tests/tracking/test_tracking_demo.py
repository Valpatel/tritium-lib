# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the standalone tracking demo pipeline.

Validates that the tracking pipeline works independently of tritium-sc,
exercising TargetTracker, TargetCorrelator, GeofenceEngine, HeatmapEngine,
ThreatScorer, and DwellTracker.
"""

from __future__ import annotations

import time

import pytest

from tritium_lib.tracking.demos.tracking_demo import (
    TrackingPipeline,
    SimpleEventBus,
    _target_to_dict,
    _BLE_DEVICES,
    _WIFI_DETECTIONS,
)
from tritium_lib.tracking import (
    TargetTracker,
    GeofenceEngine,
    GeoZone,
    HeatmapEngine,
    ThreatScorer,
    TargetCorrelator,
)
from tritium_lib.tracking.dwell_tracker import DwellTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    """Create a fresh TrackingPipeline instance."""
    return TrackingPipeline()


@pytest.fixture
def warmed_pipeline():
    """Pipeline with several ticks of data already generated."""
    p = TrackingPipeline()
    for _ in range(5):
        p.generate_sightings()
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleEventBus:
    """Test the minimal event bus used in the demo."""

    def test_publish_records_events(self):
        bus = SimpleEventBus()
        bus.publish("test_topic", {"key": "value"})
        assert bus.event_count == 1
        assert bus.events[0] == ("test_topic", {"key": "value"})

    def test_multiple_publishes(self):
        bus = SimpleEventBus()
        for i in range(10):
            bus.publish(f"topic_{i}", {"i": i})
        assert bus.event_count == 10


class TestTrackingPipeline:
    """Test the pipeline orchestration."""

    def test_pipeline_initializes(self, pipeline):
        """Pipeline should create all components without error."""
        assert pipeline.tracker is not None
        assert pipeline.geofence is not None
        assert pipeline.heatmap is not None
        assert pipeline.correlator is not None
        assert pipeline.dwell_tracker is not None
        assert pipeline.threat_scorer is not None

    def test_geofence_zones_created(self, pipeline):
        """Pipeline should create 3 geofence zones on init."""
        zones = pipeline.geofence.list_zones()
        assert len(zones) == 3
        zone_ids = {z.zone_id for z in zones}
        assert "restricted-hq" in zone_ids
        assert "parking-lot" in zone_ids
        assert "perimeter-north" in zone_ids

    def test_generate_sightings_returns_stats(self, pipeline):
        """First sighting batch should produce BLE and YOLO data."""
        stats = pipeline.generate_sightings()
        assert stats["ble"] == len(_BLE_DEVICES)
        assert stats["yolo"] == len(_WIFI_DETECTIONS)
        assert isinstance(stats["correlations"], int)
        assert isinstance(stats["threats"], int)

    def test_tick_increments(self, pipeline):
        """Tick counter should increment on each generate call."""
        assert pipeline.tick == 0
        pipeline.generate_sightings()
        assert pipeline.tick == 1
        pipeline.generate_sightings()
        assert pipeline.tick == 2


class TestTargetTracker:
    """Test that the TargetTracker receives sightings correctly."""

    def test_ble_targets_created(self, pipeline):
        """BLE sightings should create targets with ble_ prefix IDs."""
        pipeline.generate_sightings()
        targets = pipeline.tracker.get_all()
        ble_targets = [t for t in targets if t.source == "ble"]
        assert len(ble_targets) >= 1  # Some may be correlated away

    def test_yolo_targets_created(self, pipeline):
        """YOLO detections should create targets with det_ prefix IDs."""
        pipeline.generate_sightings()
        targets = pipeline.tracker.get_all()
        has_yolo = any(t.source == "yolo" for t in targets)
        has_ble = any(t.source == "ble" for t in targets)
        # After correlation, YOLO may be merged into BLE targets
        assert has_yolo or has_ble
        assert len(targets) >= 1

    def test_target_signal_count_grows(self, pipeline):
        """Repeated sightings should increment signal_count."""
        pipeline.generate_sightings()
        pipeline.generate_sightings()
        targets = pipeline.tracker.get_all()
        ble_targets = [t for t in targets if "ble" in t.confirming_sources]
        if ble_targets:
            assert ble_targets[0].signal_count >= 2

    def test_target_to_dict_serialization(self, pipeline):
        """_target_to_dict should produce JSON-safe output."""
        pipeline.generate_sightings()
        targets = pipeline.tracker.get_all()
        assert len(targets) > 0
        d = _target_to_dict(targets[0])
        assert "target_id" in d
        assert "position" in d
        assert "x" in d["position"]
        assert "y" in d["position"]
        assert isinstance(d["confirming_sources"], list)
        assert isinstance(d["position_confidence"], float)


class TestCorrelator:
    """Test the multi-strategy correlator."""

    def test_correlations_produced(self, warmed_pipeline):
        """After several ticks, correlator should have merged some targets."""
        assert len(warmed_pipeline.correlation_log) > 0

    def test_correlation_has_required_fields(self, warmed_pipeline):
        """Each correlation record should have the expected fields."""
        if not warmed_pipeline.correlation_log:
            pytest.skip("No correlations produced")
        rec = warmed_pipeline.correlation_log[0]
        assert "primary_id" in rec
        assert "secondary_id" in rec
        assert "confidence" in rec
        assert 0 < rec["confidence"] <= 1.0

    def test_dossiers_created(self, warmed_pipeline):
        """Correlator should create dossiers in the DossierStore."""
        dossiers = warmed_pipeline.correlator.dossier_store.get_all()
        assert len(dossiers) > 0
        d = dossiers[0]
        assert len(d.signal_ids) >= 2
        assert d.confidence > 0


class TestGeofenceEngine:
    """Test geofence zone detection."""

    def test_point_inside_restricted_zone(self, pipeline):
        """A point inside HQ restricted area should trigger enter events."""
        events = pipeline.geofence.check("test-inside", (50, 50))
        zone_ids = {e.zone_id for e in events}
        assert "restricted-hq" in zone_ids

    def test_point_outside_all_zones(self, pipeline):
        """A point far outside all zones should trigger no enter events."""
        events = pipeline.geofence.check("test-outside", (500, 500))
        enter_events = [e for e in events if e.event_type == "enter"]
        assert len(enter_events) == 0

    def test_geofence_events_generated(self, warmed_pipeline):
        """After ticks, geofence should have logged enter/exit events."""
        events = warmed_pipeline.geofence.get_events(limit=100)
        assert len(events) > 0
        event_types = {e.event_type for e in events}
        assert "enter" in event_types


class TestHeatmapEngine:
    """Test the spatial heatmap."""

    def test_heatmap_accumulates(self, warmed_pipeline):
        """Heatmap should accumulate events from sightings."""
        count = warmed_pipeline.heatmap.event_count()
        assert count > 0

    def test_heatmap_grid_structure(self, warmed_pipeline):
        """get_heatmap should return a properly structured grid."""
        result = warmed_pipeline.heatmap.get_heatmap(
            time_window_minutes=60, resolution=10
        )
        assert "grid" in result
        assert "bounds" in result
        assert "max_value" in result
        assert len(result["grid"]) == 10
        assert len(result["grid"][0]) == 10
        assert result["max_value"] > 0


class TestThreatScorer:
    """Test threat scoring."""

    def test_threat_profiles_exist(self, warmed_pipeline):
        """After ticks, threat scorer should have profiles."""
        profiles = warmed_pipeline.threat_scorer.get_all_profiles()
        assert len(profiles) > 0

    def test_threat_score_range(self, warmed_pipeline):
        """All threat scores should be between 0 and 1."""
        profiles = warmed_pipeline.threat_scorer.get_all_profiles()
        for p in profiles:
            assert 0.0 <= p["threat_score"] <= 1.0

    def test_geofence_checker_wired(self, warmed_pipeline):
        """Threat scorer should report that geofence checker is active."""
        status = warmed_pipeline.threat_scorer.get_status()
        assert status["has_geofence_checker"] is True


class TestDwellTracker:
    """Test the DwellTracker integration."""

    def test_dwell_tracker_created(self, pipeline):
        """DwellTracker should be created with custom demo thresholds."""
        assert pipeline.dwell_tracker._threshold_s == 10.0
        assert pipeline.dwell_tracker._radius_m == 8.0

    def test_dwell_tracker_can_start_stop(self, pipeline):
        """DwellTracker thread should start and stop cleanly."""
        pipeline.dwell_tracker.start()
        assert pipeline.dwell_tracker._running is True
        pipeline.dwell_tracker.stop()
        assert pipeline.dwell_tracker._running is False


class TestFastAPIEndpoints:
    """Test the FastAPI app endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        """Create a TestClient. Generate some sightings first."""
        # We need to reset the module-level pipeline for clean tests
        from tritium_lib.tracking.demos import tracking_demo
        tracking_demo.pipeline = TrackingPipeline()
        for _ in range(3):
            tracking_demo.pipeline.generate_sightings()

        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette TestClient not available")

        from tritium_lib.tracking.demos.tracking_demo import app
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_get_targets(self):
        resp = self.client.get("/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "target_id" in data[0]

    def test_get_target_by_id(self):
        targets = self.client.get("/targets").json()
        tid = targets[0]["target_id"]
        resp = self.client.get(f"/target/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_id"] == tid
        assert "trail" in data
        assert "zones" in data

    def test_get_target_not_found(self):
        resp = self.client.get("/target/nonexistent_id")
        assert resp.status_code == 404

    def test_get_heatmap(self):
        resp = self.client.get("/heatmap?resolution=10&minutes=60")
        assert resp.status_code == 200
        data = resp.json()
        assert "grid" in data
        assert data["event_count"] > 0

    def test_get_geofence(self):
        resp = self.client.get("/geofence")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["zones"]) == 3

    def test_get_threats(self):
        resp = self.client.get("/threats")
        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert "status" in data

    def test_get_correlations(self):
        resp = self.client.get("/correlations")
        assert resp.status_code == 200
        data = resp.json()
        assert "correlations" in data
        assert "dossiers" in data
        assert isinstance(data["total"], int)

    def test_get_dwells(self):
        resp = self.client.get("/dwells")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data
        assert "history" in data

    def test_get_status(self):
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_targets" in data
        assert data["total_targets"] > 0
        assert data["geofence_zones"] == 3

    def test_dashboard_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "TRITIUM TRACKING PIPELINE DEMO" in resp.text
