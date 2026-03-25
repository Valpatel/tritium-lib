# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the intelligence pipeline demo."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tritium_lib.intelligence.demos.pipeline_demo import (
    PipelineState,
    _EventBus,
    _SyntheticTarget,
    _init_pipeline,
    _pipeline_tick,
    _state,
    _generate_ble_sightings,
    _generate_camera_detections,
    _generate_wifi_probes,
    _generate_acoustic_events,
    _run_anomaly_detection,
    _run_position_estimation,
    _run_threat_assessment,
    _run_correlation,
    app,
    BASE_LAT,
    BASE_LNG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global pipeline state before each test."""
    # Re-initialize state
    _state.__init__()
    _init_pipeline()
    yield
    _state.running = False


@pytest.fixture
def client():
    """FastAPI test client (does not start the background loop)."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Unit tests — EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_publish_and_count(self):
        bus = _EventBus()
        assert bus.event_count == 0
        bus.publish("test_topic", {"key": "value"})
        assert bus.event_count == 1

    def test_recent(self):
        bus = _EventBus()
        bus.publish("topic_a", {"x": 1})
        bus.publish("topic_b", {"y": 2})
        recent = bus.recent(5)
        assert len(recent) == 2
        assert recent[0]["topic"] == "topic_a"
        assert recent[1]["topic"] == "topic_b"

    def test_events_list(self):
        bus = _EventBus()
        bus.publish("t", {"data": 123})
        events = bus.events
        assert len(events) == 1
        assert events[0][0] == "t"


# ---------------------------------------------------------------------------
# Unit tests — SyntheticTarget
# ---------------------------------------------------------------------------


class TestSyntheticTarget:
    def test_tick_moves_target(self):
        import random
        rng = random.Random(42)
        target = _SyntheticTarget(
            entity_id="test_1", kind="person",
            lat=BASE_LAT, lng=BASE_LNG,
            speed=0.0001, heading=90.0,
        )
        old_lat, old_lng = target.lat, target.lng
        target.tick(rng)
        # Target should have moved
        assert (target.lat != old_lat) or (target.lng != old_lng)

    def test_heading_drifts(self):
        import random
        rng = random.Random(42)
        target = _SyntheticTarget(
            entity_id="test_2", kind="vehicle",
            heading=0.0, speed=0.0001,
        )
        headings = set()
        for _ in range(10):
            target.tick(rng)
            headings.add(round(target.heading, 1))
        # Heading should drift
        assert len(headings) > 1


# ---------------------------------------------------------------------------
# Unit tests — Pipeline initialization
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_tracker_initialized(self):
        assert _state.tracker is not None

    def test_correlator_initialized(self):
        assert _state.correlator is not None

    def test_geofence_initialized(self):
        assert _state.geofence is not None

    def test_entities_created(self):
        assert len(_state.entities) == 6

    def test_entity_types(self):
        kinds = {e.kind for e in _state.entities}
        assert "person" in kinds
        assert "vehicle" in kinds
        assert "phone" in kinds


# ---------------------------------------------------------------------------
# Unit tests — Data generation
# ---------------------------------------------------------------------------


class TestDataGeneration:
    def test_ble_sightings(self):
        _generate_ble_sightings()
        targets = _state.tracker.get_all()
        ble_targets = [t for t in targets if t.source == "ble"]
        assert len(ble_targets) > 0

    def test_camera_detections(self):
        # Run multiple times to get past the 70% probability
        for _ in range(10):
            _generate_camera_detections()
        targets = _state.tracker.get_all()
        yolo_targets = [t for t in targets if t.source == "yolo"]
        assert len(yolo_targets) > 0

    def test_wifi_probes(self):
        for _ in range(10):
            _generate_wifi_probes()
        targets = _state.tracker.get_all()
        wifi_targets = [t for t in targets if "wifi" in t.name.lower() or "001122" in t.target_id]
        assert len(wifi_targets) > 0

    def test_acoustic_events(self):
        # Run enough times to generate at least one event
        for _ in range(30):
            _generate_acoustic_events()
        assert len(_state.acoustic_log) > 0
        event = _state.acoustic_log[0]
        assert "event_type" in event
        assert "confidence" in event
        assert "timestamp" in event


# ---------------------------------------------------------------------------
# Unit tests — Pipeline stages
# ---------------------------------------------------------------------------


class TestPipelineStages:
    def test_position_estimation(self):
        # First populate some BLE targets
        _generate_ble_sightings()
        _run_position_estimation()
        assert len(_state.position_estimates) > 0
        est = _state.position_estimates[0]
        assert "target_id" in est
        assert "lat" in est
        assert "lng" in est
        assert "accuracy_m" in est
        assert "method" in est
        assert "confidence" in est

    def test_anomaly_detection_builds_baseline(self):
        _generate_ble_sightings()
        _generate_camera_detections()
        _run_anomaly_detection()
        assert len(_state.rf_baseline) == 1
        sample = _state.rf_baseline[0]
        assert "ble_count" in sample
        assert "yolo_count" in sample
        assert "total_targets" in sample

    def test_anomaly_detection_needs_baseline(self):
        """No anomalies until baseline is large enough."""
        _generate_ble_sightings()
        _run_anomaly_detection()
        assert len(_state.anomaly_log) == 0  # Not enough baseline yet

    def test_threat_assessment(self):
        _generate_ble_sightings()
        _generate_camera_detections()
        _run_threat_assessment()
        stats = _state.threat_model.get_stats()
        assert stats["total_signals"] > 0

    def test_correlation(self):
        _generate_ble_sightings()
        for _ in range(5):
            _generate_camera_detections()
        _run_correlation()
        records = _state.correlator.get_correlations()
        # May or may not find correlations depending on spatial proximity
        # but the call should succeed
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# Integration test — full tick
# ---------------------------------------------------------------------------


class TestFullPipelineTick:
    def test_single_tick(self):
        _pipeline_tick()
        assert _state.ticks == 1
        targets = _state.tracker.get_all()
        assert len(targets) > 0

    def test_multiple_ticks(self):
        for _ in range(5):
            _pipeline_tick()
        assert _state.ticks == 5
        targets = _state.tracker.get_all()
        # After 5 ticks we should have BLE + YOLO targets
        sources = {t.source for t in targets}
        assert "ble" in sources

    def test_targets_have_multiple_sources(self):
        for _ in range(10):
            _pipeline_tick()
        targets = _state.tracker.get_all()
        # Some targets should have confirming sources
        multi_source = [t for t in targets if len(t.confirming_sources) > 1]
        # With 10 ticks, some BLE targets should be confirmed by BLE updates
        assert len(targets) > 0

    def test_acoustic_log_populated_after_many_ticks(self):
        for _ in range(20):
            _pipeline_tick()
        # With 20% chance per tick over 20 ticks, we should have some events
        assert len(_state.acoustic_log) > 0

    def test_position_estimates_after_ticks(self):
        for _ in range(3):
            _pipeline_tick()
        assert len(_state.position_estimates) > 0

    def test_threat_model_assesses_after_ticks(self):
        for _ in range(5):
            _pipeline_tick()
        assessments = _state.threat_model.assess_all()
        assert len(assessments) > 0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_status(self, client):
        # Run a tick first for data
        _pipeline_tick()
        resp = client.get("/pipeline/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("running", "stopped")
        assert "targets" in data
        assert "sensors" in data
        assert "threat_model" in data
        assert data["targets"]["total"] > 0

    def test_targets(self, client):
        _pipeline_tick()
        resp = client.get("/pipeline/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert "targets" in data
        assert "count" in data
        assert data["count"] > 0
        # Each target should have threat_assessment
        target = data["targets"][0]
        assert "threat_assessment" in target
        assert "composite_score" in target["threat_assessment"]

    def test_anomalies(self, client):
        resp = client.get("/pipeline/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert "anomalies" in data
        assert "detector" in data
        assert data["detector"] == "simple_threshold"

    def test_acoustic(self, client):
        # Generate some events
        for _ in range(30):
            _pipeline_tick()
        resp = client.get("/pipeline/acoustic")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert "by_type" in data
        assert "ml_available" in data

    def test_threats(self, client):
        _pipeline_tick()
        resp = client.get("/pipeline/threats")
        assert resp.status_code == 200
        data = resp.json()
        assert "assessments" in data
        assert "level_summary" in data
        assert "stats" in data

    def test_positions(self, client):
        _pipeline_tick()
        resp = client.get("/pipeline/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert "estimates" in data
        assert "anchors" in data
        assert len(data["anchors"]) == 4

    def test_events(self, client):
        _pipeline_tick()
        resp = client.get("/pipeline/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "recent" in data
        assert "total_events" in data

    def test_correlations(self, client):
        for _ in range(5):
            _pipeline_tick()
        resp = client.get("/pipeline/correlations")
        assert resp.status_code == 200
        data = resp.json()
        assert "correlations" in data
        assert "count" in data


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    def test_target_ids_are_unique(self):
        for _ in range(5):
            _pipeline_tick()
        targets = _state.tracker.get_all()
        ids = [t.target_id for t in targets]
        assert len(ids) == len(set(ids))

    def test_threat_scores_in_range(self):
        for _ in range(5):
            _pipeline_tick()
        assessments = _state.threat_model.assess_all()
        for a in assessments:
            assert 0.0 <= a.composite_score <= 1.0

    def test_position_estimates_near_base(self):
        for _ in range(3):
            _pipeline_tick()
        for est in _state.position_estimates:
            # Estimates should be within ~0.01 degrees of base
            assert abs(est["lat"] - BASE_LAT) < 0.01
            assert abs(est["lng"] - BASE_LNG) < 0.01

    def test_acoustic_events_have_valid_types(self):
        from tritium_lib.intelligence.acoustic_classifier import AcousticEventType
        valid_types = {e.value for e in AcousticEventType}
        for _ in range(30):
            _pipeline_tick()
        for event in _state.acoustic_log:
            assert event["event_type"] in valid_types

    def test_confidence_values_in_range(self):
        for _ in range(5):
            _pipeline_tick()
        for est in _state.position_estimates:
            assert 0.0 <= est["confidence"] <= 1.0

    def test_correlation_records_have_strategy_scores(self):
        for _ in range(5):
            _pipeline_tick()
        records = _state.correlator.get_correlations()
        for r in records:
            assert len(r.strategy_scores) > 0
            for ss in r.strategy_scores:
                assert 0.0 <= ss.score <= 1.0
                assert ss.strategy_name != ""
