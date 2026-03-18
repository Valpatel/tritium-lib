# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for position anchoring and sensor fusion foundation."""

import time

import pytest

from tritium_lib.models.position_anchor import (
    DetectionEdge,
    FusedPositionEstimate,
    PositionAnchor,
)
from tritium_lib.intelligence.position_estimator import (
    estimate_from_multiple_anchors,
    estimate_from_single_anchor,
    rssi_to_distance,
)


# ---------------------------------------------------------------------------
# PositionAnchor model
# ---------------------------------------------------------------------------


class TestPositionAnchor:
    def test_basic_creation(self):
        anchor = PositionAnchor(
            anchor_id="mesh_aabbccdd",
            lat=37.707724,
            lng=-121.939279,
            source="gps",
            confidence=0.95,
            device_id="!aabbccdd",
            label="T-LoRa Pager",
        )
        assert anchor.anchor_id == "mesh_aabbccdd"
        assert anchor.lat == 37.707724
        assert anchor.lng == -121.939279
        assert anchor.source == "gps"
        assert anchor.confidence == 0.95
        assert anchor.fixed is False

    def test_fixed_anchor(self):
        anchor = PositionAnchor(
            anchor_id="roof_node",
            lat=37.7,
            lng=-121.9,
            source="survey",
            confidence=1.0,
            fixed=True,
            label="Roof Node",
        )
        assert anchor.fixed is True
        assert anchor.confidence == 1.0

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            PositionAnchor(anchor_id="bad", lat=0, lng=0, confidence=1.5)
        with pytest.raises(Exception):
            PositionAnchor(anchor_id="bad", lat=0, lng=0, confidence=-0.1)

    def test_defaults(self):
        anchor = PositionAnchor(anchor_id="test", lat=0, lng=0)
        assert anchor.source == "gps"
        assert anchor.confidence == 0.9
        assert anchor.alt is None
        assert anchor.device_id is None
        assert anchor.fixed is False
        assert anchor.label == ""
        assert anchor.timestamp > 0

    def test_serialization_roundtrip(self):
        anchor = PositionAnchor(
            anchor_id="test",
            lat=37.7,
            lng=-121.9,
            alt=15.0,
            source="manual",
            confidence=0.8,
        )
        data = anchor.model_dump()
        restored = PositionAnchor(**data)
        assert restored.anchor_id == anchor.anchor_id
        assert restored.lat == anchor.lat
        assert restored.alt == 15.0


# ---------------------------------------------------------------------------
# DetectionEdge model
# ---------------------------------------------------------------------------


class TestDetectionEdge:
    def test_ble_detection(self):
        edge = DetectionEdge(
            detector_id="mesh_aabbccdd",
            detected_id="ble_11:22:33:44:55:66",
            detection_type="ble",
            rssi=-72.0,
            confidence=0.85,
        )
        assert edge.detection_type == "ble"
        assert edge.rssi == -72.0
        assert edge.snr is None

    def test_lora_detection(self):
        edge = DetectionEdge(
            detector_id="mesh_aabbccdd",
            detected_id="mesh_11223344",
            detection_type="lora",
            rssi=-95.0,
            snr=8.5,
            distance_estimate_m=500.0,
        )
        assert edge.snr == 8.5
        assert edge.distance_estimate_m == 500.0

    def test_camera_detection(self):
        edge = DetectionEdge(
            detector_id="cam_front_door",
            detected_id="det_person_1",
            detection_type="camera",
            confidence=0.92,
        )
        assert edge.rssi is None
        assert edge.distance_estimate_m is None

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            DetectionEdge(
                detector_id="a",
                detected_id="b",
                detection_type="ble",
                confidence=2.0,
            )

    def test_defaults(self):
        edge = DetectionEdge(
            detector_id="a",
            detected_id="b",
            detection_type="wifi",
        )
        assert edge.confidence == 0.8
        assert edge.timestamp > 0


# ---------------------------------------------------------------------------
# FusedPositionEstimate model
# ---------------------------------------------------------------------------


class TestFusedPositionEstimate:
    def test_basic(self):
        est = FusedPositionEstimate(
            target_id="ble_aabb",
            lat=37.7,
            lng=-121.9,
            accuracy_m=25.0,
            method="centroid",
            anchor_count=3,
            confidence=0.75,
        )
        assert est.method == "centroid"
        assert est.anchor_count == 3

    def test_defaults(self):
        est = FusedPositionEstimate(target_id="x", lat=0, lng=0)
        assert est.accuracy_m == 50.0
        assert est.method == "proximity"
        assert est.anchor_count == 1
        assert est.confidence == 0.5


# ---------------------------------------------------------------------------
# RSSI to distance conversion
# ---------------------------------------------------------------------------


class TestRSSIToDistance:
    def test_close_signal(self):
        """Strong signal = close distance."""
        dist = rssi_to_distance(-40.0, tx_power=-40.0)
        assert dist == pytest.approx(1.0, abs=0.1)

    def test_medium_signal(self):
        """-65 dBm with tx_power=-40 should be a few meters."""
        dist = rssi_to_distance(-65.0, tx_power=-40.0, path_loss_exp=2.5)
        assert 5.0 < dist < 100.0

    def test_weak_signal(self):
        """Very weak signal = far away."""
        dist = rssi_to_distance(-100.0, tx_power=-40.0)
        assert dist > 100.0

    def test_minimum_clamp(self):
        """Distance should never go below 0.5m."""
        dist = rssi_to_distance(-30.0, tx_power=-40.0)
        assert dist >= 0.5

    def test_invalid_path_loss(self):
        with pytest.raises(ValueError):
            rssi_to_distance(-70.0, path_loss_exp=0)
        with pytest.raises(ValueError):
            rssi_to_distance(-70.0, path_loss_exp=-1.0)

    def test_different_environments(self):
        """Higher path loss exponent = shorter estimated distance for same RSSI."""
        dist_outdoor = rssi_to_distance(-80.0, tx_power=-40.0, path_loss_exp=2.0)
        dist_indoor = rssi_to_distance(-80.0, tx_power=-40.0, path_loss_exp=3.5)
        assert dist_outdoor > dist_indoor


# ---------------------------------------------------------------------------
# Single anchor estimation
# ---------------------------------------------------------------------------


class TestSingleAnchorEstimation:
    def test_basic(self):
        anchor = PositionAnchor(
            anchor_id="mesh_aabb",
            lat=37.707724,
            lng=-121.939279,
            confidence=0.95,
        )
        detection = DetectionEdge(
            detector_id="mesh_aabb",
            detected_id="ble_target",
            detection_type="ble",
            rssi=-72.0,
            confidence=0.8,
        )
        est = estimate_from_single_anchor(anchor, detection)

        assert est.target_id == "ble_target"
        assert est.lat == anchor.lat
        assert est.lng == anchor.lng
        assert est.method == "proximity"
        assert est.anchor_count == 1
        assert est.accuracy_m > 0
        assert 0.0 < est.confidence <= 1.0

    def test_with_distance_estimate(self):
        anchor = PositionAnchor(
            anchor_id="a1", lat=37.7, lng=-121.9, confidence=0.9,
        )
        detection = DetectionEdge(
            detector_id="a1",
            detected_id="t1",
            detection_type="lora",
            distance_estimate_m=250.0,
            confidence=0.7,
        )
        est = estimate_from_single_anchor(anchor, detection)
        assert est.accuracy_m == 250.0

    def test_no_signal_data(self):
        anchor = PositionAnchor(
            anchor_id="a1", lat=37.7, lng=-121.9, confidence=0.9,
        )
        detection = DetectionEdge(
            detector_id="a1",
            detected_id="t1",
            detection_type="camera",
            confidence=0.9,
        )
        est = estimate_from_single_anchor(anchor, detection)
        assert est.accuracy_m == 100.0  # default fallback


# ---------------------------------------------------------------------------
# Multi-anchor estimation
# ---------------------------------------------------------------------------


class TestMultiAnchorEstimation:
    def test_two_anchors(self):
        anchors = [
            PositionAnchor(anchor_id="a1", lat=37.707, lng=-121.939, confidence=0.9),
            PositionAnchor(anchor_id="a2", lat=37.708, lng=-121.938, confidence=0.9),
        ]
        detections = [
            DetectionEdge(
                detector_id="a1", detected_id="t1",
                detection_type="ble", rssi=-65.0, confidence=0.8,
            ),
            DetectionEdge(
                detector_id="a2", detected_id="t1",
                detection_type="ble", rssi=-70.0, confidence=0.8,
            ),
        ]
        est = estimate_from_multiple_anchors(anchors, detections)

        assert est is not None
        assert est.target_id == "t1"
        assert est.anchor_count == 2
        assert est.method == "centroid"
        # Position should be between the two anchors, pulled toward a1 (stronger)
        assert 37.707 <= est.lat <= 37.708
        assert -121.939 <= est.lng <= -121.938
        assert 0.0 < est.confidence <= 1.0

    def test_three_anchors_equal_signal(self):
        anchors = [
            PositionAnchor(anchor_id="a1", lat=37.707, lng=-121.940, confidence=0.9),
            PositionAnchor(anchor_id="a2", lat=37.708, lng=-121.939, confidence=0.9),
            PositionAnchor(anchor_id="a3", lat=37.707, lng=-121.938, confidence=0.9),
        ]
        detections = [
            DetectionEdge(
                detector_id="a1", detected_id="t1",
                detection_type="ble", rssi=-70.0, confidence=0.8,
            ),
            DetectionEdge(
                detector_id="a2", detected_id="t1",
                detection_type="ble", rssi=-70.0, confidence=0.8,
            ),
            DetectionEdge(
                detector_id="a3", detected_id="t1",
                detection_type="ble", rssi=-70.0, confidence=0.8,
            ),
        ]
        est = estimate_from_multiple_anchors(anchors, detections)

        assert est is not None
        assert est.anchor_count == 3
        # Equal signal = geometric centroid
        assert est.lat == pytest.approx(
            (37.707 + 37.708 + 37.707) / 3, abs=0.0001
        )

    def test_no_matching_pairs(self):
        anchors = [
            PositionAnchor(anchor_id="a1", lat=37.7, lng=-121.9, confidence=0.9),
        ]
        detections = [
            DetectionEdge(
                detector_id="a_nonexistent", detected_id="t1",
                detection_type="ble", rssi=-70.0,
            ),
        ]
        est = estimate_from_multiple_anchors(anchors, detections)
        assert est is None

    def test_empty_inputs(self):
        assert estimate_from_multiple_anchors([], []) is None
        assert estimate_from_multiple_anchors([], [
            DetectionEdge(detector_id="a", detected_id="b", detection_type="ble"),
        ]) is None

    def test_single_match_falls_back(self):
        """With only one matching pair, falls back to single-anchor estimation."""
        anchors = [
            PositionAnchor(anchor_id="a1", lat=37.7, lng=-121.9, confidence=0.9),
            PositionAnchor(anchor_id="a2", lat=37.8, lng=-121.8, confidence=0.9),
        ]
        detections = [
            DetectionEdge(
                detector_id="a1", detected_id="t1",
                detection_type="ble", rssi=-70.0, confidence=0.8,
            ),
        ]
        est = estimate_from_multiple_anchors(anchors, detections)
        assert est is not None
        assert est.method == "proximity"
        assert est.anchor_count == 1

    def test_stronger_signal_pulls_harder(self):
        """Target should be closer to the anchor with stronger signal."""
        anchors = [
            PositionAnchor(anchor_id="a1", lat=37.700, lng=-121.900, confidence=0.9),
            PositionAnchor(anchor_id="a2", lat=37.710, lng=-121.900, confidence=0.9),
        ]
        detections = [
            DetectionEdge(
                detector_id="a1", detected_id="t1",
                detection_type="ble", rssi=-50.0, confidence=0.8,  # very close
            ),
            DetectionEdge(
                detector_id="a2", detected_id="t1",
                detection_type="ble", rssi=-90.0, confidence=0.8,  # far away
            ),
        ]
        est = estimate_from_multiple_anchors(anchors, detections)
        assert est is not None
        # Should be much closer to a1 (lat 37.700)
        assert est.lat < 37.705


# ---------------------------------------------------------------------------
# Meshtastic GPS → PositionAnchor conversion
# ---------------------------------------------------------------------------


class TestMeshtasticAnchors:
    """Test that NodeManager creates PositionAnchors from GPS nodes."""

    def _make_manager(self):
        """Import and create NodeManager with mock event bus."""
        import importlib
        import importlib.util
        import os

        # Import node_manager directly to avoid the package __init__
        # which pulls in FastAPI via router.py
        nm_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "tritium-addons",
            "meshtastic", "meshtastic_addon", "node_manager.py",
        )
        nm_path = os.path.normpath(nm_path)
        spec = importlib.util.spec_from_file_location(
            "meshtastic_addon.node_manager", nm_path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NodeManager = mod.NodeManager

        events: list = []

        class MockBus:
            def publish(self, topic, data):
                events.append((topic, data))

        mgr = NodeManager(event_bus=MockBus())
        return mgr, events

    def test_gps_node_creates_anchor(self):
        mgr, events = self._make_manager()
        mgr.set_local_node("!aabbccdd")

        raw_nodes = {
            "!aabbccdd": {
                "user": {"longName": "MyPager", "shortName": "MP"},
                "position": {
                    "latitudeI": 377077240,
                    "longitudeI": -1219392790,
                    "altitude": 15,
                    "time": int(time.time()),
                    "satsInView": 8,
                },
                "lastHeard": int(time.time()),
            },
        }
        mgr.update_nodes(raw_nodes)

        anchors = mgr.get_position_anchors()
        assert len(anchors) == 1

        a = anchors[0]
        assert a.anchor_id == "mesh_aabbccdd"
        assert a.lat == pytest.approx(37.707724, abs=0.001)
        assert a.lng == pytest.approx(-121.939279, abs=0.001)
        assert a.confidence >= 0.9  # local node with good GPS
        assert a.device_id == "!aabbccdd"
        assert a.label == "MyPager"

    def test_non_gps_node_no_anchor(self):
        mgr, events = self._make_manager()

        raw_nodes = {
            "!11223344": {
                "user": {"longName": "NoGPS"},
                "position": {},
                "lastHeard": int(time.time()),
            },
        }
        mgr.update_nodes(raw_nodes)

        anchors = mgr.get_position_anchors()
        assert len(anchors) == 0

    def test_anchor_event_emitted(self):
        mgr, events = self._make_manager()

        raw_nodes = {
            "!aabbccdd": {
                "user": {"longName": "GPSNode"},
                "position": {
                    "latitudeI": 377077240,
                    "longitudeI": -1219392790,
                },
                "lastHeard": int(time.time()),
            },
        }
        mgr.update_nodes(raw_nodes)

        anchor_events = [
            (t, d) for t, d in events if t == "addon:meshtastic:position_anchor"
        ]
        assert len(anchor_events) == 1
        data = anchor_events[0][1]
        assert data["anchor_id"] == "mesh_aabbccdd"
        assert data["lat"] == pytest.approx(37.707724, abs=0.001)

    def test_stale_position_lower_confidence(self):
        mgr, events = self._make_manager()

        # Node last heard 2 hours ago
        raw_nodes = {
            "!old_node": {
                "user": {"longName": "OldNode"},
                "position": {
                    "latitudeI": 377000000,
                    "longitudeI": -1219000000,
                },
                "lastHeard": int(time.time()) - 7200,
            },
        }
        mgr.update_nodes(raw_nodes)

        anchors = mgr.get_position_anchors()
        assert len(anchors) == 1
        # Stale node should have reduced confidence
        assert anchors[0].confidence < 0.7
