# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.target_tracker."""

import time
import pytest

from tritium_lib.tracking.target_tracker import (
    TargetTracker,
    TrackedTarget,
    _decayed_confidence,
)


class TestTrackedTarget:
    """Tests for TrackedTarget dataclass."""

    def test_create_ble_target(self):
        """Test TrackedTarget creation with BLE sighting data."""
        tracker = TargetTracker()
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "name": "TestPhone",
            "rssi": -60,
            "position": {"x": 10.0, "y": 20.0},
        })

        target = tracker.get_target("ble_aabbccddeeff")
        assert target is not None
        assert target.target_id == "ble_aabbccddeeff"
        assert target.name == "TestPhone"
        assert target.source == "ble"
        assert target.alliance == "unknown"
        assert target.position == (10.0, 20.0)
        assert target.position_source == "trilateration"
        assert target.signal_count == 1
        assert "ble" in target.confirming_sources

    def test_create_simulation_target(self):
        """Test TrackedTarget creation from simulation data."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "name": "Alpha Rover",
            "alliance": "friendly",
            "asset_type": "rover",
            "position": {"x": 50.0, "y": 100.0},
            "heading": 45.0,
            "speed": 2.0,
            "battery": 0.85,
            "status": "active",
        })

        target = tracker.get_target("rover_01")
        assert target is not None
        assert target.name == "Alpha Rover"
        assert target.alliance == "friendly"
        assert target.asset_type == "rover"
        assert target.position == (50.0, 100.0)
        assert target.heading == 45.0
        assert target.speed == 2.0
        assert target.battery == 0.85
        assert target.source == "simulation"

    def test_create_detection_target(self):
        """Test TrackedTarget creation from YOLO detection."""
        tracker = TargetTracker()
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.8,
            "center_x": 5.0,
            "center_y": 10.0,
        })

        targets = tracker.get_all()
        assert len(targets) == 1
        t = targets[0]
        assert t.source == "yolo"
        assert t.asset_type == "person"
        assert t.alliance == "hostile"
        assert t.classification == "person"


class TestConfidenceDecay:
    """Tests for confidence decay over time."""

    def test_no_decay_at_zero_elapsed(self):
        """Confidence should not decay when elapsed time is 0."""
        assert _decayed_confidence("ble", 0.8, 0.0) == 0.8

    def test_half_life_decay(self):
        """Confidence should be ~50% at the half-life time."""
        # BLE half-life is 30 seconds
        result = _decayed_confidence("ble", 1.0, 30.0)
        assert abs(result - 0.5) < 0.01

    def test_simulation_never_decays(self):
        """Simulation source has 0 half-life = never decays."""
        result = _decayed_confidence("simulation", 1.0, 1000.0)
        assert result == 1.0

    def test_decay_below_threshold_returns_zero(self):
        """Confidence below MIN_CONFIDENCE returns 0."""
        # Very long elapsed time should push below threshold
        result = _decayed_confidence("ble", 0.1, 600.0)
        assert result == 0.0

    def test_effective_confidence_decays(self):
        """TrackedTarget.effective_confidence should decay over time."""
        t = TrackedTarget(
            target_id="test",
            name="test",
            alliance="unknown",
            asset_type="ble_device",
            source="ble",
            position_confidence=0.8,
            _initial_confidence=0.8,
            last_seen=time.monotonic() - 30.0,  # 30 seconds ago
        )
        # After 30s (BLE half-life), should be roughly 50% of initial
        eff = t.effective_confidence
        assert 0.3 < eff < 0.6


class TestMultiSourceBoosting:
    """Tests for multi-source confidence boosting."""

    def test_single_source_no_boost(self):
        """Single source should not get a boost."""
        t = TrackedTarget(
            target_id="test",
            name="test",
            alliance="unknown",
            asset_type="ble_device",
            source="ble",
            position_confidence=0.5,
            _initial_confidence=0.5,
            confirming_sources={"ble"},
        )
        eff = t.effective_confidence
        # Should be close to 0.5, no boost
        assert eff <= 0.55

    def test_multi_source_boost(self):
        """Multiple confirming sources should boost confidence."""
        t = TrackedTarget(
            target_id="test",
            name="test",
            alliance="unknown",
            asset_type="ble_device",
            source="ble",
            position_confidence=0.5,
            _initial_confidence=0.5,
            confirming_sources={"ble", "yolo", "mesh"},
        )
        eff = t.effective_confidence
        # 3 sources = 2 extra, each boosts by 1.3x
        # 0.5 * 1.3^2 = 0.845
        assert eff > 0.5
        assert eff <= 0.99


class TestTargetRemoval:
    """Tests for target timeout and removal."""

    def test_stale_yolo_target_removed(self):
        """YOLO targets should be pruned after STALE_TIMEOUT."""
        tracker = TargetTracker()
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.8,
            "center_x": 5.0,
            "center_y": 10.0,
        })

        # Force the target to be stale
        with tracker._lock:
            for t in tracker._targets.values():
                t.last_seen = time.monotonic() - 60.0  # 60s ago

        targets = tracker.get_all()
        assert len(targets) == 0

    def test_manual_remove(self):
        """Should be able to manually remove a target."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "name": "Alpha",
            "alliance": "friendly",
            "asset_type": "rover",
        })

        assert tracker.remove("rover_01") is True
        assert tracker.get_target("rover_01") is None
        assert tracker.remove("nonexistent") is False

    def test_get_hostiles_and_friendlies(self):
        """Should correctly filter by alliance."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "name": "Alpha",
            "alliance": "friendly",
            "asset_type": "rover",
        })
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.8,
            "center_x": 5.0,
            "center_y": 10.0,
        })

        assert len(tracker.get_friendlies()) == 1
        assert len(tracker.get_hostiles()) == 1

    def test_update_increments_signal_count(self):
        """Updating same target should increment signal_count."""
        tracker = TargetTracker()
        for _ in range(5):
            tracker.update_from_ble({
                "mac": "AA:BB:CC:DD:EE:FF",
                "name": "TestPhone",
                "rssi": -60,
                "position": {"x": 10.0, "y": 20.0},
            })

        target = tracker.get_target("ble_aabbccddeeff")
        assert target is not None
        assert target.signal_count == 5


class TestVelocityCheck:
    """Tests for velocity/teleportation detection."""

    def test_normal_velocity_not_flagged(self):
        """Normal movement speed should not flag velocity_suspicious."""
        tracker = TargetTracker()
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -60,
            "position": {"x": 0.0, "y": 0.0},
        })
        # Move 1 meter in 1 second — 1 m/s is normal
        time.sleep(0.01)
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -60,
            "position": {"x": 0.01, "y": 0.0},
        })
        target = tracker.get_target("ble_aabbccddeeff")
        assert target is not None
        assert target.velocity_suspicious is False
