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


class TestStatusVsKinematics:
    """Wave 200: status is a discrete lifecycle field, not a kinematic blob.

    Sources that report rich state (radar range/bearing/speed, RF motion
    direction) must store it in ``kinematics`` and leave ``status`` as a
    lifecycle marker so the terminal-status filter in
    targets_unified.py keeps working.
    """

    def test_default_kinematics_is_none(self):
        target = TrackedTarget(
            target_id="t1", name="t1", alliance="unknown", asset_type="x"
        )
        assert target.kinematics is None
        assert target.status == "active"

    def test_to_dict_includes_kinematics(self):
        target = TrackedTarget(
            target_id="t1",
            name="t1",
            alliance="unknown",
            asset_type="x",
            kinematics={"range_m": 100.0, "bearing_deg": 45.0, "speed_mps": 5.0},
        )
        d = target.to_dict()
        assert "kinematics" in d
        assert d["kinematics"]["range_m"] == 100.0
        assert d["kinematics"]["bearing_deg"] == 45.0
        assert d["kinematics"]["speed_mps"] == 5.0

    def test_to_dict_kinematics_none_when_unset(self):
        target = TrackedTarget(
            target_id="t1", name="t1", alliance="unknown", asset_type="x"
        )
        d = target.to_dict()
        assert d["kinematics"] is None

    def test_rf_motion_does_not_corrupt_status(self):
        """update_from_rf_motion must NOT write motion:{direction} to status."""
        tracker = TargetTracker()
        tracker.update_from_rf_motion({
            "target_id": "rf_xy",
            "position": (5.0, 10.0),
            "confidence": 0.7,
            "direction_hint": "approaching",
            "pair_id": "pair_a",
        })
        target = tracker.get_target("rf_xy")
        assert target is not None
        assert target.status == "active"
        assert not target.status.startswith("motion:")
        assert target.kinematics is not None
        assert target.kinematics["direction_hint"] == "approaching"
        assert target.kinematics["pair_id"] == "pair_a"

    def test_rf_motion_update_preserves_lifecycle_status(self):
        """Re-updating an existing rf_motion target must not overwrite status."""
        tracker = TargetTracker()
        tracker.update_from_rf_motion({
            "target_id": "rf_xy",
            "position": (5.0, 10.0),
            "confidence": 0.7,
            "direction_hint": "approaching",
        })
        # Operator marks it eliminated; a follow-up motion event must not
        # silently reset that lifecycle state.
        tracker.get_target("rf_xy").status = "eliminated"
        tracker.update_from_rf_motion({
            "target_id": "rf_xy",
            "position": (6.0, 11.0),
            "confidence": 0.8,
            "direction_hint": "receding",
        })
        target = tracker.get_target("rf_xy")
        assert target.status == "eliminated"
        assert target.kinematics["direction_hint"] == "receding"

    def test_to_dict_status_is_terminal_safe(self):
        """The serialized status must be one the terminal-status filter
        understands — never a 'motion:foo' or 'radar:...' compound string."""
        tracker = TargetTracker()
        tracker.update_from_rf_motion({
            "target_id": "rf_xy",
            "position": (5.0, 10.0),
            "confidence": 0.7,
            "direction_hint": "approaching",
        })
        d = tracker.get_target("rf_xy").to_dict()
        assert d["status"] in {
            "active", "idle", "stationary", "arrived",
            "escaped", "neutralized", "eliminated",
            "despawned", "low_battery", "destroyed",
        }


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

    def test_acoustic_half_life_pinned_at_20s(self):
        """Wave 205: acoustic source pinned to a 20s half-life.

        Without an explicit entry, acoustic-source targets fell back to
        the default 300s decay, which is wrong for transient sounds
        (gunshot, glass break, voice).  20s sits between rf_motion (10s)
        and ble (30s) — see the ``_HALF_LIVES`` block comment in
        ``target_tracker.py`` for rationale.

        This test pins the value so future changes are intentional,
        not accidental.
        """
        from tritium_lib.tracking.target_tracker import _HALF_LIVES

        assert _HALF_LIVES["acoustic"] == 20.0
        # And confirm the decay function uses it: at 20s, confidence
        # should halve.
        result = _decayed_confidence("acoustic", 1.0, 20.0)
        assert abs(result - 0.5) < 0.01
        # And acoustic decays faster than ble: at 30s, acoustic should
        # be well below ble's 50% mark.
        acoustic_30s = _decayed_confidence("acoustic", 1.0, 30.0)
        ble_30s = _decayed_confidence("ble", 1.0, 30.0)
        assert acoustic_30s < ble_30s

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

    def test_clear_source_removes_simulation_targets(self):
        """clear_source('simulation') drops every target whose source
        field is 'simulation' (Gap-fix C GA-1)."""
        tracker = TargetTracker()
        # Two simulation-sourced targets and one BLE target.
        tracker.update_from_simulation({
            "target_id": "rover_01", "name": "R1", "alliance": "friendly",
            "asset_type": "rover",
        })
        tracker.update_from_simulation({
            "target_id": "drone_01", "name": "D1", "alliance": "hostile",
            "asset_type": "drone",
        })
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:01", "name": "Phone",
            "rssi": -60, "position": {"x": 0, "y": 0},
        })

        cleared = tracker.clear_source("simulation")
        assert cleared == 2

        remaining = tracker.get_all()
        ids = {t.target_id for t in remaining}
        assert "rover_01" not in ids
        assert "drone_01" not in ids
        assert "ble_aabbccddee01" in ids

    def test_clear_source_unknown_returns_zero(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01", "name": "R1", "alliance": "friendly",
            "asset_type": "rover",
        })
        assert tracker.clear_source("does-not-exist") == 0
        assert tracker.clear_source("") == 0
        assert len(tracker.get_all()) == 1


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


class TestVersionAndSnapshot:
    """Wave 201: ETag-supporting version counter + snapshot helper."""

    def test_initial_version_is_zero(self):
        tracker = TargetTracker()
        assert tracker.version == 0

    def test_version_bumps_on_create(self):
        tracker = TargetTracker()
        v0 = tracker.version
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 0.0, "y": 0.0},
        })
        assert tracker.version > v0

    def test_version_unchanged_on_update_existing(self):
        """Wave 201 design: version tracks SET MEMBERSHIP, not field
        updates.  Position changes stream via WebSocket and do not
        invalidate the /api/targets reconciliation cache."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 0.0, "y": 0.0},
        })
        v_after_create = tracker.version
        # Same target_id, different position — version should NOT bump
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 1.0, "y": 1.0},
        })
        assert tracker.version == v_after_create

    def test_version_bumps_on_distinct_create(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 0.0, "y": 0.0},
        })
        v_after_first = tracker.version
        tracker.update_from_simulation({
            "target_id": "rover_02",
            "position": {"x": 1.0, "y": 1.0},
        })
        assert tracker.version > v_after_first

    def test_version_bumps_on_remove(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 0.0, "y": 0.0},
        })
        v_before = tracker.version
        assert tracker.remove("rover_01") is True
        assert tracker.version > v_before

    def test_version_unchanged_on_remove_missing(self):
        tracker = TargetTracker()
        v_before = tracker.version
        assert tracker.remove("nonexistent") is False
        assert tracker.version == v_before

    def test_snapshot_returns_targets_and_version(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 1.0, "y": 2.0},
        })
        targets, version = tracker.snapshot()
        assert len(targets) == 1
        assert targets[0].target_id == "rover_01"
        assert version == tracker.version
        assert version > 0

    def test_snapshot_is_a_copy(self):
        """Mutating the returned list must not affect tracker state."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 1.0, "y": 2.0},
        })
        targets, _ = tracker.snapshot()
        targets.clear()
        # Tracker still knows about rover_01.
        targets2, _ = tracker.snapshot()
        assert len(targets2) == 1

    def test_two_consecutive_snapshots_same_version_when_quiet(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_01",
            "position": {"x": 0.0, "y": 0.0},
        })
        _, v1 = tracker.snapshot()
        _, v2 = tracker.snapshot()
        # _prune_stale() may not bump version (no stale targets) so the
        # version stays the same across two snapshots when no work was
        # done — this is what the ETag short-circuit relies on.
        assert v1 == v2
