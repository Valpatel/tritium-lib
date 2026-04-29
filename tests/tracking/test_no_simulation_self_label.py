# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Gap-fix A: simulation must not self-label as a confirming source.

The old behaviour produced a ~70% inflation in the
``multi_source_count`` headline metric on /api/fusion/status.  A
simulation-spawned rover was created with
``confirming_sources={"simulation"}``.  When the same target then
received a YOLO update via ``update_from_detection`` the
``confirming_sources`` set became ``{"simulation", "yolo"}`` — which
satisfies the >=2 multi-source threshold even though only ONE real
sensor (yolo) actually observed it.

This test pins the contract:

1. A pure-simulation target must NOT have ``"simulation"`` in
   ``confirming_sources`` after creation or repeated updates.
2. Re-feeding the SAME modality (e.g. yolo on a yolo target) must NOT
   inflate ``confirming_sources``.
3. Genuine cross-modal observations (ble + yolo on the same target_id,
   mesh + adsb, etc.) ARE recorded — that's the real product value.
"""

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker


class TestNoSimulationSelfLabel:
    """Simulation telemetry is synthetic ground truth, not a sensor."""

    def test_new_simulation_target_has_empty_confirming_sources(self):
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "rover_alpha",
            "name": "Alpha",
            "alliance": "friendly",
            "asset_type": "rover",
            "position": {"x": 10.0, "y": 20.0},
        })
        target = tracker.get_target("rover_alpha")
        assert target is not None
        assert "simulation" not in target.confirming_sources, (
            "simulation is synthetic ground truth and must never appear "
            "in confirming_sources — doing so inflates the multi-source "
            "fusion metric with non-sensor data.  "
            f"got confirming_sources={target.confirming_sources!r}"
        )
        # And no real sensor has observed it yet, so it must not look
        # like a confirmed multi-source target.
        assert len(target.confirming_sources) == 0

    def test_repeated_simulation_update_does_not_inflate_sources(self):
        tracker = TargetTracker()
        for _ in range(5):
            tracker.update_from_simulation({
                "target_id": "rover_alpha",
                "name": "Alpha",
                "alliance": "friendly",
                "asset_type": "rover",
                "position": {"x": 10.0, "y": 20.0},
            })
        target = tracker.get_target("rover_alpha")
        assert target is not None
        assert "simulation" not in target.confirming_sources
        assert len(target.confirming_sources) == 0


class TestNoSelfModalityInflation:
    """Repeating the same modality is not multi-source confirmation."""

    def test_repeated_ble_does_not_grow_confirming_sources(self):
        tracker = TargetTracker()
        for _ in range(10):
            tracker.update_from_ble({
                "mac": "AA:BB:CC:DD:EE:01",
                "name": "Phone",
                "rssi": -55,
                "position": {"x": 1.0, "y": 2.0},
            })
        target = tracker.get_target("ble_aabbccddee01")
        assert target is not None
        assert target.confirming_sources == {"ble"}, (
            "repeated BLE observations are the same sensor modality; "
            "they must not appear as multiple confirming sources.  "
            f"got {target.confirming_sources!r}"
        )

    def test_repeated_yolo_does_not_grow_confirming_sources(self):
        tracker = TargetTracker()
        # First detection creates the target.
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 0.0,
            "center_y": 0.0,
        })
        # Subsequent detections at the same location should match the
        # existing yolo target and update it — not inflate sources.
        for _ in range(10):
            tracker.update_from_detection({
                "class_name": "person",
                "confidence": 0.9,
                "center_x": 0.0,
                "center_y": 0.0,
            })
        targets = [t for t in tracker.get_all() if t.source == "yolo"]
        assert len(targets) == 1, "expected single matched yolo target"
        assert targets[0].confirming_sources == {"yolo"}


class TestGenuineCrossModalConfirmation:
    """Multi-source confirmation MUST still work for real cross-modal data."""

    def test_ble_plus_yolo_on_same_id_records_both(self):
        """When a real cross-modal observation happens, record it.

        We can't drive yolo→ble matching through positions alone (each
        update_from_X uses different ID schemes), so we exercise the
        helper directly: a BLE target gets updated by a yolo signal.
        """
        tracker = TargetTracker()
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:02",
            "name": "Phone",
            "rssi": -55,
            "position": {"x": 5.0, "y": 5.0},
        })
        target = tracker.get_target("ble_aabbccddee02")
        assert target is not None
        # Simulate a different sensor confirming the same entity.
        tracker._add_confirming_source(target, "yolo")
        assert "ble" in target.confirming_sources
        assert "yolo" in target.confirming_sources
        assert len(target.confirming_sources) == 2

    def test_simulation_on_real_sensor_target_is_rejected(self):
        """A simulation-tagged update on a BLE target must not inflate."""
        tracker = TargetTracker()
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:03",
            "name": "Phone",
            "rssi": -55,
            "position": {"x": 5.0, "y": 5.0},
        })
        target = tracker.get_target("ble_aabbccddee03")
        assert target is not None
        # Even when called explicitly, "simulation" must be rejected.
        tracker._add_confirming_source(target, "simulation")
        assert "simulation" not in target.confirming_sources
        assert target.confirming_sources == {"ble"}

    def test_mesh_target_does_not_self_inflate_on_repeat(self):
        """Mesh repeating itself stays at one source; mixing in ble adds it."""
        tracker = TargetTracker()
        tracker.update_from_mesh({
            "target_id": "mesh_node_42",
            "name": "Bravo",
            "position": {"x": 1.0, "y": 1.0},
        })
        for _ in range(3):
            tracker.update_from_mesh({
                "target_id": "mesh_node_42",
                "name": "Bravo",
                "position": {"x": 1.0, "y": 1.0},
            })
        t = tracker.get_target("mesh_node_42")
        assert t is not None
        assert t.confirming_sources == {"mesh"}

        # Now feed a BLE sighting under the same target_id (forced via
        # the helper, since update_from_ble derives its own id).
        tracker._add_confirming_source(t, "ble")
        assert t.confirming_sources == {"mesh", "ble"}
