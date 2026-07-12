# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Production north-star regression: co-located multi-modal sightings fuse to ONE
unique tracked target.

Tritium's operational mission is "fuse RF + vision + mesh + acoustic into one
tactical map — unique target ID per entity." This guards the core of that: a BLE
sighting and a camera detection of the SAME entity (co-located within the
correlation radius) must correlate into a SINGLE TrackedTarget carrying both
source types, while a well-separated entity must NOT be merged. (Measured
2026-06-18 while scoping the fusion-from-sim breadth work: the capability is
healthy; the open gap is sim-side multi-modal EMISSION, scoped separately.)
"""
import pytest

from tritium_lib.tracking.target_tracker import TargetTracker
from tritium_lib.fusion import FusionEngine

pytestmark = pytest.mark.unit


def _build():
    tracker = TargetTracker()
    fusion = FusionEngine(tracker=tracker, correlation_radius=5.0)
    return tracker, fusion


def _targets(tracker):
    return list(getattr(tracker, "_targets", {}).values())


def test_colocated_ble_and_detection_fuse_to_one_unique_id():
    tracker, fusion = _build()
    # ONE entity at (10,10) seen by BLE and a camera.
    tracker.update_from_ble({
        "mac": "AA:BB:CC:DD:EE:01", "name": "phone-1", "rssi": -55,
        "position": {"x": 10.0, "y": 10.0}, "device_type": "phone", "source": "city_sim",
    })
    tracker.update_from_detection({
        "class_name": "person", "confidence": 0.9,
        "center_x": 10.0, "center_y": 10.0, "source": "city_sim",
    })
    assert len(_targets(tracker)) == 2  # two separate tracks before fusion

    fusion.run_correlation()

    merged = [t for t in _targets(tracker)
              if len(getattr(t, "confirming_sources", set()) or set()) >= 2]
    assert len(merged) == 1, "co-located BLE + detection did not fuse to one track"
    m = merged[0]
    assert {"ble", "yolo"}.issubset(set(m.confirming_sources))
    assert m.correlated_ids, "fused track should record the correlated source id"


def test_well_separated_entities_do_not_merge():
    tracker, fusion = _build()
    tracker.update_from_ble({
        "mac": "AA:BB:CC:DD:EE:01", "name": "phone-1", "rssi": -55,
        "position": {"x": 10.0, "y": 10.0}, "device_type": "phone", "source": "city_sim",
    })
    tracker.update_from_detection({
        "class_name": "person", "confidence": 0.9,
        "center_x": 80.0, "center_y": 80.0, "source": "city_sim",  # 99m away
    })
    fusion.run_correlation()
    multi = [t for t in _targets(tracker)
             if len(getattr(t, "confirming_sources", set()) or set()) >= 2]
    assert not multi, "well-separated BLE + detection must NOT be fused"
