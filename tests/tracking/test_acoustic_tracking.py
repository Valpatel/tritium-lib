# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TargetTracker.update_from_acoustic — the ACOUSTIC modality.

The north star (CLAUDE.md) is to "Fuse RF, vision, mesh, acoustic into one
tactical map. Unique target ID per entity."  RF (ble), vision (yolo) and mesh
all had a tracker ingestion path; acoustic did not — the FusionEngine could
ingest acoustic but the unified TargetTracker (which the correlator and
``/api/targets`` use) could not, so acoustic never entered the multi-source
fused track.  These tests pin the new path.
"""
from __future__ import annotations

import time

from tritium_lib.tracking.correlator import TargetCorrelator
from tritium_lib.tracking.target_tracker import TargetTracker


def _acoustic(target_id="hostile-1", x=10.0, y=10.0, event_type="gunshot",
              sensor_id="mic-1", confidence=0.5):
    return {
        "sensor_id": sensor_id,
        "event_type": event_type,
        "position": {"x": x, "y": y},
        "confidence": confidence,
        "target_id": target_id,
        "name": "Intruder 1",
    }


def test_acoustic_creates_tracked_target():
    tr = TargetTracker()
    tr.update_from_acoustic(_acoustic())
    targets = tr.get_all()
    assert len(targets) == 1
    t = targets[0]
    assert t.source == "acoustic"
    assert "acoustic" in t.confirming_sources
    assert t.classification == "gunshot"
    assert t.position == (10.0, 10.0)
    assert t.position_source == "acoustic_doa"


def test_acoustic_event_type_maps_to_asset_type():
    tr = TargetTracker()
    tr.update_from_acoustic(_acoustic(target_id="veh-1", event_type="vehicle"))
    t = tr.get_target("acoustic_veh-1")
    assert t is not None
    assert t.asset_type == "vehicle"


def test_acoustic_repeat_increments_signal_count():
    tr = TargetTracker()
    tr.update_from_acoustic(_acoustic())
    tr.update_from_acoustic(_acoustic())
    t = tr.get_target("acoustic_hostile-1")
    assert t is not None
    assert t.signal_count == 2


def test_acoustic_missing_target_id_uses_sensor_and_type():
    tr = TargetTracker()
    ev = _acoustic()
    del ev["target_id"]
    tr.update_from_acoustic(ev)
    t = tr.get_target("acoustic_mic-1_gunshot")
    assert t is not None


def test_acoustic_no_position_or_id_is_safe():
    tr = TargetTracker()
    tr.update_from_acoustic({})  # must not raise, must not create junk
    assert tr.get_all() == []


def test_acoustic_fuses_with_other_modalities_into_one_id():
    """The whole point: ble + yolo + acoustic co-located -> ONE fused track."""
    tr = TargetTracker()
    # RF (ble) track
    tr.update_from_ble({"mac": "AA:BB:CC:DD:EE:01", "name": "phone",
                        "rssi": -45, "position": {"x": 10.0, "y": 10.0}})
    # vision (yolo) track
    tr.update_from_detection({"class_name": "person", "confidence": 0.9,
                              "center_x": 10.5, "center_y": 9.8})
    # acoustic track, co-located
    tr.update_from_acoustic(_acoustic(x=9.6, y=10.3))

    corr = TargetCorrelator(tr, radius=5.0, broad_phase_radius=50.0)
    corr.correlate()

    best = max(tr.get_all(),
               key=lambda t: len({s for s in t.confirming_sources
                                  if s != "simulation"}))
    genuine = {s for s in best.confirming_sources if s != "simulation"}
    assert {"ble", "yolo", "acoustic"} <= genuine, (
        f"expected ble+yolo+acoustic fused, got {sorted(genuine)}"
    )


def test_acoustic_track_is_prunable():
    tr = TargetTracker()
    tr.update_from_acoustic(_acoustic())
    t = tr.get_target("acoustic_hostile-1")
    assert t is not None
    # Force staleness beyond the acoustic timeout and prune.
    t.last_seen = time.monotonic() - (tr.ACOUSTIC_STALE_TIMEOUT + 5.0)
    tr._prune_stale()
    assert tr.get_target("acoustic_hostile-1") is None
