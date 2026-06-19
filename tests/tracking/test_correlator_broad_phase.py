# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Correlator broad-phase spatial gate — scale + false-merge guard (Tick O).

correlate() was O(n^2): it ran the full multi-strategy _evaluate_pair on
EVERY cross-source pair, so at hundreds of nodes a pass took seconds
(measured: 510 tracks -> 2831 ms). A broad-phase distance gate skips pairs
too far apart to be the same entity BEFORE the expensive evaluation —
turning the pass tractable (510 tracks -> ~266 ms, 10x) while merging the
EXACT same co-located pairs. It also prevents false merges of distant
co-movers (temporal strategy is distance-independent).
"""
from __future__ import annotations

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.correlator import TargetCorrelator

pytestmark = pytest.mark.unit


def _add(tr, tid, source, pos):
    t = TrackedTarget(target_id=tid, name=tid, alliance="hostile",
                      asset_type="person", position=pos, source=source,
                      position_confidence=0.9)
    with tr._lock:
        tr._targets[tid] = t


def _pairs(records):
    return {frozenset((r.primary_id, r.secondary_id)) for r in records}


class TestBroadPhaseGate:
    def test_default_gate_is_generous(self):
        tr = TargetTracker()
        c = TargetCorrelator(tr)
        assert c.broad_phase_radius == 50.0
        assert c._bp_radius_sq == 50.0 ** 2

    def test_colocated_cross_source_still_correlates(self):
        # The behavior-preserving case: co-located ble+yolo (well inside the
        # gate) must still fuse exactly as before.
        tr = TargetTracker()
        _add(tr, "ble_1", "ble", (10.0, 10.0))
        _add(tr, "det_1", "yolo", (11.0, 10.5))  # ~1.1m apart
        recs = TargetCorrelator(tr, confidence_threshold=0.3).correlate()
        assert frozenset(("ble_1", "det_1")) in _pairs(recs)

    def test_distant_pair_is_culled(self):
        # Two cross-source tracks 200m apart are NOT the same entity — the gate
        # skips them (scale win + false-merge prevention).
        tr = TargetTracker()
        _add(tr, "ble_1", "ble", (-150.0, -150.0))
        _add(tr, "det_1", "yolo", (150.0, 150.0))  # ~424m apart
        recs = TargetCorrelator(tr, confidence_threshold=0.3).correlate()
        assert frozenset(("ble_1", "det_1")) not in _pairs(recs)

    def test_gate_is_configurable_and_active(self):
        # A tight gate proves the cull is real: a 3m pair is skipped at 2m gate.
        tr = TargetTracker()
        _add(tr, "ble_1", "ble", (0.0, 0.0))
        _add(tr, "det_1", "yolo", (3.0, 0.0))
        tight = TargetCorrelator(tr, broad_phase_radius=2.0, confidence_threshold=0.3)
        assert frozenset(("ble_1", "det_1")) not in _pairs(tight.correlate())
        # ...but a generous gate evaluates it (and it fuses — within spatial radius).
        wide = TargetCorrelator(tr, broad_phase_radius=50.0, confidence_threshold=0.3)
        assert frozenset(("ble_1", "det_1")) in _pairs(wide.correlate())
