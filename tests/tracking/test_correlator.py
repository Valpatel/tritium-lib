# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.correlator."""

import time
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.correlator import (
    TargetCorrelator,
    CorrelationRecord,
    DEFAULT_WEIGHTS,
    _node_type_for,
)
from tritium_lib.tracking.correlation_strategies import StrategyScore
from tritium_lib.tracking.dossier import DossierStore


def _make_target(
    target_id: str,
    source: str,
    position: tuple[float, float] = (0.0, 0.0),
    name: str = "",
    asset_type: str = "person",
    confidence: float = 0.8,
) -> TrackedTarget:
    now = time.monotonic()
    return TrackedTarget(
        target_id=target_id,
        name=name or target_id,
        alliance="unknown",
        asset_type=asset_type,
        position=position,
        source=source,
        position_confidence=confidence,
        last_seen=now,
        first_seen=now,
        confirming_sources={source},
    )


class TestNodeTypeFor:
    def test_person(self):
        assert _node_type_for("person") == "Person"

    def test_vehicle_types(self):
        for vt in ("vehicle", "car", "motorcycle", "bicycle"):
            assert _node_type_for(vt) == "Vehicle"

    def test_device_types(self):
        for dt in ("ble_device", "rover", "drone", "turret"):
            assert _node_type_for(dt) == "Device"

    def test_unknown_defaults_to_device(self):
        assert _node_type_for("alien_craft") == "Device"


class TestCorrelationRecord:
    def test_fields(self):
        r = CorrelationRecord(
            primary_id="ble_aa",
            secondary_id="det_person_1",
            confidence=0.75,
            reason="test",
        )
        assert r.primary_id == "ble_aa"
        assert r.secondary_id == "det_person_1"
        assert r.confidence == 0.75
        assert r.dossier_uuid == ""
        assert isinstance(r.strategy_scores, list)


class TestTargetCorrelatorInit:
    def test_default_init(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker)
        assert c.radius == 5.0
        assert c.max_age == 30.0
        assert c.confidence_threshold == 0.3
        assert len(c.strategies) >= 4

    def test_custom_params(self):
        tracker = TargetTracker()
        c = TargetCorrelator(
            tracker, radius=10.0, max_age=60.0, confidence_threshold=0.5
        )
        assert c.radius == 10.0
        assert c.max_age == 60.0
        assert c.confidence_threshold == 0.5

    def test_custom_weights(self):
        tracker = TargetTracker()
        w = {"spatial": 1.0}
        c = TargetCorrelator(tracker, weights=w)
        assert c.weights == {"spatial": 1.0}


class TestCorrelate:
    def test_correlate_nearby_different_sources(self):
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_1", "yolo", position=(10.5, 10.5))
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["det_person_1"] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        # Spatial proximity should produce a correlation
        assert len(records) >= 1
        assert records[0].primary_id in ("ble_aa", "det_person_1")
        assert records[0].confidence > 0

    def test_no_correlation_same_source(self):
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("ble_bb", "ble", position=(10.1, 10.1))
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["ble_bb"] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) == 0

    def test_no_correlation_far_apart(self):
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(0.0, 0.0))
        t2 = _make_target("det_person_1", "yolo", position=(1000.0, 1000.0))
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["det_person_1"] = t2

        # Use a threshold high enough that signal_pattern alone can't trigger
        c = TargetCorrelator(tracker, confidence_threshold=0.5, max_age=9999)
        records = c.correlate()
        # Should not correlate — too far apart for high threshold
        assert len(records) == 0

    def test_correlate_creates_dossier(self):
        tracker = TargetTracker()
        store = DossierStore()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_1", "yolo", position=(10.2, 10.2))
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["det_person_1"] = t2

        c = TargetCorrelator(
            tracker, dossier_store=store,
            confidence_threshold=0.01, max_age=9999,
        )
        records = c.correlate()
        if records:
            assert records[0].dossier_uuid != ""
            assert store.count >= 1

    def test_correlate_merges_secondary_into_primary(self):
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0), confidence=0.9)
        t2 = _make_target("det_person_1", "yolo", position=(10.2, 10.2), confidence=0.7)
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["det_person_1"] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        c.correlate()

        # Secondary should be removed
        assert tracker.get_target("det_person_1") is None
        primary = tracker.get_target("ble_aa")
        assert primary is not None
        assert "det_person_1" in primary.correlated_ids

    def test_get_correlations_returns_history(self):
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_1", "yolo", position=(10.2, 10.2))
        with tracker._lock:
            tracker._targets["ble_aa"] = t1
            tracker._targets["det_person_1"] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        c.correlate()
        history = c.get_correlations()
        assert isinstance(history, list)


class TestAddStrategy:
    def test_add_strategy(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker)
        initial_count = len(c.strategies)

        class DummyStrategy:
            name = "dummy"
            def evaluate(self, a, b):
                return StrategyScore(strategy_name="dummy", score=0.5, detail="test")

        c.add_strategy(DummyStrategy(), weight=0.2)
        assert len(c.strategies) == initial_count + 1
        assert c.weights["dummy"] == 0.2


class TestWeightedScore:
    def test_weighted_score_basic(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={"a": 0.5, "b": 0.5})
        scores = [
            StrategyScore(strategy_name="a", score=1.0, detail=""),
            StrategyScore(strategy_name="b", score=0.0, detail=""),
        ]
        result = c._weighted_score(scores)
        assert abs(result - 0.5) < 0.01

    def test_weighted_score_empty(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={})
        result = c._weighted_score([])
        assert result == 0.0

    def test_weighted_score_capped_at_one(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={"a": 0.5})
        scores = [StrategyScore(strategy_name="a", score=2.0, detail="")]
        result = c._weighted_score(scores)
        assert result <= 1.0


class TestMerge:
    def test_merge_updates_primary(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker)
        p = _make_target("ble_aa", "ble", position=(10.0, 10.0), confidence=0.5)
        s = _make_target("det_p1", "yolo", position=(10.1, 10.1), confidence=0.3)
        old_conf = p.position_confidence
        c._merge(p, s)

        assert p.position_confidence > old_conf
        assert "det_p1" in p.correlated_ids
        assert "yolo" in p.confirming_sources

    def test_merge_takes_better_position(self):
        tracker = TargetTracker()
        c = TargetCorrelator(tracker)
        p = _make_target("ble_aa", "ble", position=(10.0, 10.0), confidence=0.3)
        s = _make_target("det_p1", "yolo", position=(20.0, 20.0), confidence=0.9)
        c._merge(p, s)
        # Secondary has higher confidence so primary gets its position
        assert p.position == (20.0, 20.0)
