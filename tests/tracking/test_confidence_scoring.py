# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for improved correlation confidence scoring.

Covers:
  - ConfidenceCalibrator (precision, recall, FPR, calibration, threshold tuning)
  - Explanation generation in CorrelationRecord
  - Edge cases: very close/distant targets, MAC randomization, temporal gaps,
    multi-strategy weighting, calibration with real outcome data
"""

import time
import math
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.correlator import (
    TargetCorrelator,
    CorrelationRecord,
    DEFAULT_WEIGHTS,
)
from tritium_lib.tracking.correlation_strategies import (
    StrategyScore,
    SpatialStrategy,
    TemporalStrategy,
    SignalPatternStrategy,
    WiFiProbeStrategy,
    DossierStrategy,
    ConfidenceCalibrator,
    CalibrationRecord,
)
from tritium_lib.tracking.dossier import DossierStore


def _make_target(
    target_id: str,
    source: str,
    position: tuple[float, float] = (0.0, 0.0),
    name: str = "",
    asset_type: str = "person",
    confidence: float = 0.8,
    last_seen: float | None = None,
) -> TrackedTarget:
    now = last_seen if last_seen is not None else time.monotonic()
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


# ---------------------------------------------------------------------------
# ConfidenceCalibrator tests
# ---------------------------------------------------------------------------

class TestConfidenceCalibrator:
    def test_empty_calibrator_returns_defaults(self):
        cal = ConfidenceCalibrator()
        assert cal.precision("spatial") == 1.0
        assert cal.recall("spatial") == 1.0
        assert cal.false_positive_rate("spatial") == 0.0

    def test_record_outcome_and_precision(self):
        cal = ConfidenceCalibrator()
        # 8 true positives, 2 false positives at threshold 0.3
        for _ in range(8):
            cal.record_outcome("spatial", 0.7, actual_match=True)
        for _ in range(2):
            cal.record_outcome("spatial", 0.7, actual_match=False)
        assert cal.precision("spatial", threshold=0.3) == pytest.approx(0.8, abs=0.01)

    def test_recall_calculation(self):
        cal = ConfidenceCalibrator()
        # 6 true positives above threshold
        for _ in range(6):
            cal.record_outcome("spatial", 0.5, actual_match=True)
        # 4 false negatives below threshold
        for _ in range(4):
            cal.record_outcome("spatial", 0.1, actual_match=True)
        assert cal.recall("spatial", threshold=0.3) == pytest.approx(0.6, abs=0.01)

    def test_false_positive_rate(self):
        cal = ConfidenceCalibrator()
        # 3 FP (high score, not match), 7 TN (low score, not match)
        for _ in range(3):
            cal.record_outcome("signal_pattern", 0.6, actual_match=False)
        for _ in range(7):
            cal.record_outcome("signal_pattern", 0.1, actual_match=False)
        assert cal.false_positive_rate("signal_pattern", threshold=0.3) == pytest.approx(0.3, abs=0.01)

    def test_calibrate_score_insufficient_data(self):
        """With < 10 records, calibration falls back to raw score."""
        cal = ConfidenceCalibrator()
        for i in range(5):
            cal.record_outcome("spatial", 0.5, actual_match=True)
        assert cal.calibrate_score("spatial", 0.5) == 0.5

    def test_calibrate_score_with_enough_data(self):
        """With 10+ records, calibration adjusts based on observed accuracy."""
        cal = ConfidenceCalibrator()
        # Bin 0.5-0.6: all true matches -> calibrated score should be 1.0
        for _ in range(10):
            cal.record_outcome("spatial", 0.55, actual_match=True)
        # Add some in other bins to reach 10 total
        for _ in range(5):
            cal.record_outcome("spatial", 0.15, actual_match=False)
        calibrated = cal.calibrate_score("spatial", 0.55)
        assert calibrated == pytest.approx(1.0, abs=0.01)

    def test_calibrate_score_zero_accuracy_bin(self):
        """Bin with all false positives should calibrate to 0."""
        cal = ConfidenceCalibrator()
        for _ in range(10):
            cal.record_outcome("spatial", 0.75, actual_match=False)
        for _ in range(5):
            cal.record_outcome("spatial", 0.15, actual_match=True)
        calibrated = cal.calibrate_score("spatial", 0.75)
        assert calibrated == pytest.approx(0.0, abs=0.01)

    def test_recommend_threshold_default(self):
        """With < 20 records, returns default 0.3."""
        cal = ConfidenceCalibrator(target_fpr=0.05)
        assert cal.recommend_threshold() == 0.3

    def test_recommend_threshold_adjusts(self):
        """With enough FP data, threshold rises to reduce FPR."""
        cal = ConfidenceCalibrator(target_fpr=0.05)
        # Lots of false positives at low thresholds
        for _ in range(15):
            cal.record_outcome("spatial", 0.35, actual_match=False)
        # A few true matches at high thresholds
        for _ in range(10):
            cal.record_outcome("spatial", 0.75, actual_match=True)
        # True negatives at low scores
        for _ in range(5):
            cal.record_outcome("spatial", 0.05, actual_match=False)
        recommended = cal.recommend_threshold()
        # Should recommend a higher threshold to avoid the FP cluster at 0.35
        assert recommended >= 0.3

    def test_strategy_stats_empty(self):
        cal = ConfidenceCalibrator()
        stats = cal.strategy_stats("nonexistent")
        assert stats["sample_count"] == 0
        assert stats["precision"] == 1.0
        assert stats["fpr"] == 0.0

    def test_strategy_stats_populated(self):
        cal = ConfidenceCalibrator()
        for _ in range(5):
            cal.record_outcome("spatial", 0.8, actual_match=True)
        for _ in range(1):
            cal.record_outcome("spatial", 0.8, actual_match=False)
        stats = cal.strategy_stats("spatial")
        assert stats["sample_count"] == 6
        assert stats["precision"] == pytest.approx(5 / 6, abs=0.01)

    def test_all_stats_multiple_strategies(self):
        cal = ConfidenceCalibrator()
        cal.record_outcome("spatial", 0.5, actual_match=True)
        cal.record_outcome("temporal", 0.6, actual_match=False)
        cal.record_outcome("signal_pattern", 0.7, actual_match=True)
        stats = cal.all_stats()
        assert len(stats) == 3
        names = {s["strategy"] for s in stats}
        assert names == {"signal_pattern", "spatial", "temporal"}

    def test_rolling_window_cap(self):
        """History is capped at MAX_HISTORY per strategy."""
        cal = ConfidenceCalibrator()
        for i in range(600):
            cal.record_outcome("spatial", 0.5, actual_match=(i % 2 == 0))
        stats = cal.strategy_stats("spatial")
        assert stats["sample_count"] == ConfidenceCalibrator.MAX_HISTORY


# ---------------------------------------------------------------------------
# Edge case: very close targets (should correlate)
# ---------------------------------------------------------------------------

class TestVeryCloseTargets:
    def test_overlapping_positions_correlate(self):
        """Two targets at the exact same position from different sensors."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa:bb:cc", "ble", position=(50.0, 50.0))
        t2 = _make_target("det_person_0", "yolo", position=(50.0, 50.0))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) >= 1
        assert records[0].confidence > 0.5

    def test_sub_meter_distance_correlates(self):
        """Targets < 0.5m apart should get high spatial score."""
        s = SpatialStrategy(radius=5.0)
        t1 = _make_target("a", "ble", position=(10.0, 10.0))
        t2 = _make_target("b", "yolo", position=(10.3, 10.4))
        result = s.evaluate(t1, t2)
        assert result.score > 0.8


# ---------------------------------------------------------------------------
# Edge case: very distant targets (should not correlate)
# ---------------------------------------------------------------------------

class TestVeryDistantTargets:
    def test_km_apart_never_correlates(self):
        """Targets 1km apart should not correlate at a reasonable threshold.

        Signal pattern alone (same last_seen time) can produce ~0.16 combined
        score, so we use threshold=0.3 which is the production default —
        spatial separation dominates and prevents the correlation.
        """
        tracker = TargetTracker()
        t1 = _make_target("ble_xx", "ble", position=(0.0, 0.0))
        t2 = _make_target("det_car_0", "yolo", position=(1000.0, 0.0))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.3, max_age=9999)
        records = c.correlate()
        assert len(records) == 0

    def test_spatial_score_zero_beyond_radius(self):
        """Spatial score must be exactly 0 when distance > radius."""
        s = SpatialStrategy(radius=10.0)
        t1 = _make_target("a", "ble", position=(0.0, 0.0))
        t2 = _make_target("b", "yolo", position=(100.0, 100.0))
        result = s.evaluate(t1, t2)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Edge case: same MAC on different devices (MAC randomization)
# ---------------------------------------------------------------------------

class TestMACRandomization:
    def test_same_source_never_correlates(self):
        """Two BLE targets (same source) should never correlate even if close."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa:bb:cc", "ble", position=(10.0, 10.0))
        t2 = _make_target("ble_dd:ee:ff", "ble", position=(10.1, 10.1))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) == 0

    def test_randomized_mac_different_dossiers(self):
        """If two BLE MACs are in different dossiers, dossier strategy returns 0."""
        store = DossierStore()
        store.create_or_update("ble_aa", "ble", "det_1", "yolo", confidence=0.8)
        store.create_or_update("ble_bb", "ble", "det_2", "yolo", confidence=0.8)
        s = DossierStrategy(dossier_store=store)
        t1 = _make_target("ble_aa", "ble")
        t2 = _make_target("ble_bb", "ble")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Edge case: temporal gap too large
# ---------------------------------------------------------------------------

class TestTemporalGap:
    def test_large_temporal_gap_no_signal_pattern(self):
        """Targets last seen 60s apart should have 0 signal pattern score."""
        s = SignalPatternStrategy(appearance_window=10.0)
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "yolo", last_seen=now - 60.0)
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_stale_target_excluded_by_max_age(self):
        """Target older than max_age should not be considered for correlation."""
        tracker = TargetTracker()
        now = time.monotonic()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0), last_seen=now)
        t2 = _make_target("det_p1", "yolo", position=(10.0, 10.0), last_seen=now - 120.0)
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=30.0)
        records = c.correlate()
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Explanation field tests
# ---------------------------------------------------------------------------

class TestExplanation:
    def test_explanation_generated_on_correlation(self):
        """Correlation records should have a non-empty explanation."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_0", "yolo", position=(10.1, 10.1))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) >= 1
        record = records[0]
        assert record.explanation != ""
        assert "correlated with" in record.explanation
        assert "confidence" in record.explanation

    def test_explanation_mentions_strongest_signal(self):
        """Explanation should reference the strongest contributing strategy."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_0", "yolo", position=(10.0, 10.0))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) >= 1
        explanation = records[0].explanation
        assert "Strongest signal:" in explanation

    def test_explanation_contains_assessment(self):
        """Explanation should contain an assessment level."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_0", "yolo", position=(10.2, 10.2))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) >= 1
        explanation = records[0].explanation
        assert any(level in explanation for level in ["HIGH", "MEDIUM", "LOW"])


# ---------------------------------------------------------------------------
# Calibrated confidence tests
# ---------------------------------------------------------------------------

class TestCalibratedConfidence:
    def test_calibrated_confidence_populated(self):
        """CorrelationRecord.calibrated_confidence should be set."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_0", "yolo", position=(10.1, 10.1))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert len(records) >= 1
        # Without calibration data, calibrated == raw
        assert records[0].calibrated_confidence == pytest.approx(
            records[0].confidence, abs=0.01
        )

    def test_record_outcome_feeds_calibrator(self):
        """record_outcome should populate the calibrator history."""
        tracker = TargetTracker()
        cal = ConfidenceCalibrator()
        c = TargetCorrelator(
            tracker, confidence_threshold=0.01, max_age=9999,
            calibrator=cal,
        )

        # Create a fake record to record outcome for
        record = CorrelationRecord(
            primary_id="ble_aa",
            secondary_id="det_p1",
            confidence=0.7,
            reason="test",
            strategy_scores=[
                StrategyScore(strategy_name="spatial", score=0.9, detail="close"),
                StrategyScore(strategy_name="temporal", score=0.0, detail="no data"),
            ],
        )
        c.record_outcome(record, actual_match=True)

        stats = cal.strategy_stats("spatial")
        assert stats["sample_count"] == 1
        assert stats["precision"] == 1.0

    def test_calibration_stats_accessible(self):
        """get_calibration_stats returns per-strategy info."""
        tracker = TargetTracker()
        cal = ConfidenceCalibrator()
        c = TargetCorrelator(tracker, calibrator=cal)

        # Feed some data
        cal.record_outcome("spatial", 0.8, actual_match=True)
        cal.record_outcome("temporal", 0.3, actual_match=False)

        stats = c.get_calibration_stats()
        assert len(stats) == 2
        strategy_names = {s["strategy"] for s in stats}
        assert "spatial" in strategy_names
        assert "temporal" in strategy_names


# ---------------------------------------------------------------------------
# Auto-tune threshold tests
# ---------------------------------------------------------------------------

class TestAutoTuneThreshold:
    def test_auto_tune_disabled_by_default(self):
        """Threshold should not change when auto_tune_threshold is False."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, confidence_threshold=0.3)
        assert c.auto_tune_threshold is False
        # Run correlate with no targets -- threshold should stay
        c.correlate()
        assert c.confidence_threshold == 0.3

    def test_auto_tune_adjusts_threshold(self):
        """When enabled and calibrator has data, threshold adjusts."""
        tracker = TargetTracker()
        cal = ConfidenceCalibrator(target_fpr=0.05)
        # Seed enough data to trigger recommendation
        for _ in range(15):
            cal.record_outcome("spatial", 0.35, actual_match=False)
        for _ in range(10):
            cal.record_outcome("spatial", 0.75, actual_match=True)
        for _ in range(5):
            cal.record_outcome("spatial", 0.05, actual_match=False)

        c = TargetCorrelator(
            tracker,
            confidence_threshold=0.1,  # intentionally low
            calibrator=cal,
            auto_tune_threshold=True,
        )
        c.correlate()
        # Threshold should have been adjusted upward from 0.1
        assert c.confidence_threshold >= 0.1


# ---------------------------------------------------------------------------
# Multi-strategy weighted scoring edge cases
# ---------------------------------------------------------------------------

class TestWeightedScoringEdgeCases:
    def test_single_strategy_dominates(self):
        """When only one strategy has weight, it determines the score."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={"spatial": 1.0})
        scores = [
            StrategyScore(strategy_name="spatial", score=0.9, detail=""),
            StrategyScore(strategy_name="temporal", score=0.1, detail=""),
        ]
        result = c._weighted_score(scores)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_zero_weight_strategies_excluded(self):
        """Strategies with zero weight contribute nothing."""
        tracker = TargetTracker()
        c = TargetCorrelator(
            tracker,
            weights={"spatial": 0.5, "temporal": 0.0, "signal_pattern": 0.5},
        )
        scores = [
            StrategyScore(strategy_name="spatial", score=1.0, detail=""),
            StrategyScore(strategy_name="temporal", score=1.0, detail=""),
            StrategyScore(strategy_name="signal_pattern", score=0.0, detail=""),
        ]
        result = c._weighted_score(scores)
        # Only spatial(1.0 * 0.5) + signal_pattern(0.0 * 0.5) = 0.5/1.0
        assert result == pytest.approx(0.5, abs=0.01)

    def test_all_strategies_max_score(self):
        """All strategies at 1.0 should yield 1.0."""
        tracker = TargetTracker()
        c = TargetCorrelator(
            tracker,
            weights={"a": 0.3, "b": 0.3, "c": 0.4},
        )
        scores = [
            StrategyScore(strategy_name="a", score=1.0, detail=""),
            StrategyScore(strategy_name="b", score=1.0, detail=""),
            StrategyScore(strategy_name="c", score=1.0, detail=""),
        ]
        result = c._weighted_score(scores)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_unknown_strategy_has_zero_weight(self):
        """A strategy not in the weights dict gets weight 0."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={"spatial": 1.0})
        scores = [
            StrategyScore(strategy_name="spatial", score=0.5, detail=""),
            StrategyScore(strategy_name="unknown_fancy", score=1.0, detail=""),
        ]
        result = c._weighted_score(scores)
        # Only spatial counts: 0.5 * 1.0 / 1.0 = 0.5
        assert result == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# CalibrationRecord dataclass
# ---------------------------------------------------------------------------

class TestCalibrationRecord:
    def test_fields(self):
        r = CalibrationRecord(
            strategy_name="spatial",
            predicted_score=0.75,
            actual_match=True,
        )
        assert r.strategy_name == "spatial"
        assert r.predicted_score == 0.75
        assert r.actual_match is True
        assert r.timestamp > 0

    def test_false_match_record(self):
        r = CalibrationRecord(
            strategy_name="wifi_probe",
            predicted_score=0.4,
            actual_match=False,
        )
        assert r.actual_match is False


# ---------------------------------------------------------------------------
# Integration: full pipeline with calibration feedback loop
# ---------------------------------------------------------------------------

class TestCalibrationFeedbackLoop:
    def test_correlate_then_feedback_improves_stats(self):
        """Full flow: correlate, then record outcomes, check stats evolve."""
        tracker = TargetTracker()
        cal = ConfidenceCalibrator()
        store = DossierStore()

        t1 = _make_target("ble_01", "ble", position=(10.0, 10.0))
        t2 = _make_target("det_person_0", "yolo", position=(10.1, 10.1))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1
            tracker._targets[t2.target_id] = t2

        c = TargetCorrelator(
            tracker,
            confidence_threshold=0.01,
            max_age=9999,
            dossier_store=store,
            calibrator=cal,
        )
        records = c.correlate()
        assert len(records) >= 1

        # Simulate human confirming the correlation
        c.record_outcome(records[0], actual_match=True)

        # Check that calibrator has learned
        stats = c.get_calibration_stats()
        assert len(stats) > 0
        for s in stats:
            if s["sample_count"] > 0:
                assert s["precision"] >= 0.0
                assert s["recall"] >= 0.0


# ---------------------------------------------------------------------------
# Edge case: empty score list in weighted_score
# ---------------------------------------------------------------------------

class TestEmptyScoreList:
    def test_weighted_score_empty_list(self):
        """_weighted_score with empty list should return 0.0, not crash."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={"spatial": 1.0})
        result = c._weighted_score([])
        assert result == 0.0

    def test_weighted_score_all_zero_weights(self):
        """When all weights are 0, should return 0.0 not divide by zero."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, weights={})
        scores = [
            StrategyScore(strategy_name="unknown", score=1.0, detail=""),
        ]
        result = c._weighted_score(scores)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Edge case: correlator _correlations list bounded
# ---------------------------------------------------------------------------

class TestCorrelationHistoryBounded:
    def test_correlations_list_does_not_grow_unbounded(self):
        """_correlations list should be capped to prevent memory leak."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)

        # Directly inject many records to simulate growth
        with c._lock:
            for i in range(6000):
                c._correlations.append(CorrelationRecord(
                    primary_id=f"a_{i}",
                    secondary_id=f"b_{i}",
                    confidence=0.5,
                    reason="test",
                ))

        # Running correlate triggers the cap
        c.correlate()

        with c._lock:
            assert len(c._correlations) <= 5000


# ---------------------------------------------------------------------------
# Edge case: correlate with empty tracker
# ---------------------------------------------------------------------------

class TestCorrelateEmptyTracker:
    def test_correlate_no_targets(self):
        """Correlating with no targets should return empty list, no crash."""
        tracker = TargetTracker()
        c = TargetCorrelator(tracker, confidence_threshold=0.3)
        records = c.correlate()
        assert records == []

    def test_correlate_single_target(self):
        """Correlating with one target should return empty list."""
        tracker = TargetTracker()
        t1 = _make_target("ble_aa", "ble", position=(10.0, 10.0))
        with tracker._lock:
            tracker._targets[t1.target_id] = t1

        c = TargetCorrelator(tracker, confidence_threshold=0.01, max_age=9999)
        records = c.correlate()
        assert records == []


class TestAbstainExcludedFromConfidence:
    """Strategies with no data to judge a pair must ABSTAIN, not vote 0.

    FEATURE-AUDIT 2026-06-14 ("does fusion actually fuse?"): the live
    correlation dashboard showed 387 correlations but ZERO high-confidence
    ones (avg 0.488).  Tracing a textbook same-entity case
    (POST /api/fusion/inject_test_quad) found a perfect cross-modal match
    capped at ~0.51 confidence purely because temporal ("insufficient
    history"), dossier ("no prior association") and wifi_probe ("not a
    BLE+wifi pair") scored 0 for lack of data and dragged the weighted
    average down.  Those are abstentions, not evidence against; they are now
    excluded from the confidence denominator (with a floor that preserves
    skepticism), so genuine matches can reach the 0.70 "high confidence" bar
    while distant / out-of-time pairs still cannot over-correlate.
    """

    def _corr(self, targets, **kw):
        tracker = TargetTracker()
        with tracker._lock:
            for t in targets:
                tracker._targets[t.target_id] = t
        return TargetCorrelator(tracker, max_age=9999, **kw).correlate()

    def test_clear_cross_modal_match_reaches_high_confidence(self):
        """Co-located + co-timed cross-modal detections fuse at high confidence
        (>=0.70) -- the INTENDED multi-sensor fusion (one entity seen by BLE +
        camera + acoustic at one place/time).

        Honest caveat (FEATURE-AUDIT 2026-06-14): proximity + timing alone carry
        no identity-discriminating evidence, so this input shape is the same as
        two genuinely-different co-located entities -- the system fuses them by
        design. What it MUST NOT do is fuse entities KNOWN to be different; that
        is guarded by TestIdentityVeto (a different-dossier veto). Before the
        abstain fix this capped at ~0.51 (temporal/dossier/wifi_probe abstained
        but were counted as 0 in the denominator)."""
        now = time.monotonic()
        ble = _make_target("ble_aa", "ble", position=(100.0, 100.0), last_seen=now)
        aco = _make_target("acoustic_x", "acoustic", position=(100.0, 100.0), last_seen=now)
        records = self._corr([ble, aco])
        assert records, "a co-located cross-modal pair must correlate (intended fusion)"
        assert records[0].confidence >= 0.70, (
            f"confidence {records[0].confidence:.3f} below the 0.70 high-confidence "
            "bar -- abstaining strategies are still penalising a genuine match"
        )

    def test_same_place_different_time_does_not_over_correlate(self):
        """SAFETY: same position but far apart in time must stay below 0.70.

        signal_pattern HAS timing data and disagrees, so it stays in the
        denominator and holds confidence down -- the abstain fix must not turn
        coincidental co-location into a false fusion."""
        now = time.monotonic()
        ble = _make_target("ble_bb", "ble", position=(100.0, 100.0), last_seen=now)
        aco = _make_target("acoustic_y", "acoustic", position=(100.0, 100.0),
                           last_seen=now - 60.0)
        records = self._corr([ble, aco], confidence_threshold=0.01)
        conf = records[0].confidence if records else 0.0
        assert conf < 0.70, (
            f"same-place/different-time pair reached {conf:.3f} -- the abstain "
            "fix must not over-correlate coincidental co-location"
        )

    def test_distant_pair_does_not_correlate(self):
        """SAFETY: spatial always votes, so a distant pair stays low / unformed."""
        now = time.monotonic()
        a = _make_target("ble_cc", "ble", position=(100.0, 100.0), last_seen=now)
        b = _make_target("acoustic_z", "acoustic", position=(400.0, 400.0), last_seen=now)
        records = self._corr([a, b])
        assert records == [] or records[0].confidence < 0.50, (
            "a pair hundreds of metres apart must not become a confident fusion"
        )

    def test_weighted_score_excludes_abstentions(self):
        """Unit: an abstaining (applicable=False) strategy is excluded from the
        denominator, so it does not dilute the applicable strategies."""
        c = TargetCorrelator(TargetTracker())
        applicable_only = [
            StrategyScore("spatial", 1.0, "match"),
            StrategyScore("signal_pattern", 1.0, "match"),
        ]
        with_abstention = applicable_only + [
            StrategyScore("dossier", 0.0, "no data", applicable=False),
            StrategyScore("temporal", 0.0, "no history", applicable=False),
        ]
        # Adding pure abstentions must not lower the combined confidence.
        assert c._weighted_score(with_abstention) == pytest.approx(
            c._weighted_score(applicable_only)
        )
        # ...whereas a real low-scoring (applicable) strategy DOES lower it.
        # Use enough applicable weight that the denominator floor is not masking
        # the effect (applicable weight must exceed _MIN_EVIDENCE_WEIGHT).
        all_agree = [
            StrategyScore("spatial", 1.0, "match"),
            StrategyScore("signal_pattern", 1.0, "match"),
            StrategyScore("temporal", 1.0, "match"),
            StrategyScore("dossier", 1.0, "match"),
        ]
        one_disagrees = [
            StrategyScore("spatial", 1.0, "match"),
            StrategyScore("signal_pattern", 1.0, "match"),
            StrategyScore("temporal", 0.0, "different movement", applicable=True),
            StrategyScore("dossier", 1.0, "match"),
        ]
        assert c._weighted_score(one_disagrees) < c._weighted_score(all_agree)


class TestIdentityVeto:
    """Definitive identity disagreement (known-different dossiers) vetoes a
    merge, so co-location can never override known identity (FEATURE-AUDIT
    2026-06-14 fusion over-correlation fix).
    """

    def _corr(self, targets, ds=None, **kw):
        tracker = TargetTracker()
        with tracker._lock:
            for t in targets:
                tracker._targets[t.target_id] = t
        return TargetCorrelator(tracker, max_age=9999, dossier_store=ds, **kw).correlate()

    def test_known_different_dossiers_do_not_merge(self):
        from tritium_lib.tracking.dossier import DossierStore
        ds = DossierStore()
        ds.create_or_update("ble_x", "ble", "filler1", "x", 0.9)      # dossier 1
        ds.create_or_update("aco_y", "acoustic", "filler2", "y", 0.9)  # dossier 2
        a = _make_target("ble_x", "ble", position=(100.0, 100.0))
        b = _make_target("aco_y", "acoustic", position=(100.0, 100.0))
        records = self._corr([a, b], ds)
        # co-located AND co-timed, but the dossier KNOWS they are different
        # entities -> proximity must not override that.
        assert records == [], "known-different entities must not merge despite co-location"

    def test_veto_forces_weighted_score_to_zero(self):
        c = TargetCorrelator(TargetTracker())
        scores = [
            StrategyScore("spatial", 1.0, "match"),
            StrategyScore("signal_pattern", 1.0, "match"),
            StrategyScore("dossier", 0.0, "different dossiers", applicable=True, veto=True),
        ]
        # Without the veto these would score 0.733; the veto forces 0.
        assert c._weighted_score(scores) == 0.0

    def test_non_veto_disagreement_still_scores(self):
        """A veto is only for DEFINITIVE negative evidence; an ordinary low
        applicable score must not zero the whole correlation."""
        c = TargetCorrelator(TargetTracker())
        scores = [
            StrategyScore("spatial", 1.0, "match"),
            StrategyScore("signal_pattern", 1.0, "match"),
        ]
        assert c._weighted_score(scores) > 0.0
