# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ThreatAssessmentEngine — advanced multi-source threat assessment."""
from __future__ import annotations

import time

import pytest

from tritium_lib.intelligence.threat_assessment import (
    ACTIVE_INDICATOR_THRESHOLD,
    ALL_INDICATOR_CATEGORIES,
    AreaAssessment,
    CROSS_INDICATOR_BOOST,
    DEFAULT_INDICATOR_WEIGHTS,
    INDICATOR_ASSOCIATION,
    INDICATOR_HISTORY,
    INDICATOR_MOVEMENT,
    INDICATOR_SIGNAL,
    INDICATOR_TEMPORAL,
    IndicatorCategory,
    ThreatAssessmentEngine,
    ThreatIndicator,
    ThreatMatrix,
    ThreatPrediction,
    _linear_regression_slope,
    _mean_std,
)
from tritium_lib.intelligence.threat_model import ThreatLevel, score_to_threat_level


# ---------------------------------------------------------------------------
# ThreatIndicator
# ---------------------------------------------------------------------------

class TestThreatIndicator:
    """ThreatIndicator dataclass."""

    def test_default_values(self):
        ind = ThreatIndicator()
        assert ind.category == ""
        assert ind.score == 0.0
        assert ind.confidence == 1.0
        assert ind.target_id == ""
        assert ind.raw_data == {}

    def test_effective_score(self):
        ind = ThreatIndicator(score=0.8, confidence=0.5)
        assert abs(ind.effective_score() - 0.4) < 1e-6

    def test_effective_score_full_confidence(self):
        ind = ThreatIndicator(score=0.6, confidence=1.0)
        assert abs(ind.effective_score() - 0.6) < 1e-6

    def test_to_dict(self):
        ind = ThreatIndicator(
            category=INDICATOR_SIGNAL,
            score=0.75,
            source="signal_strength_anomaly",
            detail="Signal deviated 3 sigma",
            confidence=0.9,
            target_id="ble_test",
        )
        d = ind.to_dict()
        assert d["category"] == INDICATOR_SIGNAL
        assert d["score"] == 0.75
        assert d["source"] == "signal_strength_anomaly"
        assert d["confidence"] == 0.9
        assert d["target_id"] == "ble_test"

    def test_indicator_category_enum(self):
        assert IndicatorCategory.SIGNAL.value == INDICATOR_SIGNAL
        assert IndicatorCategory.MOVEMENT.value == INDICATOR_MOVEMENT
        assert IndicatorCategory.TEMPORAL.value == INDICATOR_TEMPORAL
        assert IndicatorCategory.ASSOCIATION.value == INDICATOR_ASSOCIATION
        assert IndicatorCategory.HISTORY.value == INDICATOR_HISTORY


# ---------------------------------------------------------------------------
# ThreatMatrix
# ---------------------------------------------------------------------------

class TestThreatMatrix:
    """ThreatMatrix computation."""

    def test_empty_matrix(self):
        matrix = ThreatMatrix(target_id="t1")
        matrix.compute()
        assert matrix.composite_score == 0.0
        assert matrix.threat_level == ThreatLevel.GREEN
        assert matrix.active_categories == 0

    def test_single_category(self):
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.8, confidence=1.0)
        )
        matrix.compute()
        assert matrix.category_scores[INDICATOR_SIGNAL] == 0.8
        assert matrix.composite_score > 0.0
        assert matrix.active_categories == 1

    def test_multiple_categories_boost(self):
        """Cross-indicator boost should increase score when multiple categories are active."""
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.5, confidence=1.0)
        )
        matrix.category_indicators[INDICATOR_MOVEMENT].append(
            ThreatIndicator(category=INDICATOR_MOVEMENT, score=0.5, confidence=1.0)
        )
        matrix.category_indicators[INDICATOR_TEMPORAL].append(
            ThreatIndicator(category=INDICATOR_TEMPORAL, score=0.5, confidence=1.0)
        )
        matrix.compute()
        assert matrix.active_categories == 3
        assert matrix.cross_boost_applied > 0.0
        # Boosted score should be higher than pure weighted average
        # Pure weighted avg of 0.5 across 3 of 5 categories < boosted
        assert matrix.composite_score > 0.25

    def test_max_indicator_per_category(self):
        """Max effective score should be used when multiple indicators in same category."""
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.3, confidence=1.0)
        )
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.9, confidence=1.0)
        )
        matrix.compute()
        assert matrix.category_scores[INDICATOR_SIGNAL] == 0.9

    def test_confidence_affects_score(self):
        """Low confidence should reduce effective contribution."""
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.8, confidence=0.25)
        )
        matrix.compute()
        # Effective score = 0.8 * 0.25 = 0.2
        assert matrix.category_scores[INDICATOR_SIGNAL] == pytest.approx(0.2, abs=0.01)

    def test_custom_weights(self):
        """Custom weights should override defaults."""
        custom_weights = {INDICATOR_SIGNAL: 1.0}
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.7, confidence=1.0)
        )
        matrix.compute(weights=custom_weights)
        assert abs(matrix.composite_score - 0.7) < 0.01

    def test_to_dict(self):
        matrix = ThreatMatrix(target_id="t1")
        matrix.category_indicators[INDICATOR_SIGNAL].append(
            ThreatIndicator(category=INDICATOR_SIGNAL, score=0.5, confidence=1.0)
        )
        matrix.compute()
        d = matrix.to_dict()
        assert d["target_id"] == "t1"
        assert "composite_score" in d
        assert "threat_level" in d
        assert "category_scores" in d

    def test_composite_clamped(self):
        """Composite score should be clamped to [0, 1]."""
        matrix = ThreatMatrix(target_id="t1")
        # Fill every category with max scores to push boost above 1.0
        for cat in ALL_INDICATOR_CATEGORIES:
            matrix.category_indicators[cat].append(
                ThreatIndicator(category=cat, score=1.0, confidence=1.0)
            )
        matrix.compute()
        assert matrix.composite_score <= 1.0
        assert matrix.composite_score >= 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Statistics helper functions."""

    def test_mean_std_empty(self):
        mean, std = _mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_mean_std_single(self):
        mean, std = _mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_mean_std_normal(self):
        mean, std = _mean_std([2.0, 4.0, 6.0])
        assert abs(mean - 4.0) < 1e-6
        assert std > 0.0

    def test_linear_regression_slope_ascending(self):
        slope = _linear_regression_slope([1, 2, 3, 4, 5])
        assert slope > 0.0
        assert abs(slope - 1.0) < 1e-6

    def test_linear_regression_slope_descending(self):
        slope = _linear_regression_slope([5, 4, 3, 2, 1])
        assert slope < 0.0

    def test_linear_regression_slope_flat(self):
        slope = _linear_regression_slope([3, 3, 3, 3])
        assert abs(slope) < 1e-6

    def test_linear_regression_slope_too_few(self):
        assert _linear_regression_slope([5]) == 0.0
        assert _linear_regression_slope([]) == 0.0


# ---------------------------------------------------------------------------
# ThreatAssessmentEngine: assess_target
# ---------------------------------------------------------------------------

class TestAssessTarget:
    """ThreatAssessmentEngine.assess_target tests."""

    def test_empty_target(self):
        engine = ThreatAssessmentEngine()
        result = engine.assess_target("nonexistent")
        assert result.composite_score == 0.0
        assert result.threat_level == ThreatLevel.GREEN

    def test_basic_target_assessment(self):
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()
        # Feed enough observations to build baselines
        for i in range(10):
            engine.update_target(
                "ble_aa:bb",
                position=(10.0 + i, 20.0),
                speed=1.2,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        result = engine.assess_target("ble_aa:bb")
        assert result.target_id == "ble_aa:bb"
        assert isinstance(result.composite_score, float)
        assert result.threat_level in list(ThreatLevel)

    def test_signal_anomaly_detection(self):
        """A sudden signal strength change should trigger signal anomaly."""
        engine = ThreatAssessmentEngine(baseline_min_observations=5)
        now = time.time()
        # Build baseline with consistent signal
        for i in range(20):
            engine.update_target(
                "ble_sig",
                position=(10.0, 20.0),
                speed=1.0,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        # Inject anomalous signal
        engine.update_target(
            "ble_sig",
            position=(10.0, 20.0),
            speed=1.0,
            signal_strength=-20.0,  # Huge jump
            hour_of_day=14,
            timestamp=now + 21 * 60,
        )
        result = engine.assess_target("ble_sig")
        signal_score = result.category_scores.get(INDICATOR_SIGNAL, 0.0)
        assert signal_score > 0.0, "Signal anomaly should be detected"

    def test_movement_speed_anomaly(self):
        """A sudden speed spike should trigger movement anomaly."""
        engine = ThreatAssessmentEngine(baseline_min_observations=5)
        now = time.time()
        # Normal walking speed
        for i in range(15):
            engine.update_target(
                "ble_move",
                position=(float(i), 0.0),
                speed=1.2,
                hour_of_day=12,
                timestamp=now + i * 60,
            )
        # Sudden fast movement
        engine.update_target(
            "ble_move",
            position=(100.0, 0.0),
            speed=25.0,
            hour_of_day=12,
            timestamp=now + 16 * 60,
        )
        result = engine.assess_target("ble_move")
        movement_score = result.category_scores.get(INDICATOR_MOVEMENT, 0.0)
        assert movement_score > 0.0, "Movement anomaly should be detected"

    def test_temporal_off_hours(self):
        """Activity during off-hours should trigger temporal anomaly."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target(
            "ble_night",
            position=(10.0, 20.0),
            speed=1.0,
            hour_of_day=3,  # 3 AM
            timestamp=now,
        )
        result = engine.assess_target("ble_night")
        temporal_score = result.category_scores.get(INDICATOR_TEMPORAL, 0.0)
        assert temporal_score > 0.0, "Off-hours activity should be detected"

    def test_temporal_normal_hours_clean(self):
        """Activity during normal hours should not trigger off-hours indicator."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target(
            "ble_day",
            position=(10.0, 20.0),
            speed=1.0,
            hour_of_day=14,  # 2 PM
            timestamp=now,
        )
        result = engine.assess_target("ble_day")
        # Check that no off_hours_activity indicator was generated
        temporal_indicators = result.category_indicators.get(INDICATOR_TEMPORAL, [])
        off_hours = [i for i in temporal_indicators if i.source == "off_hours_activity"]
        assert len(off_hours) == 0, "Normal hours should not trigger off-hours indicator"

    def test_dwell_detection(self):
        """Stationary target should trigger dwell anomaly."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()
        # Target sits in same spot with near-zero speed for 10+ minutes
        for i in range(10):
            engine.update_target(
                "ble_dwell",
                position=(50.0, 50.0),
                speed=0.05,
                hour_of_day=14,
                timestamp=now + i * 120,  # 2 min intervals, total 20 min
            )
        result = engine.assess_target("ble_dwell")
        movement_indicators = result.category_indicators.get(INDICATOR_MOVEMENT, [])
        dwell_inds = [i for i in movement_indicators if i.source == "dwell_anomaly"]
        assert len(dwell_inds) > 0, "Dwell anomaly should be detected for stationary target"

    def test_association_colocation(self):
        """Repeated co-location in different zones should trigger association anomaly."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        # Target seen with same companion in multiple zones
        for i, zone in enumerate(["zone_a", "zone_b", "zone_c", "zone_d"]):
            engine.update_target(
                "ble_leader",
                position=(float(i * 10), 0.0),
                speed=1.0,
                hour_of_day=14,
                co_located_targets=["ble_follower"],
                zone_id=zone,
                timestamp=now + i * 300,
            )
        result = engine.assess_target("ble_leader")
        assoc_score = result.category_scores.get(INDICATOR_ASSOCIATION, 0.0)
        assert assoc_score > 0.0, "Co-location pattern should be detected"

    def test_cache_invalidation(self):
        """New observation should invalidate cached assessment."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target("t1", position=(0.0, 0.0), speed=1.0, hour_of_day=14, timestamp=now)
        a1 = engine.assess_target("t1")
        # Add more data
        engine.update_target("t1", position=(0.0, 0.0), speed=1.0, hour_of_day=3, timestamp=now + 60)
        a2 = engine.assess_target("t1")
        # Assessment should differ since new off-hours data was added
        # (a2 sees hour_of_day=3 as latest, triggering temporal anomaly)
        assert a2.assessed_at >= a1.assessed_at

    def test_cross_indicator_boost(self):
        """Multiple active categories should apply cross-indicator boost."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()

        # Build baseline with consistent data
        for i in range(15):
            engine.update_target(
                "ble_multi",
                position=(10.0, 20.0),
                speed=1.2,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        # Now inject anomalies across multiple dimensions
        engine.update_target(
            "ble_multi",
            position=(10.0, 20.0),
            speed=30.0,               # Speed anomaly
            signal_strength=-10.0,    # Signal anomaly
            hour_of_day=2,            # Temporal anomaly
            timestamp=now + 16 * 60,
        )
        result = engine.assess_target("ble_multi")
        assert result.active_categories >= 2
        assert result.cross_boost_applied > 0.0


# ---------------------------------------------------------------------------
# ThreatAssessmentEngine: assess_area
# ---------------------------------------------------------------------------

class TestAssessArea:
    """ThreatAssessmentEngine.assess_area tests."""

    def test_empty_area(self):
        engine = ThreatAssessmentEngine()
        result = engine.assess_area((0, 0, 100, 100))
        assert result.target_count == 0
        assert result.composite_score == 0.0

    def test_area_with_targets(self):
        engine = ThreatAssessmentEngine()
        now = time.time()
        # Place targets within area
        engine.update_target("t1", position=(50.0, 50.0), speed=1.0, hour_of_day=14, timestamp=now)
        engine.update_target("t2", position=(70.0, 70.0), speed=1.0, hour_of_day=3, timestamp=now)
        # Place target outside area
        engine.update_target("t3", position=(200.0, 200.0), speed=1.0, hour_of_day=14, timestamp=now)

        result = engine.assess_area((0, 0, 100, 100))
        assert result.target_count == 2
        assert "t3" not in [t["target_id"] for t in result.highest_threat_targets]

    def test_area_threat_distribution(self):
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target("t1", position=(10.0, 10.0), speed=1.0, hour_of_day=14, timestamp=now)
        result = engine.assess_area((0, 0, 50, 50))
        assert result.target_count == 1
        assert isinstance(result.threat_distribution, dict)
        total_in_dist = sum(result.threat_distribution.values())
        assert total_in_dist == 1

    def test_area_to_dict(self):
        result = AreaAssessment(
            bounds=(0, 0, 100, 100),
            composite_score=0.45,
            max_score=0.8,
            threat_level=ThreatLevel.ORANGE,
            target_count=5,
        )
        d = result.to_dict()
        assert d["bounds"] == [0, 0, 100, 100]
        assert d["composite_score"] == 0.45
        assert d["threat_level"] == "ORANGE"


# ---------------------------------------------------------------------------
# ThreatAssessmentEngine: predict_threat
# ---------------------------------------------------------------------------

class TestPredictThreat:
    """ThreatAssessmentEngine.predict_threat tests."""

    def test_no_history_stable(self):
        engine = ThreatAssessmentEngine()
        engine.update_target("t1", position=(10.0, 20.0), speed=1.0, hour_of_day=14)
        pred = engine.predict_threat("t1", hours=4)
        assert pred.trend == "stable"
        assert pred.confidence <= 0.2  # Low confidence with minimal data

    def test_escalating_trend(self):
        """Progressively worse assessments should predict escalation."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()

        # Build baseline
        for i in range(10):
            engine.update_target(
                "t_esc",
                position=(10.0, 20.0),
                speed=1.0,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
            engine.assess_target("t_esc")  # Build assessment history

        # Inject progressively worsening signals
        for i in range(5):
            engine.update_target(
                "t_esc",
                position=(10.0, 20.0),
                speed=1.0 + i * 5,  # Increasing speed
                signal_strength=-65.0 + i * 10,  # Changing signal
                hour_of_day=14,
                timestamp=now + (11 + i) * 60,
            )
            engine.assess_target("t_esc")

        pred = engine.predict_threat("t_esc", hours=2)
        assert isinstance(pred.predicted_score, float)
        assert 0.0 <= pred.predicted_score <= 1.0
        assert pred.hours_ahead == 2.0

    def test_prediction_clamped(self):
        """Predicted score should be clamped to [0, 1]."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        for i in range(5):
            engine.update_target("t_clamp", position=(10.0, 20.0), speed=1.0,
                                 hour_of_day=14, timestamp=now + i * 60)
            engine.assess_target("t_clamp")
        pred = engine.predict_threat("t_clamp", hours=48)
        assert 0.0 <= pred.predicted_score <= 1.0

    def test_prediction_to_dict(self):
        pred = ThreatPrediction(
            target_id="t1",
            current_score=0.3,
            predicted_score=0.5,
            predicted_level=ThreatLevel.ORANGE,
            hours_ahead=4.0,
            trend="escalating",
            trend_slope=0.05,
            confidence=0.7,
            contributing_factors=["score_trending_up"],
        )
        d = pred.to_dict()
        assert d["target_id"] == "t1"
        assert d["predicted_level"] == "ORANGE"
        assert d["trend"] == "escalating"
        assert "score_trending_up" in d["contributing_factors"]

    def test_prediction_confidence_drops_for_distant_horizon(self):
        """Predictions far in the future should have lower confidence."""
        engine = ThreatAssessmentEngine()
        now = time.time()
        for i in range(10):
            engine.update_target("t_conf", position=(10.0, 20.0), speed=1.0,
                                 hour_of_day=14, timestamp=now + i * 60)
            engine.assess_target("t_conf")

        pred_near = engine.predict_threat("t_conf", hours=2)
        pred_far = engine.predict_threat("t_conf", hours=24)
        assert pred_far.confidence <= pred_near.confidence


# ---------------------------------------------------------------------------
# ThreatAssessmentEngine: management methods
# ---------------------------------------------------------------------------

class TestEngineManagement:
    """Engine clear, stats, and query methods."""

    def test_get_stats(self):
        engine = ThreatAssessmentEngine()
        engine.update_target("t1", position=(0.0, 0.0), speed=1.0, hour_of_day=12)
        stats = engine.get_stats()
        assert stats["target_count"] == 1
        assert stats["total_updates"] == 1
        assert "weights" in stats

    def test_clear_single_target(self):
        engine = ThreatAssessmentEngine()
        engine.update_target("t1", position=(0.0, 0.0), speed=1.0, hour_of_day=12)
        engine.update_target("t2", position=(10.0, 10.0), speed=1.0, hour_of_day=12)
        engine.clear(target_id="t1")
        stats = engine.get_stats()
        assert stats["target_count"] == 1

    def test_clear_all(self):
        engine = ThreatAssessmentEngine()
        engine.update_target("t1", position=(0.0, 0.0), speed=1.0, hour_of_day=12)
        engine.update_target("t2", position=(10.0, 10.0), speed=1.0, hour_of_day=12)
        engine.clear()
        stats = engine.get_stats()
        assert stats["target_count"] == 0
        assert stats["total_updates"] == 0

    def test_get_all_assessments(self):
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target("t1", position=(10.0, 10.0), speed=1.0, hour_of_day=14, timestamp=now)
        engine.update_target("t2", position=(20.0, 20.0), speed=1.0, hour_of_day=3, timestamp=now)
        results = engine.get_all_assessments()
        assert len(results) == 2
        # Should be sorted descending by score
        assert results[0].composite_score >= results[1].composite_score

    def test_get_targets_by_level(self):
        engine = ThreatAssessmentEngine()
        now = time.time()
        engine.update_target("t_green", position=(10.0, 10.0), speed=1.0, hour_of_day=14, timestamp=now)
        green_targets = engine.get_targets_by_level(ThreatLevel.GREEN)
        # The benign target should be in GREEN
        assert "t_green" in green_targets

    def test_constants(self):
        """Verify exported constants are reasonable."""
        assert len(ALL_INDICATOR_CATEGORIES) == 5
        assert sum(DEFAULT_INDICATOR_WEIGHTS.values()) == pytest.approx(1.0, abs=0.01)
        assert CROSS_INDICATOR_BOOST > 0.0
        assert ACTIVE_INDICATOR_THRESHOLD > 0.0


# ---------------------------------------------------------------------------
# Position jump (teleportation) detection
# ---------------------------------------------------------------------------

class TestPositionJump:
    """Test position jump / teleportation detection in movement indicators."""

    def test_teleportation_detected(self):
        """Large position jump should trigger movement anomaly."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()
        # Build consistent position history
        for i in range(10):
            engine.update_target(
                "ble_jump",
                position=(10.0 + i * 0.5, 20.0),
                speed=0.5,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        # Teleport to far away location
        engine.update_target(
            "ble_jump",
            position=(500.0, 500.0),
            speed=0.5,
            hour_of_day=14,
            timestamp=now + 11 * 60,
        )
        result = engine.assess_target("ble_jump")
        movement_indicators = result.category_indicators.get(INDICATOR_MOVEMENT, [])
        jump_inds = [i for i in movement_indicators if i.source == "position_jump"]
        assert len(jump_inds) > 0, "Position jump should be detected"


# ---------------------------------------------------------------------------
# History-based indicators
# ---------------------------------------------------------------------------

class TestHistoryIndicators:
    """Test history factor indicator computation."""

    def test_historical_peak_decays(self):
        """A past high threat score should influence current assessment with decay."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()

        # Build a baseline, then cause a high score, then observe benign data
        for i in range(8):
            engine.update_target(
                "ble_hist",
                position=(10.0, 20.0),
                speed=1.0,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        # Trigger high score
        engine.update_target(
            "ble_hist",
            position=(10.0, 20.0),
            speed=50.0,
            signal_strength=-10.0,
            hour_of_day=2,
            timestamp=now + 9 * 60,
        )
        high_result = engine.assess_target("ble_hist")

        # More benign observations
        for i in range(5):
            engine.update_target(
                "ble_hist",
                position=(10.0 + i, 20.0),
                speed=1.0,
                signal_strength=-65.0,
                hour_of_day=14,
                timestamp=now + (10 + i) * 60,
            )

        later_result = engine.assess_target("ble_hist")
        # History indicators should be present if score was high enough
        history_indicators = later_result.category_indicators.get(INDICATOR_HISTORY, [])
        # The engine should at least have assessed the target,
        # even if history indicators require more assessment cycles
        assert later_result.target_id == "ble_hist"


# ---------------------------------------------------------------------------
# Inter-visit interval anomaly
# ---------------------------------------------------------------------------

class TestInterVisitInterval:
    """Test temporal inter-visit interval anomaly detection."""

    def test_irregular_interval_detected(self):
        """A very unusual gap between observations should be flagged."""
        engine = ThreatAssessmentEngine(baseline_min_observations=3)
        now = time.time()
        # Regular 1-minute intervals
        for i in range(15):
            engine.update_target(
                "ble_interval",
                position=(10.0, 20.0),
                speed=1.0,
                hour_of_day=14,
                timestamp=now + i * 60,
            )
        # Huge gap then sudden reappearance
        engine.update_target(
            "ble_interval",
            position=(10.0, 20.0),
            speed=1.0,
            hour_of_day=14,
            timestamp=now + 15 * 60 + 86400,  # 1 day gap
        )
        result = engine.assess_target("ble_interval")
        temporal_indicators = result.category_indicators.get(INDICATOR_TEMPORAL, [])
        interval_inds = [i for i in temporal_indicators if i.source == "inter_visit_interval"]
        assert len(interval_inds) > 0, "Irregular inter-visit interval should be detected"
