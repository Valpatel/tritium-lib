# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.target_prediction."""

import math
import pytest

from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.target_prediction import (
    PredictedPosition,
    predict_target,
    predict_all_targets,
    DEFAULT_HORIZONS,
    MIN_SPEED_THRESHOLD,
    BASE_CONFIDENCE,
    CONE_GROWTH_RATE,
    MIN_SAMPLES,
    VELOCITY_WINDOW_S,
    _get_rl_cone_scale,
)


class TestPredictedPosition:
    def test_fields(self):
        p = PredictedPosition(
            x=10.0, y=20.0, horizon_minutes=5, confidence=0.8,
            cone_radius_m=50.0, heading_deg=90.0, speed_mps=3.0,
        )
        assert p.x == 10.0
        assert p.horizon_minutes == 5

    def test_to_dict(self):
        p = PredictedPosition(
            x=10.123, y=20.456, horizon_minutes=1, confidence=0.85,
            cone_radius_m=10.0, heading_deg=45.5, speed_mps=2.5,
        )
        d = p.to_dict()
        assert d["x"] == 10.12
        assert d["y"] == 20.46
        assert d["horizon_minutes"] == 1
        assert d["confidence"] == 0.85
        assert d["heading_deg"] == 45.5


class TestPredictTarget:
    def _build_trail(self, target_id, history, speed=5.0, n=10, heading_deg=90.0):
        rad = math.radians(heading_deg)
        dx = speed * math.sin(rad)
        dy = speed * math.cos(rad)
        for i in range(n):
            x = float(i * dx)
            y = float(i * dy)
            history.record(target_id, (x, y), timestamp=100.0 + i)

    def test_insufficient_history(self):
        h = TargetHistory()
        h.record("t1", (0.0, 0.0), timestamp=100.0)
        result = predict_target("t1", h)
        assert result == []

    def test_returns_default_horizons(self):
        h = TargetHistory()
        self._build_trail("t1", h)
        preds = predict_target("t1", h, rl_weighted=False)
        assert len(preds) == len(DEFAULT_HORIZONS)
        for p, horizon in zip(preds, DEFAULT_HORIZONS):
            assert p.horizon_minutes == horizon

    def test_custom_horizons(self):
        h = TargetHistory()
        self._build_trail("t1", h)
        preds = predict_target("t1", h, horizons=[2, 7], rl_weighted=False)
        assert len(preds) == 2
        assert preds[0].horizon_minutes == 2

    def test_stationary_target_no_predictions(self):
        h = TargetHistory()
        for i in range(10):
            h.record("still", (5.0, 5.0), timestamp=100.0 + i)
        preds = predict_target("still", h, rl_weighted=False)
        assert preds == []

    def test_confidence_decays_with_horizon(self):
        h = TargetHistory()
        self._build_trail("t1", h)
        preds = predict_target("t1", h, rl_weighted=False)
        assert preds[0].confidence > preds[-1].confidence

    def test_cone_grows_with_horizon(self):
        h = TargetHistory()
        self._build_trail("t1", h)
        preds = predict_target("t1", h, rl_weighted=False)
        assert preds[-1].cone_radius_m > preds[0].cone_radius_m

    def test_prediction_direction_matches_movement(self):
        h = TargetHistory()
        # Moving east (positive x)
        for i in range(10):
            h.record("t1", (float(i * 5), 0.0), timestamp=100.0 + i)
        preds = predict_target("t1", h, rl_weighted=False)
        # Predicted x should be ahead of last position
        assert preds[0].x > 9 * 5

    def test_velocity_window(self):
        h = TargetHistory()
        # Old positions far in the past
        h.record("t1", (0.0, 0.0), timestamp=1.0)
        h.record("t1", (100.0, 0.0), timestamp=2.0)
        # Recent positions at different speed
        for i in range(5):
            h.record("t1", (100.0 + i * 2.0, 0.0), timestamp=100.0 + i)
        preds = predict_target("t1", h, rl_weighted=False)
        # Should use recent velocity, not the old fast velocity
        if preds:
            assert preds[0].speed_mps < 50

    def test_zero_dt_returns_empty(self):
        h = TargetHistory()
        for i in range(5):
            h.record("t1", (float(i), 0.0), timestamp=100.0)  # all same time
        preds = predict_target("t1", h, rl_weighted=False)
        assert preds == []


class TestPredictAllTargets:
    def test_multiple_targets(self):
        h = TargetHistory()
        for i in range(10):
            h.record("moving_a", (float(i * 3), 0.0), timestamp=100.0 + i)
            h.record("moving_b", (0.0, float(i * 4)), timestamp=100.0 + i)
        results = predict_all_targets(["moving_a", "moving_b"], h)
        assert "moving_a" in results
        assert "moving_b" in results

    def test_skips_stationary(self):
        h = TargetHistory()
        for i in range(10):
            h.record("mover", (float(i * 5), 0.0), timestamp=100.0 + i)
            h.record("sitter", (0.0, 0.0), timestamp=100.0 + i)
        results = predict_all_targets(["mover", "sitter"], h)
        assert "mover" in results
        assert "sitter" not in results

    def test_empty_list(self):
        h = TargetHistory()
        results = predict_all_targets([], h)
        assert results == {}


class TestRLConeScale:
    def test_default_no_learner(self):
        scale = _get_rl_cone_scale("test_target")
        assert scale == 1.0

    def test_custom_learner_not_trained(self):
        class FakeLearner:
            is_trained = False
            accuracy = 0.0
        scale = _get_rl_cone_scale("t", correlation_learner_fn=lambda: FakeLearner())
        assert scale == 1.0

    def test_high_accuracy_learner(self):
        class FakeLearner:
            is_trained = True
            accuracy = 0.9
        scale = _get_rl_cone_scale("t", correlation_learner_fn=lambda: FakeLearner())
        assert scale < 1.0  # tighter cone

    def test_low_accuracy_learner(self):
        class FakeLearner:
            is_trained = True
            accuracy = 0.1
        scale = _get_rl_cone_scale("t", correlation_learner_fn=lambda: FakeLearner())
        assert scale > 1.0  # wider cone
