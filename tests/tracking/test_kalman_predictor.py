# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.kalman_predictor."""

import math
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.kalman_predictor import (
    KalmanState,
    kalman_update,
    predict_target_kalman,
    predict_all_targets_kalman,
    clear_kalman_state,
    get_kalman_state,
    _kalman_states,
)
from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.target_prediction import PredictedPosition


@pytest.fixture(autouse=True)
def _clean_kalman_state():
    """Clear global Kalman state before and after each test."""
    _kalman_states.clear()
    yield
    _kalman_states.clear()


class TestKalmanState:
    def test_default_values(self):
        s = KalmanState()
        assert s.x == 0.0
        assert s.y == 0.0
        assert s.vx == 0.0
        assert not s.initialized

    def test_custom_values(self):
        s = KalmanState(x=10.0, y=20.0, vx=1.0, vy=2.0, initialized=True)
        assert s.x == 10.0
        assert s.initialized


class TestKalmanUpdate:
    def test_first_update_initializes(self):
        state = kalman_update("test_1", 10.0, 20.0, timestamp=100.0)
        assert state.x == 10.0
        assert state.y == 20.0
        assert state.initialized
        assert state.vx == 0.0

    def test_second_update_estimates_velocity(self):
        kalman_update("test_1", 0.0, 0.0, timestamp=100.0)
        state = kalman_update("test_1", 10.0, 0.0, timestamp=101.0)
        assert state.initialized
        # Should have positive vx after moving right
        assert state.vx > 0.0

    def test_zero_dt_no_change(self):
        kalman_update("test_1", 0.0, 0.0, timestamp=100.0)
        state = kalman_update("test_1", 10.0, 0.0, timestamp=100.0)
        # Same timestamp — should not update
        assert state.x == 0.0

    def test_large_dt_resets(self):
        kalman_update("test_1", 0.0, 0.0, timestamp=100.0)
        state = kalman_update("test_1", 50.0, 50.0, timestamp=200.0)
        # 100s gap > 60s threshold — should reset
        assert state.x == 50.0
        assert state.vx == 0.0

    def test_sequential_updates_converge(self):
        # Feed a target moving at constant velocity
        for i in range(20):
            kalman_update("linear", float(i * 5), 0.0, timestamp=100.0 + i)
        state = get_kalman_state("linear")
        # Velocity should converge near 5 u/s
        assert abs(state.vx - 5.0) < 2.0

    def test_acceleration_clamped(self):
        kalman_update("accel", 0.0, 0.0, timestamp=100.0)
        # Huge jump to trigger acceleration
        state = kalman_update("accel", 1000.0, 1000.0, timestamp=100.1)
        assert abs(state.ax) <= 10.0
        assert abs(state.ay) <= 10.0


class TestPredictTargetKalman:
    def _build_moving_history(self, target_id, history, speed=5.0, n=10):
        for i in range(n):
            x = float(i * speed)
            history.record(target_id, (x, 0.0), timestamp=100.0 + i)

    def test_insufficient_history(self):
        h = TargetHistory()
        h.record("t1", (0.0, 0.0), timestamp=100.0)
        result = predict_target_kalman("t1", h)
        assert result == []

    def test_returns_predictions(self):
        h = TargetHistory()
        self._build_moving_history("t1", h)
        preds = predict_target_kalman("t1", h, rl_weighted=False)
        assert len(preds) == 3  # default horizons [1, 5, 15]
        for p in preds:
            assert isinstance(p, PredictedPosition)
            assert p.confidence > 0

    def test_custom_horizons(self):
        h = TargetHistory()
        self._build_moving_history("t1", h)
        preds = predict_target_kalman("t1", h, horizons=[2, 10], rl_weighted=False)
        assert len(preds) == 2
        assert preds[0].horizon_minutes == 2
        assert preds[1].horizon_minutes == 10

    def test_confidence_decays_with_horizon(self):
        h = TargetHistory()
        self._build_moving_history("t1", h)
        preds = predict_target_kalman("t1", h, rl_weighted=False)
        assert preds[0].confidence > preds[-1].confidence

    def test_stationary_target_no_predictions(self):
        h = TargetHistory()
        for i in range(10):
            h.record("still", (5.0, 5.0), timestamp=100.0 + i)
        preds = predict_target_kalman("still", h, rl_weighted=False)
        assert preds == []


class TestPredictAllTargetsKalman:
    def test_multiple_targets(self):
        h = TargetHistory()
        for i in range(10):
            h.record("t1", (float(i * 3), 0.0), timestamp=100.0 + i)
            h.record("t2", (0.0, float(i * 4)), timestamp=100.0 + i)
        results = predict_all_targets_kalman(["t1", "t2"], h)
        assert "t1" in results
        assert "t2" in results

    def test_skips_stationary(self):
        h = TargetHistory()
        for i in range(10):
            h.record("moving", (float(i * 5), 0.0), timestamp=100.0 + i)
            h.record("still", (0.0, 0.0), timestamp=100.0 + i)
        results = predict_all_targets_kalman(["moving", "still"], h)
        assert "moving" in results
        assert "still" not in results


class TestClearAndGetState:
    def test_clear_specific(self):
        kalman_update("a", 0.0, 0.0, timestamp=100.0)
        kalman_update("b", 0.0, 0.0, timestamp=100.0)
        clear_kalman_state("a")
        assert get_kalman_state("a") is None
        assert get_kalman_state("b") is not None

    def test_clear_all(self):
        kalman_update("a", 0.0, 0.0, timestamp=100.0)
        kalman_update("b", 0.0, 0.0, timestamp=100.0)
        clear_kalman_state()
        assert get_kalman_state("a") is None
        assert get_kalman_state("b") is None

    def test_get_nonexistent(self):
        assert get_kalman_state("nonexistent") is None
