# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Kalman filter predictor — replaces linear extrapolation with a proper
state estimator that accounts for velocity changes and turning.

State vector: [x, y, vx, vy, ax, ay] (position, velocity, acceleration)
Measurement: [x, y] (observed position)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .target_history import TargetHistory
from .target_prediction import (
    PredictedPosition,
    DEFAULT_HORIZONS,
    MIN_SPEED_THRESHOLD,
    BASE_CONFIDENCE,
    CONE_GROWTH_RATE,
    MIN_SAMPLES,
    _get_rl_cone_scale,
)


@dataclass(slots=True)
class KalmanState:
    """Internal Kalman filter state for a single target."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0

    p_x: float = 100.0
    p_y: float = 100.0
    p_vx: float = 10.0
    p_vy: float = 10.0
    p_ax: float = 1.0
    p_ay: float = 1.0

    last_update: float = 0.0
    initialized: bool = False


# Process noise
Q_POS = 0.1
Q_VEL = 1.0
Q_ACC = 5.0

# Measurement noise
R_POS = 2.0

# Acceleration damping factor
ACC_DECAY = 0.9

# Cache of Kalman states per target
_kalman_states: dict[str, KalmanState] = {}


def _get_or_create_state(target_id: str) -> KalmanState:
    if target_id not in _kalman_states:
        _kalman_states[target_id] = KalmanState()
    return _kalman_states[target_id]


def kalman_update(
    target_id: str,
    x: float,
    y: float,
    timestamp: float | None = None,
) -> KalmanState:
    """Feed a new position measurement into the Kalman filter."""
    if timestamp is None:
        timestamp = time.monotonic()

    state = _get_or_create_state(target_id)

    if not state.initialized:
        state.x = x
        state.y = y
        state.vx = 0.0
        state.vy = 0.0
        state.ax = 0.0
        state.ay = 0.0
        state.last_update = timestamp
        state.initialized = True
        return state

    dt = timestamp - state.last_update
    if dt <= 0:
        return state
    if dt > 60.0:
        state.x = x
        state.y = y
        state.vx = 0.0
        state.vy = 0.0
        state.ax = 0.0
        state.ay = 0.0
        state.last_update = timestamp
        return state

    # --- Predict step ---
    dt2 = 0.5 * dt * dt
    pred_x = state.x + state.vx * dt + state.ax * dt2
    pred_y = state.y + state.vy * dt + state.ay * dt2
    pred_vx = state.vx + state.ax * dt
    pred_vy = state.vy + state.ay * dt
    decay = ACC_DECAY ** dt
    pred_ax = state.ax * decay
    pred_ay = state.ay * decay

    pp_x = state.p_x + state.p_vx * dt * dt + Q_POS * dt
    pp_y = state.p_y + state.p_vy * dt * dt + Q_POS * dt
    pp_vx = state.p_vx + Q_VEL * dt
    pp_vy = state.p_vy + Q_VEL * dt
    pp_ax = state.p_ax + Q_ACC * dt
    pp_ay = state.p_ay + Q_ACC * dt

    # --- Update step ---
    innov_x = x - pred_x
    innov_y = y - pred_y

    s_x = pp_x + R_POS
    s_y = pp_y + R_POS

    k_x = pp_x / s_x
    k_y = pp_y / s_y

    alpha_v = 0.3
    alpha_a = 0.1

    state.x = pred_x + k_x * innov_x
    state.y = pred_y + k_y * innov_y

    if dt > 0.01:
        state.vx = pred_vx + alpha_v * innov_x / dt
        state.vy = pred_vy + alpha_v * innov_y / dt
        state.ax = pred_ax + alpha_a * innov_x / (dt * dt)
        state.ay = pred_ay + alpha_a * innov_y / (dt * dt)
    else:
        state.vx = pred_vx
        state.vy = pred_vy
        state.ax = pred_ax
        state.ay = pred_ay

    max_acc = 10.0
    state.ax = max(-max_acc, min(max_acc, state.ax))
    state.ay = max(-max_acc, min(max_acc, state.ay))

    state.p_x = (1.0 - k_x) * pp_x
    state.p_y = (1.0 - k_y) * pp_y
    state.p_vx = pp_vx * 0.95
    state.p_vy = pp_vy * 0.95
    state.p_ax = pp_ax * 0.98
    state.p_ay = pp_ay * 0.98

    state.last_update = timestamp
    return state


def predict_target_kalman(
    target_id: str,
    history: TargetHistory,
    horizons: list[int] | None = None,
    sample_count: int = 20,
    rl_weighted: bool = True,
) -> list[PredictedPosition]:
    """Predict future positions using Kalman filter state estimation."""
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    trail = history.get_trail(target_id, max_points=sample_count)
    if len(trail) < MIN_SAMPLES:
        return []

    state = _get_or_create_state(target_id)
    if not state.initialized:
        for x, y, t in trail:
            kalman_update(target_id, x, y, t)
    else:
        x, y, t = trail[-1]
        kalman_update(target_id, x, y, t)

    state = _kalman_states[target_id]

    speed = math.hypot(state.vx, state.vy)
    if speed < MIN_SPEED_THRESHOLD:
        return []

    heading = math.degrees(math.atan2(state.vx, state.vy)) % 360
    rl_scale = _get_rl_cone_scale(target_id) if rl_weighted else 1.0

    predictions = []
    for h_min in horizons:
        dt_s = h_min * 60.0
        dt2 = 0.5 * dt_s * dt_s

        if abs(1.0 - ACC_DECAY) > 1e-9:
            ln_decay = math.log(ACC_DECAY)
            decay_integral = (ACC_DECAY ** dt_s - 1.0) / ln_decay
            decay_double_integral = (
                (ACC_DECAY ** dt_s - 1.0) / ln_decay - dt_s
            ) / ln_decay
        else:
            decay_integral = dt_s
            decay_double_integral = dt2

        pred_x = state.x + state.vx * dt_s + state.ax * decay_double_integral
        pred_y = state.y + state.vy * dt_s + state.ay * decay_double_integral

        pred_vx = state.vx + state.ax * decay_integral
        pred_vy = state.vy + state.ay * decay_integral
        pred_speed = math.hypot(pred_vx, pred_vy)

        confidence = BASE_CONFIDENCE * math.exp(-0.08 * h_min)
        confidence = max(0.05, confidence)

        cov_scale = math.sqrt(state.p_x + state.p_y) * dt_s * 0.1
        cone_radius = (CONE_GROWTH_RATE * 0.7 * h_min + cov_scale) * rl_scale

        predictions.append(PredictedPosition(
            x=pred_x,
            y=pred_y,
            horizon_minutes=h_min,
            confidence=confidence,
            cone_radius_m=cone_radius,
            heading_deg=heading,
            speed_mps=pred_speed,
        ))

    return predictions


def predict_all_targets_kalman(
    target_ids: list[str],
    history: TargetHistory,
    horizons: list[int] | None = None,
) -> dict[str, list[PredictedPosition]]:
    """Predict future positions for multiple targets using Kalman filter."""
    results: dict[str, list[PredictedPosition]] = {}
    for tid in target_ids:
        preds = predict_target_kalman(tid, history, horizons=horizons)
        if preds:
            results[tid] = preds
    return results


def clear_kalman_state(target_id: str | None = None) -> None:
    """Clear Kalman filter state for a target or all targets."""
    if target_id is None:
        _kalman_states.clear()
    else:
        _kalman_states.pop(target_id, None)


def get_kalman_state(target_id: str) -> KalmanState | None:
    """Get the current Kalman state for a target."""
    return _kalman_states.get(target_id)
