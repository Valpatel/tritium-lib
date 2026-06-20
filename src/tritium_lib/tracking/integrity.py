# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Track integrity monitoring — RAIM-style innovation gating against spoofing.

A fixed max-speed threshold ("anything over 50 m/s is suspicious") cannot tell a
legitimate fast, *steady* mover (an aircraft at 200 m/s, a highway vehicle) from
a GPS-spoofed teleport — it flags the aircraft and, for slow drift spoofs, misses
the attack. The standard fix from navigation integrity monitoring (RAIM) and
Kalman tracking is to gate the measurement against a motion *prediction*: keep a
constant-velocity estimate, predict where the target should be after dt, and test
the normalized innovation (how far the new fix lands from the prediction, in units
of the expected uncertainty) against a chi-square threshold.

This is the algorithm a filterpy/Kalman gate implements, in dependency-free form
(pure Python, no numpy needed for the 2-D case). It turns the display-only
``velocity_suspicious`` flag into a real, self-calibrating integrity test.
"""

from __future__ import annotations

import math
from typing import Tuple

# Chi-square critical values for 2 degrees of freedom (2-D position innovation).
#   p=0.99  -> 9.210     p=0.999 -> 13.816
CHI2_2DOF_99 = 9.210
CHI2_2DOF_999 = 13.816

# Default isotropic measurement noise (meters) and the unknown-acceleration
# process-noise bound (m/s^2) used to inflate the gate as dt grows.
DEFAULT_MEAS_NOISE_M = 2.0
DEFAULT_ACCEL_SIGMA = 30.0


def innovation_mahalanobis_sq(
    prev_pos: Tuple[float, float],
    new_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    dt: float,
    accel_sigma: float = DEFAULT_ACCEL_SIGMA,
    meas_noise_m: float = DEFAULT_MEAS_NOISE_M,
) -> float:
    """Squared Mahalanobis distance of a position fix from its CV prediction.

    Predicts ``prev_pos + velocity * dt`` and returns the squared innovation
    normalized by the expected position uncertainty over ``dt``. The uncertainty
    grows with dt (an unaccounted acceleration of up to ``accel_sigma`` moves the
    target ~0.5*a*dt^2), so the SAME steady velocity stays in-gate at any dt while
    a teleport blows past the threshold.

    Returns a value distributed ~chi-square(2) for a correctly-modeled target;
    compare against :data:`CHI2_2DOF_999` to gate at p=0.999.
    """
    if dt <= 0.0:
        return 0.0
    pred_x = prev_pos[0] + velocity[0] * dt
    pred_y = prev_pos[1] + velocity[1] * dt
    inn_x = new_pos[0] - pred_x
    inn_y = new_pos[1] - pred_y
    sigma = 0.5 * accel_sigma * dt * dt + meas_noise_m
    sigma = max(sigma, meas_noise_m)
    return (inn_x * inn_x + inn_y * inn_y) / (sigma * sigma)


def update_velocity_ewma(
    velocity: Tuple[float, float],
    prev_pos: Tuple[float, float],
    new_pos: Tuple[float, float],
    dt: float,
    alpha: float = 0.4,
) -> Tuple[float, float]:
    """Exponentially-weighted update of the constant-velocity estimate (m/s)."""
    if dt <= 0.0:
        return velocity
    vx = (new_pos[0] - prev_pos[0]) / dt
    vy = (new_pos[1] - prev_pos[1]) / dt
    return (
        alpha * vx + (1.0 - alpha) * velocity[0],
        alpha * vy + (1.0 - alpha) * velocity[1],
    )


def is_spoofed(
    mahalanobis_sq: float,
    samples: int,
    *,
    threshold: float = CHI2_2DOF_999,
    min_samples: int = 2,
) -> bool:
    """Gate decision: reject as implausible once a velocity baseline exists.

    Before ``min_samples`` velocity deltas are observed there is no model to gate
    against (a freshly-acquired fast mover must not be flagged), so this returns
    False until the baseline is established.
    """
    if samples < min_samples:
        return False
    return mahalanobis_sq > threshold


def spoof_score(mahalanobis_sq: float, threshold: float = CHI2_2DOF_999) -> float:
    """Map a Mahalanobis^2 to a bounded 0..1 plausibility-of-spoof score."""
    if mahalanobis_sq <= 0.0:
        return 0.0
    # 0 at the gate threshold, saturating toward 1 well above it.
    ratio = mahalanobis_sq / threshold
    return max(0.0, min(1.0, 1.0 - math.exp(-(max(0.0, ratio - 1.0)))))
