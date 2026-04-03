# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""IDM -- Intelligent Driver Model (Treiber 2000).

Computes longitudinal acceleration for a vehicle based on its speed, the gap
to the vehicle ahead, and the speed difference.  Pure math, no rendering.

    a_IDM = a * [1 - (v/v0)^delta - (s*(v, dv) / s)^2]
    s*(v, dv) = s0 + max(0, v*T + v*dv / (2*sqrt(a*b)))

This is the Python port of ``web/sim/idm.js``.

Reference: Treiber, Hennecke, Helbing (2000)
"Congested traffic states in empirical observations and microscopic simulations"

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IDMParams:
    """Parameters for the Intelligent Driver Model.

    Attributes
    ----------
    v0 : float
        Desired speed in m/s.
    a : float
        Maximum acceleration in m/s^2.
    b : float
        Comfortable deceleration in m/s^2 (positive value).
    s0 : float
        Minimum gap when stopped in meters.
    T : float
        Safe time headway in seconds.
    delta : int
        Acceleration exponent (default 4).
    """

    v0: float = 12.0     # ~43 km/h residential
    a: float = 1.4       # max acceleration
    b: float = 2.0       # comfortable braking
    s0: float = 2.0      # minimum gap stopped
    T: float = 1.5       # safe time headway
    delta: int = 4        # acceleration exponent


# Default IDM parameters for a residential road car.
IDM_DEFAULTS = IDMParams()


# Speed limits by road class (m/s).
ROAD_SPEEDS: dict[str, float] = {
    "motorway": 30.0,       # 108 km/h
    "trunk": 25.0,          # 90 km/h
    "primary": 18.0,        # 65 km/h
    "secondary": 15.0,      # 54 km/h
    "tertiary": 13.0,       # 47 km/h
    "residential": 10.0,    # 36 km/h
    "service": 5.0,         # 18 km/h
    "unclassified": 10.0,
    "living_street": 5.0,
}


# Pre-computed sqrt value for IDM acceleration.
_MAX_BRAKING = 9.0  # 1g braking limit


def idm_acceleration(
    v: float,
    gap: float,
    v_leader: float,
    params: IDMParams | None = None,
) -> float:
    """Compute IDM acceleration.

    Parameters
    ----------
    v : float
        Current speed in m/s.
    gap : float
        Bumper-to-bumper distance to leader in meters.
    v_leader : float
        Leader speed in m/s.
    params : IDMParams, optional
        IDM parameters.  Defaults to :data:`IDM_DEFAULTS`.

    Returns
    -------
    float
        Acceleration in m/s^2.  Negative values mean braking.
    """
    if params is None:
        params = IDM_DEFAULTS

    v0 = params.v0
    a = params.a
    b = params.b
    s0 = params.s0
    T = params.T
    delta = params.delta

    # Free-road term: how much headroom we have to accelerate
    free_road = 1.0 - (v / v0) ** delta if v0 > 0 else 0.0

    # Desired gap s*
    dv = v - v_leader
    interaction_term = (v * dv) / (2.0 * math.sqrt(a * b)) if a > 0 and b > 0 else 0.0
    s_star = s0 + max(0.0, v * T + interaction_term)

    # Interaction term -- clamp gap to prevent division by zero
    effective_gap = max(gap, 0.5)
    interaction = (s_star / effective_gap) ** 2

    # Clamp to physical limits
    acc = a * (free_road - interaction)
    return max(-_MAX_BRAKING, min(a, acc))


def idm_free_flow(
    v: float,
    params: IDMParams | None = None,
) -> float:
    """Compute free-flow acceleration (no leader ahead).

    Parameters
    ----------
    v : float
        Current speed in m/s.
    params : IDMParams, optional
        IDM parameters.  Defaults to :data:`IDM_DEFAULTS`.

    Returns
    -------
    float
        Acceleration in m/s^2.
    """
    if params is None:
        params = IDM_DEFAULTS

    v0 = params.v0
    a = params.a
    delta = params.delta

    if v0 <= 0:
        return 0.0

    return a * (1.0 - (v / v0) ** delta)


@dataclass
class IDMStepResult:
    """Result of an IDM integration step."""

    v: float    # new speed (m/s)
    ds: float   # distance traveled (m)


def idm_step(
    v: float,
    acc: float,
    dt: float,
) -> IDMStepResult:
    """Update speed and position using IDM acceleration.

    Euler integration with non-negative speed/distance clamping.

    Parameters
    ----------
    v : float
        Current speed in m/s.
    acc : float
        IDM acceleration in m/s^2.
    dt : float
        Timestep in seconds.

    Returns
    -------
    IDMStepResult
        New speed and distance traveled.
    """
    new_v = max(0.0, v + acc * dt)
    ds = max(0.0, v * dt + 0.5 * acc * dt * dt)
    return IDMStepResult(v=new_v, ds=ds)


# ---------------------------------------------------------------------------
# Vehicle profile presets (IDM params tuned per vehicle subtype)
# ---------------------------------------------------------------------------

VEHICLE_IDM_PROFILES: dict[str, IDMParams] = {
    "sedan": IDMParams(v0=12.0, a=1.4, b=2.0, s0=2.0, T=1.5),
    "suv": IDMParams(v0=11.0, a=1.2, b=2.0, s0=2.0, T=1.6),
    "truck": IDMParams(v0=9.0, a=0.8, b=1.5, s0=3.0, T=2.0),
    "motorcycle": IDMParams(v0=14.0, a=2.5, b=3.0, s0=1.5, T=1.0),
    "van": IDMParams(v0=10.0, a=1.0, b=1.8, s0=2.5, T=1.7),
    "bus": IDMParams(v0=8.0, a=0.7, b=1.5, s0=3.0, T=2.2),
    "emergency": IDMParams(v0=20.0, a=2.0, b=3.0, s0=1.5, T=1.0),
}


def get_idm_for_road(
    road_class: str,
    base_params: IDMParams | None = None,
    speed_variation: float = 0.1,
) -> IDMParams:
    """Create IDM params adjusted for a road class.

    Parameters
    ----------
    road_class : str
        Road classification (e.g. "residential", "motorway").
    base_params : IDMParams, optional
        Base parameters.  Only ``v0`` is overridden.
    speed_variation : float
        Random variation range (fraction of base speed, default 10%).
        Set to 0 for deterministic behavior (tests).

    Returns
    -------
    IDMParams
        New params with ``v0`` set for the road class.
    """
    if base_params is None:
        base_params = IDM_DEFAULTS

    import random

    base_speed = ROAD_SPEEDS.get(road_class, 10.0)
    if speed_variation > 0:
        variation = 1.0 - speed_variation + random.random() * 2 * speed_variation
    else:
        variation = 1.0

    return IDMParams(
        v0=base_speed * variation,
        a=base_params.a,
        b=base_params.b,
        s0=base_params.s0,
        T=base_params.T,
        delta=base_params.delta,
    )
