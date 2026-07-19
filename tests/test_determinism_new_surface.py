# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Determinism pins for the new lib surface (gait, hitscan, depth, engagement, body).

The repo holds its sim goldens byte-identical, and has been bitten before by
hash-seed and wall-clock nondeterminism.  This suite extends that discipline
to the recently landed modules:

  * ``models/gait_trajectory``  — ``joint_targets_at`` / ``sample_cycle``
  * ``geo/hitscan``             — ``ray_sphere`` / ``ray_aabb`` /
                                  ``muzzle_from_body`` / ``resolve_shot``
  * ``perception/depth``        — ``range_for_bbox`` / ``deproject_pixel``
  * ``perception/projection``   — ``world_from_camera_xyz``
  * ``perception/depth_pipeline`` — ``process_depth_frame``
  * ``tracking/engagement``     — ``ShotEvent`` / ``EngagementLog``
  * ``models/body``             — ``intent_from_motors`` / ``motors_from_intent``
  * ``control/step_reflex``     — ``capture_point`` / ``velocity_deviation``
                                  / ``step_target`` / ``StepReflex.decide``
                                  (deviation-gated revision: ``decide`` takes
                                  a required ``nominal_vel_xy``)
  * ``control/yaw_regulator``   — ``heading_error_deg`` /
                                  ``YawRegulator.correct`` / ``turn_intent``
  * ``control/yaw_rate_tracker`` — ``YawRateTracker.track`` (threaded
                                  integrator state — the payload runs a
                                  scripted multi-tick sequence so the state
                                  threading itself is pinned, not just one
                                  stateless call)
  * ``control/gait_speed``      — ``GaitSpeedTracker.track`` (scripted
                                  multi-tick drives pinning the threaded
                                  integral / slew-reference / slip-anchor
                                  state) / ``StrideSpeedEstimator`` /
                                  ``GaitPhaseClock``
  * ``perception/depth_codec``  — ``encode_depth16_png`` /
                                  ``decode_depth16_png`` / ``colorize_depth_bgr``
                                  (metric uint16-mm ROS ``16UC1`` transport —
                                  pinned for BYTE stability of the blob AND
                                  bit-exact round-trip fidelity on the mm grid;
                                  a lossy depth codec is silent data
                                  corruption, not a style issue)
  * ``fleet/frame_push``        — ``FramePushPolicy`` (scripted offer/sent/
                                  failed drive at injected times) /
                                  ``frame_push_path``
  * ``fleet/scan_pump``         — ``ScanPump.offer`` refusal ladder +
                                  forwarded ``/api/sighting`` payloads
  * ``fleet/sensor_rig``        — ``registration_plan`` (pull + push) /
                                  ``summarize_bringup`` / ``feed_source_id``
  * detection provenance        — ``BackgroundMotionDetector`` (scripted
                                  frames), ``CameraDetection.is_classified``
                                  / ``display_label``, the
                                  ``FrameDetectionPipeline`` payload carrying
                                  ``class_source``/``shape_hint``, and
                                  ``TargetTracker.update_from_detection``
                                  (heuristic-confidence zeroing, keyed
                                  identity, camera provenance stamping)

Four guarantees, each tested honestly:

1. **Repeatability** — one canonical-JSON payload exercises every surface;
   identical inputs must produce a byte-identical payload (compared by
   SHA-256 digest, so a float-ordering or repr change is caught).
2. **Hash-seed independence** — the same payload is rebuilt in SUBPROCESSES
   under ``PYTHONHASHSEED=0`` and ``=99`` and the digests compared.  Running
   the computation under genuinely different seeds is the only honest way to
   test this; the test skips cleanly if a subprocess cannot be spawned.
3. **No wall-clock dependence** — the payload is rebuilt after a real sleep
   and must not move.  ``ShotEvent.from_shot`` may consult ``time.time()``
   ONLY through its documented ``timestamp=None`` default — never when an
   explicit timestamp is supplied.
4. **Float discipline** — representative values are pinned with explicit
   tolerances so a refactor that changes the math fails loudly, not silently.

Known, deliberate exclusions (documented nondeterminism, not bugs):

  * ``ShotEvent.shot_id`` comes from a process-local monotonic counter
    (uniqueness is its contract), so it is stripped before digesting and its
    monotonicity is asserted separately.
  * The depth pipeline's ground-fallback lat/lng derive from the module-level
    geo reference singleton; the payload builder calls ``geo.reset()`` so the
    digest cannot depend on test ordering.
  * ``BackgroundMotionDetector`` stamps each detection with
    ``datetime.now(timezone.utc)``.  The timestamp is stripped before
    digesting and its contract (tz-aware UTC, current) is asserted
    separately — like ``ShotEvent``'s ``timestamp=None`` default, the clock
    read is documented API, not hidden state.
  * ``TrackedTarget.first_seen`` / ``last_seen`` come from
    ``time.monotonic()`` and ``to_dict()`` derives ``status`` and staleness
    from them; the provenance payload digests only explicitly listed
    time-free fields, never a whole ``to_dict()``.
  * The depth PNG blob's exact bytes depend on which image codec is
    installed (cv2 preferred, Pillow fallback).  Within one environment the
    bytes are pinned by digest; ACROSS codecs only value-identity holds
    (asserted separately) — the two encoders emit different, equally valid
    PNG streams for the same pixels.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pytest

import tritium_lib.geo as geo
from tritium_lib.control import (
    GaitPhaseClock,
    GaitSpeedState,
    GaitSpeedTracker,
    LegPlacement,
    ReachLimits,
    StepReflex,
    StrideSpeedEstimator,
    YawRateState,
    YawRateTracker,
    YawRegulator,
    capture_point,
    heading_error_deg,
    step_target,
    velocity_deviation,
    velocity_from_impulse,
)
from tritium_lib.fleet.frame_push import (
    FRAME_PUSH_PATH,
    FramePushPolicy,
    frame_push_path,
)
from tritium_lib.fleet.scan_pump import REFUSALS, ScanPump
from tritium_lib.fleet.sensor_rig import (
    OUTCOMES,
    RigSensor,
    registration_plan,
    summarize_bringup,
)
from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.hitscan import (
    BoxTarget,
    Muzzle,
    SphereTarget,
    muzzle_from_body,
    ray_aabb,
    ray_sphere,
    resolve_shot,
)
from tritium_lib.geo.isaac_frame import LocalPose
from tritium_lib.models.body import (
    ControlIntent,
    intent_from_motors,
    motors_from_intent,
)
from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.models.gait_trajectory import (
    JOINT_NAMES,
    QuadrupedGaitCycle,
    joint_targets_at,
)
from tritium_lib.perception.depth import (
    CameraIntrinsics,
    deproject_pixel,
    range_for_bbox,
)
from tritium_lib.perception.depth_codec import (
    DEPTH_SCALE_MM,
    colorize_depth_bgr,
    decode_depth16_png,
    encode_depth16_png,
)
from tritium_lib.perception.depth_pipeline import process_depth_frame
from tritium_lib.perception.detector import (
    MOTION_CLASS,
    BackgroundMotionDetector,
)
from tritium_lib.perception.pipeline import FrameDetectionPipeline
from tritium_lib.perception.projection import (
    CameraWorldPose,
    world_from_camera_xyz,
)
from tritium_lib.tracking import TargetTracker
from tritium_lib.tracking.engagement import EngagementLog, ShotEvent


# --------------------------------------------------------------------------
# Canonical serialization — sorted keys, tight separators, NaN forbidden.
# Two runs that differ in ANY emitted float, key order, or structure produce
# different digests.
# --------------------------------------------------------------------------

def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


# --------------------------------------------------------------------------
# Payload builders — one per pinned surface, pure functions of constants.
# --------------------------------------------------------------------------

def _gait_payload() -> dict:
    out: dict = {}
    for gait, speed in (("trot", None), ("walk", 0.7), ("bound", 1.9)):
        key = f"{gait}_{speed}"
        out[key] = [
            joint_targets_at(t, gait=gait, speed=speed)
            for t in (0.0, 0.125, 0.5, 1.75)
        ]
    cycle = QuadrupedGaitCycle("trot")
    out["sample_cycle_8"] = [
        [phase, angles] for phase, angles in cycle.sample_cycle(8)
    ]
    out["trot_params"] = {
        "stride_hz": cycle.stride_hz,
        "speed_mps": cycle.speed_mps,
        "thigh_amp_rad": cycle.thigh_amp_rad,
        "hip_amp_rad": cycle.hip_amp_rad,
        "duty_factor": cycle.duty_factor,
    }
    return out


_HITSCAN_TARGETS: list = [
    SphereTarget("s_near", 0.0, 30.0, 1.0, 0.5),
    SphereTarget("s_far", 0.0, 60.0, 1.0, 2.0),
    BoxTarget("b_wall", -2.0, 40.0, 0.0, 2.0, 42.0, 3.0),
]

_MUZZLE_N = Muzzle(
    east_m=0.0, north_m=0.0, up_m=1.0, heading_deg=0.0, elevation_deg=0.0,
)


def _hitscan_payload() -> dict:
    body = LocalPose(east_m=10.0, north_m=20.0, up_m=0.5, heading_deg=90.0)
    mount = CameraMount(forward_m=0.3, up_m=0.2, tilt_deg=10.0)
    muzzle = muzzle_from_body(body, mount, barrel_m=0.5)
    shot_hit = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
    miss_muzzle = Muzzle(
        east_m=0.0, north_m=0.0, up_m=1.0, heading_deg=90.0, elevation_deg=0.0,
    )
    shot_miss = resolve_shot(miss_muzzle, _HITSCAN_TARGETS, 100.0)
    return {
        "ray_sphere_hit": ray_sphere((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0), 1.0),
        "ray_sphere_miss": ray_sphere((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 5.0, 0.0), 1.0),
        "ray_sphere_inside": ray_sphere((10.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0), 1.0),
        "ray_aabb_hit": ray_aabb((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, -1.0, -1.0), (7.0, 1.0, 1.0)),
        "ray_aabb_parallel_miss": ray_aabb((0.0, 5.0, 0.0), (1.0, 0.0, 0.0), (5.0, -1.0, -1.0), (7.0, 1.0, 1.0)),
        "muzzle": muzzle.to_dict(),
        "muzzle_aim": list(muzzle.direction()),
        "shot_hit": shot_hit.to_dict(),
        "shot_miss": shot_miss.to_dict(),
    }


def _depth_frame() -> np.ndarray:
    """A synthetic depth frame with structure + every invalid-pixel kind."""
    depth = np.linspace(2.0, 40.0, 480, dtype=np.float64)[:, None] * np.ones(
        (1, 640), dtype=np.float64,
    )
    depth[100:110, 100:110] = 0.0      # sensor no-return
    depth[200:205, 300:305] = np.nan   # dropout
    depth[240:245, 320:325] = np.inf   # overflow
    return depth


_INTR = CameraIntrinsics.from_fov(640, 480, horizontal_fov_deg=90.0)


def _depth_payload() -> dict:
    depth = _depth_frame()
    return {
        "range_center": range_for_bbox(depth, (300.0, 195.0, 40.0, 90.0)),
        "range_p25_full_box": range_for_bbox(
            depth, {"x": 90.0, "y": 90.0, "w": 40.0, "h": 40.0},
            percentile=25.0, inner=1.0,
        ),
        "range_mm_scale": range_for_bbox(
            np.full((480, 640), 8000.0), (300.0, 195.0, 40.0, 90.0),
            depth_scale=0.001,
        ),
        "range_offframe": range_for_bbox(depth, (10_000.0, 10_000.0, 5.0, 5.0)),
        "deproject_center": deproject_pixel(320.0, 240.0, 10.0, _INTR),
        "deproject_offaxis": deproject_pixel(400.0, 300.0, 10.0, _INTR),
        "intrinsics": [_INTR.fx, _INTR.fy, _INTR.cx, _INTR.cy],
    }


def _projection_payload() -> dict:
    flat = world_from_camera_xyz(
        (0.0, 0.0, 10.0), CameraWorldPose(heading_deg=0.0, height_m=2.0),
    )
    posed = world_from_camera_xyz(
        (2.0, 1.0, 10.0),
        CameraWorldPose(
            lat=37.0, lng=-122.0, heading_deg=45.0, pitch_deg=-10.0,
            height_m=3.0,
        ),
    )
    return {"flat": flat, "posed": posed}


class _FixedDetector:
    """Injected detector emitting the same two boxes every call."""

    def detect(self, frame, source_id=""):
        return [
            CameraDetection(
                source_id=source_id or "cam", class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x=300.0, y=195.0, w=40.0, h=90.0),
            ),
            CameraDetection(
                source_id=source_id or "cam", class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x=100.0, y=300.0, w=120.0, h=60.0),
            ),
        ]


# Kinematics keys that carry the pipeline's MEASURED output.  Time-derived
# bookkeeping (first_seen / last_seen live on the target, not in this set)
# is deliberately not digested.
_PIPELINE_KIN_KEYS = (
    "range_m", "world_enu", "elevation_m", "bearing_deg", "distance_m",
    "depth_source", "camera_id", "world_lat", "world_lng",
)


def _pipeline_payload() -> dict:
    # The ground-fallback lat/lng path reads the module-level geo reference
    # singleton; reset it so the digest cannot depend on what an earlier
    # test happened to configure.
    geo.reset()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 10.0)
    depth[280:380, 80:240] = 0.0  # car bbox dropout -> ground fallback path
    tracker = TargetTracker()
    ids = process_depth_frame(
        rgb, depth, _INTR, CameraWorldPose(heading_deg=0.0, height_m=2.0),
        tracker, _FixedDetector(), source="isaac_depth", cam_id="cam-01",
    )
    rows = []
    for tid in ids:
        target = tracker.get_target(tid)
        rows.append({
            "id": tid,
            "asset_type": target.asset_type,
            "position": list(target.position),
            "kinematics": {
                k: target.kinematics[k]
                for k in _PIPELINE_KIN_KEYS if k in target.kinematics
            },
        })
    geo.reset()
    return {"ids": ids, "targets": rows}


def _shot_dict_no_id(event: ShotEvent) -> dict:
    d = event.to_dict()
    d.pop("shot_id")  # process-local monotonic counter — unique by contract
    return d


def _engagement_payload() -> dict:
    shot_hit = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
    miss_muzzle = Muzzle(
        east_m=0.0, north_m=0.0, up_m=1.0, heading_deg=90.0, elevation_deg=0.0,
    )
    shot_miss = resolve_shot(miss_muzzle, _HITSCAN_TARGETS, 100.0)

    log = EngagementLog(max_events=4)
    for i, shot in enumerate((shot_hit, shot_miss, shot_hit, shot_hit, shot_miss, shot_hit)):
        log.record(ShotEvent.from_shot(shot, shooter_id="unit-7", timestamp=1000.0 + i))

    wire = shot_hit.to_dict() | {"shooter_id": "unit-9", "timestamp": 42.5}
    rebuilt = ShotEvent.from_payload(wire)

    return {
        "hit_event": _shot_dict_no_id(
            ShotEvent.from_shot(shot_hit, shooter_id="unit-7", timestamp=1234.5),
        ),
        "miss_event": _shot_dict_no_id(
            ShotEvent.from_shot(shot_miss, shooter_id="unit-7", timestamp=1234.5),
        ),
        "recent": [
            _shot_dict_no_id(e) for e in log.recent(limit=3, since=1001.5)
        ],
        "stats": log.stats(),
        "wire_roundtrip": _shot_dict_no_id(rebuilt),
    }


def _body_payload() -> dict:
    motor_grid = (
        (1.0, 1.0), (-1.0, -1.0), (0.6, -0.2), (0.0, 0.9),
        (1.5, -3.0),  # out-of-envelope input: clamps, deterministically
    )
    intent_grid = (
        {"forward": 0.5, "turn": 1.0, "climb": 0.0},
        {"forward": -0.25, "turn": -0.5, "climb": 1.0},
        {"forward": 1.0, "turn": 1.0, "climb": 0.0},  # saturating demand
    )
    return {
        "intent_from_motors": [
            intent_from_motors(left, right).model_dump()
            for left, right in motor_grid
        ],
        "motors_from_intent": [
            list(motors_from_intent(ControlIntent(**kw))) for kw in intent_grid
        ],
    }


_REFLEX_LEGS = (
    LegPlacement("FL", 0.19, 0.13),
    LegPlacement("FR", 0.19, -0.13),
    LegPlacement("RL", -0.19, 0.13),
    LegPlacement("RR", -0.19, -0.13),
)


def _step_reflex_payload() -> dict:
    limits = ReachLimits(max_dx=0.1, max_dy=0.08)
    reflex = StepReflex(com_height_m=0.31)
    walk = (0.60, 0.0)  # commanded gait velocity for the walking cases
    cases = (
        # (measured, nominal)
        ((0.0, 0.0), (0.0, 0.0)),    # standing, captured — gate closed
        ((0.62, -0.03), walk),       # walking at nominal — gate closed
        ((0.60, 0.50), walk),        # lateral shove on the walk — steps
        ((0.20, 0.0), walk),         # blocked/tripped — steps backward
        ((0.0, 0.6), (0.0, 0.0)),    # standing shove — steps, input-order tie
        ((-0.5, 0.4), (0.0, 0.0)),   # standing diagonal shove — steps
        ((2.1, 0.0), walk),          # violent — clamped, honest residual
    )
    return {
        "capture": [
            list(capture_point(m, 0.31)) for m, _ in cases
        ] + [list(capture_point((0.42, -0.15), 0.36, 9.81))],
        "deviation": [
            list(velocity_deviation(m, n)) for m, n in cases
        ],
        "impulse_velocity": list(velocity_from_impulse((5.0, -2.4), 12.0)),
        "step_target": list(
            step_target(_REFLEX_LEGS, (0.05, 0.30), reach_limits=limits)
        ),
        "decisions": [
            reflex.decide(
                measured, _REFLEX_LEGS,
                nominal_vel_xy=nominal, reach_limits=limits,
                leg_height_offsets={"FL": 0.01, "FR": -0.01},
            ).as_dict()
            for measured, nominal in cases
        ],
    }


def _yaw_payload() -> dict:
    regulator = YawRegulator()  # documented defaults — a retune must face this
    hot = YawRegulator(kp=4.0, kd=1.0, max_correction_dps=20.0)
    cases = (
        # (measured, commanded, rate or None, dt or None)
        (90.0, 90.0, None, None),      # on heading — the byte-zero no-op
        (359.0, 1.0, None, None),      # wrap across north, small positive
        (1.0, 359.0, None, None),      # wrap the other way
        (0.0, 179.0, None, None),      # saturating demand
        (0.0, 180.0, None, None),      # half-open boundary: error -180
        (10.0, 40.0, 5.0, None),       # damped mid-turn
        (0.0, 1.0, 0.0, 0.02),         # deadbeat cap active
        (45.0, 45.0, -20.0, 0.02),     # pure damping at zero error
        (-355.0, 3.0, 2.5, 0.05),      # unnormalized compass input
    )
    corrections = []
    for measured, commanded, rate, dt in cases:
        corrections.append(
            regulator.correct(
                measured, commanded, measured_yaw_rate_dps=rate, dt=dt,
            ).as_dict()
        )
        corrections.append(
            hot.correct(
                measured, commanded, measured_yaw_rate_dps=rate, dt=dt,
            ).as_dict()
        )
    return {
        "errors": [
            heading_error_deg(m, c)
            for m, c in (
                (359.0, 1.0), (1.0, 359.0), (180.0, -180.0),
                (0.0, 180.0), (725.0, 5.0), (-90.0, 270.0),
            )
        ],
        "corrections": corrections,
        "turn_intents": [
            regulator.correct(0.0, e).turn_intent(60.0)
            for e in (0.0, 2.0, 45.0, -45.0, 179.0)
        ],
        "hold": [
            regulator.hold(0.0, e, 0.25, 60.0)
            for e in (0.0, 10.0, -10.0, 170.0)
        ],
    }


def _yaw_rate_payload() -> dict:
    """A scripted multi-tick run so the STATE THREADING is what gets pinned.

    Each row carries the integral the tick handed forward; a change to the
    anti-windup condition, the integral clamp, or the entry clamp moves a
    digest even if any single stateless call still agrees.
    """
    default = YawRateTracker(turn_rate_dps=60.0)  # documented defaults
    weak_authority = YawRateTracker(
        turn_rate_dps=90.0, kp=2.0, ki=3.0, max_turn=0.5,
        integral_limit_deg=4.0,
    )
    script = (
        # (demanded_dps, measured_dps, dt_s)
        (0.0, 0.0, 0.02),     # the byte-zero no-op
        (6.0, 0.0, 0.02),     # opening tick of a step demand
        (6.0, 2.0, 0.02),     # partial delivery — integral building
        (40.0, 5.0, 0.02),    # saturating demand — anti-windup path
        (40.0, 5.0, 0.02),    # still pinned: the integral must not move
        (4.0, 6.0, 0.02),     # desaturating reversal
        (-6.0, 1.0, 0.05),    # sign flip on a coarser tick
        (3.0, 3.0, 0.0),      # repeated timestamp: integrator holds
    )
    out: dict = {}
    for name, tracker in (
        ("default", default), ("weak_authority", weak_authority),
    ):
        state: YawRateState | None = None
        ticks = []
        for demanded, measured, dt in script:
            cmd = tracker.track(demanded, measured, dt, state=state)
            state = cmd.state
            ticks.append(cmd.as_dict())
        out[name] = ticks
    out["stale_state_clamp"] = default.track(
        0.0, 0.0, 0.02, state=YawRateState(integral_deg=50.0),
    ).as_dict()
    out["limits"] = [
        default.effective_integral_limit_deg,
        weak_authority.effective_integral_limit_deg,
    ]
    return out


def _gait_speed_payload() -> dict:
    """Scripted multi-tick drives so the THREADED STATE is what gets pinned
    — integral, slew reference, slip anchor/latch, ceiling ticks — plus the
    windowed estimator and the phase-continuous clock.  All time injected.
    """
    default = GaitSpeedTracker(nominal_mps=1.6)  # documented defaults
    tight = GaitSpeedTracker(
        nominal_mps=1.0, kp=0.2, ki=0.8, max_cadence_scale=1.5,
        max_slew_mps_per_s=0.25, deadband_mps=0.02,
        slip_probe_delta_mps=0.04, slip_tol_mps=0.05, slip_release_s=1.0,
    )
    script = (
        # (commanded_mps, measured_mps, dt_s)
        (1.2, 1.20, 0.4),    # the byte-exact inert tick
        (1.2, 0.45, 0.4),    # opening shortfall — integral building
        (1.2, 0.50, 0.4),
        (1.2, 0.48, 0.0),    # repeated timestamp: integrator holds
        (2.0, 0.60, 0.4),    # unreachable ask — toward the stop
        (2.0, 0.40, 0.4),    # demand up, speed DOWN: slip evidence
        (2.0, 0.30, 0.4),
        (2.0, 0.30, 0.4),
        # Persistent overshoot: demand slews down to the FLOOR, then the
        # strictly-downward amplitude trim engages.
    ) + ((0.4, 1.00, 0.4),) * 12
    out: dict = {}
    for name, tracker in (("default", default), ("tight", tight)):
        state: GaitSpeedState | None = None
        ticks = []
        for commanded, measured, dt in script:
            cmd = tracker.track(commanded, measured, dt, state=state)
            state = cmd.state
            ticks.append(cmd.as_dict())
        out[name] = ticks
    out["stale_state_clamp"] = default.track(
        1.0, 1.0, 0.4, state=GaitSpeedState(integral_mps=50.0),
    ).as_dict()
    out["limits"] = [
        default.min_demand_mps,
        default.max_demand_mps,
        default.effective_integral_limit_mps,
        tight.effective_integral_limit_mps,
    ]
    est = StrideSpeedEstimator(window_s=0.8)
    out["estimator"] = [
        est.update(0.25 * i, 1.1 * 0.25 * i, -0.4 * 0.25 * i)
        for i in range(9)
    ]
    clock = GaitPhaseClock(2.6)
    trace = [clock.phase_at(t) for t in (0.0, 0.5, 1.0)]
    clock.retime(1.0, 3.25)
    trace += [clock.phase_at(t) for t in (1.0, 1.5, 2.0)]
    clock.retime(2.0, 1.3)
    trace.append(clock.phase_at(3.0))
    out["phase_clock"] = trace
    return out


_DEPTH_UNITS = np.array(
    [
        [0, 1, 2, 999, 1000],
        [1001, 12500, 40000, 65534, 65535],
        [3, 0, 250, 65000, 7],
    ],
    dtype=np.uint16,
)


def _units_to_metres(units: np.ndarray, scale: float = DEPTH_SCALE_MM) -> np.ndarray:
    """The exact inverse domain of the wire format: values ON the mm grid."""
    return units.astype(np.float32) / np.float32(scale)


def _recovered_units(decoded: np.ndarray, scale: float = DEPTH_SCALE_MM) -> np.ndarray:
    """Decoded metres -> integer units, with holes (NaN) mapped to -1."""
    filled = np.nan_to_num(decoded.astype(np.float64), nan=-1.0 / scale)
    return np.rint(filled * scale).astype(np.int64)


def _depth_codec_payload() -> dict:
    metres = _units_to_metres(_DEPTH_UNITS)
    blob = encode_depth16_png(metres)
    decoded = decode_depth16_png(blob)

    ramp_units = np.arange(0, 65536, 257, dtype=np.uint16).reshape(16, 16)
    ramp_blob = encode_depth16_png(_units_to_metres(ramp_units))

    cm_units = np.array([[1, 30000, 65535], [0, 2, 65534]], dtype=np.uint16)
    cm_blob = encode_depth16_png(_units_to_metres(cm_units, 100.0), scale=100.0)
    cm_decoded = decode_depth16_png(cm_blob, scale=100.0)

    return {
        # The BYTES are the contract on the wire: pinned by digest so any
        # drift in the encoder (filter choice, bit depth, channel count) is
        # caught even when the decoded values happen to survive it.
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "blob_len": len(blob),
        "ramp_blob_sha256": hashlib.sha256(ramp_blob).hexdigest(),
        "cm_blob_sha256": hashlib.sha256(cm_blob).hexdigest(),
        "recovered_units": _recovered_units(decoded).tolist(),
        "nan_mask": np.isnan(decoded).tolist(),
        "cm_recovered_units": _recovered_units(cm_decoded, 100.0).tolist(),
        "sub_half_mm_lift": float(
            decode_depth16_png(
                encode_depth16_png(np.array([[0.0004]], dtype=np.float32)),
            )[0, 0]
        ),
        "saturation_m": float(
            decode_depth16_png(
                encode_depth16_png(np.array([[70.0]], dtype=np.float32)),
            )[0, 0]
        ),
        "colorize_sha256": hashlib.sha256(
            colorize_depth_bgr(metres, near=0.5, far=60.0).tobytes()
        ).hexdigest(),
    }


def _frame_push_payload() -> dict:
    """A scripted offer/sent/failed drive at INJECTED times.

    ``FramePushPolicy`` never reads a clock — ``now`` is always a parameter —
    so the whole state machine (rate gate, in-flight drop, exponential
    backoff, recovery) is digestible as pure logic.
    """
    policy = FramePushPolicy(target_fps=5.0, base_backoff_s=0.5, max_backoff_s=4.0)
    decisions: list[list] = []

    def offer(t: float) -> None:
        d = policy.offer(t)
        decisions.append([t, d.send, d.reason])

    offer(0.0)
    policy.sent(0.05)          # first frame lands
    offer(0.1)                 # 0.05 s since send < 0.2 s interval -> rate gate
    offer(0.30)
    policy.failed(0.32)        # refused -> backoff 0.5 s (until 0.82)
    offer(0.5)                 # inside backoff
    offer(0.9)
    policy.failed(0.95)        # 2nd consecutive -> backoff 1.0 s (until 1.95)
    offer(1.5)                 # still inside backoff
    offer(2.0)                 # clear again
    offer(2.05)                # previous send unresolved -> in_flight drop
    policy.sent(2.1)           # recovery clears the failure run
    offer(2.15)                # rate gate re-anchored on the 2.1 send
    offer(2.4)
    policy.sent(2.45)

    growth = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5, max_backoff_s=4.0)
    backoffs = []
    for i in range(6):
        t = 100.0 * (i + 1)    # spaced far beyond any backoff
        assert growth.offer(t).send
        growth.failed(t)
        backoffs.append(growth.backoff_remaining(t))

    stats = policy.stats
    return {
        "path_template": FRAME_PUSH_PATH,
        "paths": [
            frame_push_path("isaac_rgb"),
            frame_push_path("isaac_depth16"),
            frame_push_path("a/b c"),     # hostile id: must be %-encoded
            frame_push_path("d%e+f"),
        ],
        "decisions": decisions,
        "backoff_growth": backoffs,
        "backoff_cleared": policy.backoff_remaining(2.5),
        "stats": {
            "sent": stats.sent,
            "dropped_rate_limited": stats.dropped_rate_limited,
            "dropped_in_flight": stats.dropped_in_flight,
            "dropped_backoff": stats.dropped_backoff,
            "failed": stats.failed,
            "consecutive_failures": stats.consecutive_failures,
            "offered": stats.offered,
        },
        "healthy": policy.healthy,
    }


_SWEEP_A = {
    "ranges": [1.5, 2.0, 3.5, 10.0],
    "angle_min": -3.14159, "angle_increment": 1.5708,
    "range_min": 0.1, "range_max": 10.0,
}
_SWEEP_B = {
    "ranges": [1.5, 2.0, 3.5, 9.0],
    "angle_min": -3.14159, "angle_increment": 1.5708,
    "range_min": 0.1, "range_max": 10.0,
}


def _scan_pump_payload() -> dict:
    pump = ScanPump(
        "lidar-01", sensor_x=2.0, sensor_y=-3.0, sensor_yaw_deg=90.0,
        max_failures=2,
    )
    offers: list[list] = []

    def offer(scan) -> None:
        d = pump.offer(scan)
        offers.append([d.forward, d.reason, d.payload])

    offer(None)                          # malformed: not a dict
    offer({"no_ranges": True})           # malformed: no ranges
    offer({"ranges": ["bogus", 1.0]})    # malformed: non-numeric
    offer(_SWEEP_A)                      # forward
    offer(_SWEEP_A)                      # stale (identical)
    offer(dict(_SWEEP_A))                # STILL stale — value equality, and the
    pump.set_sensor_pose(4.5, 1.25, 180.0)  # baseline was deliberately kept
    offer(_SWEEP_B)                      # forward, with the updated pose
    offer({"ranges": [10.0, 10.0 - 5e-7, 10.0], "range_max": 10.0})  # no_returns
    pump.record_result(False)
    pump.record_result(False)            # trips the breaker (max_failures=2)
    offer({"ranges": [4.0, 5.0], "range_max": 10.0})  # tripped
    pump.record_result(True)             # breaker resets on a success
    offer({"ranges": [4.0, 5.0], "range_max": 10.0})  # forward again

    return {
        "refusal_names": list(REFUSALS),
        "offers": offers,
        "stats": pump.stats(),
    }


_RIG = (
    RigSensor(role="camera", host="10.0.0.5", port=8081, ready=True),
    RigSensor(role="depth", host="10.0.0.5", port=8081, ready=True,
              attach_to="unit-go2"),
    RigSensor(role="stereo_right", host="10.0.0.5", port=8081, ready=False),
    RigSensor(role="lidar", host="10.0.0.5", port=8082, ready=True),
    RigSensor(role="body", host="10.0.0.5", port=8090, ready=True),
    RigSensor(role="thermal", host="10.0.0.5", port=8083, ready=True),
    RigSensor(role="camera", host="10.0.0.6", port=8081, ready=True,
              source_id="custom_cam"),
)


def _rig_report_row(report) -> dict:
    return {
        "registered": report.registered,
        "already": report.already,
        "failed": report.failed,
        "skipped": report.skipped,
        "detail": report.detail,
        "ok": report.ok,
        "str": str(report),
    }


def _sensor_rig_payload() -> dict:
    def call_dicts(calls) -> list[dict]:
        return [
            {"method": c.method, "path": c.path, "payload": c.payload,
             "role": c.role}
            for c in calls
        ]

    reports = {
        "mixed": summarize_bringup([
            ("isaac_rgb", "registered"),
            ("isaac_depth16", "already_registered"),
            ("isaac_right", "failed"),
            ("go2_lidar", "skipped"),
        ]),
        "empty": summarize_bringup([]),
        "all_skipped": summarize_bringup([("a", "skipped"), ("b", "skipped")]),
        "clean": summarize_bringup([("isaac_rgb", "registered")]),
    }
    return {
        "pull_plan": call_dicts(registration_plan(_RIG)),
        "pull_plan_nodetect": call_dicts(registration_plan(_RIG, detect=False)),
        "push_plan": call_dicts(registration_plan(_RIG, push=True)),
        "feed_source_ids": [s.feed_source_id() for s in _RIG],
        "outcome_names": sorted(OUTCOMES),
        "reports": {name: _rig_report_row(r) for name, r in reports.items()},
    }


def _motion_frames() -> list[np.ndarray]:
    """A scripted static-camera sequence: flat background, then a tall
    moving blob.  MOG2 uses no RNG at inference, so the detections are a
    pure function of this sequence."""
    frames = []
    for i in range(6):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        if i >= 3:
            frame[30:90, 20 + 10 * i:40 + 10 * i] = 220
        frames.append(frame)
    return frames


class _ProvenanceDetector:
    """Injected detector emitting one of each provenance kind per tick."""

    backend_name = "fixed"

    def detect(self, frame, source_id=""):
        return [
            CameraDetection(
                source_id=source_id, class_name=MOTION_CLASS,
                class_source="heuristic", shape_hint="tall", confidence=0.75,
                bbox=BoundingBox(x=10.0, y=20.0, w=30.0, h=60.0),
            ),
            CameraDetection(
                source_id=source_id, class_name="person",
                class_source="classifier", confidence=0.92,
                bbox=BoundingBox(x=200.0, y=100.0, w=40.0, h=90.0),
            ),
            CameraDetection(  # below the pipeline's min_confidence gate
                source_id=source_id, class_name="car", confidence=0.30,
                bbox=BoundingBox(x=0.0, y=0.0, w=5.0, h=5.0),
            ),
        ]


# Time-free fields of a tracked target.  first_seen / last_seen come from
# time.monotonic() and to_dict() derives staleness from them, so the digest
# never touches a whole to_dict() — only this explicit list.
_TRACK_FIELDS = (
    "target_id", "name", "classification", "classification_confidence",
    "class_source", "asset_type", "alliance", "source", "position_source",
    "signal_count",
)


def _provenance_payload() -> dict:
    detector = BackgroundMotionDetector(min_area=100, history=10)
    detector_rows = []
    for frame in _motion_frames():
        for det in detector.detect(frame, "cam-x"):
            detector_rows.append({
                # det.timestamp is quarantined: datetime.now(timezone.utc)
                # by documented contract, asserted separately in the pins.
                "class_name": det.class_name,
                "class_source": det.class_source,
                "shape_hint": det.shape_hint,
                "confidence": det.confidence,
                "bbox": [det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h],
                "is_classified": det.is_classified,
                "display_label": det.display_label,
            })

    hints = [
        [w, h, detector._shape_hint(w, h)]
        for w, h in (
            (10, 13), (10, 12), (14, 10), (13, 10), (4, 5), (7, 7), (0, 1),
        )
    ]

    sink: list[dict] = []
    pipeline = FrameDetectionPipeline(
        _ProvenanceDetector(),
        lambda: np.zeros((480, 640, 3), dtype=np.uint8),
        sink.append,
        source_id="cam-7",
    )
    emitted = pipeline.tick()

    tracker = TargetTracker()
    ids = [tracker.update_from_detection(payload) for payload in sink]
    keyed = [
        tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.88,
             "center_x": 5.0, "center_y": 6.0},
            detection_key="bytetrack:9",
        ),
        # Same key, 70 m away: keyed identity must hold, not re-mint.
        tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.88,
             "center_x": 55.0, "center_y": 66.0},
            detection_key="bytetrack:9",
        ),
        # A DIFFERENT key co-located with the first: keys never merge.
        tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.88,
             "center_x": 5.0, "center_y": 6.0,
             "source_track_id": "radar-4"},
        ),
    ]
    dedupe_id = tracker.update_from_detection(sink[1])  # unkeyed re-observation
    unstamped_id = tracker.update_from_detection(
        # Legacy caller with NO class_source: confidence must be KEPT.
        {"class_name": "car", "confidence": 0.66,
         "center_x": 40.0, "center_y": 40.0},
    )
    rejected = tracker.update_from_detection(
        {"class_name": "person", "confidence": 0.2,
         "center_x": 0.0, "center_y": 0.0},
    )

    tracks = []
    for tid in sorted(set(ids + keyed + [dedupe_id, unstamped_id])):
        target = tracker.get_target(tid)
        row = {field: getattr(target, field) for field in _TRACK_FIELDS}
        row["position"] = list(target.position)
        row["confirming_sources"] = sorted(target.confirming_sources)
        row["kinematics"] = {
            k: target.kinematics[k]
            for k in ("camera_id", "bbox", "bearing_deg", "distance_m")
            if k in (target.kinematics or {})
        }
        tracks.append(row)

    return {
        "motion_class": MOTION_CLASS,
        "detector_rows": detector_rows,
        "shape_hints": hints,
        "pipeline_sink": sink,
        "pipeline_emitted": emitted,
        "ids": ids,
        "keyed_ids": keyed,
        "dedupe_id": dedupe_id,
        "unstamped_id": unstamped_id,
        "low_confidence_rejected": rejected,
        "tracks": tracks,
    }


def build_payload() -> dict:
    """Every pinned surface, computed from constants — no time, no randomness."""
    return {
        "gait": _gait_payload(),
        "hitscan": _hitscan_payload(),
        "depth": _depth_payload(),
        "projection": _projection_payload(),
        "pipeline": _pipeline_payload(),
        "engagement": _engagement_payload(),
        "body": _body_payload(),
        "step_reflex": _step_reflex_payload(),
        "yaw": _yaw_payload(),
        "yaw_rate": _yaw_rate_payload(),
        "gait_speed": _gait_speed_payload(),
        "depth_codec": _depth_codec_payload(),
        "frame_push": _frame_push_payload(),
        "scan_pump": _scan_pump_payload(),
        "sensor_rig": _sensor_rig_payload(),
        "provenance": _provenance_payload(),
    }


def payload_digest() -> str:
    """SHA-256 of the canonical payload — the subprocess entry point."""
    return _digest(build_payload())


# --------------------------------------------------------------------------
# 1. Repeatability within a run
# --------------------------------------------------------------------------

class TestRepeatability:
    def test_payload_is_byte_identical_across_repeated_calls(self):
        first = _canonical_bytes(build_payload())
        for _ in range(3):
            assert _canonical_bytes(build_payload()) == first

    def test_digest_is_stable(self):
        assert payload_digest() == payload_digest()

    def test_payload_covers_every_pinned_surface(self):
        payload = build_payload()
        assert set(payload) == {
            "gait", "hitscan", "depth", "projection", "pipeline",
            "engagement", "body", "step_reflex", "yaw", "yaw_rate",
            "gait_speed", "depth_codec", "frame_push", "scan_pump",
            "sensor_rig", "provenance",
        }
        # The gait section must emit all 12 canonical joints.
        sample = payload["gait"]["trot_None"][0]
        assert set(sample) == set(JOINT_NAMES)


# --------------------------------------------------------------------------
# 2. Hash-seed independence (subprocess — the only honest way)
# --------------------------------------------------------------------------

_WORKER = (
    "import sys; sys.path.insert(0, {test_dir!r}); "
    "import test_determinism_new_surface as m; "
    "print(m.payload_digest())"
)


def _digest_under_hashseed(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    worker = _WORKER.format(test_dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            env=env, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"cannot spawn subprocess for hash-seed check: {exc}")
    assert proc.returncode == 0, (
        f"hash-seed worker (PYTHONHASHSEED={seed}) failed:\n{proc.stderr}"
    )
    return proc.stdout.strip()


class TestHashSeedIndependence:
    def test_digest_identical_under_different_hash_seeds(self):
        local = payload_digest()
        seed0 = _digest_under_hashseed("0")
        seed99 = _digest_under_hashseed("99")
        assert len(seed0) == 64  # a real sha256, not an error string
        assert seed0 == seed99, (
            "payload digest varies with PYTHONHASHSEED — some pinned surface "
            "iterates a set or hash-ordered structure"
        )
        assert local == seed0, (
            "in-process digest differs from subprocess digest — hidden "
            "process-local state leaked into a pinned surface"
        )


# --------------------------------------------------------------------------
# 3. No wall-clock dependence
# --------------------------------------------------------------------------

class TestNoWallClock:
    def test_payload_unchanged_after_real_sleep(self):
        before = payload_digest()
        time.sleep(1.1)  # crosses a wall-second boundary
        assert payload_digest() == before

    def test_shot_event_never_reads_clock_with_explicit_timestamp(self, monkeypatch):
        """time.time() is reachable ONLY via the documented timestamp=None default."""
        import tritium_lib.tracking.engagement as engagement_mod

        calls = []
        real_time = time.time

        def spy() -> float:
            calls.append(1)
            return real_time()

        monkeypatch.setattr(engagement_mod.time, "time", spy)
        shot = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
        ShotEvent.from_shot(shot, shooter_id="u", timestamp=5.0)
        ShotEvent.from_payload(
            shot.to_dict() | {"shooter_id": "u", "timestamp": 6.0},
        )
        assert calls == [], (
            "ShotEvent consulted time.time() despite an explicit timestamp"
        )

        # And the documented default DOES use the clock — an API contract,
        # not a silent stamp.
        event = ShotEvent.from_shot(shot)
        assert calls, "timestamp=None default no longer reads the clock"
        assert event.timestamp == pytest.approx(real_time(), abs=5.0)


# --------------------------------------------------------------------------
# 4. Float pins — representative values with explicit tolerances
# --------------------------------------------------------------------------

REL = 1e-12  # closed-form math: far above libm ulp noise, far below drift


class TestGaitPins:
    def test_trot_t0_joint_targets(self):
        angles = joint_targets_at(0.0)
        assert angles["FL_hip"] == pytest.approx(0.0, abs=1e-15)
        assert angles["FL_thigh"] == pytest.approx(0.42266462599716476, rel=REL)
        assert angles["FL_calf"] == pytest.approx(-1.7453292519943295, rel=REL)
        # FR is the anti-phase diagonal at trot phase 0 (mid-swing region).
        assert angles["FR_thigh"] == pytest.approx(1.2408464441789828, rel=REL)
        assert angles["RR_calf"] == pytest.approx(-1.7453292519943295, rel=REL)

    def test_walk_speed_scaled_sample(self):
        angles = joint_targets_at(0.25, gait="walk", speed=0.7)
        assert angles["FL_thigh"] == pytest.approx(0.9026646259971648, rel=REL)
        assert angles["RL_calf"] == pytest.approx(-1.7453292519943295, rel=REL)

    def test_sample_cycle_phases_and_swing_tuck(self):
        cycle = QuadrupedGaitCycle("trot")
        samples = cycle.sample_cycle(8)
        assert [phase for phase, _ in samples] == [i / 8 for i in range(8)]
        assert samples[3][1]["FR_calf"] == pytest.approx(
            -2.013444807085972, rel=REL,
        )

    def test_trot_operating_point(self):
        cycle = QuadrupedGaitCycle("trot")
        assert cycle.stride_hz == pytest.approx(2.6, rel=REL)
        assert cycle.speed_mps == pytest.approx(1.6, rel=REL)
        assert cycle.thigh_amp_rad == pytest.approx(0.45, rel=REL)
        assert cycle.duty_factor == pytest.approx(0.55, rel=REL)


class TestHitscanPins:
    def test_ray_sphere_surface_range(self):
        assert ray_sphere(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0), 1.0,
        ) == pytest.approx(9.0, rel=REL)

    def test_ray_sphere_inside_reports_zero(self):
        assert ray_sphere(
            (10.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0), 1.0,
        ) == pytest.approx(0.0, abs=1e-15)

    def test_ray_aabb_near_face(self):
        assert ray_aabb(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, -1.0, -1.0), (7.0, 1.0, 1.0),
        ) == pytest.approx(5.0, rel=REL)

    def test_muzzle_from_body_world_pose(self):
        muzzle = muzzle_from_body(
            LocalPose(east_m=10.0, north_m=20.0, up_m=0.5, heading_deg=90.0),
            CameraMount(forward_m=0.3, up_m=0.2, tilt_deg=10.0),
            barrel_m=0.5,
        )
        assert muzzle.east_m == pytest.approx(10.792403876506105, rel=REL)
        assert muzzle.north_m == pytest.approx(20.0, abs=1e-12)
        assert muzzle.up_m == pytest.approx(0.7868240888334651, rel=REL)
        assert muzzle.heading_deg == pytest.approx(90.0, rel=REL)
        assert muzzle.elevation_deg == pytest.approx(10.0, rel=REL)

    def test_resolve_shot_nearest_target_wins(self):
        result = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
        assert result.hit and result.target_id == "s_near"
        assert result.range_m == pytest.approx(29.5, rel=REL)
        assert result.impact() == pytest.approx((0.0, 29.5, 1.0), rel=REL)


class TestDepthPins:
    def test_fov_intrinsics(self):
        assert _INTR.fx == pytest.approx(320.0, rel=REL)
        assert _INTR.fy == pytest.approx(320.0, rel=REL)
        assert (_INTR.cx, _INTR.cy) == (320.0, 240.0)

    def test_range_for_bbox_constant_frame(self):
        depth = np.full((480, 640), 10.0)
        assert range_for_bbox(
            depth, (300.0, 195.0, 40.0, 90.0),
        ) == pytest.approx(10.0, rel=REL)

    def test_range_for_bbox_millimetre_scale(self):
        depth = np.full((480, 640), 8000.0)
        assert range_for_bbox(
            depth, (300.0, 195.0, 40.0, 90.0), depth_scale=0.001,
        ) == pytest.approx(8.0, rel=REL)

    def test_deproject_pixel_center_and_offaxis(self):
        assert deproject_pixel(320.0, 240.0, 10.0, _INTR) == pytest.approx(
            (0.0, 0.0, 10.0), abs=1e-12,
        )
        x, y, z = deproject_pixel(400.0, 300.0, 10.0, _INTR)
        assert x == pytest.approx(2.5, rel=1e-9)
        assert y == pytest.approx(1.875, rel=1e-9)
        assert z == pytest.approx(10.0, rel=REL)


class TestProjectionPins:
    def test_flat_north_facing_camera(self):
        world = world_from_camera_xyz(
            (0.0, 0.0, 10.0), CameraWorldPose(heading_deg=0.0, height_m=2.0),
        )
        assert world["east"] == pytest.approx(0.0, abs=1e-12)
        assert world["north"] == pytest.approx(10.0, rel=REL)
        assert world["up"] == pytest.approx(2.0, rel=REL)
        assert world["lat"] is None and world["lng"] is None

    def test_posed_camera_with_geo(self):
        world = world_from_camera_xyz(
            (2.0, 1.0, 10.0),
            CameraWorldPose(
                lat=37.0, lng=-122.0, heading_deg=45.0, pitch_deg=-10.0,
                height_m=3.0,
            ),
        )
        assert world["east"] == pytest.approx(8.255068161604312, rel=REL)
        assert world["north"] == pytest.approx(5.426641036858122, rel=REL)
        assert world["up"] == pytest.approx(0.27871047031848883, rel=REL)
        assert world["lat"] == pytest.approx(37.00004880291907, abs=1e-11)
        assert world["lng"] == pytest.approx(-121.99990704187545, abs=1e-11)


class TestPipelinePins:
    def test_measured_and_fallback_targets_land_where_pinned(self):
        payload = _pipeline_payload()
        rows = {row["asset_type"]: row for row in payload["targets"]}
        person, car = rows["person"], rows["vehicle"]

        assert person["id"].startswith("det_person_")
        assert person["kinematics"]["depth_source"] == "isaac_depth"
        assert person["position"][0] == pytest.approx(0.0, abs=1e-9)
        assert person["position"][1] == pytest.approx(10.0, rel=1e-9)
        assert person["kinematics"]["range_m"] == pytest.approx(10.0, rel=1e-9)
        assert person["kinematics"]["world_enu"][2] == pytest.approx(2.0, rel=1e-9)

        # The car bbox sits in the depth dropout: it must arrive via the
        # honest flat-ground fallback, at the pinned approximate position.
        assert car["kinematics"]["depth_source"] == "2d_ground"
        assert car["position"][0] == pytest.approx(-12.48313356374922, rel=1e-9)
        assert car["position"][1] == pytest.approx(30.136950350518173, rel=1e-9)
        assert car["kinematics"]["bearing_deg"] == pytest.approx(337.5, rel=1e-9)


class TestEngagementPins:
    def test_miss_terminus_falls_back_to_range_gate(self):
        muzzle = Muzzle(
            east_m=3.0, north_m=4.0, up_m=1.0, heading_deg=0.0,
            elevation_deg=0.0,
        )
        shot = resolve_shot(muzzle, [], 100.0)
        event = ShotEvent.from_shot(shot, timestamp=1.0)
        assert not event.hit
        assert event.origin == pytest.approx((3.0, 4.0, 1.0), rel=REL)
        assert event.terminus == pytest.approx((3.0, 104.0, 1.0), rel=REL)

    def test_shot_ids_are_monotonic_and_unique(self):
        shot = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
        events = [ShotEvent.from_shot(shot, timestamp=1.0) for _ in range(3)]
        numbers = [int(e.shot_id.split("_")[1]) for e in events]
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == 3

    def test_bounded_log_stats(self):
        shot = resolve_shot(_MUZZLE_N, _HITSCAN_TARGETS, 100.0)
        log = EngagementLog(max_events=4)
        for i in range(6):
            log.record(ShotEvent.from_shot(shot, timestamp=float(i)))
        assert len(log) == 4
        stats = log.stats()
        assert stats == {"shots": 4, "hits": 4, "accuracy": 1.0}


class TestStepReflexPins:
    def test_capture_point_pins(self):
        cx, cy = capture_point((0.42, -0.15), 0.31)
        assert cx == pytest.approx(0.07467406604068445, rel=REL)
        assert cy == pytest.approx(-0.02666930930024445, rel=REL)

    def test_step_target_clamps_to_pinned_corner(self):
        leg, target = step_target(
            _REFLEX_LEGS, (0.05, 0.30),
            reach_limits=ReachLimits(max_dx=0.1, max_dy=0.1),
        )
        assert leg == "FL"
        assert target[0] == pytest.approx(0.09, rel=REL)
        assert target[1] == pytest.approx(0.23, rel=REL)

    def test_gated_decision_pins(self):
        reflex = StepReflex(com_height_m=0.31)
        # Quiet case runs AT WALKING SPEED with the nominal supplied — the
        # regime the absolute gate measured 0/6 upright in (2026-07-17).
        # Its absolute capture point (~0.110 m) is deep inside the old
        # failure band; the deviation gate must read only the residual.
        quiet = reflex.decide(
            (0.62, -0.03), _REFLEX_LEGS,
            nominal_vel_xy=(0.60, 0.0),
            reach_limits=ReachLimits(max_dx=0.1, max_dy=0.08),
        )
        assert quiet.step is None
        assert quiet.deviation_distance_m == pytest.approx(
            0.006410504144216, rel=REL,
        )
        shoved = reflex.decide(
            (0.0, 0.6), _REFLEX_LEGS,
            nominal_vel_xy=(0.0, 0.0),  # standing: deviation == absolute
            reach_limits=ReachLimits(max_dx=0.1, max_dy=0.08),
        )
        assert shoved.step is not None
        assert shoved.deviation_distance_m == pytest.approx(
            0.1066772372009778, rel=REL,
        )
        assert shoved.step.leg == "FL"
        assert shoved.step.foot_target[0] == pytest.approx(0.09, rel=REL)
        assert shoved.step.foot_target[1] == pytest.approx(
            0.1066772372009778, rel=REL,
        )
        assert shoved.step.residual_m == pytest.approx(0.09, rel=REL)


class TestYawRegulatorPins:
    def test_wrap_pins(self):
        assert heading_error_deg(359.0, 1.0) == pytest.approx(2.0, rel=REL)
        assert heading_error_deg(1.0, 359.0) == pytest.approx(-2.0, rel=REL)
        assert heading_error_deg(0.0, 180.0) == pytest.approx(-180.0, rel=REL)
        assert heading_error_deg(180.0, -180.0) == pytest.approx(0.0, abs=1e-15)

    def test_default_law_pins(self):
        regulator = YawRegulator()
        plain = regulator.correct(0.0, 10.0)
        assert plain.correction_dps == pytest.approx(15.0, rel=REL)  # kp=1.5
        assert not plain.saturated
        damped = regulator.correct(0.0, 10.0, measured_yaw_rate_dps=5.0)
        assert damped.correction_dps == pytest.approx(12.0, rel=REL)  # -kd*5
        clamped = regulator.correct(0.0, 179.0)
        assert clamped.correction_dps == pytest.approx(30.0, rel=REL)
        assert clamped.saturated

    def test_zero_error_is_exactly_zero(self):
        corr = YawRegulator().correct(90.0, 90.0)
        assert corr.correction_dps == 0.0
        assert corr.turn_intent(60.0) == 0.0

    def test_deadbeat_and_fold_in_pins(self):
        capped = YawRegulator(
            kp=50.0, kd=0.0, max_correction_dps=90.0,
        ).correct(0.0, 1.0, dt=0.1)
        assert capped.correction_dps == pytest.approx(10.0, rel=REL)
        regulator = YawRegulator()
        assert regulator.correct(0.0, 2.0).turn_intent(60.0) == pytest.approx(
            0.05, rel=REL,
        )
        assert regulator.hold(0.0, 10.0, 0.25, 60.0) == pytest.approx(
            0.5, rel=REL,
        )


class TestYawRateTrackerPins:
    """Hand-computed single-tick pins for the rate tracker's PI law."""

    def test_no_op_is_exactly_zero(self):
        cmd = YawRateTracker(turn_rate_dps=60.0).track(0.0, 0.0, 0.02)
        assert cmd.turn == 0.0
        assert cmd.state.integral_deg == 0.0
        assert not cmd.saturated

    def test_opening_tick_of_a_step_demand(self):
        # error 6; integrate first: I = 6 * 0.02 = 0.12; then
        # comp = 6 + 1.0*6 + 6.0*0.12 = 12.72; turn = 12.72 / 60 = 0.212.
        cmd = YawRateTracker(turn_rate_dps=60.0).track(6.0, 0.0, 0.02)
        assert cmd.compensated_dps == pytest.approx(12.72, rel=REL)
        assert cmd.turn == pytest.approx(0.212, rel=REL)
        assert cmd.state.integral_deg == pytest.approx(0.12, rel=REL)
        assert not cmd.saturated

    def test_saturating_tick_freezes_the_integral(self):
        # raw = 120 + 120 + 0 = 240 > 60, same-sign error: conditional
        # integration must leave the integral at EXACTLY zero.
        cmd = YawRateTracker(turn_rate_dps=60.0).track(120.0, 0.0, 0.02)
        assert cmd.turn == 1.0
        assert cmd.saturated
        assert cmd.state.integral_deg == 0.0

    def test_desaturating_tick_with_a_warm_integral(self):
        # error -2; I = 2.0 - 2*0.02 = 1.96;
        # comp = 4 - 2 + 6*1.96 = 13.76; turn = 13.76 / 60.
        cmd = YawRateTracker(turn_rate_dps=60.0).track(
            4.0, 6.0, 0.02, state=YawRateState(integral_deg=2.0),
        )
        assert cmd.compensated_dps == pytest.approx(13.76, rel=REL)
        assert cmd.turn == pytest.approx(13.76 / 60.0, rel=REL)
        assert cmd.state.integral_deg == pytest.approx(1.96, rel=REL)

    def test_default_integral_limit_derivation(self):
        assert YawRateTracker(
            turn_rate_dps=60.0
        ).effective_integral_limit_deg == pytest.approx(10.0, rel=REL)


class TestGaitSpeedPins:
    """Hand-computed single-tick pins for the gait speed tracker.

    Defaults: nominal 1.6 -> band [0.32, 2.0] m/s, kp 0.4, ki 1.5,
    slew 0.5 (m/s)/s, deadband 0.05, integral limit 2.0/1.5.
    """

    def test_inert_tick_is_byte_exact(self):
        cmd = GaitSpeedTracker(nominal_mps=1.6).track(1.2, 1.2, 0.4)
        assert cmd.demand_mps == 1.2          # ==, not approx: pass-through
        assert cmd.amp_scale == 1.0
        assert cmd.state.integral_mps == 0.0
        assert not cmd.saturated
        assert not cmd.at_ceiling

    def test_slew_limited_opening_tick_is_not_reported_as_ceiling(self):
        # cold ref = 1.2, slew band tops at 1.4; raw = 1.2 + 0.4*0.75 =
        # 1.5 > 1.4 -> frozen integral, demand rides the slew bound, and
        # at_ceiling stays False (a ramp is a transient, not a wall).
        cmd = GaitSpeedTracker(nominal_mps=1.6).track(1.2, 0.45, 0.4)
        assert cmd.demand_mps == pytest.approx(1.4, rel=REL)
        assert cmd.state.integral_mps == 0.0
        assert cmd.saturated
        assert not cmd.at_ceiling

    def test_unsaturated_tick_integrates_then_recomputes(self):
        # error 0.1; I = 0.1 * 0.4 = 0.04;
        # demand = 1.2 + 0.4*0.1 + 1.5*0.04 = 1.30.
        cmd = GaitSpeedTracker(nominal_mps=1.6).track(1.2, 1.1, 0.4)
        assert cmd.demand_mps == pytest.approx(1.30, rel=REL)
        assert cmd.state.integral_mps == pytest.approx(0.04, rel=REL)

    def test_authority_ceiling_reports_honestly(self):
        # cold ref clamps to hi = 2.0; raw = 2.0 + 0.4*1.4 = 2.56 > 2.0:
        # frozen integral, demand parked at the stop, shortfall told.
        cmd = GaitSpeedTracker(nominal_mps=1.6).track(2.0, 0.6, 0.4)
        assert cmd.demand_mps == pytest.approx(2.0, rel=REL)
        assert cmd.at_ceiling
        assert cmd.ceiling_ticks == 1
        assert cmd.shortfall_mps == pytest.approx(1.4, rel=REL)
        assert cmd.state.integral_mps == 0.0

    def test_floor_tick_freezes_integral_and_trims_amplitude(self):
        # raw = 0.4 + 0.4*(-0.6) = 0.16 < floor with error < 0: the
        # floor-side conditional integration holds I at exactly zero,
        # demand pins at lo = 0.32, amp trims to commanded/measured = 0.4.
        cmd = GaitSpeedTracker(nominal_mps=1.6).track(0.4, 1.0, 0.4)
        assert cmd.demand_mps == pytest.approx(0.32, rel=REL)
        assert cmd.amp_scale == pytest.approx(0.4, rel=REL)
        assert cmd.state.integral_mps == 0.0
        assert cmd.saturated
        assert not cmd.at_ceiling

    def test_default_integral_limit_derivation(self):
        assert GaitSpeedTracker(
            nominal_mps=1.6
        ).effective_integral_limit_mps == pytest.approx(2.0 / 1.5, rel=REL)

    def test_estimator_window_pin(self):
        est = StrideSpeedEstimator(window_s=0.8)
        assert est.update(0.0, 0.0, 0.0) is None
        assert est.update(0.5, 0.55, -0.2) is None    # half window: None
        speed = est.update(1.0, 1.1, -0.4)
        assert speed == pytest.approx(math.hypot(1.1, -0.4), rel=REL)

    def test_phase_clock_pin(self):
        clock = GaitPhaseClock(2.6)
        assert clock.phase_at(1.0) == pytest.approx(2.6, rel=REL)
        clock.retime(1.0, 3.25)
        assert clock.phase_at(1.0) == pytest.approx(2.6, rel=REL)  # C0
        assert clock.phase_at(2.0) == pytest.approx(5.85, rel=REL)
        clock.retime(2.0, 1.3)
        assert clock.phase_at(3.0) == pytest.approx(7.15, rel=REL)


class TestBodyPins:
    def test_intent_from_motors_pins(self):
        straight = intent_from_motors(1.0, 1.0)
        assert straight.forward == pytest.approx(1.0, rel=REL)
        assert straight.turn == pytest.approx(0.0, abs=1e-15)
        mixed = intent_from_motors(0.6, -0.2)
        assert mixed.forward == pytest.approx(0.2, rel=1e-9)
        assert mixed.turn == pytest.approx(0.8, rel=1e-9)
        clamped = intent_from_motors(1.5, -3.0)  # clamps to (1, -1)
        assert clamped.forward == pytest.approx(0.0, abs=1e-15)
        assert clamped.turn == pytest.approx(1.0, rel=REL)

    def test_motors_from_intent_saturates(self):
        left, right = motors_from_intent(ControlIntent(forward=0.5, turn=1.0))
        assert left == pytest.approx(1.0, rel=REL)
        assert right == pytest.approx(0.0, abs=1e-15)

    def test_round_trip_inside_envelope(self):
        intent = intent_from_motors(0.5, 0.1)
        left, right = motors_from_intent(intent)
        assert left == pytest.approx(0.5, rel=1e-12)
        assert right == pytest.approx(0.1, rel=1e-12)
        again = intent_from_motors(left, right)
        assert again.forward == pytest.approx(intent.forward, rel=1e-12)
        assert again.turn == pytest.approx(intent.turn, rel=1e-12)


class TestDepthCodecFidelityPins:
    """Round-trip FIDELITY, not just determinism.

    The codec moves metric uint16-mm depth (ROS ``16UC1``).  A codec that is
    not exactly round-trip stable is a silent data-corruption bug: the
    operator would be holding a picture of depth, not depth — which is
    precisely the defect that shipped this module (SC decoded the depth PNG
    with ``IMREAD_COLOR`` and destroyed the sender's lossless frame).

    Where the codec is lossy it is lossy BY DESIGN, at the encode boundary
    only, and each loss is a refusal to lie rather than an erosion:

      * metres quantize to the ``1/scale`` grid (1 mm by default) — the grid
        IS the wire format; values already on it are exact;
      * ``NaN``/``inf``/``<= 0`` fold to the 0 no-return sentinel and decode
        to ``NaN`` (one-way: a hole is not a distance);
      * a valid reading under half a unit lifts to 1 unit so it cannot
        masquerade as a hole;
      * beyond the uint16 ceiling saturates to 65535, never wraps.

    On the uint16 grid itself — the transport's native domain — the round
    trip must be BIT-EXACT, and that is what these tests hold, including the
    edges (0, 1, 65535), a full-range ramp, and non-square shapes.
    """

    def _assert_units_roundtrip(
        self, units: np.ndarray, scale: float = DEPTH_SCALE_MM,
    ) -> None:
        metres = _units_to_metres(units, scale)
        decoded = decode_depth16_png(
            encode_depth16_png(metres, scale=scale), scale=scale,
        )
        assert decoded.shape == units.shape
        assert decoded.dtype == np.float32
        valid = units != 0
        assert np.isnan(decoded[~valid]).all(), (
            "no-return sentinel pixels must decode to NaN, not 0 m"
        )
        recovered = np.rint(
            decoded[valid].astype(np.float64) * scale
        ).astype(np.uint16)
        assert np.array_equal(recovered, units[valid]), (
            "uint16 depth did not survive its own wire format bit-exact"
        )

    def test_full_range_ramp_bit_exact(self):
        # Every representable millimetre value once, in a non-square frame.
        self._assert_units_roundtrip(
            np.arange(65536, dtype=np.uint16).reshape(128, 512),
        )

    def test_edge_values_bit_exact(self):
        self._assert_units_roundtrip(
            np.array([[0, 1, 2], [65534, 65535, 12500]], dtype=np.uint16),
        )

    def test_non_square_shapes_bit_exact(self):
        for shape in ((1, 5), (7, 3), (24, 56), (1, 1)):
            n = shape[0] * shape[1]
            units = np.linspace(1, 65535, n).astype(np.uint16).reshape(shape)
            self._assert_units_roundtrip(units)

    def test_centimetre_scale_bit_exact(self):
        self._assert_units_roundtrip(
            np.array([[1, 30000, 65535], [0, 2, 65534]], dtype=np.uint16),
            scale=100.0,
        )

    def test_reencode_is_byte_stable(self):
        """A relay hop (decode -> re-encode) must reproduce the exact bytes."""
        metres = _units_to_metres(
            np.arange(65536, dtype=np.uint16).reshape(256, 256),
        )
        first = encode_depth16_png(metres)
        assert encode_depth16_png(metres) == first
        assert encode_depth16_png(decode_depth16_png(first)) == first

    def test_colour_decode_destroys_depth_the_original_defect(self):
        """The failure mode this codec exists to prevent, proven live.

        SC originally decoded the depth PNG with ``cv2.IMREAD_COLOR``.  This
        test decodes the SAME blob both ways and shows the colour path
        cannot pass the bit-exact assertion the metric path passes — i.e.
        the fidelity tests above would have caught the original defect.
        """
        cv2 = pytest.importorskip("cv2")
        units = np.arange(65536, dtype=np.uint16).reshape(128, 512)
        blob = encode_depth16_png(_units_to_metres(units))

        # The metric path recovers every level bit-exact...
        good = decode_depth16_png(blob)
        valid = units != 0
        assert np.array_equal(
            np.rint(good[valid].astype(np.float64) * 1000.0).astype(np.uint16),
            units[valid],
        )

        # ...the colour path — what SC used to do — provably cannot.
        bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.ndim == 3 and bgr.dtype == np.uint8, (
            "IMREAD_COLOR yields a picture of depth: 3-channel uint8"
        )
        channel = bgr[..., 0].astype(np.uint16)
        assert not np.array_equal(channel, units)
        assert not np.array_equal(channel << np.uint16(8), units), (
            "even rescaling the colour channel cannot recover the metric "
            "frame — the low byte of every reading is gone"
        )
        # 65536 distinct millimetre levels collapse into a uint8 container:
        # ~256-to-1, unconditionally lossy.
        assert np.unique(channel).size <= 256 < np.unique(units).size

    def test_cross_codec_value_identity_cv2_pillow(self):
        """cv2 and Pillow emit DIFFERENT bytes for the same pixels (both are
        valid PNG streams; compression differs), but each must decode the
        other's blob to the identical uint16 values.  Byte identity across
        codecs is deliberately NOT pinned — only value identity is the
        contract a relay hop relies on."""
        cv2 = pytest.importorskip("cv2")
        Image = pytest.importorskip("PIL.Image")
        units = np.linspace(0, 65535, 31 * 302).astype(np.uint16).reshape(31, 302)

        ok, buf = cv2.imencode(".png", units)
        assert ok
        cv2_blob = buf.tobytes()
        out = io.BytesIO()
        Image.fromarray(units).save(out, format="PNG")
        pil_blob = out.getvalue()

        for blob in (cv2_blob, pil_blob):
            via_cv2 = cv2.imdecode(
                np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED,
            )
            via_pil = np.array(Image.open(io.BytesIO(blob)))
            assert np.array_equal(via_cv2.astype(np.uint16), units)
            assert np.array_equal(via_pil.astype(np.uint16), units)


class TestFramePushPins:
    def test_backoff_grows_exponentially_and_caps(self):
        policy = FramePushPolicy(
            target_fps=1000.0, base_backoff_s=0.5, max_backoff_s=4.0,
        )
        got = []
        for i in range(6):
            t = 100.0 * (i + 1)
            assert policy.offer(t).send
            policy.failed(t)
            got.append(policy.backoff_remaining(t))
        assert got == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]

    def test_deep_outage_saturates_instead_of_overflowing(self):
        """Past 32 doublings the exponent saturates straight to the cap —
        pinned with a cap large enough that the min() path alone would give
        a DIFFERENT (smaller) answer, so the guard branch is observable."""
        policy = FramePushPolicy(
            target_fps=1000.0, base_backoff_s=1.0, max_backoff_s=1e18,
        )
        t = 0.0
        last_backoff = 0.0
        for i in range(34):
            assert policy.offer(t).send
            policy.failed(t)
            last_backoff = policy.backoff_remaining(t)
            t += last_backoff + 1.0
        # 34 consecutive failures -> exponent 33 > 32 -> straight to the cap
        # (2**33 * 1.0 s would be ~8.6e9, far below 1e18 — so the saturation
        # branch, not the min(), is what this value proves).
        assert policy.stats.consecutive_failures == 34
        assert last_backoff == pytest.approx(1e18, rel=1e-9)

    def test_strict_cycle_is_enforced(self):
        policy = FramePushPolicy(target_fps=10.0)
        with pytest.raises(RuntimeError):
            policy.sent(0.0)
        with pytest.raises(RuntimeError):
            policy.failed(0.0)
        assert policy.offer(0.0).send
        policy.sent(0.1)
        # A refused offer does not open a cycle either.
        assert not policy.offer(0.11).send
        with pytest.raises(RuntimeError):
            policy.sent(0.12)

    def test_healthy_requires_a_delivered_frame(self):
        policy = FramePushPolicy(target_fps=10.0)
        assert not policy.healthy          # never delivered != starting up
        assert policy.offer(0.0).send
        policy.sent(0.1)
        assert policy.healthy
        assert policy.offer(1.0).send
        policy.failed(1.1)
        assert not policy.healthy          # failing right now
        assert policy.offer(5.0).send
        policy.sent(5.1)
        assert policy.healthy              # recovery restores it

    def test_frame_push_path_percent_encodes_hostile_ids(self):
        assert frame_push_path("isaac_rgb") == (
            "/api/camera-feeds/sources/isaac_rgb/frame"
        )
        assert frame_push_path("a/b c") == (
            "/api/camera-feeds/sources/a%2Fb%20c/frame"
        )
        with pytest.raises(ValueError):
            frame_push_path("")
        with pytest.raises(ValueError):
            frame_push_path("   ")


class TestScanPumpPins:
    _LIVE = {"ranges": [1.0, 2.0, 3.0], "range_max": 10.0}

    def test_stuck_sweep_stays_refused_forever(self):
        """A stopped LiDAR serving its last sweep must LOOK dead: the stale
        refusal must not alternate forward/stale by re-baselining."""
        pump = ScanPump("l1")
        assert pump.offer(self._LIVE).forward
        for _ in range(5):
            decision = pump.offer(dict(self._LIVE))
            assert not decision.forward
            assert decision.reason == "stale"
            assert decision.payload is None

    def test_no_return_epsilon_boundary(self):
        # All beams within 1e-6 of range_max: no information either way.
        pump = ScanPump("l2")
        d = pump.offer({"ranges": [10.0, 10.0 - 5e-7], "range_max": 10.0})
        assert not d.forward and d.reason == "no_returns"
        # One beam clearly short of the ceiling: a real return, forwarded.
        pump2 = ScanPump("l3")
        d2 = pump2.offer({"ranges": [10.0, 10.0 - 1e-3], "range_max": 10.0})
        assert d2.forward

    def test_payload_carries_pose_and_sweep_geometry(self):
        pump = ScanPump("go2_lidar", sensor_x=1.0, sensor_y=2.0,
                        sensor_yaw_deg=45.0)
        pump.set_sensor_pose(3.5, -1.5, 270.0)
        scan = {
            "ranges": [2.0, 4.0], "angle_min": -1.5, "angle_increment": 0.1,
            "range_min": 0.05, "range_max": 12.0,
        }
        decision = pump.offer(scan)
        assert decision.forward
        assert decision.payload == {
            "source": "lidar",
            "lidar_id": "go2_lidar",
            "ranges": [2.0, 4.0],
            "sensor_x": 3.5,
            "sensor_y": -1.5,
            "sensor_yaw_deg": 270.0,
            "angle_min": -1.5,
            "angle_increment": 0.1,
            "range_min": 0.05,
            "range_max": 12.0,
        }
        # Geometry keys absent from the scan stay absent — never defaulted.
        pump2 = ScanPump("l4")
        bare = pump2.offer({"ranges": [1.0, 2.0]})
        assert bare.forward
        assert "angle_min" not in bare.payload

    def test_breaker_trips_and_resets(self):
        pump = ScanPump("l5", max_failures=1)
        assert pump.offer(self._LIVE).forward
        pump.record_result(False)
        assert pump.tripped
        refused = pump.offer({"ranges": [9.0], "range_max": 10.0})
        assert not refused.forward and refused.reason == "tripped"
        pump.record_result(True)
        assert not pump.tripped
        assert pump.offer({"ranges": [8.0], "range_max": 10.0}).forward

    def test_malformed_input_never_raises(self):
        pump = ScanPump("l6")
        for bad in (None, 7, "scan", {}, {"ranges": None}, {"ranges": []},
                    {"ranges": 5}, {"ranges": ["x", 1.0]}):
            decision = pump.offer(bad)
            assert not decision.forward
            assert decision.reason == "malformed"
        assert pump.stats()["refusals"]["malformed"] == 8


class TestSensorRigPins:
    def test_push_plan_carries_no_address(self):
        """Push exists BECAUSE the advertised address is unreachable; a
        push payload leaking host/port would reintroduce the failure."""
        for call in registration_plan(_RIG, push=True):
            assert "host" not in call.payload
            assert "port" not in call.payload
            assert call.payload["source_type"] == "push"
        for call in registration_plan(_RIG):
            assert "host" in call.payload  # pull mode: SC dials the sensor
            assert "port" in call.payload

    def test_attach_to_is_omitted_never_null(self):
        for plan in (registration_plan(_RIG), registration_plan(_RIG, push=True)):
            by_role = {c.role: c.payload for c in plan}
            assert by_role["depth"]["attach_to"] == "unit-go2"
            assert "attach_to" not in by_role["camera"]
            assert None not in by_role["camera"].values()

    def test_non_pixel_and_unready_roles_get_no_feed(self):
        roles = [c.role for c in registration_plan(_RIG)]
        assert "lidar" not in roles      # streams sightings, not frames
        assert "body" not in roles       # streams pose
        assert "thermal" not in roles    # unknown role: refused, not guessed
        assert "stereo_right" not in roles  # not ready: never registered
        assert roles == ["camera", "depth", "camera"]

    def test_feed_source_ids_disambiguate_streams(self):
        assert [s.feed_source_id() for s in _RIG] == [
            "isaac_rgb", "isaac_depth16", "isaac_right", "isaac_lidar",
            "isaac_body", "isaac_thermal", "custom_cam",
        ]

    def test_empty_rig_is_not_ok(self):
        """``all([])`` is True — the trap this report exists to refuse."""
        assert not summarize_bringup([]).ok
        assert not summarize_bringup([("a", "skipped")]).ok
        assert not summarize_bringup(
            [("a", "registered"), ("b", "failed")]
        ).ok
        assert summarize_bringup([("a", "already_registered")]).ok

    def test_unknown_outcome_raises(self):
        with pytest.raises(ValueError):
            summarize_bringup([("cam", "grand_success")])


class TestProvenancePins:
    def test_heuristic_confidence_is_zeroed_but_unstamped_kept(self):
        tracker = TargetTracker()
        heuristic = tracker.update_from_detection({
            "class_name": MOTION_CLASS, "class_source": "heuristic",
            "confidence": 0.9, "center_x": 0.0, "center_y": 0.0,
        })
        classified = tracker.update_from_detection({
            "class_name": "person", "class_source": "classifier",
            "confidence": 0.92, "center_x": 10.0, "center_y": 0.0,
        })
        unstamped = tracker.update_from_detection({
            "class_name": "car", "confidence": 0.66,
            "center_x": 20.0, "center_y": 0.0,
        })
        h = tracker.get_target(heuristic)
        assert h.classification_confidence == 0.0
        assert h.class_source == "heuristic"
        c = tracker.get_target(classified)
        assert c.classification_confidence == pytest.approx(0.92)
        assert c.class_source == "classifier"
        # A legacy caller with no class_source keeps its confidence — the
        # SC YOLO plugin publishes real verdicts with no stamp yet, and
        # silently zeroing those would discard genuine classifier output.
        u = tracker.get_target(unstamped)
        assert u.classification_confidence == pytest.approx(0.66)
        assert u.class_source == ""
        assert tracker.get_target(heuristic).to_dict()["class_source"] == (
            "heuristic"
        )

    def test_shape_hint_reports_nothing_when_ambiguous(self):
        """The old ``_classify`` made 'person' the catch-all; the square
        blob that reproduced the defect must now yield NO hint."""
        detector = BackgroundMotionDetector()
        assert detector._shape_hint(10, 13) == "tall"   # 13 >= 10 * 1.25
        assert detector._shape_hint(10, 12) is None     # under both ratios
        assert detector._shape_hint(14, 10) == "wide"   # 14 >= 10 * 1.4
        assert detector._shape_hint(13, 10) is None
        assert detector._shape_hint(7, 7) is None       # the square blob
        assert detector._shape_hint(4, 5) == "tall"     # exact boundary

    def test_motion_detector_never_speaks_coco(self):
        detector = BackgroundMotionDetector(min_area=100, history=10)
        seen = []
        for frame in _motion_frames():
            seen.extend(detector.detect(frame, "cam-x"))
        assert seen, "scripted sequence must produce detections"
        for det in seen:
            assert det.class_name == MOTION_CLASS
            assert det.class_source == "heuristic"
            assert not det.is_classified
            assert det.display_label == "MOTION"

    def test_detection_timestamps_are_utc_and_current(self):
        """The quarantined field's contract, asserted separately: the
        detector stamps datetime.now(timezone.utc) — documented API, like
        ShotEvent's timestamp=None default."""
        detector = BackgroundMotionDetector(min_area=100, history=10)
        stamped = []
        for frame in _motion_frames():
            stamped.extend(detector.detect(frame, "cam-x"))
        assert stamped
        now = datetime.now(timezone.utc)
        for det in stamped:
            assert det.timestamp is not None
            assert det.timestamp.tzinfo is not None
            assert det.timestamp.utcoffset().total_seconds() == 0.0
            assert abs((now - det.timestamp).total_seconds()) < 60.0

    def test_pipeline_payload_carries_provenance(self):
        sink: list[dict] = []
        pipeline = FrameDetectionPipeline(
            _ProvenanceDetector(),
            lambda: np.zeros((480, 640, 3), dtype=np.uint8),
            sink.append,
            source_id="cam-7",
        )
        assert pipeline.tick() == 2  # the 0.30-confidence car is gated
        motion, person = sink
        assert motion["class_source"] == "heuristic"
        assert motion["shape_hint"] == "tall"
        assert person["class_source"] == "classifier"
        assert person["shape_hint"] is None
        assert motion["source_camera"] == "cam-7"

    def test_keyed_identity_is_strict_and_never_merges(self):
        tracker = TargetTracker()
        first = tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.9,
             "center_x": 5.0, "center_y": 6.0},
            detection_key="bytetrack:9",
        )
        # Same key, far away: SAME track follows (no re-mint, no proximity).
        moved = tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.9,
             "center_x": 55.0, "center_y": 66.0},
            detection_key="bytetrack:9",
        )
        assert moved == first
        assert tracker.get_target(first).position == (55.0, 66.0)
        # Different key, co-located: keys assert identity -> two tracks.
        other = tracker.update_from_detection(
            {"class_name": "person", "confidence": 0.9,
             "center_x": 55.0, "center_y": 66.0,
             "source_track_id": "radar-4"},
        )
        assert other != first
