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
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import tritium_lib.geo as geo
from tritium_lib.control import (
    LegPlacement,
    ReachLimits,
    StepReflex,
    YawRateState,
    YawRateTracker,
    YawRegulator,
    capture_point,
    heading_error_deg,
    step_target,
    velocity_deviation,
    velocity_from_impulse,
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
from tritium_lib.perception.depth_pipeline import process_depth_frame
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
