# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Body-agnostic closed-loop controllers.

Controllers here take a *measurement* and return a *correction*, and know
nothing about the body producing either — no joint names, no leg counts, no
simulator, no ROS.  A quadruped in Isaac, a rover on broken ground and a
headless unit test drive the identical object.  That is what makes this lib
code rather than addon code: the plant is data (see
:class:`~tritium_lib.control.attitude_stabilizer.LegPlacement`), not a subclass.

Stdlib only, so it imports on a bare Jetson alongside the rest of the robot
brain.
"""

from tritium_lib.control.attitude_stabilizer import (
    AttitudeCorrection,
    AttitudeStabilizer,
    LegPlacement,
    roll_pitch_deg,
)
from tritium_lib.control.disturbance import (
    DisturbanceSchedule,
    Impulse,
    KickVerdict,
    RecoveryScore,
    kick_landed,
    score_recovery,
)
from tritium_lib.control.route_trace import (
    RouteScore,
    score_route_trace,
)
from tritium_lib.control.step_reflex import (
    ReachLimits,
    ReflexDecision,
    StepDecision,
    StepReflex,
    capture_point,
    step_target,
    velocity_from_impulse,
)
from tritium_lib.control.stride_filter import (
    StrideFilter,
)
from tritium_lib.control.yaw_rate_loop import (
    YawRateCorrection,
    YawRateLoop,
    yaw_rate_from_headings,
)
from tritium_lib.control.yaw_rate_tracker import (
    TurnCorrection,
    YawRateState,
    YawRateTracker,
)
from tritium_lib.control.yaw_regulator import (
    YawCorrection,
    YawRegulator,
    heading_error_deg,
)
from tritium_lib.control.waypoint_follower import (
    FollowState,
    PurePursuitFollower,
    StrideBias,
    TwistCommand,
    differential_stride,
)
from tritium_lib.control.command_link import (
    CommandLimits,
    CommandLink,
)
from tritium_lib.control.teleop import (
    AxisMap,
    GamepadState,
    SlewLimiter,
    TeleopProfile,
    TeleopWatchdog,
    apply_deadzone,
    apply_expo,
    shape_axis,
    twist_command_from_intent,
    twist_from_stick,
)

__all__ = [
    "AttitudeCorrection",
    "AxisMap",
    "CommandLimits",
    "CommandLink",
    "GamepadState",
    "SlewLimiter",
    "TeleopProfile",
    "TeleopWatchdog",
    "apply_deadzone",
    "apply_expo",
    "shape_axis",
    "twist_command_from_intent",
    "twist_from_stick",
    "AttitudeStabilizer",
    "DisturbanceSchedule",
    "FollowState",
    "Impulse",
    "KickVerdict",
    "LegPlacement",
    "PurePursuitFollower",
    "ReachLimits",
    "RecoveryScore",
    "ReflexDecision",
    "RouteScore",
    "StepDecision",
    "StepReflex",
    "StrideBias",
    "StrideFilter",
    "TwistCommand",
    "capture_point",
    "differential_stride",
    "kick_landed",
    "roll_pitch_deg",
    "score_recovery",
    "score_route_trace",
    "step_target",
    "velocity_from_impulse",
    "TurnCorrection",
    "YawRateCorrection",
    "YawRateLoop",
    "YawRateState",
    "YawRateTracker",
    "yaw_rate_from_headings",
    "YawCorrection",
    "YawRegulator",
    "heading_error_deg",
]
