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

__all__ = [
    "AttitudeCorrection",
    "AttitudeStabilizer",
    "LegPlacement",
    "roll_pitch_deg",
]
