# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Has the body fallen over?  Attitude metrics that displacement cannot fake.

This module exists because of one specific way a locomotion test lies.  Score a
gait on how far the body travelled and how much height it kept, and a robot
that flips onto its back and skitters along on its shoulders scores as a
healthy walk: it covers ground, and an inverted quadruped sits at very nearly
the same height as a standing one, so a height-retention check never fires.
That is not a hypothetical — a Go2 trotting in a live Newton sim scored
displacement 1.27 m, height retained 0.89, collapsed False, verdict MOVED,
while the rendered frame showed it upside down with its legs in the air.

Height cannot see rotation, so the metric has to watch rotation directly.  The
quantity here is the angle between the body's own up axis and world up: 0 deg
standing, ~90 on its side, ~180 on its back.  Sliding, bouncing and skittering
cannot move it, which is exactly the property a locomotion gate needs.

Conventions:
  * Quaternions are ``(w, x, y, z)``, matching
    :mod:`tritium_lib.geo.isaac_frame`.  They need not be unit length —
    solver-read quaternions drift, and normalising is this module's job.
  * The world is Z-up, matching Isaac/Newton and the rest of ``geo``.
  * Yaw is deliberately invisible here.  A walking body changes heading
    constantly and that is not a fall, which is why this measures the up axis
    rather than a generic distance from the identity rotation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "DEFAULT_MAX_TILT_DEG",
    "is_upright",
    "tilt_from_upright_deg",
]

# A trotting quadruped pitches and rolls a few degrees every stride, and a
# rover on broken ground more than that.  The threshold has to clear normal
# gait lean without reaching anything a person would call a fall; 45 deg sits
# an order of magnitude above the former and half way to the latter.
DEFAULT_MAX_TILT_DEG = 45.0


def _body_up_axis(quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """The body's own +Z axis expressed in world coordinates.

    This is the third column of the rotation matrix for ``quat_wxyz``, which is
    all that is needed — building the full matrix to use one column of it would
    be waste.
    """
    if len(quat_wxyz) != 4:
        raise ValueError(
            f"quaternion must have 4 components (w, x, y, z), got {len(quat_wxyz)}"
        )
    w, x, y, z = (float(c) for c in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError(
            "quaternion is all zeros, which is not a rotation. This usually means "
            "the pose read failed; treating it as upright would hide that."
        )
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def tilt_from_upright_deg(quat_wxyz: Sequence[float]) -> float:
    """Angle in degrees between the body's up axis and world up.

    0 is level, 90 is on its side, 180 is fully inverted.  The result is
    independent of heading, so a body that turns while staying level reads 0.
    """
    _, _, up_z = _body_up_axis(quat_wxyz)
    # The dot product of the body's up axis with world up (0, 0, 1) is just its
    # z component.  Clamp before acos: normalisation leaves round-off that can
    # push it a hair outside [-1, 1] and make acos raise.
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def is_upright(
    quat_wxyz: Sequence[float],
    *,
    max_tilt_deg: float = DEFAULT_MAX_TILT_DEG,
) -> bool:
    """Whether the body is still the right way up, within ``max_tilt_deg``.

    Intended as the gate a locomotion run must pass *in addition to* covering
    ground — neither test is sufficient alone.  Distance without this certifies
    a tumble; this without distance certifies standing still.
    """
    return tilt_from_upright_deg(quat_wxyz) <= max_tilt_deg
