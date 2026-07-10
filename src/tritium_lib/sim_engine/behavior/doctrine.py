# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Engagement-range doctrine helpers — LOS-recovery repositioning.

When a GROUND unit has a target inside weapon range but its fire solution is
MASKED by a building (fire-time line-of-sight blocked), a good shooter does
not stand there wasting the engagement — it side-steps to a PEEK/FLANK point
that restores line of sight, then re-engages.

``find_peek_position`` is a pure, deterministic candidate search: it samples
lateral offsets perpendicular to the shooter->target bearing, alternating
left/right at increasing distance, and returns the FIRST (closest) offset
whose cell is walkable, keeps the target within weapon range, AND has clear
line of sight to the target.  No RNG — a deterministic candidate order is
both reproducible (golden-replay safe) and better-behaved than seeded noise.

The same helper drives two worlds:
  - FUN: urban stand-in fights come alive; units flank and peek like a player.
  - PRODUCTION: this is the LOS-recovery maneuver a real ground robot needs
    when its fire solution is masked — the identical routine can run against a
    live costmap (any object exposing ``line_of_sight`` + a walkability probe).

Terrain contract
----------------
The ``terrain_map`` argument must expose:
  - ``line_of_sight(pos_a, pos_b) -> bool``  (clear == True)
  - ``get_movement_cost(x, y) -> float``     (``inf`` == impassable: building/water)
  - optional ``bounds`` (half-extent, metres) — candidates outside are rejected.

:class:`tritium_lib.sim_engine.world.terrain_map.TerrainMap` satisfies this.
"""

from __future__ import annotations

import math
from typing import Optional


def _is_walkable(terrain_map, x: float, y: float) -> bool:
    """True when world ``(x, y)`` is inside bounds and not impassable.

    Impassable == infinite movement cost (building or water in TerrainMap).
    Out-of-bounds cells (when the map exposes ``bounds``) are not walkable —
    a peek must stay on the playable field.
    """
    bounds = getattr(terrain_map, "bounds", None)
    if bounds is not None and (abs(x) > bounds or abs(y) > bounds):
        return False
    try:
        cost = terrain_map.get_movement_cost(x, y)
    except Exception:
        return True  # No cost model -> assume open (LOS check still gates).
    return not math.isinf(cost)


def find_peek_position(
    terrain_map,
    shooter_pos: tuple[float, float],
    target_pos: tuple[float, float],
    weapon_range: float,
    *,
    max_offset: float = 30.0,
    step: float = 2.5,
    anchor: Optional[tuple[float, float]] = None,
    max_anchor_dist: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    """Find the closest lateral peek point that restores LOS to the target.

    Deterministic search: lateral offsets perpendicular to the shooter->target
    bearing, sampled at ``step`` increments out to ``max_offset``, trying the
    left side (+perp) then the right side (-perp) at each distance.  The first
    candidate satisfying ALL of:

      (a) walkable / not inside a building (``terrain_map`` walkability),
      (b) clear line of sight to ``target_pos``,
      (c) still within ``weapon_range`` of ``target_pos``,
      (d) (optional) within ``max_anchor_dist`` of ``anchor`` — BOUNDED
          EXPOSURE, so a wounded unit leaning out of cover only exposes itself
          up to a fixed lean-out distance from its cover point.

    is returned as ``(x, y)``.  Returns ``None`` when no such point exists
    (fully masked / boxed in / every restoring point is out of range).

    Because the search walks outward from the smallest offset, the first hit is
    also the peek NEAREST ``shooter_pos`` (and, when ``anchor`` is the cover
    point the unit is standing at, the peek nearest the cover point).

    Args:
        terrain_map: Object exposing ``line_of_sight`` + ``get_movement_cost``
            (see module docstring).
        shooter_pos: Current ``(x, y)`` of the shooter.
        target_pos: ``(x, y)`` of the engagement target.
        weapon_range: Max distance the shooter may be from the target and
            still fire — the peek must stay within it.
        max_offset: Largest lateral offset to try, metres.
        step: Lateral sampling increment, metres (smaller == finer, slower).
        anchor: Optional ``(x, y)`` reference point (e.g. the cover point the
            unit is holding at).  When given with ``max_anchor_dist`` the peek
            is constrained to stay within that radius of the anchor — bounded
            exposure so a unit leans out of cover instead of abandoning it.
        max_anchor_dist: Optional max distance (metres) a peek candidate may be
            from ``anchor``.  Ignored when ``anchor`` is ``None``.

    Returns:
        The closest valid ``(x, y)`` peek point, or ``None``.
    """
    sx, sy = float(shooter_pos[0]), float(shooter_pos[1])
    tx, ty = float(target_pos[0]), float(target_pos[1])

    dx = tx - sx
    dy = ty - sy
    bearing = math.hypot(dx, dy)
    if bearing < 1e-6:
        return None  # Shooter is on top of the target — nothing to peek around.

    # Unit vector perpendicular to the shooter->target bearing.
    perp_x = -dy / bearing
    perp_y = dx / bearing

    wr2 = float(weapon_range) * float(weapon_range)
    steps = max(1, int(max_offset / step))

    # (d) bounded-exposure setup: constrain candidates near the cover anchor.
    anchor_bounded = anchor is not None and max_anchor_dist is not None
    if anchor_bounded:
        ax, ay = float(anchor[0]), float(anchor[1])
        mad2 = float(max_anchor_dist) * float(max_anchor_dist)

    for i in range(1, steps + 1):
        offset = step * i
        for sign in (1.0, -1.0):
            cx = sx + perp_x * offset * sign
            cy = sy + perp_y * offset * sign

            # (a) must be somewhere the unit could actually stand.
            if not _is_walkable(terrain_map, cx, cy):
                continue

            # (d) bounded exposure: stay within the lean-out radius of cover.
            if anchor_bounded:
                adx = cx - ax
                ady = cy - ay
                if (adx * adx + ady * ady) > mad2:
                    continue

            # (c) must keep the target in weapon range (cheap; before LOS).
            cdx = cx - tx
            cdy = cy - ty
            if (cdx * cdx + cdy * cdy) > wr2:
                continue

            # (b) must actually restore the fire solution.
            if not terrain_map.line_of_sight((cx, cy), (tx, ty)):
                continue

            return (cx, cy)

    return None
