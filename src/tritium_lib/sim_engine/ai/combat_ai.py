# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Combat AI behaviors for tactical simulation units.

Provides cover-seeking, flanking, suppression, squad coordination, and
engagement decision-making.  Integrates with the behavior_tree module
(decides WHAT) and steering module (decides HOW).

Obstacles are represented as (center, radius) tuples — same format used
by steering.avoid_obstacles.

Usage::

    from tritium_lib.sim_engine.ai.combat_ai import find_cover, make_assault_tree

    cover = find_cover((10, 10), (50, 50), obstacles=[((30, 30), 5.0)])
    tree = make_assault_tree()
    ctx = {"unit_pos": (10, 10), "threats": [...], "health": 0.8}
    tree.tick(ctx)
"""

from __future__ import annotations

import math
from typing import Optional

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    _sub,
    _add,
    _scale,
)
from tritium_lib.sim_engine.ai.behavior_tree import (
    Node,
    Status,
    Sequence,
    Selector,
    Inverter,
    Cooldown,
    Action,
    Condition,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angle_between(a: Vec2, b: Vec2) -> float:
    """Angle in radians from a to b (atan2)."""
    d = _sub(b, a)
    return math.atan2(d[1], d[0])


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference in radians."""
    d = (b - a) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def _point_behind_obstacle(
    point: Vec2,
    threat_pos: Vec2,
    obs_center: Vec2,
    obs_radius: float,
) -> bool:
    """True if *obs* blocks the line of sight from *threat_pos* to *point*."""
    # Vector from threat to point
    tp = _sub(point, threat_pos)
    tp_len = magnitude(tp)
    if tp_len < 1e-12:
        return False

    # Vector from threat to obstacle center
    to = _sub(obs_center, threat_pos)

    # Project obstacle center onto threat->point line
    tp_norm = normalize(tp)
    proj_len = to[0] * tp_norm[0] + to[1] * tp_norm[1]

    # Obstacle must be between threat and point
    if proj_len < 0 or proj_len > tp_len:
        return False

    # Perpendicular distance from obstacle center to line
    proj_point = _add(threat_pos, _scale(tp_norm, proj_len))
    perp_dist = distance(obs_center, proj_point)

    return perp_dist < obs_radius


# ---------------------------------------------------------------------------
# Cover system
# ---------------------------------------------------------------------------

def find_cover(
    position: Vec2,
    threat_pos: Vec2,
    obstacles: list[tuple[Vec2, float]],
    max_range: float = 50.0,
) -> Optional[Vec2]:
    """Find nearest position behind an obstacle relative to threat.

    Searches candidate positions on the far side of each obstacle
    (away from threat) and returns the closest reachable one within
    *max_range* of *position*.
    """
    best: Optional[Vec2] = None
    best_dist = float("inf")

    for obs_center, obs_radius in obstacles:
        # Candidate: point on the far side of obstacle from threat
        angle_from_threat = _angle_between(threat_pos, obs_center)
        # Place candidate just beyond the obstacle radius
        offset = obs_radius + 2.0
        candidate = (
            obs_center[0] + math.cos(angle_from_threat) * offset,
            obs_center[1] + math.sin(angle_from_threat) * offset,
        )

        d = distance(position, candidate)
        if d > max_range:
            continue

        if d < best_dist:
            best_dist = d
            best = candidate

    return best


def is_in_cover(
    position: Vec2,
    threat_pos: Vec2,
    obstacles: list[tuple[Vec2, float]],
) -> bool:
    """Check if position is behind cover from threat's perspective.

    Returns True if any obstacle blocks line of sight from threat to position.
    """
    for obs_center, obs_radius in obstacles:
        if _point_behind_obstacle(position, threat_pos, obs_center, obs_radius):
            return True
    return False


def rate_cover_position(
    pos: Vec2,
    threat_pos: Vec2,
    obstacles: list[tuple[Vec2, float]],
) -> float:
    """Score a cover position 0-1 based on concealment quality.

    Considers: number of obstacles providing cover, how well-centered
    the position is behind cover, and distance from threat (farther is
    slightly better up to a point).
    """
    if not obstacles:
        return 0.0

    score = 0.0
    blocking_count = 0

    for obs_center, obs_radius in obstacles:
        if _point_behind_obstacle(pos, threat_pos, obs_center, obs_radius):
            blocking_count += 1

            # How centered are we behind this obstacle?
            threat_to_obs = _angle_between(threat_pos, obs_center)
            threat_to_pos = _angle_between(threat_pos, pos)
            ang_diff = abs(_angle_diff(threat_to_obs, threat_to_pos))
            # Smaller angle = better centered (max contribution 0.3)
            centering = max(0.0, 1.0 - ang_diff / (math.pi / 4)) * 0.3
            score += centering

    if blocking_count == 0:
        return 0.0

    # Base score for having cover
    score += 0.4

    # Bonus for multiple blocking obstacles (up to 0.2)
    score += min(blocking_count - 1, 2) * 0.1

    # Distance factor: 10-40m is ideal
    d = distance(pos, threat_pos)
    if d < 5.0:
        score *= 0.5  # Too close
    elif d > 60.0:
        score *= 0.8  # A bit far

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Flanking
# ---------------------------------------------------------------------------

def compute_flank_position(
    target_pos: Vec2,
    target_facing: float,
    attacker_pos: Vec2,
    flank_distance: float = 20.0,
) -> Vec2:
    """Compute a position to the side/rear of a target.

    Picks the flank side (left or right of target's facing) that is
    closer to the attacker, then places a point *flank_distance* from
    the target at 90-135 degrees off the target's facing.
    """
    # Target's right and left perpendicular
    right_angle = target_facing - math.pi / 2
    left_angle = target_facing + math.pi / 2

    right_pos = (
        target_pos[0] + math.cos(right_angle) * flank_distance,
        target_pos[1] + math.sin(right_angle) * flank_distance,
    )
    left_pos = (
        target_pos[0] + math.cos(left_angle) * flank_distance,
        target_pos[1] + math.sin(left_angle) * flank_distance,
    )

    # Pick the closer flank
    if distance(attacker_pos, right_pos) < distance(attacker_pos, left_pos):
        # Bias toward rear: average of perpendicular and rear
        rear_angle = target_facing + math.pi
        avg_angle = (right_angle + rear_angle) / 2
        return (
            target_pos[0] + math.cos(avg_angle) * flank_distance,
            target_pos[1] + math.sin(avg_angle) * flank_distance,
        )
    else:
        rear_angle = target_facing + math.pi
        avg_angle = (left_angle + rear_angle) / 2
        return (
            target_pos[0] + math.cos(avg_angle) * flank_distance,
            target_pos[1] + math.sin(avg_angle) * flank_distance,
        )


def is_flanking(
    attacker_pos: Vec2,
    target_pos: Vec2,
    target_facing: float,
    angle_threshold: float = 90.0,
) -> bool:
    """Check if attacker is outside target's frontal arc.

    Returns True if the angle between the target's facing direction
    and the direction to the attacker exceeds *angle_threshold* degrees
    (i.e., the attacker is to the side or rear).
    """
    to_attacker = _angle_between(target_pos, attacker_pos)
    diff = abs(_angle_diff(target_facing, to_attacker))
    return math.degrees(diff) > angle_threshold


# ---------------------------------------------------------------------------
# Engagement decisions
# ---------------------------------------------------------------------------

def optimal_engagement_range(
    weapon_range: float,
    accuracy_falloff: float = 0.7,
) -> float:
    """Sweet spot distance for engaging.

    Returns *weapon_range* * *accuracy_falloff* — close enough for good
    accuracy, far enough that the enemy can't easily rush.
    """
    return weapon_range * accuracy_falloff


def should_engage(
    dist: float,
    health_ratio: float,
    ammo_ratio: float,
    in_cover: bool,
    num_allies_nearby: int,
) -> bool:
    """Decision function: fight or flight based on tactical situation.

    Computes a confidence score from multiple factors and returns True
    if the score exceeds 0.5.
    """
    score = 0.0

    # Health: 0-1 contributes 0-0.3
    score += health_ratio * 0.3

    # Ammo: need at least some
    if ammo_ratio < 0.1:
        return False
    score += ammo_ratio * 0.2

    # Cover bonus
    if in_cover:
        score += 0.2

    # Allies
    score += min(num_allies_nearby, 4) * 0.05

    # Distance penalty: too close or too far hurts
    if dist < 3.0:
        score -= 0.1  # danger close
    elif dist > 100.0:
        score -= 0.15  # too far for effective fire

    return score > 0.5


def should_retreat(
    health_ratio: float,
    ammo_ratio: float,
    enemies_visible: int,
    allies_nearby: int,
) -> bool:
    """Decision function: tactical retreat conditions.

    Returns True when the tactical situation is untenable.
    """
    # Critical health
    if health_ratio < 0.2:
        return True

    # Out of ammo
    if ammo_ratio < 0.05:
        return True

    # Badly outnumbered and hurt
    if enemies_visible > (allies_nearby + 1) * 2 and health_ratio < 0.5:
        return True

    # Outnumbered with low ammo
    if enemies_visible > allies_nearby + 1 and ammo_ratio < 0.2:
        return True

    return False


# ---------------------------------------------------------------------------
# Squad coordination
# ---------------------------------------------------------------------------

_FORMATION_OFFSETS: dict[str, list[tuple[float, float]]] = {
    # Offsets in leader-local space: (forward, lateral)
    # Negative forward = behind leader, positive lateral = left
    "wedge": [
        (-1.0, -1.0),
        (-1.0, 1.0),
        (-2.0, -2.0),
        (-2.0, 2.0),
        (-3.0, -3.0),
        (-3.0, 3.0),
        (-3.0, 0.0),
    ],
    "line": [
        (0.0, -1.0),
        (0.0, 1.0),
        (0.0, -2.0),
        (0.0, 2.0),
        (0.0, -3.0),
        (0.0, 3.0),
        (0.0, -4.0),
    ],
    "column": [
        (-1.0, 0.0),
        (-2.0, 0.0),
        (-3.0, 0.0),
        (-4.0, 0.0),
        (-5.0, 0.0),
        (-6.0, 0.0),
        (-7.0, 0.0),
    ],
    "diamond": [
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-2.0, 0.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (-3.0, 0.0),
    ],
    "echelon": [
        (-1.0, -1.0),
        (-2.0, -2.0),
        (-3.0, -3.0),
        (-4.0, -4.0),
        (-5.0, -5.0),
        (-6.0, -6.0),
        (-7.0, -7.0),
    ],
}


def formation_positions(
    leader_pos: Vec2,
    leader_heading: float,
    num_members: int,
    formation: str = "wedge",
    spacing: float = 5.0,
) -> list[Vec2]:
    """Compute positions for squad formation.

    Supported formations: wedge, line, column, diamond, echelon.
    Returns *num_members* world-space positions (excluding the leader).

    Args:
        leader_pos: Leader's world position.
        leader_heading: Leader's facing in radians (0 = +x, CCW positive).
        num_members: Number of squad members (excluding leader).
        formation: One of "wedge", "line", "column", "diamond", "echelon".
        spacing: Distance multiplier for formation spread.
    """
    offsets = _FORMATION_OFFSETS.get(formation, _FORMATION_OFFSETS["wedge"])

    cos_h = math.cos(leader_heading)
    sin_h = math.sin(leader_heading)

    positions: list[Vec2] = []
    for i in range(min(num_members, len(offsets))):
        fwd, lat = offsets[i]
        # Scale by spacing
        local_x = fwd * spacing
        local_y = lat * spacing
        # Rotate to world space
        wx = local_x * cos_h - local_y * sin_h
        wy = local_x * sin_h + local_y * cos_h
        positions.append((leader_pos[0] + wx, leader_pos[1] + wy))

    # If more members than predefined offsets, stack behind
    for i in range(len(offsets), num_members):
        row = (i - len(offsets)) + len(offsets) + 1
        local_x = -row * spacing
        wx = local_x * cos_h
        wy = local_x * sin_h
        positions.append((leader_pos[0] + wx, leader_pos[1] + wy))

    return positions


def assign_targets(
    squad_positions: list[Vec2],
    enemy_positions: list[Vec2],
) -> list[int]:
    """Optimal target assignment — minimize total engagement distance.

    Returns a list of enemy indices, one per squad member.  Uses a
    greedy nearest-enemy assignment.  If there are more squad members
    than enemies, multiple members may share a target.
    """
    if not enemy_positions:
        return [-1] * len(squad_positions)
    if not squad_positions:
        return []

    assignments: list[int] = []
    for sp in squad_positions:
        best_idx = 0
        best_dist = float("inf")
        for ei, ep in enumerate(enemy_positions):
            d = distance(sp, ep)
            if d < best_dist:
                best_dist = d
                best_idx = ei
        assignments.append(best_idx)

    return assignments


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def suppression_cone(
    shooter_pos: Vec2,
    target_pos: Vec2,
    cone_half_angle: float = 15.0,
    range_m: float = 50.0,
) -> list[Vec2]:
    """Area suppressed by fire from shooter toward target.

    Returns a list of 3 vertices forming a triangle (the suppression
    cone): the shooter position and two points at *range_m* distance
    at +/- *cone_half_angle* degrees from the line of fire.
    """
    base_angle = _angle_between(shooter_pos, target_pos)
    half_rad = math.radians(cone_half_angle)

    left = (
        shooter_pos[0] + math.cos(base_angle + half_rad) * range_m,
        shooter_pos[1] + math.sin(base_angle + half_rad) * range_m,
    )
    right = (
        shooter_pos[0] + math.cos(base_angle - half_rad) * range_m,
        shooter_pos[1] + math.sin(base_angle - half_rad) * range_m,
    )

    return [shooter_pos, left, right]


def _point_in_triangle(
    p: Vec2,
    a: Vec2,
    b: Vec2,
    c: Vec2,
) -> bool:
    """Barycentric point-in-triangle test."""
    def sign(p1: Vec2, p2: Vec2, p3: Vec2) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def is_suppressed(
    position: Vec2,
    suppression_zones: list[list[Vec2]],
) -> bool:
    """Check if position is within any suppression cone.

    Each zone in *suppression_zones* is a 3-vertex triangle as returned
    by :func:`suppression_cone`.
    """
    for zone in suppression_zones:
        if len(zone) >= 3:
            if _point_in_triangle(position, zone[0], zone[1], zone[2]):
                return True
    return False


# ---------------------------------------------------------------------------
# Combat behavior tree predicates and actions
# ---------------------------------------------------------------------------

def _has_enemies(ctx: dict) -> bool:
    return bool(ctx.get("enemies"))


def _has_ammo(ctx: dict) -> bool:
    return ctx.get("ammo_ratio", 1.0) > 0.05


def _is_healthy_combat(ctx: dict) -> bool:
    return ctx.get("health", 1.0) > 0.3


def _enemy_in_range(ctx: dict) -> bool:
    return bool(ctx.get("enemy_in_range"))


def _is_in_cover_ctx(ctx: dict) -> bool:
    return bool(ctx.get("in_cover"))


def _is_flanking_ctx(ctx: dict) -> bool:
    return bool(ctx.get("is_flanking"))


def _is_suppressed_ctx(ctx: dict) -> bool:
    return bool(ctx.get("is_suppressed"))


def _has_squad(ctx: dict) -> bool:
    return bool(ctx.get("squad_members"))


def _is_stalled(ctx: dict) -> bool:
    """True when unit has been engaging without progress."""
    return ctx.get("engage_duration", 0) > ctx.get("stall_threshold", 10.0)


def _should_retreat_ctx(ctx: dict) -> bool:
    return should_retreat(
        ctx.get("health", 1.0),
        ctx.get("ammo_ratio", 1.0),
        ctx.get("enemies_visible", 0),
        ctx.get("allies_nearby", 0),
    )


def _set_combat_decision(decision: str):
    """Return an action that sets ctx['decision'] and succeeds."""
    def _action(ctx: dict) -> Status:
        ctx["decision"] = decision
        return Status.SUCCESS
    _action.__name__ = f"decide_{decision}"
    return _action


# ---------------------------------------------------------------------------
# Pre-built combat behavior trees
# ---------------------------------------------------------------------------

def make_assault_tree() -> Node:
    """Aggressive: advance -> find cover -> engage -> flank if stalled -> push.

    Priority:
      1. Retreat if critically hurt / out of ammo
      2. Flank if stalled in engagement
      3. Engage from cover if enemy in range
      4. Seek cover if under fire
      5. Advance toward enemy
      6. Push (default aggressive advance)
    """
    return Selector([
        # 1. Retreat when situation is untenable
        Sequence([
            Condition(_should_retreat_ctx, "should_retreat"),
            Action(_set_combat_decision("retreat")),
        ]),
        # 2. Flank if stalled
        Sequence([
            Condition(_enemy_in_range, "enemy_in_range"),
            Condition(_is_stalled, "is_stalled"),
            Action(_set_combat_decision("flank")),
        ]),
        # 3. Engage from cover
        Sequence([
            Condition(_enemy_in_range, "enemy_in_range"),
            Condition(_is_in_cover_ctx, "in_cover"),
            Action(_set_combat_decision("engage")),
        ]),
        # 4. Seek cover if enemy visible and not in cover
        Sequence([
            Condition(_has_enemies, "has_enemies"),
            Inverter(Condition(_is_in_cover_ctx, "in_cover")),
            Action(_set_combat_decision("seek_cover")),
        ]),
        # 5. Advance toward enemy
        Sequence([
            Condition(_has_enemies, "has_enemies"),
            Action(_set_combat_decision("advance")),
        ]),
        # 6. Default: push forward
        Action(_set_combat_decision("push")),
    ])


def make_defender_tree() -> Node:
    """Defensive: hold position -> engage -> fall back if overwhelmed -> rally.

    Priority:
      1. Retreat if critically hurt
      2. Fall back if overwhelmed (outnumbered and hurt)
      3. Engage from current position if enemy in range
      4. Seek better cover if suppressed
      5. Hold position (default)
    """
    return Selector([
        # 1. Critical retreat
        Sequence([
            Condition(_should_retreat_ctx, "should_retreat"),
            Action(_set_combat_decision("retreat")),
        ]),
        # 2. Fall back if suppressed and not in cover
        Sequence([
            Condition(_is_suppressed_ctx, "is_suppressed"),
            Inverter(Condition(_is_in_cover_ctx, "in_cover")),
            Action(_set_combat_decision("fall_back")),
        ]),
        # 3. Engage if enemy in range
        Sequence([
            Condition(_enemy_in_range, "enemy_in_range"),
            Condition(_has_ammo, "has_ammo"),
            Action(_set_combat_decision("engage")),
        ]),
        # 4. Seek cover if enemies spotted but not in cover
        Sequence([
            Condition(_has_enemies, "has_enemies"),
            Inverter(Condition(_is_in_cover_ctx, "in_cover")),
            Action(_set_combat_decision("seek_cover")),
        ]),
        # 5. Hold position
        Action(_set_combat_decision("hold")),
    ])


def make_sniper_tree() -> Node:
    """Patient: find vantage -> wait -> engage high-value -> relocate after shots.

    Priority:
      1. Retreat if discovered and hurt
      2. Relocate after firing (cooldown-gated)
      3. Engage if enemy in range and in cover
      4. Find vantage point if no cover
      5. Wait (observe)
    """
    return Selector([
        # 1. Emergency retreat
        Sequence([
            Condition(_should_retreat_ctx, "should_retreat"),
            Action(_set_combat_decision("retreat")),
        ]),
        # 2. Relocate after shots (every 8 seconds)
        Sequence([
            Condition(lambda ctx: ctx.get("shots_fired", 0) > 0, "has_fired"),
            Cooldown(
                Action(_set_combat_decision("relocate")),
                seconds=8.0,
            ),
        ]),
        # 3. Engage from concealment
        Sequence([
            Condition(_enemy_in_range, "enemy_in_range"),
            Condition(_is_in_cover_ctx, "in_cover"),
            Condition(_has_ammo, "has_ammo"),
            Action(_set_combat_decision("engage")),
        ]),
        # 4. Find vantage
        Sequence([
            Inverter(Condition(_is_in_cover_ctx, "in_cover")),
            Action(_set_combat_decision("find_vantage")),
        ]),
        # 5. Wait and observe
        Action(_set_combat_decision("observe")),
    ])


def make_squad_leader_tree() -> Node:
    """Coordination: assess -> assign targets -> call formations -> manage retreat.

    Priority:
      1. Order retreat if squad situation is bad
      2. Call formation change if engaged
      3. Assign targets if enemies visible and have squad
      4. Advance squad toward objective
      5. Hold and assess (default)
    """
    return Selector([
        # 1. Order squad retreat
        Sequence([
            Condition(_should_retreat_ctx, "should_retreat"),
            Condition(_has_squad, "has_squad"),
            Action(_set_combat_decision("order_retreat")),
        ]),
        # 2. Formation change under fire
        Sequence([
            Condition(_has_enemies, "has_enemies"),
            Condition(_has_squad, "has_squad"),
            Condition(_is_suppressed_ctx, "is_suppressed"),
            Action(_set_combat_decision("change_formation")),
        ]),
        # 3. Assign targets
        Sequence([
            Condition(_has_enemies, "has_enemies"),
            Condition(_has_squad, "has_squad"),
            Action(_set_combat_decision("assign_targets")),
        ]),
        # 4. Advance squad
        Sequence([
            Condition(_has_squad, "has_squad"),
            Inverter(Condition(_has_enemies, "has_enemies")),
            Action(_set_combat_decision("advance_squad")),
        ]),
        # 5. Hold and assess
        Action(_set_combat_decision("assess")),
    ])
