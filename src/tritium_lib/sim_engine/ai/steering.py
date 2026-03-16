"""Craig Reynolds' steering behaviors as pure Python functions.

Each function takes position/velocity vectors and returns a steering force vector.
All vectors are Vec2 = tuple[float, float] in meters. Functions are self-contained
and composable — combine forces by adding them together.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import random

Vec2 = tuple[float, float]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def distance(a: Vec2, b: Vec2) -> float:
    """Euclidean distance between two points."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.hypot(dx, dy)


def magnitude(v: Vec2) -> float:
    """Length of a vector."""
    return math.hypot(v[0], v[1])


def normalize(vector: Vec2) -> Vec2:
    """Return unit vector. Returns (0, 0) for zero-length input."""
    m = magnitude(vector)
    if m < 1e-12:
        return (0.0, 0.0)
    return (vector[0] / m, vector[1] / m)


def truncate(vector: Vec2, max_length: float) -> Vec2:
    """Clamp vector length to *max_length*, preserving direction."""
    m = magnitude(vector)
    if m <= max_length or m < 1e-12:
        return vector
    scale = max_length / m
    return (vector[0] * scale, vector[1] * scale)


def heading_to_vec(heading_rad: float) -> Vec2:
    """Convert heading angle (radians, 0 = +x, CCW positive) to unit vector."""
    return (math.cos(heading_rad), math.sin(heading_rad))


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _scale(v: Vec2, s: float) -> Vec2:
    return (v[0] * s, v[1] * s)


# ---------------------------------------------------------------------------
# Basic behaviors
# ---------------------------------------------------------------------------

def seek(position: Vec2, target: Vec2, max_speed: float) -> Vec2:
    """Steer toward *target* at full speed.

    Returns a desired-velocity vector pointing from *position* to *target*
    with magnitude *max_speed*. If already at the target, returns (0, 0).
    """
    desired = _sub(target, position)
    d = magnitude(desired)
    if d < 1e-12:
        return (0.0, 0.0)
    return _scale(normalize(desired), max_speed)


def flee(position: Vec2, threat: Vec2, max_speed: float) -> Vec2:
    """Steer directly away from *threat* at full speed."""
    desired = _sub(position, threat)
    d = magnitude(desired)
    if d < 1e-12:
        return (0.0, 0.0)
    return _scale(normalize(desired), max_speed)


def arrive(position: Vec2, target: Vec2, max_speed: float, slow_radius: float) -> Vec2:
    """Seek *target* but decelerate within *slow_radius*.

    Outside *slow_radius* behaves like seek. Inside, speed ramps linearly
    from *max_speed* at the edge down to 0 at the target.
    """
    to_target = _sub(target, position)
    d = magnitude(to_target)
    if d < 1e-12:
        return (0.0, 0.0)
    speed = max_speed if d >= slow_radius else max_speed * (d / slow_radius)
    return _scale(normalize(to_target), speed)


def wander(
    position: Vec2,
    velocity: Vec2,
    wander_radius: float,
    wander_distance: float,
    jitter: float,
) -> Vec2:
    """Produce a gentle, random-looking meandering force.

    Projects a circle of *wander_radius* a distance *wander_distance* ahead
    of the agent, then picks a random point on that circle perturbed by
    *jitter*. Returns a steering force toward that point.
    """
    heading = normalize(velocity) if magnitude(velocity) > 1e-12 else (1.0, 0.0)
    circle_center = _add(position, _scale(heading, wander_distance))
    angle = random.uniform(0, 2 * math.pi)
    jitter_vec = (
        math.cos(angle) * wander_radius + random.uniform(-jitter, jitter),
        math.sin(angle) * wander_radius + random.uniform(-jitter, jitter),
    )
    wander_target = _add(circle_center, jitter_vec)
    return seek(position, wander_target, magnitude(velocity) if magnitude(velocity) > 1e-12 else 1.0)


def pursue(
    position: Vec2,
    velocity: Vec2,
    target_pos: Vec2,
    target_vel: Vec2,
    max_speed: float,
) -> Vec2:
    """Intercept a moving target by seeking its predicted future position."""
    to_target = _sub(target_pos, position)
    d = magnitude(to_target)
    speed = magnitude(velocity)
    look_ahead = d / max_speed if max_speed > 1e-12 else 0.0
    future_pos = _add(target_pos, _scale(target_vel, look_ahead))
    return seek(position, future_pos, max_speed)


def evade(
    position: Vec2,
    velocity: Vec2,
    threat_pos: Vec2,
    threat_vel: Vec2,
    max_speed: float,
) -> Vec2:
    """Flee from a moving threat's predicted future position."""
    to_threat = _sub(threat_pos, position)
    d = magnitude(to_threat)
    look_ahead = d / max_speed if max_speed > 1e-12 else 0.0
    future_pos = _add(threat_pos, _scale(threat_vel, look_ahead))
    return flee(position, future_pos, max_speed)


# ---------------------------------------------------------------------------
# Path following
# ---------------------------------------------------------------------------

def follow_path(
    position: Vec2,
    velocity: Vec2,
    path: list[Vec2],
    path_radius: float,
    max_speed: float,
) -> Vec2:
    """Follow a polyline path.

    Seeks the nearest waypoint that is ahead of the agent (beyond
    *path_radius* of the current closest segment point). When close to
    the final waypoint, uses *arrive* to decelerate.
    """
    if not path:
        return (0.0, 0.0)

    # Find closest waypoint
    min_dist = float("inf")
    closest_idx = 0
    for i, wp in enumerate(path):
        d = distance(position, wp)
        if d < min_dist:
            min_dist = d
            closest_idx = i

    # Advance to next waypoint if within path_radius
    target_idx = closest_idx
    while target_idx < len(path) - 1 and distance(position, path[target_idx]) < path_radius:
        target_idx += 1

    target_wp = path[target_idx]

    # Arrive at final waypoint, seek intermediate ones
    if target_idx == len(path) - 1:
        return arrive(position, target_wp, max_speed, path_radius * 2)
    return seek(position, target_wp, max_speed)


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------

def avoid_obstacles(
    position: Vec2,
    velocity: Vec2,
    obstacles: list[tuple[Vec2, float]],
    detection_range: float,
) -> Vec2:
    """Steer away from nearby circular obstacles.

    *obstacles* is a list of ((cx, cy), radius). Only obstacles within
    *detection_range* ahead of the agent are considered. Returns a lateral
    force to avoid the nearest threatening obstacle.
    """
    speed = magnitude(velocity)
    if speed < 1e-12:
        return (0.0, 0.0)

    heading = normalize(velocity)
    # Perpendicular (left-hand normal)
    perp = (-heading[1], heading[0])

    steer = (0.0, 0.0)
    nearest_dist = float("inf")

    for (ox, oy), radius in obstacles:
        local_x = (ox - position[0]) * heading[0] + (oy - position[1]) * heading[1]
        local_y = (ox - position[0]) * perp[0] + (oy - position[1]) * perp[1]

        # Behind agent or too far ahead
        if local_x < 0 or local_x > detection_range:
            continue

        # Check lateral clearance
        expanded = radius + 0.5  # agent half-width approximation
        if abs(local_y) > expanded:
            continue

        if local_x < nearest_dist:
            nearest_dist = local_x
            # Steer laterally away from obstacle center
            lateral_force = 1.0 if local_y >= 0 else -1.0
            # Scale inversely with distance
            strength = (detection_range - local_x) / detection_range
            steer = _scale(perp, lateral_force * strength * speed)

    return steer


# ---------------------------------------------------------------------------
# Group behaviors
# ---------------------------------------------------------------------------

def separate(position: Vec2, neighbors: list[Vec2], desired_separation: float) -> Vec2:
    """Steer away from neighbors that are too close."""
    force = (0.0, 0.0)
    count = 0
    for n in neighbors:
        d = distance(position, n)
        if 1e-12 < d < desired_separation:
            diff = normalize(_sub(position, n))
            # Weight inversely by distance
            diff = _scale(diff, 1.0 / d)
            force = _add(force, diff)
            count += 1
    if count > 0:
        force = _scale(force, 1.0 / count)
    return force


def align(velocity: Vec2, neighbor_velocities: list[Vec2]) -> Vec2:
    """Steer toward the average heading of neighbors."""
    if not neighbor_velocities:
        return (0.0, 0.0)
    avg = (0.0, 0.0)
    for nv in neighbor_velocities:
        avg = _add(avg, nv)
    avg = _scale(avg, 1.0 / len(neighbor_velocities))
    return avg


def cohere(position: Vec2, neighbors: list[Vec2]) -> Vec2:
    """Steer toward the centroid of neighbors."""
    if not neighbors:
        return (0.0, 0.0)
    cx = sum(n[0] for n in neighbors) / len(neighbors)
    cy = sum(n[1] for n in neighbors) / len(neighbors)
    return _sub((cx, cy), position)


def flock(
    position: Vec2,
    velocity: Vec2,
    neighbors: list[tuple[Vec2, Vec2]],
    separation_dist: float,
    max_speed: float,
) -> Vec2:
    """Combined flocking: separation + alignment + cohesion.

    *neighbors* is a list of (position, velocity) pairs for nearby agents.
    Weights: separation 1.5, alignment 1.0, cohesion 1.0.
    """
    if not neighbors:
        return (0.0, 0.0)

    n_positions = [n[0] for n in neighbors]
    n_velocities = [n[1] for n in neighbors]

    sep = separate(position, n_positions, separation_dist)
    ali = align(velocity, n_velocities)
    coh = cohere(position, n_positions)

    # Weighted combination
    result = _add(_scale(sep, 1.5), _add(ali, coh))
    return truncate(result, max_speed)


# ---------------------------------------------------------------------------
# Formation
# ---------------------------------------------------------------------------

def formation_offset(leader_pos: Vec2, leader_heading: float, offset: Vec2) -> Vec2:
    """Compute world-space position for a formation slot.

    *offset* is relative to the leader's local frame: positive x = forward,
    positive y = left. Returns the world-space target position.
    """
    cos_h = math.cos(leader_heading)
    sin_h = math.sin(leader_heading)
    # Rotate offset by leader heading
    wx = offset[0] * cos_h - offset[1] * sin_h
    wy = offset[0] * sin_h + offset[1] * cos_h
    return (leader_pos[0] + wx, leader_pos[1] + wy)
