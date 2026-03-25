# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Formation geometry and constants for tactical command.

Pure geometry/math utilities for formation positioning. The actual
registration with SimulationEngine is done by the SC integration layer.

Constants:
  VALID_FORMATIONS -- set of supported formation types
  VALID_ORDERS     -- set of supported squad orders
  RALLY_RADIUS     -- max distance for rally command
  SCATTER_MIN/MAX  -- scatter distance bounds

Functions:
  compute_formation_offsets(formation_type, n_units) -> list of (dx, dy)
  compute_scatter_positions(center, n_units) -> list of (x, y)
"""

from __future__ import annotations

import math
import random


# Valid formation types
VALID_FORMATIONS = {"wedge", "line", "column", "circle"}

# Valid squad orders
VALID_ORDERS = {"advance", "hold", "flank_left", "flank_right", "retreat"}

# Rally radius in meters
RALLY_RADIUS = 30.0

# Scatter distance bounds
SCATTER_MIN_DISTANCE = 8.0
SCATTER_MAX_DISTANCE = 15.0

# Formation spacing
FORMATION_SPACING = 5.0


def compute_formation_offsets(
    formation_type: str,
    n_units: int,
    spacing: float = FORMATION_SPACING,
) -> list[tuple[float, float]]:
    """Compute (dx, dy) offsets for each unit in a formation.

    Args:
        formation_type: One of "wedge", "line", "column", "circle".
        n_units: Number of units in the formation.
        spacing: Distance between units.

    Returns:
        List of (dx, dy) tuples, one per unit. First unit is the leader.
    """
    if n_units <= 0:
        return []
    if n_units == 1:
        return [(0.0, 0.0)]

    offsets: list[tuple[float, float]] = []

    if formation_type == "line":
        # Horizontal line centered on leader
        total_width = (n_units - 1) * spacing
        start_x = -total_width / 2
        for i in range(n_units):
            offsets.append((start_x + i * spacing, 0.0))

    elif formation_type == "column":
        # Vertical column, leader at front
        for i in range(n_units):
            offsets.append((0.0, -i * spacing))

    elif formation_type == "wedge":
        # V-shape, leader at point
        offsets.append((0.0, 0.0))
        for i in range(1, n_units):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets.append((side * row * spacing * 0.7, -row * spacing))

    elif formation_type == "circle":
        # Circle around center
        for i in range(n_units):
            angle = 2.0 * math.pi * i / n_units
            radius = spacing * max(1, n_units / (2 * math.pi))
            offsets.append((math.cos(angle) * radius, math.sin(angle) * radius))

    else:
        # Default: column
        for i in range(n_units):
            offsets.append((0.0, -i * spacing))

    return offsets


def compute_scatter_positions(
    center: tuple[float, float],
    n_units: int,
    min_dist: float = SCATTER_MIN_DISTANCE,
    max_dist: float = SCATTER_MAX_DISTANCE,
) -> list[tuple[float, float]]:
    """Compute scatter positions for units moving away from center.

    Each unit gets a unique direction away from center with jitter.

    Args:
        center: (x, y) center point.
        n_units: Number of units.
        min_dist: Minimum distance from center.
        max_dist: Maximum distance from center.

    Returns:
        List of (x, y) positions.
    """
    positions: list[tuple[float, float]] = []
    cx, cy = center

    for i in range(n_units):
        angle = (2.0 * math.pi * i) / max(n_units, 1)
        angle += random.uniform(-0.3, 0.3)
        dist = random.uniform(min_dist, max_dist)
        positions.append((cx + math.cos(angle) * dist, cy + math.sin(angle) * dist))

    return positions


def is_within_rally_radius(
    point: tuple[float, float],
    rally_point: tuple[float, float],
    radius: float = RALLY_RADIUS,
) -> bool:
    """Check if a point is within rally radius of the rally point."""
    dx = point[0] - rally_point[0]
    dy = point[1] - rally_point[1]
    return math.hypot(dx, dy) <= radius
