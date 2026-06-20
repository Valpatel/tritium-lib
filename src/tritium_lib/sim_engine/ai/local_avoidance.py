# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reusable inter-agent local avoidance (separation) — the REACTIVE layer of
individual-unit movement.

Two-layer movement stack (see also world/pathfinding.py for the DELIBERATIVE
layer — day-routes on the OSM street/sidewalk graph, and ai/city_sim.py for
daily schedules):

    deliberative  : plan_path() -> a route of waypoints toward a goal
    reactive      : follow the route, BUT push apart from crowders (this module)
                    and avoid buildings (entity swept-collision)

This keeps units from overlapping/stacking so a dense crowd JOSTLES instead of
ghosting through itself.  Built on steering.separate() + a caller-supplied
neighbour query, so it is engine-agnostic and efficient: back the query with a
spatial grid and the whole pass is O(n*k) (k = local crowders), never O(n^2).
"""
from __future__ import annotations

from typing import Callable

from .steering import separate, magnitude

Vec2 = tuple[float, float]


def separation_deltas(
    agents: list[tuple[str, Vec2]],
    neighbors_fn: Callable[[Vec2, float], list[Vec2]],
    desired_separation: float = 1.6,
    strength: float = 0.5,
    max_push: float = 0.35,
) -> dict[str, Vec2]:
    """Bounded per-agent position nudges that push overlapping agents apart.

    Args:
        agents: ``(id, position)`` for each unit that should avoid its peers.
        neighbors_fn: ``(position, radius) -> [neighbour positions]`` within
            radius.  May include the agent's OWN position — coincident points
            are ignored by ``separate`` (distance ~0 is skipped), so a lone
            agent never self-repels.
        desired_separation: personal-space radius in meters; only neighbours
            closer than this exert a push.
        strength: scales the raw steering force into a per-tick push (a gentle
            ramp — bigger overlap pushes harder, up to ``max_push``).
        max_push: hard cap on |delta| in meters per tick (no teleporting).

    Returns:
        ``{id: (dx, dy)}`` for agents that are crowded; uncrowded agents are
        omitted (a true no-op so the caller skips them).
    """
    if max_push <= 0.0:
        return {}
    out: dict[str, Vec2] = {}
    for aid, pos in agents:
        neighbors = neighbors_fn(pos, desired_separation)
        if not neighbors:
            continue
        fx, fy = separate(pos, neighbors, desired_separation)
        if fx == 0.0 and fy == 0.0:
            continue
        # Scale to a gentle push, then hard-cap to max_push (inline truncate to
        # avoid teleports while keeping the small-overlap ramp).
        px, py = fx * strength, fy * strength
        mag = magnitude((px, py))
        if mag <= 1e-9:
            continue
        if mag > max_push:
            scale = max_push / mag
            px, py = px * scale, py * scale
        out[aid] = (px, py)
    return out
