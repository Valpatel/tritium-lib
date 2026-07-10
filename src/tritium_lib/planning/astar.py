# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Open baseline A* global planner over a :class:`Costmap`.

This is the **open-source baseline** route planner.  An advanced flow-field
planner exists privately elsewhere; this module deliberately implements only
a clean, deterministic 8-connected A* so any fleet unit can route on a cost
grid with zero extra dependencies.

Cost convention:
    - Move cost ``a -> b`` = ``step_len_m * 0.5 * (cost_a + cost_b)`` where
      ``step_len_m`` is ``resolution`` (orthogonal) or ``sqrt(2)*resolution``
      (diagonal).  Costs are therefore in **meters-scaled** units, matching
      the octile heuristic which is also in meters.
    - Heuristic: octile distance (meters) * ``min_traversable_cost`` —
      admissible because no cell is cheaper than ``min_traversable_cost``.
    - Diagonal corner-cutting is forbidden: a diagonal move is illegal if
      either shared orthogonal neighbor is lethal.
    - Deterministic tie-breaking via an ``itertools.count`` counter and a
      fixed neighbor order.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

__all__ = ["RouteResult", "plan_route"]

_SQRT2 = math.sqrt(2.0)

# Fixed neighbor order: (dcol, drow, is_diagonal).  Orthogonals first.
_NEIGHBORS = [
    (1, 0, False),
    (-1, 0, False),
    (0, 1, False),
    (0, -1, False),
    (1, 1, True),
    (1, -1, True),
    (-1, 1, True),
    (-1, -1, True),
]


@dataclass
class RouteResult:
    """Result of a route request.

    Attributes:
        success: Whether a route was found.
        path: World-coordinate waypoints.  On success the first point is the
            exact requested start and the last is the exact requested goal.
        cost: Optimal grid cost of the route (meters-scaled), 0.0 on failure.
        expansions: Number of nodes expanded by A*.
        reason: One of ``ok``, ``start_out_of_bounds``, ``goal_out_of_bounds``,
            ``start_blocked``, ``goal_blocked``, ``no_path``, ``max_expansions``.
    """

    success: bool
    path: list[tuple[float, float]] = field(default_factory=list)
    cost: float = 0.0
    expansions: int = 0
    reason: str = "ok"


def _octile(c0: int, r0: int, c1: int, r1: int) -> float:
    """Octile distance in cells between two grid cells."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    return (dc + dr) + (_SQRT2 - 2.0) * min(dc, dr)


def _snap_to_free(
    costmap, col: int, row: int, radius_cells: int
) -> tuple[int, int] | None:
    """Ring-search for the nearest non-lethal cell within ``radius_cells``.

    Returns the original cell if already free, else the closest free cell,
    or ``None`` if none within radius.
    """
    if not costmap.is_lethal(col, row):
        return (col, row)
    best: tuple[int, int] | None = None
    best_d = math.inf
    for radius in range(1, radius_cells + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                # Only the ring at Chebyshev distance == radius.
                if max(abs(dr), abs(dc)) != radius:
                    continue
                nc, nr = col + dc, row + dr
                if costmap.in_bounds(nc, nr) and not costmap.is_lethal(nc, nr):
                    d = math.hypot(dc, dr)
                    if d < best_d:
                        best_d = d
                        best = (nc, nr)
        if best is not None:
            return best
    return None


def plan_route(
    costmap,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    smooth: bool = True,
    max_expansions: int | None = None,
    snap_radius_m: float | None = None,
) -> RouteResult:
    """Plan a route from ``start`` to ``goal`` over ``costmap``.

    Args:
        costmap: A :class:`~tritium_lib.planning.costmap.Costmap`.
        start: World ``(x, y)`` start.
        goal: World ``(x, y)`` goal.
        smooth: Apply cost-aware shortcut smoothing to the result.
        max_expansions: Node-expansion cap.  Defaults to
            ``min(200_000, width * height * 4)``.
        snap_radius_m: If start/goal falls in a lethal cell, search for the
            nearest free cell within this radius.  Defaults to
            ``3 * resolution``.

    Returns:
        A :class:`RouteResult`.
    """
    res = costmap.resolution
    if snap_radius_m is None:
        snap_radius_m = 3.0 * res
    snap_cells = max(0, int(math.floor(snap_radius_m / res + 1e-9)))

    start_cell = costmap.world_to_grid(start[0], start[1])
    if start_cell is None:
        return RouteResult(False, reason="start_out_of_bounds")
    goal_cell = costmap.world_to_grid(goal[0], goal[1])
    if goal_cell is None:
        return RouteResult(False, reason="goal_out_of_bounds")

    sc = _snap_to_free(costmap, start_cell[0], start_cell[1], snap_cells)
    if sc is None:
        return RouteResult(False, reason="start_blocked")
    gc = _snap_to_free(costmap, goal_cell[0], goal_cell[1], snap_cells)
    if gc is None:
        return RouteResult(False, reason="goal_blocked")

    if max_expansions is None:
        max_expansions = min(200_000, costmap.width * costmap.height * 4)

    min_cost = costmap.min_traversable_cost()

    # -- A* ----------------------------------------------------------------
    counter = itertools.count()
    goal_c, goal_r = gc
    start_node = sc

    def h(col: int, row: int) -> float:
        return _octile(col, row, goal_c, goal_r) * res * min_cost

    open_heap: list[tuple[float, float, int, tuple[int, int]]] = []
    g_score: dict[tuple[int, int], float] = {start_node: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heapq.heappush(open_heap, (h(*start_node), 0.0, next(counter), start_node))
    closed: set[tuple[int, int]] = set()

    expansions = 0
    found = False
    while open_heap:
        _f, g, _cnt, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == gc:
            found = True
            break
        closed.add(node)
        expansions += 1
        if expansions > max_expansions:
            return RouteResult(False, expansions=expansions, reason="max_expansions")

        col, row = node
        cost_a = costmap.cost_at(col, row)
        for dc, dr, diag in _NEIGHBORS:
            nc, nr = col + dc, row + dr
            if not costmap.in_bounds(nc, nr):
                continue
            if costmap.is_lethal(nc, nr):
                continue
            if diag:
                # Forbid corner-cutting: both shared orthogonal cells free.
                if costmap.is_lethal(col + dc, row) or costmap.is_lethal(col, row + dr):
                    continue
            neighbor = (nc, nr)
            if neighbor in closed:
                continue
            cost_b = costmap.cost_at(nc, nr)
            step_len = _SQRT2 * res if diag else res
            move_cost = step_len * 0.5 * (cost_a + cost_b)
            tentative = g + move_cost
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = node
                heapq.heappush(
                    open_heap,
                    (tentative + h(nc, nr), tentative, next(counter), neighbor),
                )

    if not found:
        return RouteResult(False, expansions=expansions, reason="no_path")

    # -- Reconstruct -------------------------------------------------------
    grid_path: list[tuple[int, int]] = [gc]
    cur = gc
    while cur in came_from:
        cur = came_from[cur]
        grid_path.append(cur)
    grid_path.reverse()

    cost = g_score[gc]

    world_path = [costmap.grid_to_world(c, r) for (c, r) in grid_path]
    if len(world_path) == 1:
        world_path = [start, goal]
    else:
        world_path[0] = start
        world_path[-1] = goal

    if smooth:
        world_path = _smooth_path(costmap, world_path)

    return RouteResult(
        success=True,
        path=world_path,
        cost=cost,
        expansions=expansions,
        reason="ok",
    )


# ---------------------------------------------------------------------------
# Supercover traversal + cost-aware smoothing
# ---------------------------------------------------------------------------

def _supercover_cells(
    costmap,
    p: tuple[float, float],
    q: tuple[float, float],
    include_corner_cells: bool = True,
) -> list[tuple[int, int]]:
    """Grid cells the world segment ``p -> q`` passes through.

    Amanatides & Woo grid traversal in continuous cell coordinates.  When
    ``include_corner_cells`` is True (the default, used for lethal-crossing
    safety), an exact grid-corner crossing also emits both shared orthogonal
    cells so no lethal corner cell is ever skipped — a true supercover.  When
    False (used for cost comparison), only the cells the segment's interior
    actually traverses are emitted, so a diagonal that merely clips a corner
    does not pick up the cost of the neighbouring cell.
    """
    res = costmap.resolution
    x0 = (p[0] - costmap.origin_x) / res
    y0 = (p[1] - costmap.origin_y) / res
    x1 = (q[0] - costmap.origin_x) / res
    y1 = (q[1] - costmap.origin_y) / res

    cx = int(math.floor(x0))
    cy = int(math.floor(y0))
    ex = int(math.floor(x1))
    ey = int(math.floor(y1))

    cells = [(cx, cy)]
    dx = x1 - x0
    dy = y1 - y0
    if cx == ex and cy == ey:
        return cells

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    if dx != 0:
        next_bx = cx + (1 if step_x > 0 else 0)
        t_max_x = (next_bx - x0) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if dy != 0:
        next_by = cy + (1 if step_y > 0 else 0)
        t_max_y = (next_by - y0) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = math.inf
        t_delta_y = math.inf

    # ``t`` is parametrised so ``t == 1`` is the segment endpoint; never step
    # past it (would emit spurious — possibly out-of-bounds — cells).
    guard = 4 * (abs(ex - cx) + abs(ey - cy)) + 8
    while (cx, cy) != (ex, ey) and guard > 0:
        guard -= 1
        if abs(t_max_x - t_max_y) < 1e-12:
            if t_max_x > 1.0 - 1e-12:
                break  # corner is at/after the endpoint — stop.
            # Mid-segment corner — include both orthogonal cells so a lethal
            # pinch is never skipped, then step diagonally.
            if include_corner_cells:
                cells.append((cx + step_x, cy))
                cells.append((cx, cy + step_y))
            t_max_x += t_delta_x
            t_max_y += t_delta_y
            cx += step_x
            cy += step_y
        elif t_max_x < t_max_y:
            if t_max_x > 1.0 + 1e-12:
                break
            t_max_x += t_delta_x
            cx += step_x
        else:
            if t_max_y > 1.0 + 1e-12:
                break
            t_max_y += t_delta_y
            cy += step_y
        cells.append((cx, cy))

    # Guarantee the endpoint cell is present (it holds the segment's end).
    if cells[-1] != (ex, ey):
        cells.append((ex, ey))
    return cells


def _crosses_lethal(costmap, p, q) -> bool:
    """True if the segment ``p -> q`` touches any lethal cell (corner-safe)."""
    for c, r in _supercover_cells(costmap, p, q, include_corner_cells=True):
        if costmap.is_lethal(c, r):
            return True
    return False


def _cost_max(costmap, p, q) -> float:
    """Max non-lethal cell cost along the interior of segment ``p -> q``.

    Uses corner-exclusive traversal so a diagonal that merely clips a cell
    corner does not inherit that neighbour's cost — this keeps road
    preference intact when comparing a shortcut against the original path.
    """
    max_cost = 0.0
    for c, r in _supercover_cells(costmap, p, q, include_corner_cells=False):
        if not costmap.is_lethal(c, r):
            v = costmap.cost_at(c, r)
            if v > max_cost:
                max_cost = v
    return max_cost


def _smooth_path(
    costmap, path: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Greedy cost-aware shortcutting.

    From waypoint ``i`` take the farthest ``j`` where the straight segment
    ``i -> j`` (supercover over cells) crosses only non-lethal cells AND the
    max cell cost on that segment does not exceed the max cell cost along the
    original subpath ``i..j``.  This preserves road preference — a shortcut
    across expensive grass can never replace a cheap road detour.
    """
    n = len(path)
    if n <= 2:
        return list(path)

    result = [path[0]]
    i = 0
    while i < n - 1:
        best_j = i + 1
        for j in range(n - 1, i + 1, -1):
            if _crosses_lethal(costmap, path[i], path[j]):
                continue
            seg_max = _cost_max(costmap, path[i], path[j])
            # Max cost along the original polyline i..j.
            orig_max = 0.0
            orig_lethal = False
            for k in range(i, j):
                if _crosses_lethal(costmap, path[k], path[k + 1]):
                    orig_lethal = True
                    break
                m = _cost_max(costmap, path[k], path[k + 1])
                if m > orig_max:
                    orig_max = m
            if orig_lethal:
                # Original itself touches lethal (e.g. snapped endpoint) — do
                # not reason about cost; fall back to the trivial next step.
                continue
            if seg_max <= orig_max + 1e-9:
                best_j = j
                break
        result.append(path[best_j])
        i = best_j
    return result
