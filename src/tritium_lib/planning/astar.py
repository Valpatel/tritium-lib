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

# ``strategy="auto"`` switches to the hierarchical (coarse-to-fine) planner at
# or above this many grid cells.  Flat A* is optimal and cheap below it; above
# it a single flat solve blows the expansion cap, so the corridor planner keeps
# global planning in bounded time.  250k cells ≈ a 500² grid — comfortably above
# every simulator map (≤ ~201²/40k cells), so ``auto`` is byte-for-byte flat A*
# for all existing callers and only engages hierarchical on large real AOs.
_AUTO_HIERARCHICAL_MIN_CELLS = 250_000

# Cap the shortcut lookahead in :func:`_smooth_path` to this many waypoints
# ahead of the current index.  Bounds the smoother at O(n * w^2) instead of
# O(n^3) on very long paths.  Chosen large enough that ordinary routes (well
# under this many waypoints) smooth identically to the uncapped version.
_SMOOTH_WINDOW = 40

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
        clearance_relaxed: True when a ``clearance_m > 0`` request could not be
            satisfied and the planner fell back to a clearance-0 route so the
            dispatch still succeeds (see :func:`plan_route`).  Always False for
            clearance-0 requests and for clearance requests that were met.
        strategy: Which planner actually solved the route — ``"flat"`` (the
            8-connected A* below) or ``"hierarchical"`` (the coarse-to-fine
            corridor planner engaged for large AOs).  Reported to operator
            telemetry so the UI can show which planner ran and how hard it
            worked (``expansions``).  A hierarchical request that degenerates to
            flat on a tiny map honestly reports ``"flat"``.
    """

    success: bool
    path: list[tuple[float, float]] = field(default_factory=list)
    cost: float = 0.0
    expansions: int = 0
    reason: str = "ok"
    clearance_relaxed: bool = False
    strategy: str = "flat"


def _octile(c0: int, r0: int, c1: int, r1: int) -> float:
    """Octile distance in cells between two grid cells."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    return (dc + dr) + (_SQRT2 - 2.0) * min(dc, dr)


def _cell_blocked(costmap, col: int, row: int, clearance_m: float) -> bool:
    """True if ``(col, row)`` is impassable for a unit needing ``clearance_m``.

    A cell is blocked when it is lethal (or out of bounds) OR — only when
    ``clearance_m > 0`` — when its distance to the nearest lethal cell is below
    the requested standoff.  At ``clearance_m == 0.0`` this is EXACTLY
    :meth:`Costmap.is_lethal` (the clearance branch short-circuits and the
    distance field is never even computed), so clearance-0 planning is
    byte-for-byte today's behavior.
    """
    if costmap.is_lethal(col, row):
        return True
    if clearance_m > 0.0 and costmap.clearance_m(col, row) < clearance_m:
        return True
    return False


def _snap_to_free(
    costmap, col: int, row: int, radius_cells: int, clearance_m: float = 0.0
) -> tuple[int, int] | None:
    """Ring-search for the nearest passable cell within ``radius_cells``.

    "Passable" respects ``clearance_m`` (see :func:`_cell_blocked`): with a
    positive clearance a sub-clearance cell is not a valid snap target.
    Returns the original cell if already passable, else the closest passable
    cell, or ``None`` if none within radius.
    """
    if not _cell_blocked(costmap, col, row, clearance_m):
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
                if costmap.in_bounds(nc, nr) and not _cell_blocked(
                    costmap, nc, nr, clearance_m
                ):
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
    clearance_m: float = 0.0,
    strategy: str = "auto",
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
        clearance_m: Unit-radius standoff (meters).  When ``> 0`` every
            non-lethal cell whose distance to the nearest lethal cell
            (:meth:`Costmap.clearance_m`) is below this value is treated as
            blocked — for snapping, A* traversal AND the smoothing pass — so a
            wide unit (an APC) keeps more wall clearance than a person.  A value
            of ``0.0`` (the default) is EXACTLY the historical behavior (the
            distance field is never computed), which golden replays depend on.
        strategy: ``"flat"`` forces the flat 8-connected A* below (the classic,
            optimal baseline).  ``"hierarchical"`` forces the coarse-to-fine
            corridor planner (:func:`~tritium_lib.planning.hierarchical.plan_route_hierarchical`),
            which keeps global planning in bounded time on large AOs.  ``"auto"``
            (the default) uses flat A* for maps under
            :data:`_AUTO_HIERARCHICAL_MIN_CELLS` cells and hierarchical at or
            above it — so every simulator-scale map is byte-for-byte flat A* and
            only large real AOs pay for the hierarchy.

    Graceful fallback:
        A positive ``clearance_m`` never makes a previously-routable dispatch
        fail.  If the clearance-constrained plan fails for any reason, the
        planner retries ONCE with ``clearance_m = 0``; a successful retry is
        returned with :attr:`RouteResult.clearance_relaxed` set True.

    Returns:
        A :class:`RouteResult`.
    """
    if strategy == "hierarchical" or (
        strategy == "auto"
        and costmap.width * costmap.height >= _AUTO_HIERARCHICAL_MIN_CELLS
    ):
        # Lazy import breaks the astar <-> hierarchical import cycle; the
        # hierarchical planner calls back into this function with strategy="flat".
        from .hierarchical import plan_route_hierarchical

        return plan_route_hierarchical(
            costmap, start, goal, smooth=smooth, max_expansions=max_expansions,
            snap_radius_m=snap_radius_m, clearance_m=clearance_m,
        )

    res = costmap.resolution
    if snap_radius_m is None:
        snap_radius_m = 3.0 * res
    snap_cells = max(0, int(math.floor(snap_radius_m / res + 1e-9)))

    result = _plan_once(
        costmap, start, goal, smooth, max_expansions, snap_cells, clearance_m
    )
    if clearance_m > 0.0 and not result.success:
        # Standoff could not be honored — never fail a routable dispatch.
        relaxed = _plan_once(
            costmap, start, goal, smooth, max_expansions, snap_cells, 0.0
        )
        if relaxed.success:
            relaxed.clearance_relaxed = True
            return relaxed
    return result


def _plan_once(
    costmap,
    start: tuple[float, float],
    goal: tuple[float, float],
    smooth: bool,
    max_expansions: int | None,
    snap_cells: int,
    clearance_m: float,
) -> RouteResult:
    """One A* solve at a fixed clearance (no fallback).  See :func:`plan_route`."""
    res = costmap.resolution

    start_cell = costmap.world_to_grid(start[0], start[1])
    if start_cell is None:
        return RouteResult(False, reason="start_out_of_bounds")
    goal_cell = costmap.world_to_grid(goal[0], goal[1])
    if goal_cell is None:
        return RouteResult(False, reason="goal_out_of_bounds")

    sc = _snap_to_free(costmap, start_cell[0], start_cell[1], snap_cells, clearance_m)
    if sc is None:
        return RouteResult(False, reason="start_blocked")
    gc = _snap_to_free(costmap, goal_cell[0], goal_cell[1], snap_cells, clearance_m)
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
            if _cell_blocked(costmap, nc, nr, clearance_m):
                continue
            if diag:
                # Forbid corner-cutting: both shared orthogonal cells passable.
                if _cell_blocked(costmap, col + dc, row, clearance_m) or _cell_blocked(
                    costmap, col, row + dr, clearance_m
                ):
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
        world_path = _smooth_path(costmap, world_path, clearance_m)

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


def _crosses_blocked(costmap, p, q, clearance_m: float) -> bool:
    """True if segment ``p -> q`` touches a cell blocked at ``clearance_m``.

    Corner-safe supercover.  At ``clearance_m == 0.0`` this is identical to
    :func:`_crosses_lethal` (the clearance test short-circuits), so smoothing
    is unchanged for clearance-0 routes.
    """
    for c, r in _supercover_cells(costmap, p, q, include_corner_cells=True):
        if _cell_blocked(costmap, c, r, clearance_m):
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
    costmap, path: list[tuple[float, float]], clearance_m: float = 0.0
) -> list[tuple[float, float]]:
    """Greedy cost-aware shortcutting.

    From waypoint ``i`` take the farthest ``j`` where the straight segment
    ``i -> j`` (supercover over cells) crosses only passable cells AND the
    max cell cost on that segment does not exceed the max cell cost along the
    original subpath ``i..j``.  This preserves road preference — a shortcut
    across expensive grass can never replace a cheap road detour.

    ``clearance_m`` extends "passable" to reject a shortcut that cuts through a
    sub-clearance cell, so the smoothed route keeps the same unit-radius
    standoff as the A* path.  At ``clearance_m == 0.0`` this is the historical
    lethal-only smoother.

    ``j`` is capped to :data:`_SMOOTH_WINDOW` waypoints ahead of ``i`` so the
    worst case stays O(n * w^2) rather than O(n^3); ordinary routes (far
    shorter than the window) are unaffected and remain deterministic.
    """
    n = len(path)
    if n <= 2:
        return list(path)

    result = [path[0]]
    i = 0
    while i < n - 1:
        best_j = i + 1
        hi = min(n - 1, i + _SMOOTH_WINDOW)
        for j in range(hi, i + 1, -1):
            if _crosses_blocked(costmap, path[i], path[j], clearance_m):
                continue
            seg_max = _cost_max(costmap, path[i], path[j])
            # Max cost along the original polyline i..j.
            orig_max = 0.0
            orig_lethal = False
            for k in range(i, j):
                if _crosses_blocked(costmap, path[k], path[k + 1], clearance_m):
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
