# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Hierarchical (coarse-to-fine) global planner for large AOs.

The open baseline :func:`~tritium_lib.planning.astar.plan_route` is a flat
8-connected A*.  It is optimal and cheap on the small maps the simulator uses
(≈200² cells) but does not scale: on a city-scale AO (600²+ cells, >9 km² at
5 m resolution) a single flat solve exceeds the 200k-expansion cap and returns
no path.  This module keeps global planning in **bounded time** on those maps
with a standard **coarse-to-fine corridor** strategy — plain hierarchical grid
A*, the open baseline only (the advanced flow-field planner lives privately and
is neither referenced nor reimplemented here):

    1. **Coarsen** the fine costmap by an integer ``factor`` (default 8).  Each
       coarse cell aggregates a ``factor``×``factor`` block of fine cells: the
       soft traversal cost is the **mean** of the block's non-lethal cells, and
       the block is lethal only when **every** fine cell in it is lethal
       (optimistic aggregation — a coarse cell stays open if any sliver of it
       is passable, so sub-``factor``-width corridors like a single road are
       not sealed at the coarse level).  A mild congestion term nudges the
       coarse plan toward genuinely open ground.

    2. **Plan coarse first.**  A flat A* over the coarse costmap is tiny
       (a 600²/8 = 75² grid) and finishes in well under the expansion cap.

    3. **Refine within a corridor.**  Dilate the coarse path into a band, map
       it back to fine cells, and run the SAME flat A* restricted to that
       corridor (every out-of-corridor cell is masked lethal).  Fine expansions
       are therefore bounded by the corridor size — a thin band around the
       coarse route — not the whole grid.  Obstacle avoidance, clearance and
       road/ped precedence are byte-for-byte the flat planner's, because the
       fine solve runs the real costmap's costs and real obstacle-distance
       clearance field inside the corridor.

    4. **Completeness backstop.**  If the corridor is too tight to contain a
       fine path the band is widened and retried; only if every widening fails
       does the planner fall back to a full flat solve.  So hierarchical never
       reports ``no_path`` where flat would have found one — worst case it
       degrades to flat.

Pure stdlib — no third-party dependencies, matching the rest of this package.
Deterministic: every step reuses the deterministic flat A* over deterministic
inputs, so replays are stable.
"""

from __future__ import annotations

import math

from .astar import RouteResult, plan_route
from .costmap import Costmap

__all__ = [
    "coarsen_costmap",
    "plan_route_hierarchical",
    "DEFAULT_COARSE_FACTOR",
    "DEFAULT_CORRIDOR_RADIUS_CELLS",
]

# Downsample factor: a coarse cell spans FACTOR×FACTOR fine cells.  8 keeps the
# coarse grid ~64× smaller (a 600² map → 75²) so the coarse solve is trivial,
# while a single road (≈1-2 fine cells wide) still survives coarsening thanks to
# the optimistic lethal rule below.
DEFAULT_COARSE_FACTOR = 8

# Corridor half-width around the coarse path, in COARSE cells.  3 coarse cells
# at factor 8 = 24 fine cells of margin on each side of the coarse route —
# generous enough to contain the true fine-optimal path in realistic city maps
# while keeping the fine solve a thin band.
DEFAULT_CORRIDOR_RADIUS_CELLS = 3

# Corridor-widening ladder (multipliers on the base radius) tried in order
# before the full-flat-A* completeness backstop.
_WIDEN_LADDER = (1, 2, 4)


# ---------------------------------------------------------------------------
# Coarsening
# ---------------------------------------------------------------------------

def coarsen_costmap(costmap: Costmap, factor: int) -> Costmap:
    """Downsample ``costmap`` by ``factor`` into a coarser :class:`Costmap`.

    Each coarse cell ``(cc, cr)`` aggregates the fine block
    ``rows [cr*factor : (cr+1)*factor)`` × ``cols [cc*factor : (cc+1)*factor)``:

        - **soft cost** = mean of the block's non-lethal fine costs, plus a mild
          congestion term ``* (1 + lethal_fraction)`` so the coarse plan gently
          prefers open blocks over half-blocked ones.
        - **lethal** iff *every* in-bounds fine cell in the block is lethal
          (optimistic: a block with any passable sliver stays traversable so
          narrow corridors are not sealed at the coarse level).

    The coarse grid shares the fine grid's south-west ``origin`` and frame;
    ``resolution`` scales by ``factor``.  A trailing partial block (when the
    fine dimension is not a multiple of ``factor``) is aggregated over whatever
    fine cells it covers.  Returns a plain :class:`Costmap` so the existing flat
    :func:`plan_route` runs over it unchanged.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        # Identity: copy the grid so callers never alias the fine grid.
        return Costmap(
            origin_x=costmap.origin_x,
            origin_y=costmap.origin_y,
            resolution=costmap.resolution,
            width=costmap.width,
            height=costmap.height,
            grid=[list(row) for row in costmap.grid],
        )

    lethal = costmap.LETHAL
    fw, fh = costmap.width, costmap.height
    cw = max(1, (fw + factor - 1) // factor)
    ch = max(1, (fh + factor - 1) // factor)
    fine = costmap.grid

    coarse_grid: list[list[float]] = [[lethal] * cw for _ in range(ch)]
    for cr in range(ch):
        r0 = cr * factor
        r1 = min(r0 + factor, fh)
        for cc in range(cw):
            c0 = cc * factor
            c1 = min(c0 + factor, fw)
            total = 0
            n_lethal = 0
            cost_sum = 0.0
            for r in range(r0, r1):
                frow = fine[r]
                for c in range(c0, c1):
                    total += 1
                    v = frow[c]
                    if v == lethal:
                        n_lethal += 1
                    else:
                        cost_sum += v
            if total == 0 or n_lethal == total:
                coarse_grid[cr][cc] = lethal
                continue
            n_open = total - n_lethal
            mean_open = cost_sum / n_open
            lethal_frac = n_lethal / total
            coarse_grid[cr][cc] = mean_open * (1.0 + lethal_frac)

    return Costmap(
        origin_x=costmap.origin_x,
        origin_y=costmap.origin_y,
        resolution=costmap.resolution * factor,
        width=cw,
        height=ch,
        grid=coarse_grid,
    )


def _cached_coarse(costmap: Costmap, factor: int) -> Costmap:
    """Coarsen ``costmap`` at ``factor``, caching the result on the instance.

    The fine costmap is immutable after :meth:`CostmapBuilder.build`, so the
    coarse map is safe to memoise on the instance (same pattern as the clearance
    field cache).  Production dispatches many routes over one version-cached
    costmap, so this avoids re-coarsening a 600² grid per plan.
    """
    cache = getattr(costmap, "_coarse_cache", None)
    if cache is None:
        cache = {}
        try:
            costmap._coarse_cache = cache  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - exotic costmap that rejects attrs
            return coarsen_costmap(costmap, factor)
    coarse = cache.get(factor)
    if coarse is None:
        coarse = coarsen_costmap(costmap, factor)
        cache[factor] = coarse
    return coarse


# ---------------------------------------------------------------------------
# Corridor mask
# ---------------------------------------------------------------------------

class _CorridorCostmap:
    """A read-only view of ``base`` masking every cell outside ``corridor`` lethal.

    Delegates the full :class:`Costmap` query surface the flat planner uses to
    ``base`` — coordinate conversion, bounds, ``min_traversable_cost`` and,
    crucially, ``clearance_m`` (so unit-radius standoff reflects REAL obstacle
    distances, never the artificial corridor boundary).  Only ``cost_at`` /
    ``is_lethal`` are overridden: an in-corridor cell keeps the base cost; any
    other in-bounds cell reads as ``LETHAL`` so A* never leaves the band.  A
    catch-all ``__getattr__`` forwards anything else to ``base``.
    """

    def __init__(self, base: Costmap, corridor: set[tuple[int, int]]) -> None:
        self._base = base
        self._corridor = corridor
        self.LETHAL = base.LETHAL
        # Mirror the plain attributes A* reads directly.
        self.origin_x = base.origin_x
        self.origin_y = base.origin_y
        self.resolution = base.resolution
        self.width = base.width
        self.height = base.height

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._base, name)

    def in_bounds(self, col: int, row: int) -> bool:
        return self._base.in_bounds(col, row)

    def world_to_grid(self, x: float, y: float):
        return self._base.world_to_grid(x, y)

    def grid_to_world(self, col: int, row: int):
        return self._base.grid_to_world(col, row)

    def bounds(self):
        return self._base.bounds()

    def min_traversable_cost(self) -> float:
        return self._base.min_traversable_cost()

    def clearance_m(self, col: int, row: int) -> float:
        # Real obstacle clearance — the corridor boundary is not a wall.
        return self._base.clearance_m(col, row)

    def cost_at(self, col: int, row: int) -> float:
        if (col, row) in self._corridor:
            return self._base.cost_at(col, row)
        return self.LETHAL

    def is_lethal(self, col: int, row: int) -> bool:
        return self.cost_at(col, row) == self.LETHAL


def _fine_corridor(
    coarse_cells: list[tuple[int, int]],
    factor: int,
    radius_coarse: int,
    fw: int,
    fh: int,
    extra_fine: list[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """Fine-cell corridor: dilate the coarse path, expand each cell to its block.

    ``coarse_cells`` are the ``(col, row)`` coarse cells the coarse route passes
    through.  Each is dilated by ``radius_coarse`` in coarse space (Chebyshev),
    then every resulting coarse cell is expanded to its ``factor``×``factor``
    fine block, clipped to the fine grid ``[0, fw) × [0, fh)``.  ``extra_fine``
    cells (e.g. the snapped start/goal neighbourhood) are added directly so the
    endpoints are always inside the band.
    """
    dilated: set[tuple[int, int]] = set()
    for (cc, cr) in coarse_cells:
        for dr in range(-radius_coarse, radius_coarse + 1):
            for dc in range(-radius_coarse, radius_coarse + 1):
                dilated.add((cc + dc, cr + dr))

    corridor: set[tuple[int, int]] = set()
    for (cc, cr) in dilated:
        c0 = cc * factor
        r0 = cr * factor
        for r in range(max(0, r0), min(r0 + factor, fh)):
            for c in range(max(0, c0), min(c0 + factor, fw)):
                corridor.add((c, r))

    if extra_fine:
        for (c, r) in extra_fine:
            if 0 <= c < fw and 0 <= r < fh:
                corridor.add((c, r))
    return corridor


def _neighbourhood(col: int, row: int, radius: int, fw: int, fh: int):
    """Fine cells within a Chebyshev ``radius`` of ``(col, row)``, clipped."""
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            c, r = col + dc, row + dr
            if 0 <= c < fw and 0 <= r < fh:
                out.append((c, r))
    return out


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------

def plan_route_hierarchical(
    costmap: Costmap,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    smooth: bool = True,
    coarse_factor: int = DEFAULT_COARSE_FACTOR,
    corridor_radius_cells: int = DEFAULT_CORRIDOR_RADIUS_CELLS,
    max_expansions: int | None = None,
    snap_radius_m: float | None = None,
    clearance_m: float = 0.0,
) -> RouteResult:
    """Coarse-to-fine A* over ``costmap`` — bounded-time global planning.

    Drop-in for :func:`~tritium_lib.planning.astar.plan_route` with the same
    signature plus ``coarse_factor`` / ``corridor_radius_cells``.  Returns a
    :class:`~tritium_lib.planning.astar.RouteResult` whose ``expansions`` is the
    TOTAL node work (coarse solve + fine solve) so it is directly comparable to
    the flat planner's expansion count.

    Guarantees:
        - **Bounded time** on large AOs: the fine solve is restricted to a thin
          corridor around the coarse path, so expansions scale with the
          corridor size, not the whole grid.
        - **Obstacle avoidance / clearance / road precedence preserved**: the
          fine solve runs the real costmap costs and real obstacle-distance
          clearance inside the corridor — identical semantics to flat A*.
        - **Completeness**: falls back (widen corridor, then full flat A*) so it
          never reports ``no_path`` where flat A* would have found one.

    Small or degenerate maps (fewer than ``2*coarse_factor`` cells on a side)
    coarsen to nothing useful and are routed by the flat planner directly.
    """
    fw, fh = costmap.width, costmap.height

    # Degenerate to flat when coarsening cannot buy anything.
    if coarse_factor < 2 or fw < 2 * coarse_factor or fh < 2 * coarse_factor:
        return plan_route(
            costmap, start, goal, smooth=smooth, max_expansions=max_expansions,
            snap_radius_m=snap_radius_m, clearance_m=clearance_m, strategy="flat",
        )

    coarse = _cached_coarse(costmap, coarse_factor)

    # 1. Coarse plan (unsmoothed so we recover the cell-by-cell trail).  No
    # clearance at the coarse level — clearance is enforced by the fine solve;
    # the corridor width already gives a wide unit room.  A generous snap radius
    # (a couple of coarse cells) lets a start/goal buried in a coarse obstacle
    # block seed the corridor from a nearby open coarse cell.
    coarse_snap = 2.0 * coarse.resolution
    coarse_res = plan_route(
        coarse, start, goal, smooth=False, max_expansions=None,
        snap_radius_m=coarse_snap, clearance_m=0.0, strategy="flat",
    )
    if not coarse_res.success:
        # Coarse (optimistic) says disconnected -> fine is almost certainly
        # disconnected too, but honour completeness with a bounded flat solve.
        flat = plan_route(
            costmap, start, goal, smooth=smooth, max_expansions=max_expansions,
            snap_radius_m=snap_radius_m, clearance_m=clearance_m, strategy="flat",
        )
        flat.expansions += coarse_res.expansions
        return flat

    # Coarse route cells (unsmoothed path == coarse cell centres) + explicit
    # start/goal coarse cells so the band always spans both endpoints.
    coarse_cells: list[tuple[int, int]] = []
    for (x, y) in coarse_res.path:
        cell = coarse.world_to_grid(x, y)
        if cell is not None:
            coarse_cells.append(cell)
    for pt in (start, goal):
        cell = coarse.world_to_grid(pt[0], pt[1])
        if cell is not None:
            coarse_cells.append(cell)

    # Fine start/goal neighbourhood guaranteed inside the corridor so snapping
    # to a free fine cell never lands outside the band.
    sc = costmap.world_to_grid(start[0], start[1])
    gc = costmap.world_to_grid(goal[0], goal[1])
    if snap_radius_m is None:
        snap_cells = int(math.floor(3.0 * costmap.resolution / costmap.resolution))
    else:
        snap_cells = max(0, int(math.floor(snap_radius_m / costmap.resolution)))
    snap_cells = max(snap_cells, 2)
    extra_fine: list[tuple[int, int]] = []
    if sc is not None:
        extra_fine += _neighbourhood(sc[0], sc[1], snap_cells, fw, fh)
    if gc is not None:
        extra_fine += _neighbourhood(gc[0], gc[1], snap_cells, fw, fh)

    # 2. Refine within the corridor, widening on failure.
    total_coarse_exp = coarse_res.expansions
    for mult in _WIDEN_LADDER:
        radius = corridor_radius_cells * mult
        corridor = _fine_corridor(
            coarse_cells, coarse_factor, radius, fw, fh, extra_fine
        )
        masked = _CorridorCostmap(costmap, corridor)
        fine_cap = max_expansions
        if fine_cap is None:
            fine_cap = min(200_000, max(2000, len(corridor) * 2))
        fine = plan_route(
            masked, start, goal, smooth=smooth, max_expansions=fine_cap,
            snap_radius_m=snap_radius_m, clearance_m=clearance_m, strategy="flat",
        )
        if fine.success:
            fine.expansions += total_coarse_exp
            return fine
        total_coarse_exp += fine.expansions

    # 3. Completeness backstop: the corridor could not contain a fine path at
    # any width -> full flat solve (may itself hit the cap on a huge disconnected
    # map, but never reports no_path where flat would have succeeded).
    flat = plan_route(
        costmap, start, goal, smooth=smooth, max_expansions=max_expansions,
        snap_radius_m=snap_radius_m, clearance_m=clearance_m, strategy="flat",
    )
    flat.expansions += total_coarse_exp
    return flat
