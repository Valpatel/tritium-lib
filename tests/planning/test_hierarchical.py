# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.hierarchical — coarse-to-fine A*.

Proves the scaling fix: hierarchical planning stays correct within tolerance of
flat A* on small maps, preserves obstacle avoidance / clearance / road
precedence, and plans large AOs (600²+) in bounded time where a single flat A*
solve blows the expansion cap and returns no path.
"""

import math
import random
import time

import pytest

from tritium_lib.planning import (
    Costmap,
    CostmapBuilder,
    CostmapWeights,
    coarsen_costmap,
    plan_route,
    plan_route_hierarchical,
)
from tritium_lib.planning.astar import (
    _AUTO_HIERARCHICAL_MIN_CELLS,
    _supercover_cells,
)
from tritium_lib.planning.hierarchical import (
    DEFAULT_COARSE_FACTOR,
    _cached_coarse,
)

LETHAL = float("inf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon_fc(rings, props=None):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props or {},
             "geometry": {"type": "Polygon", "coordinates": [r]}}
            for r in rings
        ],
    }


def _line_fc(coords, props=None):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props or {},
             "geometry": {"type": "LineString", "coordinates": coords}},
        ],
    }


def _path_touches_lethal(cm, path):
    for a, b in zip(path, path[1:]):
        for c, r in _supercover_cells(cm, a, b, include_corner_cells=True):
            if cm.is_lethal(c, r):
                return True
    return False


def _path_min_clearance(cm, path):
    """Minimum obstacle clearance (m) over all cells the path traverses."""
    best = math.inf
    for a, b in zip(path, path[1:]):
        for c, r in _supercover_cells(cm, a, b, include_corner_cells=False):
            if cm.in_bounds(c, r) and not cm.is_lethal(c, r):
                best = min(best, cm.clearance_m(c, r))
    return best


def _y_at_x(path, xq):
    """Y-values where the polyline crosses the vertical line ``x == xq``."""
    ys = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if x0 == x1:
            continue
        if (x0 - xq) * (x1 - xq) <= 0:
            t = (xq - x0) / (x1 - x0)
            ys.append(y0 + t * (y1 - y0))
    return ys


def _soft_costmap(n, seed=7, resolution=5.0):
    """A continuous soft-cost field (like slope/weather): every cell traversable,
    costs vary ~1..6.  This is the flat-A* scaling killer — the octile heuristic
    is a weak lower bound so flat A* expands almost the whole reachable area."""
    rnd = random.Random(seed)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            v = 1.0 + 2.5 * (1 + math.sin(c / 40.0) * math.cos(r / 40.0)) + rnd.random() * 0.5
            row.append(v)
        grid.append(row)
    return Costmap(origin_x=0.0, origin_y=0.0, resolution=resolution,
                   width=n, height=n, grid=grid)


def _building_grid(n, resolution=5.0, block=70.0, pitch=120.0):
    """A city-block clutter map: a lattice of lethal building blocks with road
    gaps between them.  Traversable throughout the gaps."""
    span = n * resolution
    b = CostmapBuilder((0, 0, span, span), resolution=resolution)
    rings = []
    x = 60.0
    while x < span - 60:
        y = 60.0
        while y < span - 60:
            rings.append([(x, y), (x + block, y),
                          (x + block, y + block), (x, y + block)])
            y += pitch
        x += pitch
    b.add_obstacles(_polygon_fc(rings))
    return b.build()


# ---------------------------------------------------------------------------
# Coarsening
# ---------------------------------------------------------------------------

class TestCoarsen:
    def test_dimensions_and_frame(self):
        cm = CostmapBuilder((0, 0, 800, 800), resolution=5.0).build()  # 160x160
        co = coarsen_costmap(cm, 8)
        assert co.width == 20 and co.height == 20
        assert co.resolution == 40.0
        assert co.origin_x == cm.origin_x and co.origin_y == cm.origin_y

    def test_partial_trailing_block(self):
        # 25 fine cells / factor 8 -> ceil = 4 coarse cells (last block covers 1).
        cm = CostmapBuilder((0, 0, 250, 250), resolution=10.0).build()  # 25x25
        co = coarsen_costmap(cm, 8)
        assert co.width == 4 and co.height == 4

    def test_optimistic_lethal_preserves_narrow_gap(self):
        # A wall one coarse-block thick (fine cols 32..39 = coarse col 4) with a
        # 1-fine-cell-tall gap at row 30.  Optimistic coarsening must NOT seal
        # the gap: the coarse block holding the open gap row stays traversable,
        # while a coarse block wholly inside the wall is lethal.
        n = 64
        grid = [[1.0] * n for _ in range(n)]
        gap_row = 30
        for r in range(n):
            if r == gap_row:
                continue
            for c in range(32, 40):  # coarse col 4
                grid[r][c] = LETHAL
        cm = Costmap(origin_x=0, origin_y=0, resolution=5.0,
                     width=n, height=n, grid=grid)
        co = coarsen_costmap(cm, 8)
        # coarse col 4 (fine 32..39); coarse row 3 (fine 24..31) holds gap row 30.
        assert not co.is_lethal(4, 3), "narrow gap must survive coarsening"
        # coarse row 0 (fine 0..7) is wholly inside the wall -> lethal.
        assert co.is_lethal(4, 0)

    def test_fully_lethal_block_is_lethal(self):
        n = 16
        grid = [[LETHAL] * n for _ in range(n)]
        cm = Costmap(origin_x=0, origin_y=0, resolution=5.0,
                     width=n, height=n, grid=grid)
        co = coarsen_costmap(cm, 8)
        assert all(co.is_lethal(c, r) for r in range(co.height)
                   for c in range(co.width))

    def test_congestion_term_raises_half_blocked_block(self):
        # A block with a lethal half costs more than an all-open block of the
        # same open-cost, so the coarse plan prefers open ground.
        n = 16
        open_grid = [[2.0] * n for _ in range(n)]
        cm_open = Costmap(origin_x=0, origin_y=0, resolution=5.0,
                          width=n, height=n, grid=open_grid)
        half_grid = [[2.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if (c % 8) < 4:  # left half of each 8-cell coarse block lethal
                    half_grid[r][c] = LETHAL
        cm_half = Costmap(origin_x=0, origin_y=0, resolution=5.0,
                          width=n, height=n, grid=half_grid)
        co_open = coarsen_costmap(cm_open, 8)
        co_half = coarsen_costmap(cm_half, 8)
        assert co_open.cost_at(0, 0) == pytest.approx(2.0)
        # left coarse block is half lethal -> mean 2.0 * (1 + 0.5) = 3.0
        assert co_half.cost_at(0, 0) == pytest.approx(3.0)

    def test_factor_one_is_identity_copy(self):
        cm = _building_grid(40)
        co = coarsen_costmap(cm, 1)
        assert co.width == cm.width and co.height == cm.height
        assert co.grid == cm.grid
        assert co.grid is not cm.grid  # a copy, not an alias


# ---------------------------------------------------------------------------
# Correctness vs flat A*
# ---------------------------------------------------------------------------

class TestCorrectnessVsFlat:
    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_soft_field_cost_matches_flat_within_tolerance(self, seed):
        n = 200
        cm = _soft_costmap(n, seed=seed)
        span = n * cm.resolution
        s, g = (10, 10), (span - 10, span - 10)
        flat = plan_route(cm, s, g, strategy="flat")
        hier = plan_route_hierarchical(cm, s, g)
        assert flat.success and hier.success
        # Corridor-restricted A* is never cheaper than the global optimum, and
        # here the corridor contains it -> equal within float noise.
        assert hier.cost >= flat.cost - 1e-6
        assert hier.cost <= flat.cost * 1.15, (hier.cost, flat.cost)

    def test_endpoints_exact(self):
        cm = _building_grid(200)
        span = cm.width * cm.resolution
        s, g = (20.0, 20.0), (span - 20.0, span - 20.0)
        hier = plan_route_hierarchical(cm, s, g)
        assert hier.success
        assert hier.path[0] == s
        assert hier.path[-1] == g

    def test_building_clutter_cost_within_tolerance(self):
        cm = _building_grid(200)
        span = cm.width * cm.resolution
        s, g = (20.0, 20.0), (span - 20.0, span - 20.0)
        flat = plan_route(cm, s, g, strategy="flat")
        hier = plan_route_hierarchical(cm, s, g)
        assert flat.success and hier.success
        assert hier.cost >= flat.cost - 1e-6
        assert hier.cost <= flat.cost * 1.15, (hier.cost, flat.cost)


# ---------------------------------------------------------------------------
# Obstacle avoidance preserved
# ---------------------------------------------------------------------------

class TestObstacleAvoidance:
    def test_hier_path_never_crosses_lethal_soft_or_hard(self):
        cm = _building_grid(240)
        span = cm.width * cm.resolution
        s, g = (20.0, 20.0), (span - 20.0, span - 20.0)
        hier = plan_route_hierarchical(cm, s, g)
        assert hier.success
        assert not _path_touches_lethal(cm, hier.path)

    def test_routes_around_wall_with_gap(self):
        # A tall wall with a single gap: the route must detour through the gap,
        # not cross the wall.
        n = 240
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        wx = span * 0.5
        gap_lo, gap_hi = span * 0.45, span * 0.55
        b.add_obstacles(_polygon_fc([
            [(wx - 10, 40), (wx + 10, 40), (wx + 10, gap_lo), (wx - 10, gap_lo)],
            [(wx - 10, gap_hi), (wx + 10, gap_hi),
             (wx + 10, span - 40), (wx - 10, span - 40)],
        ]))
        cm = b.build()
        hier = plan_route_hierarchical(cm, (40, span * 0.5), (span - 40, span * 0.5))
        assert hier.success
        assert not _path_touches_lethal(cm, hier.path)
        # Where the route crosses the wall x-line it must be inside the gap band
        # (robust to smoothing, which need not place a vertex exactly at wx).
        ys = _y_at_x(hier.path, wx)
        assert ys, "route should cross the wall x-line"
        assert all(gap_lo - 6 <= y <= gap_hi + 6 for y in ys), ys


# ---------------------------------------------------------------------------
# Clearance preserved through the corridor
# ---------------------------------------------------------------------------

class TestClearancePreserved:
    def test_clearance_standoff_maintained(self):
        # A gap wide enough for a person but a wide unit must keep standoff.
        n = 160
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        wx = span * 0.5
        gap = 60.0
        b.add_obstacles(_polygon_fc([
            [(wx - 10, 40), (wx + 10, 40),
             (wx + 10, span / 2 - gap / 2), (wx - 10, span / 2 - gap / 2)],
            [(wx - 10, span / 2 + gap / 2), (wx + 10, span / 2 + gap / 2),
             (wx + 10, span - 40), (wx - 10, span - 40)],
        ]))
        cm = b.build()
        s, g = (40, span * 0.5), (span - 40, span * 0.5)
        clr = 12.0
        hier = plan_route_hierarchical(cm, s, g, clearance_m=clr)
        assert hier.success
        assert not _path_touches_lethal(cm, hier.path)
        # If clearance was honoured (not relaxed), every traversed cell keeps at
        # least ~clr-1cell standoff from lethal.
        if not hier.clearance_relaxed:
            assert _path_min_clearance(cm, hier.path) >= clr - cm.resolution

    def test_clearance_relaxed_never_fails_routable_dispatch(self):
        # A gap narrower than the requested clearance: the planner must still
        # return a route (relaxed) rather than failing the dispatch.
        n = 120
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        wx = span * 0.5
        gap = 14.0
        b.add_obstacles(_polygon_fc([
            [(wx - 10, 20), (wx + 10, 20),
             (wx + 10, span / 2 - gap / 2), (wx - 10, span / 2 - gap / 2)],
            [(wx - 10, span / 2 + gap / 2), (wx + 10, span / 2 + gap / 2),
             (wx + 10, span - 20), (wx - 10, span - 20)],
        ]))
        cm = b.build()
        s, g = (30, span * 0.5), (span - 30, span * 0.5)
        hier = plan_route_hierarchical(cm, s, g, clearance_m=30.0)
        assert hier.success
        assert not _path_touches_lethal(cm, hier.path)


# ---------------------------------------------------------------------------
# Scaling — the whole point
# ---------------------------------------------------------------------------

class TestScaling:
    def test_flat_fails_hierarchical_succeeds_at_600(self):
        """600² soft-cost AO: flat A* blows the 200k-expansion cap and returns
        no path; hierarchical plans it in bounded time and expansions."""
        n = 600
        cm = _soft_costmap(n, seed=7)
        span = n * cm.resolution
        s, g = (10, 10), (span - 10, span - 10)

        flat = plan_route(cm, s, g, strategy="flat")
        assert not flat.success
        assert flat.reason == "max_expansions"

        t0 = time.perf_counter()
        hier = plan_route(cm, s, g, strategy="hierarchical")
        elapsed = time.perf_counter() - t0
        assert hier.success, hier.reason
        assert not _path_touches_lethal(cm, hier.path)
        # Bounded: total (coarse+fine) expansions well under the flat cap, and a
        # wall-clock budget comfortably above the observed ~0.6s with headroom.
        assert hier.expansions < 150_000, hier.expansions
        assert elapsed < 8.0, elapsed
        # Endpoints exact.
        assert hier.path[0] == s and hier.path[-1] == g

    def test_expansions_grow_sublinearly_vs_flat(self):
        # At 400² flat still succeeds but hierarchical expands far fewer nodes —
        # the corridor bound, not the whole grid.
        n = 400
        cm = _soft_costmap(n, seed=3)
        span = n * cm.resolution
        s, g = (10, 10), (span - 10, span - 10)
        flat = plan_route(cm, s, g, strategy="flat")
        hier = plan_route(cm, s, g, strategy="hierarchical")
        assert flat.success and hier.success
        assert hier.expansions < flat.expansions * 0.6, (hier.expansions, flat.expansions)
        assert hier.cost <= flat.cost * 1.10


# ---------------------------------------------------------------------------
# Auto strategy dispatch
# ---------------------------------------------------------------------------

class TestAutoStrategy:
    def test_small_map_auto_is_byte_identical_to_flat(self):
        cm = _building_grid(200)  # 40k cells, below threshold
        assert cm.width * cm.height < _AUTO_HIERARCHICAL_MIN_CELLS
        span = cm.width * cm.resolution
        s, g = (20.0, 20.0), (span - 20.0, span - 20.0)
        auto = plan_route(cm, s, g)                      # strategy="auto"
        flat = plan_route(cm, s, g, strategy="flat")
        assert auto.path == flat.path
        assert auto.cost == flat.cost
        assert auto.expansions == flat.expansions

    def test_large_map_auto_engages_hierarchical(self):
        n = 600  # 360k cells, above threshold
        cm = _soft_costmap(n, seed=7)
        assert cm.width * cm.height >= _AUTO_HIERARCHICAL_MIN_CELLS
        span = n * cm.resolution
        s, g = (10, 10), (span - 10, span - 10)
        auto = plan_route(cm, s, g)                      # auto -> hierarchical
        hier = plan_route(cm, s, g, strategy="hierarchical")
        # Same deterministic result as forcing hierarchical.
        assert auto.success and hier.success
        assert auto.cost == pytest.approx(hier.cost)
        assert auto.expansions == hier.expansions


# ---------------------------------------------------------------------------
# Completeness backstop
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_disconnected_map_fails_like_flat(self):
        # A wall sealing the map completely: both planners must report failure,
        # not a bogus route.
        n = 120
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        wx = span * 0.5
        b.add_obstacles(_polygon_fc([
            [(wx - 15, 0), (wx + 15, 0), (wx + 15, span), (wx - 15, span)],
        ]))
        cm = b.build()
        s, g = (30, span * 0.5), (span - 30, span * 0.5)
        hier = plan_route_hierarchical(cm, s, g)
        flat = plan_route(cm, s, g, strategy="flat")
        assert not flat.success
        assert not hier.success

    def test_narrow_gap_still_routable(self):
        # A wall with a single narrow gap: hierarchical must find it (the coarse
        # map keeps the gap open, the corridor contains it).
        n = 200
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        wx = span * 0.5
        gap = 20.0
        b.add_obstacles(_polygon_fc([
            [(wx - 10, 20), (wx + 10, 20),
             (wx + 10, span / 2 - gap / 2), (wx - 10, span / 2 - gap / 2)],
            [(wx - 10, span / 2 + gap / 2), (wx + 10, span / 2 + gap / 2),
             (wx + 10, span - 20), (wx - 10, span - 20)],
        ]))
        cm = b.build()
        s, g = (30, span * 0.5), (span - 30, span * 0.5)
        hier = plan_route_hierarchical(cm, s, g)
        assert hier.success
        assert not _path_touches_lethal(cm, hier.path)


# ---------------------------------------------------------------------------
# Coarse-cache invalidation under a storm re-plan (GIS-version costmap rebuild)
# ---------------------------------------------------------------------------

class TestCoarseCacheInvalidation:
    """Pin the coarse-map cache lifecycle so a storm re-plan never serves stale
    coarse geometry.

    The hierarchical planner memoises the coarsened costmap on the fine costmap
    instance (``costmap._coarse_cache``) so many dispatches over one version-
    cached costmap don't re-coarsen a 600² grid each time.  The engine rebuilds
    a **new** Costmap instance whenever the GIS/terrain version changes (a storm
    injection stamps flood zones lethal -> ``builder.build()`` returns a fresh
    object).  These tests prove:

      1. Repeat coarsening of the SAME instance returns the identical cached
         object (the perf win).
      2. A freshly built costmap carries its OWN cache — the coarse map reflects
         the NEW (storm) geometry, never the pre-storm instance's cache.
      3. End-to-end: after planning over the calm costmap (which populates its
         coarse cache), planning over the rebuilt storm costmap detours through
         the only gap in the flood wall — the coarse plan saw the fresh wall.
    """

    # Gap sits near the bottom, OFF the start->goal straight line (both at mid
    # height), so the storm route must genuinely detour down to thread it.
    GAP_BAND = (0.10, 0.20)

    @classmethod
    def _wall_builder(cls, n=200, gap_band=None, with_wall=True):
        """A costmap with (optionally) a tall lethal 'flood' wall + one gap.

        ``with_wall=False`` is the calm (pre-storm) map: fully open where the
        wall would later be.  ``with_wall=True`` stamps the storm flood wall
        (3 coarse-blocks thick so it is lethal at the coarse level) leaving a
        single gap band, so the route must detour through the gap.
        """
        if gap_band is None:
            gap_band = cls.GAP_BAND
        span = n * 5.0
        b = CostmapBuilder((0, 0, span, span), resolution=5.0)
        if with_wall:
            wx = span * 0.5
            gap_lo, gap_hi = span * gap_band[0], span * gap_band[1]
            # +-30m wall = 12 fine cells = 1.5 coarse blocks each side of centre
            # -> at least one fully-lethal coarse column, so the coarse solve
            # cannot cut straight through it.
            b.add_obstacles(_polygon_fc([
                [(wx - 30, 40), (wx + 30, 40), (wx + 30, gap_lo), (wx - 30, gap_lo)],
                [(wx - 30, gap_hi), (wx + 30, gap_hi),
                 (wx + 30, span - 40), (wx - 30, span - 40)],
            ]))
        return b.build(), span

    def test_same_instance_coarse_is_memoised(self):
        cm, _ = self._wall_builder(with_wall=False)
        c1 = _cached_coarse(cm, DEFAULT_COARSE_FACTOR)
        c2 = _cached_coarse(cm, DEFAULT_COARSE_FACTOR)
        assert c1 is c2, "same instance must reuse the cached coarse map"
        assert getattr(cm, "_coarse_cache", None) is not None
        assert DEFAULT_COARSE_FACTOR in cm._coarse_cache

    def test_rebuilt_costmap_has_independent_fresh_cache(self):
        # Calm map: plan once so its coarse cache is populated.
        calm, span = self._wall_builder(with_wall=False)
        plan_route_hierarchical(calm, (40, span * 0.5), (span - 40, span * 0.5))
        assert getattr(calm, "_coarse_cache", None), "calm plan should cache coarse"

        # Storm rebuild: a brand-new Costmap instance with the flood wall.
        storm, _ = self._wall_builder(with_wall=True)
        # A fresh instance starts with no coarse cache (no cross-instance leak).
        assert getattr(storm, "_coarse_cache", None) in (None, {})

        calm_coarse = _cached_coarse(calm, DEFAULT_COARSE_FACTOR)
        storm_coarse = _cached_coarse(storm, DEFAULT_COARSE_FACTOR)
        assert storm_coarse is not calm_coarse
        assert storm._coarse_cache is not calm._coarse_cache

        # The storm coarse map must show the wall as lethal coarse cells where
        # the calm coarse map is open — i.e. no stale 'open ground' is served.
        # Probe at mid-height (y=span/2), squarely inside the walled band.
        wx, wy = span * 0.5, span * 0.5
        wall_cc = storm_coarse.world_to_grid(wx, wy)
        assert wall_cc is not None
        assert storm_coarse.is_lethal(*wall_cc), "storm wall must appear in coarse map"
        calm_cc = calm_coarse.world_to_grid(wx, wy)
        assert calm_cc is not None
        assert not calm_coarse.is_lethal(*calm_cc), "calm coarse map stays open"

    def test_storm_replan_detours_through_gap_not_stale_straight_line(self):
        # Calm plan: straight across (no wall), populates the coarse cache.
        calm, span = self._wall_builder(with_wall=False)
        s, g = (40, span * 0.5), (span - 40, span * 0.5)
        calm_route = plan_route_hierarchical(calm, s, g)
        assert calm_route.success
        wx = span * 0.5
        gap_lo, gap_hi = span * self.GAP_BAND[0], span * self.GAP_BAND[1]

        # Storm rebuild -> route must detour through the gap band (proves the
        # coarse plan used the fresh storm geometry, not the cached calm coarse).
        storm, _ = self._wall_builder(with_wall=True)
        storm_route = plan_route_hierarchical(storm, s, g)
        assert storm_route.success
        assert not _path_touches_lethal(storm, storm_route.path)
        ys = _y_at_x(storm_route.path, wx)
        assert ys, "storm route should cross the wall x-line"
        assert all(gap_lo - 6 <= y <= gap_hi + 6 for y in ys), (
            "storm route must thread the gap, not the stale open corridor", ys)
        # And it is a genuine detour: strictly costlier than the calm straight run.
        assert storm_route.cost > calm_route.cost
