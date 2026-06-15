# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Animal home-range metric tests for AmbientSpawner / SimulationTarget.

FEATURE-AUDIT 2026-06-15: neutral animals (dogs/cats) had no purpose — they
spawned a yard center anywhere in 80% of bounds, appended a random map-edge
exit, and DESPAWNED at the edge (measured: 100% edge-despawn, p90 167m / max
343m from spawn).  These tests pin the new contract:

  - animal_has_home_anchor_rate: every spawned dog/cat has a non-None
    home_anchor immediately after spawn (target 100%).
  - animal_stays_in_home_range: over a >=5-min no-danger run, a wandering /
    resting animal never wanders more than HOME_RANGE_R + a small margin from
    its anchor, and NEVER despawns at a map edge (target p90 < 40m, max < 55m).
  - animal_building_clip_rate: an animal is never observed strictly inside a
    building footprint (target 0%).

Peds + vehicles are NOT exercised here — their edge-to-edge transit + despawn
contract is unchanged (see test_sim_game / SC test_ambient).
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.game.ambient import AmbientSpawner, HOME_RANGE_R
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.tracking.obstacles import BuildingObstacles


class _MockEngine:
    """Minimal engine stand-in for AmbientSpawner (duck-typed)."""

    def __init__(self, map_bounds: float = 200.0, obstacles=None) -> None:
        self.targets: list[SimulationTarget] = []
        self._map_bounds = map_bounds
        self._obstacles = obstacles
        self.spawners_paused = False

    def add_target(self, target: SimulationTarget) -> None:
        # Wire the swept building checker the same way the real engine does so
        # animal moves are building-aware in these tests.
        if self._obstacles is not None:
            target.set_collision_check(self._obstacles.point_in_building)
            target.set_segment_collision_check(self._obstacles.path_crosses_building)
        self.targets.append(target)

    def get_targets(self) -> list[SimulationTarget]:
        return list(self.targets)


def _grid_of_buildings(bounds: float = 200.0, spacing: float = 40.0,
                       half: float = 6.0) -> BuildingObstacles:
    """A dense building grid covering the map (mirrors the baseline harness)."""
    polys: list[list[tuple[float, float]]] = []
    pos = -bounds + spacing
    while pos < bounds:
        p = -bounds + spacing
        while p < bounds:
            polys.append([
                (pos - half, p - half), (pos + half, p - half),
                (pos + half, p + half), (pos - half, p + half),
            ])
            p += spacing
        pos += spacing
    obs = BuildingObstacles()
    obs.polygons = polys
    obs._compute_aabbs()
    return obs


# ---------------------------------------------------------------------------
# Metric 1: animal_has_home_anchor_rate
# ---------------------------------------------------------------------------

class TestHomeAnchorRate:
    def test_dog_has_home_anchor_on_spawn(self):
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        sp._spawn_dog()
        t = eng.targets[0]
        assert t.asset_type == "animal"
        assert t.home_anchor is not None
        assert len(t.home_anchor) == 2

    def test_cat_has_home_anchor_on_spawn(self):
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        sp._spawn_cat()
        t = eng.targets[0]
        assert t.home_anchor is not None

    def test_anchor_rate_is_100pct(self):
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        for _ in range(30):
            sp._spawn_dog()
            sp._spawn_cat()
        animals = [t for t in eng.targets if t.asset_type == "animal"]
        with_anchor = [t for t in animals if t.home_anchor is not None]
        assert len(animals) == 60
        assert len(with_anchor) == 60  # 100%

    def test_anchor_near_spawn_position(self):
        """Anchor is the yard center; the spawn position starts within range."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        for _ in range(20):
            sp._spawn_dog()
        for t in eng.targets:
            d = math.hypot(
                t.position[0] - t.home_anchor[0],
                t.position[1] - t.home_anchor[1],
            )
            assert d <= HOME_RANGE_R + 5.0


# ---------------------------------------------------------------------------
# Metric 2: animal_stays_in_home_range (no edge despawn)
#
# SCOPE NOTE (MINOR B, FEATURE-AUDIT 2026-06-15): the multi-tick tests in this
# class and in TestBuildingClipRate re-implement the home-range REFRESH LOOP
# INLINE (tick -> if depleted, call sp._home_range_loop -> reassign waypoints).
# That inline loop is a TEST FIXTURE, not the production refresh path.  These
# tests therefore exercise only the LIB PRIMITIVES — _yard_wander /
# _home_range_loop bound waypoints to HOME_RANGE_R, and SimulationTarget.tick +
# _neutral_terminal_status keep an anchored animal alive at a depleted loop.
# The ENGINE-DRIVEN refresh (SimulationEngine._tick_animals actually calling
# _home_range_loop each tick) is covered separately by the SC engine test
# tritium-sc/tests/engine/simulation/test_animal_homerange.py.
# ---------------------------------------------------------------------------

class TestStaysInHomeRangePrimitives:
    def test_no_edge_exit_waypoint(self):
        """_yard_wander must NOT append a map-edge exit anymore — every
        waypoint stays within HOME_RANGE_R of the anchor."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        for _ in range(50):
            anchor, waypoints = sp._yard_wander()
            for (wx, wy) in waypoints:
                d = math.hypot(wx - anchor[0], wy - anchor[1])
                assert d <= HOME_RANGE_R + 5.0, (
                    f"waypoint {d:.0f}m from anchor exceeds home range"
                )

    def test_animal_does_not_despawn_over_5min_no_danger(self):
        """A 300s no-danger run: the dog stays alive (no edge despawn) and
        keeps looping its home range."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        sp._spawn_dog()
        t = eng.targets[0]
        # 300s at 10Hz; refresh waypoints when depleted (the engine's
        # _tick_animals does this; emulate the home-range refresh here).
        for i in range(3000):
            t.tick(0.1)
            if t.status == "idle" or t._waypoint_index >= len(t.waypoints):
                _, wps = sp._home_range_loop(t.home_anchor)
                t.waypoints = wps
                t._waypoint_index = 0
                if t.status in ("idle", "stationary"):
                    t.status = "active"
        assert t.status != "despawned", "animal despawned at a map edge"

    def test_max_distance_under_55m(self):
        """Over a 300s no-danger run the animal never exceeds 55m from
        its anchor (target: p90 < 40m, max < 55m)."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        for _ in range(20):
            sp._spawn_dog()
        animals = list(eng.targets)
        dists: list[float] = []
        for _ in range(3000):
            for t in animals:
                t.tick(0.1)
                if t.status in ("idle", "stationary") or t._waypoint_index >= len(t.waypoints):
                    _, wps = sp._home_range_loop(t.home_anchor)
                    t.waypoints = wps
                    t._waypoint_index = 0
                    if t.status in ("idle", "stationary"):
                        t.status = "active"
                d = math.hypot(
                    t.position[0] - t.home_anchor[0],
                    t.position[1] - t.home_anchor[1],
                )
                dists.append(d)
        dists.sort()
        p90 = dists[int(len(dists) * 0.9)]
        mx = dists[-1]
        assert p90 < 40.0, f"p90 distance {p90:.0f}m exceeds 40m"
        assert mx < 55.0, f"max distance {mx:.0f}m exceeds 55m"
        # And nothing despawned at an edge.
        assert all(t.status != "despawned" for t in animals)


# ---------------------------------------------------------------------------
# Metric 3: animal_building_clip_rate
#
# SCOPE NOTE (MINOR B): like TestStaysInHomeRangePrimitives, the run below
# re-implements the home-range refresh loop INLINE as a test fixture — it
# exercises the LIB PRIMITIVES (_home_range_loop bounds + SimulationTarget.tick
# swept-collision), NOT the engine's _tick_animals refresh.  The engine-driven
# zero-clip proof lives in the SC test_animal_homerange.py engine test.
# ---------------------------------------------------------------------------

class TestBuildingClipRatePrimitives:
    def test_zero_clips_over_full_run(self):
        """With a dense building grid, no animal is ever observed strictly
        inside a building footprint over a full no-danger run (LIB PRIMITIVES;
        engine-driven refresh covered by SC test_animal_homerange.py)."""
        obs = _grid_of_buildings()
        eng = _MockEngine(obstacles=obs)
        sp = AmbientSpawner(eng)
        for _ in range(20):
            sp._spawn_dog()
            sp._spawn_cat()
        animals = list(eng.targets)
        clipped = 0
        for _ in range(2000):
            for t in animals:
                t.tick(0.1)
                if t.status in ("idle", "stationary") or t._waypoint_index >= len(t.waypoints):
                    _, wps = sp._home_range_loop(t.home_anchor)
                    t.waypoints = wps
                    t._waypoint_index = 0
                    if t.status in ("idle", "stationary"):
                        t.status = "active"
                if obs.point_in_building(t.position[0], t.position[1]):
                    clipped += 1
        assert clipped == 0, f"{clipped} building-clip observations (target 0)"


# ---------------------------------------------------------------------------
# Metric 4: per-species turnover (MAJOR 2 — immortal-neutral leak class)
#
# Anchored animals never edge-despawn (Metric 2), so without a per-species cap
# AND graceful turnover they accumulate forever and starve the global
# MAX_NEUTRALS=80 cap — the spawner then stops producing ANY new neutral
# (peds/vehicles too).  These tests pin the LIB PRIMITIVES of the fix:
#   - MAX_ANIMALS sub-cap exists and is well below MAX_NEUTRALS.
#   - _spawn_random stops adding animals at the sub-cap but still adds
#     peds/vehicles (so the global cap can fill with non-animals).
#   - every spawned animal carries a finite home_lifespan_s (turnover budget).
# The engine-driven turnover CYCLE (animals retire + the global count does not
# lock up) is covered by the SC engine test_animal_homerange.py.
# ---------------------------------------------------------------------------

class TestAnimalSubCapPrimitives:
    def test_max_animals_subcap_exists_and_below_global(self):
        assert hasattr(AmbientSpawner, "MAX_ANIMALS")
        assert 0 < AmbientSpawner.MAX_ANIMALS < AmbientSpawner.MAX_NEUTRALS

    def test_spawn_random_stops_animals_at_subcap_but_keeps_peds(self):
        """Once MAX_ANIMALS live animals exist, _spawn_random never adds another
        animal, but DOES still add a ped/vehicle (the global cap can still
        fill with non-animals)."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        # Force the spawner to the animal sub-cap with live animals.
        for _ in range(AmbientSpawner.MAX_ANIMALS):
            sp._spawn_dog()
        animals_before = sum(1 for t in eng.targets if t.asset_type == "animal")
        assert animals_before == AmbientSpawner.MAX_ANIMALS
        # Now drive 60 random spawns with the animal-favoring rolls (0.60/0.75
        # would normally pick dog/cat).  None may add an animal; some must add
        # a ped/vehicle so the population keeps growing.
        rolls = [0.60, 0.75] * 40  # all animal-favoring rolls
        idx = {"i": 0}
        import random as _r
        orig = _r.random
        def fake_random():
            v = rolls[idx["i"] % len(rolls)]
            idx["i"] += 1
            return v
        _r.random = fake_random
        try:
            for _ in range(60):
                sp._spawn_random()
        finally:
            _r.random = orig
        animals_after = sum(1 for t in eng.targets if t.asset_type == "animal")
        non_animal_after = sum(
            1 for t in eng.targets if t.asset_type != "animal"
        )
        assert animals_after == AmbientSpawner.MAX_ANIMALS, (
            f"sub-cap breached: {animals_after} animals"
        )
        assert non_animal_after > 0, (
            "spawner refused to add peds/vehicles when animals at sub-cap"
        )

    def test_spawned_animals_have_finite_lifespan(self):
        """Every spawned dog/cat carries a finite home_lifespan_s turnover
        budget (None/inf would mean no turnover -> immortal leak)."""
        eng = _MockEngine()
        sp = AmbientSpawner(eng)
        for _ in range(20):
            sp._spawn_dog()
            sp._spawn_cat()
        animals = [t for t in eng.targets if t.asset_type == "animal"]
        assert len(animals) == 40
        for t in animals:
            ls = getattr(t, "home_lifespan_s", None)
            assert ls is not None, "animal has no home_lifespan_s turnover budget"
            assert 0.0 < ls < float("inf"), f"non-finite lifespan {ls!r}"
