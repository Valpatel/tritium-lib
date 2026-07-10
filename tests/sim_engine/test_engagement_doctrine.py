# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Engagement-range doctrine — LOS-recovery peek/flank for ground stand-ins.

Two layers:

  * ``find_peek_position`` (pure helper): deterministic lateral search for the
    closest point that restores line of sight to a masked target while keeping
    it in weapon range and staying out of buildings.
  * ``UnitBehaviors`` integration: a ground unit whose shot is LOS-blocked by a
    building steers to a peek point and re-engages once LOS is restored; a unit
    with a clear shot never detours; recompute is hysteresis-throttled.

FUN: urban stand-in fights come alive — units flank and peek like a player
expects.  PRODUCTION: this is the LOS-recovery maneuver a real ground robot
needs when its fire solution is masked; the same helper drives a live costmap.
"""

from __future__ import annotations

import math
import random

import pytest

import tritium_lib.sim_engine.behavior.behaviors as behaviors_mod
from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.behavior.doctrine import find_peek_position
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.world.terrain_map import TerrainMap


# ---------------------------------------------------------------------------
# Test doubles / fixtures
# ---------------------------------------------------------------------------


class _RecordingBus:
    """Event bus that records every publish for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def count(self, topic: str) -> int:
        return sum(1 for t, _ in self.events if t == topic)


def _short_wall_map() -> TerrainMap:
    """One building cell at (20, 0): blocks LOS along the x-axis but a small
    lateral step clears its edge."""
    tm = TerrainMap(map_bounds=100.0, resolution=5.0)
    tm.set_cell(20.0, 0.0, "building")
    return tm


def _long_wall_map() -> TerrainMap:
    """A full-height wall at x=20: no lateral peek can get around it."""
    tm = TerrainMap(map_bounds=100.0, resolution=5.0)
    for y in range(-100, 101, 5):
        tm.set_cell(20.0, float(y), "building")
    return tm


def _rover(target_id: str = "r", pos=(0.0, 0.0), speed=10.0) -> SimulationTarget:
    return SimulationTarget(
        target_id=target_id, name="Rover", alliance="friendly",
        asset_type="rover", position=pos, speed=speed, weapon_range=100.0,
    )


def _inert_hostile(target_id: str = "t", pos=(40.0, 0.0)) -> SimulationTarget:
    """A hostile the rover can target but that never moves or fires back
    (morale 0.0 -> broken -> its behavior returns immediately, so no random
    dodge perturbs the scene)."""
    return SimulationTarget(
        target_id=target_id, name="Hostile", alliance="hostile",
        asset_type="person", position=pos, speed=0.0, morale=0.0,
        weapon_range=15.0,
    )


def _make_behaviors(tm: TerrainMap | None, seed: int = 1):
    bus = _RecordingBus()
    cs = CombatSystem(event_bus=bus, rng=random.Random(seed))
    if tm is not None:
        cs.set_terrain_map(tm)
    beh = UnitBehaviors(cs)
    if tm is not None:
        beh.set_terrain_map(tm)
    beh.set_game_mode_type("battle")
    return beh, cs, bus


# ===================================================================
# find_peek_position — pure helper
# ===================================================================


class TestFindPeekPosition:
    def test_finds_lateral_peek_around_wall(self):
        tm = _short_wall_map()
        # Precondition: the straight shot is masked.
        assert tm.line_of_sight((0.0, 0.0), (40.0, 0.0)) is False
        peek = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 100.0)
        assert peek is not None
        # The peek genuinely restores line of sight to the target.
        assert tm.line_of_sight(peek, (40.0, 0.0)) is True
        # It is a lateral side-step (offset perpendicular to +x -> along y).
        assert abs(peek[1]) > 0.0
        assert peek[0] == pytest.approx(0.0, abs=1e-9)

    def test_returns_none_when_fully_masked(self):
        tm = _long_wall_map()
        assert tm.line_of_sight((0.0, 0.0), (40.0, 0.0)) is False
        peek = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 200.0)
        assert peek is None

    def test_candidate_is_not_inside_a_building(self):
        tm = _short_wall_map()
        peek = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 100.0)
        assert peek is not None
        # Walkable == finite movement cost (buildings/water are inf).
        assert not math.isinf(tm.get_movement_cost(peek[0], peek[1]))

    def test_respects_weapon_range(self):
        tm = _short_wall_map()
        # With a generous range we get the closest restoring peek.
        peek = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 1000.0)
        assert peek is not None
        d = math.hypot(peek[0] - 40.0, peek[1] - 0.0)
        assert d <= 1000.0
        # Shrink the range below that peek's distance: every restoring peek is
        # now out of range, so the doctrine declines to over-extend -> None.
        none = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), d - 1.0)
        assert none is None

    def test_deterministic(self):
        tm = _short_wall_map()
        a = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 100.0)
        b = find_peek_position(tm, (0.0, 0.0), (40.0, 0.0), 100.0)
        assert a == b

    def test_closest_offset_returned_first(self):
        tm = _short_wall_map()
        peek = find_peek_position(
            tm, (0.0, 0.0), (40.0, 0.0), 100.0, step=2.5, max_offset=30.0,
        )
        assert peek is not None
        # No valid restoring peek should exist at a strictly smaller offset.
        chosen_offset = math.hypot(peek[0], peek[1])
        assert chosen_offset <= 2.5 + 1e-9  # first ring cleared the edge

    def test_shooter_on_top_of_target_returns_none(self):
        tm = _short_wall_map()
        assert find_peek_position(tm, (5.0, 5.0), (5.0, 5.0), 100.0) is None


# ===================================================================
# UnitBehaviors integration — peek / flank then re-engage
# ===================================================================


class TestDoctrineIntegration:
    def _drive(self, beh, cs, targets, unit, ticks, dt=0.1):
        """Run behavior + entity + combat ticks together, like the engine."""
        for _ in range(ticks):
            beh.tick(dt, targets)
            unit.tick(dt)
            cs.tick(dt, targets)

    def test_blocked_rover_steers_toward_peek(self):
        tm = _short_wall_map()
        beh, cs, bus = _make_behaviors(tm)
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        # First tick: masked -> reposition, no shot yet.
        beh.tick(0.1, targets)
        assert "r" in beh._peek_targets, "rover should have adopted a peek target"
        peek = beh._peek_targets["r"][2]
        assert rover.waypoints == [peek]

        y_start = rover.position[1]
        rover.tick(0.1)
        rover.tick(0.1)
        # It moved laterally toward the peek (off the masked firing line).
        assert abs(rover.position[1] - peek[1]) < abs(y_start - peek[1])

    def test_rover_fires_once_los_restored(self):
        tm = _short_wall_map()
        beh, cs, bus = _make_behaviors(tm)
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        self._drive(beh, cs, targets, rover, ticks=40)
        assert bus.count("projectile_fired") >= 1, (
            "rover never re-engaged after repositioning for LOS"
        )
        # Once it regained LOS it dropped its peek state and fired normally.
        assert tm.line_of_sight(rover.position, tgt.position) is True

    def test_clear_los_does_not_detour(self):
        # No terrain wall in the flight path -> immediate shot, no peek.
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        beh, cs, bus = _make_behaviors(tm)
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        beh.tick(0.1, targets)
        assert beh._peek_targets == {}, "clear-LOS unit must not adopt a peek"
        assert rover.waypoints == [], "clear-LOS unit must not detour"
        assert bus.count("projectile_fired") == 1

    def test_no_terrain_map_fires_normally(self):
        beh, cs, bus = _make_behaviors(tm=None)
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        beh.tick(0.1, targets)
        assert beh._peek_targets == {}
        assert bus.count("projectile_fired") == 1

    def test_peek_not_recomputed_every_tick(self, monkeypatch):
        tm = _short_wall_map()
        beh, cs, bus = _make_behaviors(tm)
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        calls = {"n": 0}

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return find_peek_position(*args, **kwargs)

        monkeypatch.setattr(behaviors_mod, "find_peek_position", _counting)

        # Keep the rover frozen (do NOT tick the entity) so LOS stays masked
        # and the reposition path runs every behavior tick.
        for _ in range(5):  # 0.5 sim-seconds < _PEEK_RECOMPUTE_S (1.5)
            beh.tick(0.1, targets)
        assert calls["n"] == 1, "peek recomputed inside the hysteresis window"

        # Cross the recompute window -> exactly one more solve.
        for _ in range(15):  # advance well past 1.5 s total
            beh.tick(0.1, targets)
        assert calls["n"] == 2, "peek never recomputed after the window elapsed"

    def test_determinism_identical_positions_two_runs(self):
        def _run() -> list[tuple[float, float]]:
            tm = _short_wall_map()
            beh, cs, bus = _make_behaviors(tm, seed=1234)
            rover = _rover()
            tgt = _inert_hostile()
            targets = {"r": rover, "t": tgt}
            trace: list[tuple[float, float]] = []
            for _ in range(30):
                beh.tick(0.1, targets)
                rover.tick(0.1)
                cs.tick(0.1, targets)
                trace.append(rover.position)
            return trace

        assert _run() == _run()


# ===================================================================
# Per-unit state cleanup
# ===================================================================


class TestPeekStateCleanup:
    def test_remove_unit_clears_peek_state(self):
        beh, cs, bus = _make_behaviors(_short_wall_map())
        beh._peek_targets["ghost"] = ("t", 0.0, (1.0, 2.0))
        beh.remove_unit("ghost")
        assert "ghost" not in beh._peek_targets

    def test_clear_dodge_state_clears_all_peeks(self):
        beh, cs, bus = _make_behaviors(_short_wall_map())
        beh._peek_targets["a"] = ("t", 0.0, (1.0, 2.0))
        beh._peek_targets["b"] = ("t", 0.0, (3.0, 4.0))
        beh.clear_dodge_state()
        assert beh._peek_targets == {}
