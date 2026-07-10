# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Cover-vs-peek arbitration for DAMAGED ground stand-ins (UX Loop 4).

Two movement drives compete for a wounded ground unit whose fire solution is
MASKED by a building:

  * cover-seek  — a damaged unit ducks toward the nearest building edge;
  * engagement-range peek — a masked shot is un-masked by leaning out to a
    point that restores line of sight.

Left un-arbitrated they thrash (a peek waypoint pulling one way while cover-seek
drags the unit the other).  This module pins the explicit phase machine:

    seeking  -> move to cover, NO peek waypoints issued;
    holding  -> commit at cover for a minimum sim-time dwell (hysteresis);
    peeking  -> lean out to a BOUNDED peek (<= exposure radius from the cover
                anchor) that restores LOS, then re-engage.

FUN: wounded units visibly duck behind buildings then lean out to fire —
readable drama.  PRODUCTION: priority arbitration between competing behavior
drives is a core autonomy capability of the stand-in driver stack (the riot
lane's police reuse the identical doctrine).
"""

from __future__ import annotations

import math
import random
import types

import pytest

import tritium_lib.sim_engine.behavior.behaviors as behaviors_mod
from tritium_lib.sim_engine.behavior.behaviors import (
    UnitBehaviors,
    _COVER_ARRIVAL_RADIUS,
    _COVER_PEEK_EXPOSURE,
    _COVER_PHASE_DWELL_S,
)
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


def _mask_map() -> TerrainMap:
    """One building cell at (20, 0): masks the (0,0)->(40,0) shot along +x,
    but a small lateral lean clears its edge."""
    tm = TerrainMap(map_bounds=100.0, resolution=5.0)
    tm.set_cell(20.0, 0.0, "building")
    return tm


def _obstacles(polys):
    """Minimal cover source — _nearest_cover_point/_seek_cover only read
    ``.polygons`` (a list of (x, y) footprint rings)."""
    return types.SimpleNamespace(polygons=polys)


def _wall_at(x: float, half_y: float = 6.0, thick: float = 2.0):
    """A cover wall footprint whose nearest edge to the origin sits at ``x``."""
    return [[(x, -half_y), (x + thick, -half_y), (x + thick, half_y), (x, half_y)]]


def _rover(tid="r", pos=(0.0, 0.0), speed=10.0, health=20.0) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name="Rover", alliance="friendly", asset_type="rover",
        position=pos, speed=speed, weapon_range=100.0,
    )
    t.health = health
    t.max_health = 100.0
    return t


def _drone(tid="d", pos=(0.0, 0.0), speed=12.0, health=20.0) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name="Drone", alliance="friendly", asset_type="drone",
        position=pos, speed=speed, weapon_range=100.0, altitude=30.0,
    )
    t.health = health
    t.max_health = 100.0
    return t


def _inert_hostile(tid="t", pos=(40.0, 0.0)) -> SimulationTarget:
    """A hostile the unit can target but that never moves or fires back
    (morale 0.0 -> broken -> its behavior returns immediately)."""
    return SimulationTarget(
        target_id=tid, name="Hostile", alliance="hostile", asset_type="person",
        position=pos, speed=0.0, morale=0.0, weapon_range=15.0,
    )


def _make(tm, obstacles=None, seed=1):
    bus = _RecordingBus()
    cs = CombatSystem(event_bus=bus, rng=random.Random(seed))
    if tm is not None:
        cs.set_terrain_map(tm)
    beh = UnitBehaviors(cs)
    if tm is not None:
        beh.set_terrain_map(tm)
    if obstacles is not None:
        beh.set_obstacles(obstacles)
    beh.set_game_mode_type("battle")
    return beh, cs, bus


# ===================================================================
# (1) damaged + masked -> cover WINS, no peek while seeking
# ===================================================================


class TestCoverWinsWhileSeeking:
    def test_damaged_masked_enters_seeking_toward_cover(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        # Precondition: the straight shot is masked.
        assert tm.line_of_sight(rover.position, tgt.position) is False

        beh.tick(0.1, targets)

        # Cover WON: the unit is in the seeking phase, steering to the cover
        # point — NOT peeking.
        assert beh._cover_phase["r"][0] == "seeking"
        # No engagement-range peek was issued while seeking (the core rule).
        assert beh._peek_targets == {}
        # The waypoint is the cover point: on the firing axis (y~0) and toward
        # the wall (+x), not a lateral peek offset.
        cover = beh._nearest_cover_point((0.0, 0.0))
        assert rover.waypoints[-1] == cover
        assert abs(cover[1]) < 1.0
        assert cover[0] > rover.position[0]
        # It held fire this tick (repositioning, not shooting).
        assert bus.count("projectile_fired") == 0

    def test_seeking_unit_moves_toward_cover_and_never_peeks(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}
        cover = beh._nearest_cover_point((0.0, 0.0))

        d0 = math.hypot(rover.position[0] - cover[0], rover.position[1] - cover[1])
        # Drive a few ticks — still inside the seeking window (cover ~10m away,
        # arrival radius 3m, ~1 m/tick).
        for _ in range(5):
            beh.tick(0.1, targets)
            rover.tick(0.1)
            # Invariant: while seeking, the peek doctrine is never engaged.
            assert beh._cover_phase["r"][0] == "seeking"
            assert beh._peek_targets == {}
        d1 = math.hypot(rover.position[0] - cover[0], rover.position[1] - cover[1])
        assert d1 < d0, "damaged unit did not close on cover while seeking"


# ===================================================================
# (2) at cover + masked -> bounded peek FROM cover
# ===================================================================


class TestPeekFromCover:
    def _hold_until_peek(self, beh, targets, unit_id="r"):
        # Freeze the entity (do NOT tick it) so it stays masked at cover while
        # sim-time advances past the holding dwell, then leans out.
        ticks = int((_COVER_PHASE_DWELL_S + 0.6) / 0.1) + 2
        for _ in range(ticks):
            beh.tick(0.1, targets)

    def test_at_cover_leans_out_within_exposure_bound(self):
        tm = _mask_map()
        # Cover edge 2m away -> within arrival radius -> holds on tick 1.
        beh, cs, bus = _make(tm, _obstacles(_wall_at(2.0)))
        rover = _rover(pos=(0.0, 0.0))
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}
        cover = beh._nearest_cover_point((0.0, 0.0))
        assert math.hypot(*cover) <= _COVER_ARRIVAL_RADIUS

        self._hold_until_peek(beh, targets)

        phase, _, tid, anchor = beh._cover_phase["r"]
        assert phase == "peeking"
        assert anchor == cover, "peek must lean out from the committed cover anchor"
        peek = rover.waypoints[-1]
        # BOUNDED exposure: the lean-out stays within the exposure radius of the
        # cover anchor (leans out, does not abandon cover).
        exposure = math.hypot(peek[0] - anchor[0], peek[1] - anchor[1])
        assert exposure <= _COVER_PEEK_EXPOSURE + 1e-9, (
            f"peek {peek} is {exposure:.2f}m from cover {anchor} "
            f"(> exposure bound {_COVER_PEEK_EXPOSURE})"
        )
        # And it genuinely restores the fire solution.
        assert tm.line_of_sight(peek, tgt.position) is True

    def test_holding_commits_before_peeking_dwell(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(2.0)))
        rover = _rover(pos=(0.0, 0.0))
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        beh.tick(0.1, targets)
        assert beh._cover_phase["r"][0] == "holding"
        # Within the dwell window it stays committed at cover (no premature peek).
        for _ in range(int(_COVER_PHASE_DWELL_S / 0.1) - 2):
            beh.tick(0.1, targets)
            assert beh._cover_phase["r"][0] == "holding"
            assert not rover.waypoints, "unit must hold at cover, not lean out early"

    def test_no_cover_available_peeks_from_position(self):
        # Obstacles wired but with no polygons -> cover none available; the unit
        # peeks from where it stands (bounded to exposure of its own position).
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles([]))
        rover = _rover(pos=(0.0, 0.0))
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        beh.tick(0.1, targets)
        phase, _, _, anchor = beh._cover_phase["r"]
        assert phase == "peeking"
        assert anchor == (0.0, 0.0)
        peek = rover.waypoints[-1]
        assert math.hypot(peek[0], peek[1]) <= _COVER_PEEK_EXPOSURE + 1e-9
        assert tm.line_of_sight(peek, tgt.position) is True


# ===================================================================
# (3) anti-thrash: bounded waypoint churn, drives are mutually exclusive
# ===================================================================


class TestAntiThrash:
    def test_waypoint_churn_bounded_over_full_cycle(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        rover = _rover()
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        dests: list[tuple | None] = []
        for _ in range(60):
            beh.tick(0.1, targets)
            rover.tick(0.1)
            dest = tuple(rover.waypoints[-1]) if rover.waypoints else None
            dests.append(dest)
            # The two drives are MUTUALLY EXCLUSIVE — a unit is never steered by
            # cover-vs-peek and the raw peek doctrine at the same time (thrash).
            assert not ("r" in beh._cover_phase and "r" in beh._peek_targets)
            # While seeking, the destination is always the cover anchor (no peek
            # intrusion) and the peek doctrine is idle.
            state = beh._cover_phase.get("r")
            if state is not None and state[0] == "seeking":
                assert dest == tuple(state[3])
                assert beh._peek_targets == {}

        # Destination reassignments are bounded by the dwell-gated phase
        # transitions (seek -> hold -> peek -> settle), NOT one-per-tick thrash.
        changes = sum(1 for i in range(1, len(dests)) if dests[i] != dests[i - 1])
        assert changes <= 8, f"excessive waypoint churn: {changes} reassignments/60 ticks"

        # Fun payoff: it leaned out and actually took the shot.
        assert bus.count("projectile_fired") >= 1, "unit never fired from the peek"
        assert tm.line_of_sight(rover.position, tgt.position) is True

    def test_phase_transitions_respect_min_dwell(self):
        # Peeking commits for at least one dwell before falling back to holding.
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(2.0)))
        rover = _rover(pos=(0.0, 0.0))
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        # Reach the peeking phase (entity frozen so it never clears LOS on its
        # own — isolates the phase machine's own dwell timing).
        phases: list[str] = []
        for _ in range(45):
            beh.tick(0.1, targets)
            st = beh._cover_phase.get("r")
            phases.append(st[0] if st else "none")

        assert "seeking" not in phases  # cover was within arrival radius
        assert "holding" in phases and "peeking" in phases
        # No single-tick phase flips: every phase run is at least a few ticks.
        runs = []
        cur, n = phases[0], 1
        for p in phases[1:]:
            if p == cur:
                n += 1
            else:
                runs.append((cur, n))
                cur, n = p, 1
        runs.append((cur, n))
        for name, length in runs:
            assert length >= 2, f"phase {name} flipped after {length} tick(s) (thrash)"


# ===================================================================
# (5) aerial exemption + healthy-unit fall-through
# ===================================================================


class TestExemptions:
    def test_aerial_type_never_cover_or_peek(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        drone = _drone()  # asset_type "drone" -> LOS-exempt, in _NO_DOCTRINE_TYPES
        tgt = _inert_hostile()
        assert tm.line_of_sight(drone.position, tgt.position) is False

        # Both doctrine layers must decline an aerial unit outright.
        assert beh._arbitrate_cover_peek(drone, tgt) is False
        assert beh._reposition_for_los(drone, tgt) is False
        assert beh._cover_phase == {}
        assert beh._peek_targets == {}

        # And through a real tick the drone gets no cover/peek waypoints.
        targets = {"d": drone, "t": tgt}
        before = list(drone.waypoints)
        beh.tick(0.1, targets)
        assert beh._cover_phase == {}
        assert beh._peek_targets == {}
        assert drone.waypoints == before

    def test_static_unit_never_arbitrates(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        turret = _rover()          # borrow the rover fixture...
        turret.speed = 0.0         # ...but immobile: cover-vs-peek must decline.
        tgt = _inert_hostile()
        assert beh._arbitrate_cover_peek(turret, tgt) is False

    def test_healthy_unit_uses_peek_doctrine_not_cover(self):
        tm = _mask_map()
        beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)))
        rover = _rover(health=100.0)  # full health -> not damaged
        tgt = _inert_hostile()
        targets = {"r": rover, "t": tgt}

        # Arbitration declines a healthy unit; it falls through to the raw
        # engagement-range peek doctrine (unchanged tick-2 behavior).
        assert beh._arbitrate_cover_peek(rover, tgt) is False
        beh.tick(0.1, targets)
        assert "r" not in beh._cover_phase
        assert "r" in beh._peek_targets, "healthy masked unit must use peek doctrine"


# ===================================================================
# Determinism + per-unit state cleanup
# ===================================================================


class TestDeterminismAndCleanup:
    def test_deterministic_two_runs(self):
        def _run() -> list[tuple[float, float]]:
            tm = _mask_map()
            beh, cs, bus = _make(tm, _obstacles(_wall_at(10.0)), seed=7)
            rover = _rover()
            tgt = _inert_hostile()
            targets = {"r": rover, "t": tgt}
            trace: list[tuple[float, float]] = []
            for _ in range(40):
                beh.tick(0.1, targets)
                rover.tick(0.1)
                trace.append(rover.position)
            return trace

        assert _run() == _run()

    def test_remove_unit_clears_cover_phase(self):
        beh, cs, bus = _make(_mask_map(), _obstacles(_wall_at(10.0)))
        beh._cover_phase["ghost"] = ("seeking", 0.0, "t", (10.0, 0.0))
        beh.remove_unit("ghost")
        assert "ghost" not in beh._cover_phase

    def test_clear_dodge_state_clears_cover_phase(self):
        beh, cs, bus = _make(_mask_map(), _obstacles(_wall_at(10.0)))
        beh._cover_phase["a"] = ("holding", 0.0, "t", (2.0, 0.0))
        beh._cover_phase["b"] = ("peeking", 0.0, "t", (2.0, 0.0))
        beh.clear_dodge_state()
        assert beh._cover_phase == {}
