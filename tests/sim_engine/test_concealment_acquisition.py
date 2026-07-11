# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Seasonal concealment degrades combat target-acquisition.

Integration seam: the GIS lane's ``LandCoverGrid.concealment_at`` (per-point,
season-folded) reduces the range at which a combatant can acquire a hostile
standing in vegetation. A hostile in a *summer* forest (full canopy, high
concealment) is acquired only when a unit closes much nearer than the SAME
hostile in the SAME forest in *winter* (bare, low concealment).

Gating (byte-identical guarantee): with no concealment provider attached
(the default, and what every canonical scenario runs), the acquisition path
is unchanged — proven here selection-for-selection.
"""

from __future__ import annotations

import random

from tritium_lib.geo.gis.landcover import LandCoverGrid
from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.perception import (
    CONCEALMENT_FLOOR,
    ConcealmentField,
    detectability_multiplier,
)


# --- fixtures ---------------------------------------------------------------

def _behaviors(seed: int = 1) -> UnitBehaviors:
    cs = CombatSystem(event_bus=None, rng=random.Random(seed))
    beh = UnitBehaviors(cs)
    beh.set_game_mode_type("battle")
    return beh


def _turret(weapon_range: float = 90.0) -> SimulationTarget:
    return SimulationTarget(
        target_id="def", name="Turret", alliance="friendly",
        asset_type="turret", position=(0.0, 0.0), speed=0.0,
        weapon_range=weapon_range,
    )


def _hostile(dist: float, tid: str = "h") -> SimulationTarget:
    # Position uses (x, y) == (lon, lat) so the test grid samples it directly
    # (ConcealmentField with to_lonlat=None treats position as lon/lat).
    return SimulationTarget(
        target_id=tid, name="Hostile", alliance="hostile",
        asset_type="person", position=(dist, 0.0), speed=0.0, morale=0.0,
        weapon_range=15.0,
    )


def _forest_grid() -> LandCoverGrid:
    """A single-cell all-evergreen-forest (code 42) grid covering a wide bbox.

    ``concealment_at`` snaps to the nearest cell, so every position in the
    bbox samples evergreen forest — summer canopy vs winter bare is driven
    entirely by the foliage argument.
    """
    return LandCoverGrid(
        west=-1000.0, south=-1000.0, east=1000.0, north=1000.0,
        ncols=1, nrows=1, codes=[42], source="test",
    )


# --- detectability math -----------------------------------------------------

def test_exposed_target_multiplier_is_exactly_one():
    # concealment 0.0 (exposed) => range unchanged bit-for-bit (gating no-op).
    assert detectability_multiplier(0.0) == 1.0


def test_multiplier_shrinks_with_concealment_and_floors():
    assert abs(detectability_multiplier(0.81) - 0.19) < 1e-9
    assert detectability_multiplier(1.0) == CONCEALMENT_FLOOR
    assert detectability_multiplier(-0.5) == 1.0   # clamped, never extends


# --- summer vs winter acquisition (the headline) ----------------------------

def test_summer_forest_acquired_later_than_winter_bare():
    """Same seed, same hostile, same forest: summer hides it, winter reveals it."""
    grid = _forest_grid()
    summer = ConcealmentField(grid, foliage_fn=lambda: 1.0)
    winter = ConcealmentField(grid, foliage_fn=lambda: 0.13)

    # concealment_at(any point) in evergreen forest:
    #   summer 0.90 -> mult 0.10 -> acquisition range 90*0.10 = 9.0m
    #   winter 0.117 -> mult 0.883 -> acquisition range 90*0.883 ~ 79.5m
    c_summer = grid.concealment_at(0.0, 60.0, foliage=1.0)
    c_winter = grid.concealment_at(0.0, 60.0, foliage=0.13)
    assert c_summer > c_winter
    summer_range = 90.0 * detectability_multiplier(c_summer)
    winter_range = 90.0 * detectability_multiplier(c_winter)
    assert winter_range > summer_range * 3   # dramatic seasonal swing

    # A hostile at 60m: acquired in winter, NOT in summer.
    beh_s = _behaviors(seed=7)
    beh_s.set_concealment_provider(summer)
    beh_w = _behaviors(seed=7)
    beh_w.set_concealment_provider(winter)

    turret = _turret(90.0)
    assert beh_w._nearest_in_range(turret, {"h": _hostile(60.0)}) is not None
    assert beh_s._nearest_in_range(turret, {"h": _hostile(60.0)}) is None

    # In summer the hostile must CLOSE inside ~9m before it's acquired.
    assert beh_s._nearest_in_range(turret, {"h": _hostile(8.0)}) is not None
    assert beh_s._nearest_in_range(turret, {"h": _hostile(20.0)}) is None


def test_concealment_measured_acquisition_range_summer_vs_winter():
    """Bisect the acquisition threshold to report concrete summer/winter ranges."""
    grid = _forest_grid()
    turret = _turret(90.0)

    def max_acq(field) -> float:
        beh = _behaviors(seed=3)
        beh.set_concealment_provider(field)
        lo, hi = 0.0, 90.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if beh._nearest_in_range(turret, {"h": _hostile(mid)}) is not None:
                lo = mid
            else:
                hi = mid
        return lo

    summer = max_acq(ConcealmentField(grid, foliage_fn=lambda: 1.0))
    winter = max_acq(ConcealmentField(grid, foliage_fn=lambda: 0.13))
    # Documented numbers: summer ~9.0m, winter ~79.5m.
    assert 8.5 < summer < 9.5
    assert 78.0 < winter < 81.0
    assert winter > summer * 5


# --- gating: no provider == byte-identical selection ------------------------

def test_no_provider_is_byte_identical_selection():
    """Default (no field) picks EXACTLY the same target as pre-feature."""
    grid = _forest_grid()
    turret = _turret(90.0)
    enemies = {"a": _hostile(60.0, "a"), "b": _hostile(40.0, "b")}

    beh_off = _behaviors(seed=5)                       # no provider
    pick_off = beh_off._nearest_in_range(turret, dict(enemies))
    cands_off = beh_off._candidates_in_range(turret, dict(enemies))

    # With a summer forest field, the far target drops out (concealed) — proves
    # the field is actually doing something (not a silent no-op).
    beh_on = _behaviors(seed=5)
    beh_on.set_concealment_provider(ConcealmentField(grid, foliage_fn=lambda: 1.0))
    pick_on = beh_on._nearest_in_range(turret, dict(enemies))

    assert pick_off is not None and pick_off.target_id == "b"   # nearest, unfiltered
    assert len(cands_off) == 2                                   # both in weapon_range
    assert pick_on is None    # both concealed beyond 9m summer range


def test_nearer_concealed_yields_to_farther_exposed():
    """A near hidden target is un-acquirable while a far exposed one is engaged.

    Mixed grid: forest column on the left (x<0), open water/grassland-free open
    space on the right so the exposed target has concealment 0. The nearer
    forest hostile is beyond summer acquisition range; the farther exposed one
    is inside it -> the exposed one wins even though it is farther.
    """
    # Two-column grid: col0 (west, lon<0) forest 42, col1 (east, lon>0) open (71 grass=0.2)
    # Use developed-open-space code 21 (concealment 0.2, season-invariant) on the
    # right so an exposed hostile there stays acquirable.
    grid = LandCoverGrid(
        west=-100.0, south=-10.0, east=100.0, north=10.0,
        ncols=2, nrows=1, codes=[42, 21], source="test",
    )
    turret = _turret(90.0)
    beh = _behaviors(seed=9)
    beh.set_concealment_provider(ConcealmentField(grid, foliage_fn=lambda: 1.0))

    near_forest = _hostile(-30.0, "near")   # lon=-30 -> forest, summer conceal 0.9
    far_open = _hostile(50.0, "far")        # lon=+50 -> developed open, conceal 0.2
    picked = beh._nearest_in_range(
        turret, {"near": near_forest, "far": far_open})
    assert picked is not None and picked.target_id == "far"


# --- ConcealmentField adapter ----------------------------------------------

def test_field_foliage_defaults_and_none_grid():
    grid = _forest_grid()
    f_default = ConcealmentField(grid)             # no foliage_fn -> summer 1.0
    assert f_default.foliage() == 1.0
    assert f_default(_hostile(10.0)) == grid.concealment_at(0.0, 10.0, foliage=1.0)

    f_none = ConcealmentField(None)
    assert f_none(_hostile(10.0)) == 0.0           # no grid -> exposed


def test_field_to_lonlat_mapper():
    """A world-metres->lonlat mapper is applied before sampling the grid."""
    grid = _forest_grid()
    # Map every world position to a fixed forest point.
    field = ConcealmentField(grid, foliage_fn=lambda: 1.0,
                             to_lonlat=lambda x, y: (0.0, 0.0))
    assert field(_hostile(500.0)) == grid.concealment_at(0.0, 0.0, foliage=1.0)
