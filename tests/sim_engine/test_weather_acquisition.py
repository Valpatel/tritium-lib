# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Weather target-acquisition degradation in UnitBehaviors.

Fog/snow shrink the range at which a combatant can open fire, so the enemy
closes before anyone shoots. The modifier defaults to exactly 1.0 (weather
off), and ``weapon_range * 1.0 == weapon_range`` in IEEE754, so every
weather-off drive is byte-identical (the 6 canonical goldens depend on it).
"""

from __future__ import annotations

import random

from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget


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


def _hostile(dist: float) -> SimulationTarget:
    return SimulationTarget(
        target_id="h", name="Hostile", alliance="hostile",
        asset_type="person", position=(dist, 0.0), speed=0.0, morale=0.0,
        weapon_range=15.0,
    )


def test_default_modifier_is_one():
    assert _behaviors()._detection_modifier == 1.0


def test_clear_weather_acquires_at_full_range():
    beh = _behaviors()
    turret = _turret(90.0)
    enemies = {"h": _hostile(70.0)}   # inside weapon_range
    assert beh._nearest_in_range(turret, enemies) is not None


def test_fog_denies_long_range_acquisition():
    beh = _behaviors()
    beh.set_detection_modifier(0.489)   # heavy fog: eff range ~44m
    turret = _turret(90.0)
    # 70m is inside weapon_range but OUTSIDE the fog-degraded acquisition range.
    assert beh._nearest_in_range(turret, {"h": _hostile(70.0)}) is None
    # A hostile that has closed to 40m is now acquirable again.
    assert beh._nearest_in_range(turret, {"h2": _hostile(40.0)}) is not None


def test_candidates_mirror_nearest_gating():
    beh = _behaviors()
    beh.set_detection_modifier(0.489)
    turret = _turret(90.0)
    far = beh._candidates_in_range(turret, {"h": _hostile(70.0)})
    near = beh._candidates_in_range(turret, {"h2": _hostile(40.0)})
    assert far == []            # fog-obscured, no candidate
    assert len(near) == 1       # closed the distance -> candidate


def test_modifier_has_a_floor():
    beh = _behaviors()
    beh.set_detection_modifier(0.0)     # never fully blind
    assert beh._detection_modifier >= 0.05


def test_modifier_one_is_exactly_weapon_range():
    """weapon_range * 1.0 must be bit-exact so weather-off stays byte-identical."""
    beh = _behaviors()
    turret = _turret(90.0)
    # A hostile exactly at weapon_range is acquirable at modifier 1.0 just as
    # it was before the feature (the multiply is a no-op).
    assert turret.weapon_range * beh._detection_modifier == turret.weapon_range
