# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Ballistics tests (WP1) — real dispersion, unguided direct fire, occlusion.

Covers the WP1 combat changes:
  - TerrainMap.raycast() first-building-cell hit.
  - Seedable dispersion (bit-for-bit determinism under a fixed rng seed).
  - accuracy 1.0 -> zero dispersion.
  - Dispersion self-calibrated so hit probability ~= weapon accuracy.
  - Fire-time LOS via the stored terrain map (ground blocked, mortar arcs over).
  - Aerial exemption from LOS/occlusion.
  - In-flight building occlusion -> projectile_impact.
  - Unguided direct fire (does not follow a teleporting target).
  - Guided (missile) fire (still homes to a moved target).

Mirrors tests/sim_engine/test_combat_systems.py — the fake-target stub is
replicated here (importing the real SimulationTarget pulls in heavy optional
deps not available in the lib test environment).
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field

import pytest

from tritium_lib.sim_engine.combat import (
    CombatSystem,
    HIT_RADIUS,
    Projectile,
    Weapon,
    WeaponSystem,
)
from tritium_lib.sim_engine.world.terrain_map import TerrainMap


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class _FakeTarget:
    """Minimal target stub matching the fields CombatSystem.fire() needs."""

    target_id: str
    name: str = "Unit"
    asset_type: str = "rover"
    alliance: str = "friendly"
    position: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    health: float = 100.0
    max_health: float = 100.0
    weapon_range: float = 100.0
    weapon_cooldown: float = 0.0  # instant fire for testing
    weapon_damage: float = 10.0
    last_fired: float = 0.0
    kills: int = 0
    is_combatant: bool = True
    status: str = "active"
    ammo_count: int = -1  # unlimited
    inventory: object = None

    def can_fire(self) -> bool:
        if self.status not in ("active", "idle", "stationary"):
            return False
        if self.weapon_range <= 0 or self.weapon_damage <= 0:
            return False
        if not self.is_combatant:
            return False
        now = time.time()
        return (now - self.last_fired) >= self.weapon_cooldown

    def apply_damage(self, amount: float) -> bool:
        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.status = "eliminated"
            return True
        return False


class _RecordingBus:
    """Event bus that records every (topic, data) publish for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def by_topic(self, topic: str) -> list[dict]:
        return [d for t, d in self.events if t == topic]


def _tick_to_resolution(cs: CombatSystem, proj: Projectile, targets: dict,
                        dt: float = 0.1, max_ticks: int = 120) -> None:
    """Tick until the projectile hits, misses, or is removed."""
    for _ in range(max_ticks):
        cs.tick(dt, targets)
        if proj.hit or proj.missed or proj.id not in cs._projectiles:
            return


# ===================================================================
# (a) TerrainMap.raycast
# ===================================================================


class TestRaycast:
    def test_clear_map_returns_none(self):
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        assert tm.raycast((0.0, 0.0), (40.0, 0.0)) is None

    def test_wall_between_returns_impact_near_wall(self):
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        tm.set_cell(20.0, 0.0, "building")
        impact = tm.raycast((0.0, 0.0), (40.0, 0.0))
        assert impact is not None
        # Returns the building cell's world-center — within one cell of the wall.
        assert math.hypot(impact[0] - 20.0, impact[1] - 0.0) <= tm.resolution

    def test_adjacent_cell_wall_off_ray_is_clear(self):
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        # Wall one full cell off the horizontal ray — Bresenham along y=0
        # must not report it (only cells actually on the line count).
        tm.set_cell(20.0, 10.0, "building")
        assert tm.raycast((0.0, 0.0), (40.0, 0.0)) is None

    def test_wall_on_diagonal_ray_is_hit(self):
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        tm.set_cell(20.0, 20.0, "building")
        impact = tm.raycast((0.0, 0.0), (40.0, 40.0))
        assert impact is not None
        assert math.hypot(impact[0] - 20.0, impact[1] - 20.0) <= tm.resolution


# ===================================================================
# (b) seeded determinism
# ===================================================================


class TestSeededDeterminism:
    @staticmethod
    def _run_shots(seed: int) -> list[tuple[float, float]]:
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus, rng=random.Random(seed))
        out: list[tuple[float, float]] = []
        for i in range(10):
            src = _FakeTarget(target_id="s", position=(0.0, 0.0))
            tgt = _FakeTarget(
                target_id=f"t{i}", position=(30.0, 5.0),
                alliance="hostile", heading=90.0, speed=4.0,
            )
            proj = cs.fire(src, tgt)
            assert proj is not None
            out.append(proj.target_pos)
            cs.clear()
        return out

    def test_same_seed_same_dispersion(self):
        a = self._run_shots(42)
        b = self._run_shots(42)
        assert a == b

    def test_different_seed_differs(self):
        a = self._run_shots(1)
        b = self._run_shots(2)
        assert a != b


# ===================================================================
# (c) accuracy 1.0 -> no dispersion
# ===================================================================


class TestPerfectAccuracy:
    def test_accuracy_one_hits_aim_point_exactly(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("s", Weapon(
            name="perfect", accuracy=1.0, damage=10.0,
            weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws, rng=random.Random(9))
        src = _FakeTarget(target_id="s", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t", position=(25.0, 0.0), alliance="hostile")
        proj = cs.fire(src, tgt, aim_pos=(25.0, 3.0))
        assert proj is not None
        # Perfect accuracy -> the dispersed aim point IS the intended point.
        assert proj.target_pos == (25.0, 3.0)
        fired = bus.by_topic("projectile_fired")[0]
        assert fired["aim_error_deg"] == 0.0


# ===================================================================
# (d) calibration — hit fraction tracks weapon accuracy
# ===================================================================


class TestDispersionCalibration:
    def test_accuracy_half_hits_about_half(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("s", Weapon(
            name="half", accuracy=0.5, damage=1.0,
            weapon_range=100.0, ammo=1_000_000, max_ammo=1_000_000,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws, rng=random.Random(7))
        trials = 300
        hits = 0
        for _ in range(trials):
            src = _FakeTarget(target_id="s", position=(0.0, 0.0))
            tgt = _FakeTarget(
                target_id="t", position=(20.0, 0.0),
                alliance="hostile", speed=0.0, health=1e9,
            )
            proj = cs.fire(src, tgt)
            assert proj is not None
            _tick_to_resolution(cs, proj, {"s": src, "t": tgt})
            if proj.hit:
                hits += 1
            cs.clear()
        frac = hits / trials
        assert 0.35 <= frac <= 0.65, f"hit fraction {frac} outside calibrated band"


# ===================================================================
# (e) fire-time LOS via the stored terrain map
# ===================================================================


class TestFireTimeLOS:
    def _walled_map(self) -> TerrainMap:
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        tm.set_cell(20.0, 0.0, "building")
        return tm

    def test_ground_unit_blocked(self):
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus)
        cs.set_terrain_map(self._walled_map())
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t", position=(40.0, 0.0), alliance="hostile")
        assert cs.fire(rover, tgt) is None

    def test_mortar_capable_beyond_30pct_fires_over_wall(self):
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus)
        cs.set_terrain_map(self._walled_map())
        # Turret is mortar-capable; 40m > 30% of 100m range -> indirect arc.
        turret = _FakeTarget(
            target_id="q", asset_type="turret", position=(0.0, 0.0),
            weapon_range=100.0,
        )
        tgt = _FakeTarget(target_id="t", position=(40.0, 0.0), alliance="hostile")
        proj = cs.fire(turret, tgt)
        assert proj is not None
        assert proj.is_mortar is True


# ===================================================================
# (f) aerial exemption
# ===================================================================


class TestAerialExemption:
    def test_drone_source_fires_over_wall(self):
        bus = _RecordingBus()
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        tm.set_cell(20.0, 0.0, "building")
        cs = CombatSystem(event_bus=bus)
        cs.set_terrain_map(tm)
        drone = _FakeTarget(target_id="d", asset_type="drone", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t", position=(40.0, 0.0), alliance="hostile")
        proj = cs.fire(drone, tgt)
        assert proj is not None
        assert proj.aerial is True


# ===================================================================
# (g) in-flight occlusion
# ===================================================================


class TestInFlightOcclusion:
    def test_projectile_impacts_building_mid_flight(self):
        bus = _RecordingBus()
        tm = TerrainMap(map_bounds=100.0, resolution=5.0)
        ws = WeaponSystem()
        # accuracy 1.0 -> deterministic straight flight through the wall cell.
        ws.assign_weapon("r", Weapon(
            name="rifle", accuracy=1.0, damage=10.0,
            weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws)
        cs.set_terrain_map(tm)
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(40.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        # Clear LOS at fire time.
        proj = cs.fire(rover, tgt)
        assert proj is not None
        assert proj.hit is False
        # Drop a wall into the flight path AFTER firing.
        tm.set_cell(20.0, 0.0, "building")
        cs.tick(0.5, {"r": rover, "t": tgt})  # 40m step crosses the wall
        impacts = bus.by_topic("projectile_impact")
        assert len(impacts) == 1
        impact = impacts[0]
        assert impact["surface"] == "building"
        assert impact["projectile_id"] == proj.id
        assert impact["source_id"] == "r"
        # Shot stopped at the wall — target unharmed, projectile gone.
        assert tgt.health == 100.0
        assert proj.id not in cs._projectiles


# ===================================================================
# (h) unguided direct fire
# ===================================================================


class TestUnguidedDirectFire:
    def test_ballistic_round_does_not_follow_teleporting_target(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("r", Weapon(
            name="rifle", accuracy=1.0, damage=10.0,
            weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws)
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(rover, tgt)
        assert proj is not None
        assert proj.guided is False
        # Target teleports 30m sideways AFTER the shot commits.
        tgt.position = (20.0, 30.0)
        _tick_to_resolution(cs, proj, {"r": rover, "t": tgt}, dt=0.5, max_ticks=30)
        assert tgt.health == 100.0  # never hit — ballistic round flew straight
        assert proj.hit is False
        assert proj.missed is True
        assert proj.id not in cs._projectiles


# ===================================================================
# (i) guided missile fire
# ===================================================================


class TestGuidedMissile:
    def test_missile_homes_to_moved_target(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("r", Weapon(
            name="missile", weapon_class="missile", accuracy=1.0,
            damage=10.0, weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws)
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(rover, tgt)
        assert proj is not None
        assert proj.guided is True
        # Target moves; the homing munition tracks its live position.
        tgt.position = (20.0, 30.0)
        _tick_to_resolution(cs, proj, {"r": rover, "t": tgt}, dt=0.1, max_ticks=60)
        assert proj.hit is True
        assert tgt.health < 100.0


# ===================================================================
# (j) landed-dart terminal window
# ===================================================================


class TestTerminalWindow:
    """A spent ballistic round lingers on its impact point only briefly.

    An UNGUIDED direct-fire round (and a landed mortar) that reaches its
    committed aim point is physically spent.  It stays live for
    ``TERMINAL_WINDOW`` sim-seconds — long enough to clip a mover crossing the
    impact point right then — after which it is removed.  A walker who strolls
    onto the landing point later takes NO damage (no ghost darts on the floor).
    Guided (missile) rounds home and never "arrive", so they are unaffected.
    """

    @staticmethod
    def _rifle_ws() -> WeaponSystem:
        # accuracy 1.0 -> zero dispersion, so the committed aim point is exact.
        ws = WeaponSystem()
        ws.assign_weapon("r", Weapon(
            name="rifle", accuracy=1.0, damage=10.0,
            weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        return ws

    def test_dart_lands_and_mover_arriving_later_takes_no_damage(self):
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus, weapon_system=self._rifle_ws())
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        # Commit the shot to (20, 0); accuracy 1.0 means no dispersion.
        proj = cs.fire(rover, tgt, aim_pos=(20.0, 0.0))
        assert proj is not None
        assert proj.guided is False
        # Target dodges clear before the dart arrives — it lands on empty ground.
        tgt.position = (20.0, 40.0)
        # Fly the dart to its committed landing point and let the fuse expire.
        for _ in range(5):  # 2.5 sim-seconds at dt=0.5
            cs.tick(0.5, {"r": rover, "t": tgt})
        assert proj.arrived_at is not None  # stamped on arrival
        assert proj.missed is True          # spent after the terminal window
        assert proj.id not in cs._projectiles
        # ~2s after the round landed, the target strolls onto the impact point.
        # The dart is long gone, so it takes no damage.
        tgt.position = (20.0, 0.0)
        cs.tick(0.5, {"r": rover, "t": tgt})
        assert tgt.health == 100.0

    def test_mover_within_window_is_still_clipped(self):
        # Pins the pre-existing behavior: a near-miss can still connect if the
        # target crosses the impact point while the fuse is still live.
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus, weapon_system=self._rifle_ws())
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 40.0),  # off the flight path — no early hit
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(rover, tgt, aim_pos=(20.0, 0.0))
        assert proj is not None
        # Fly to the landing point (3 ticks of dt=0.1: 8, 16, 20 -> arrives).
        for _ in range(3):
            cs.tick(0.1, {"r": rover, "t": tgt})
        assert proj.arrived_at is not None
        assert proj.hit is False
        # WITHIN the terminal window, the target crosses the impact point.
        tgt.position = (20.0, 0.0)
        cs.tick(0.1, {"r": rover, "t": tgt})  # 0.1s after arrival, still fresh
        assert proj.hit is True
        assert tgt.health < 100.0

    def test_landed_mortar_round_expires(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("m", Weapon(
            name="mortar", accuracy=1.0, damage=10.0,
            weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws)
        turret = _FakeTarget(
            target_id="m", asset_type="turret", position=(0.0, 0.0),
            weapon_range=100.0,
        )
        tgt = _FakeTarget(
            target_id="t", position=(40.0, 0.0),  # 40m > 30% of range -> mortar arc
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(turret, tgt)
        assert proj is not None
        assert proj.is_mortar is True
        assert proj.guided is False
        # Target clears the impact point before the round lands.
        tgt.position = (40.0, 40.0)
        for _ in range(6):  # 3 sim-seconds at dt=0.5
            cs.tick(0.5, {"m": turret, "t": tgt})
        assert proj.arrived_at is not None
        assert proj.missed is True
        assert proj.id not in cs._projectiles
        # A spent mortar round is gone — a mover onto the crater is safe.
        tgt.position = (40.0, 0.0)
        cs.tick(0.5, {"m": turret, "t": tgt})
        assert tgt.health == 100.0

    def test_guided_missile_unaffected_by_terminal_window(self):
        bus = _RecordingBus()
        ws = WeaponSystem()
        ws.assign_weapon("r", Weapon(
            name="missile", weapon_class="missile", accuracy=1.0,
            damage=10.0, weapon_range=100.0, ammo=10, max_ammo=10,
        ))
        cs = CombatSystem(event_bus=bus, weapon_system=ws)
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(rover, tgt)
        assert proj is not None
        assert proj.guided is True
        # Target relocates; the missile homes for ~2 sim-seconds and connects.
        tgt.position = (20.0, 30.0)
        for _ in range(20):  # up to 2.0 sim-seconds at dt=0.1
            cs.tick(0.1, {"r": rover, "t": tgt})
            if proj.hit or proj.id not in cs._projectiles:
                break
        # A guided round never "arrives" at a committed point — no terminal fuse.
        assert proj.arrived_at is None
        assert proj.hit is True
        assert tgt.health < 100.0

    def test_spent_dart_gone_from_active_projectiles(self):
        # The frontend must not render a lingering dart after it is spent.
        bus = _RecordingBus()
        cs = CombatSystem(event_bus=bus, weapon_system=self._rifle_ws())
        rover = _FakeTarget(target_id="r", asset_type="rover", position=(0.0, 0.0))
        tgt = _FakeTarget(
            target_id="t", position=(20.0, 0.0),
            alliance="hostile", speed=0.0, health=100.0,
        )
        proj = cs.fire(rover, tgt, aim_pos=(20.0, 0.0))
        assert proj is not None
        tgt.position = (20.0, 40.0)  # dodge -> lands on empty ground
        for _ in range(3):  # fly to the impact point
            cs.tick(0.1, {"r": rover, "t": tgt})
        assert proj.arrived_at is not None
        # Just arrived, still inside the window -> frontend still renders it.
        assert proj.id in {p["id"] for p in cs.get_active_projectiles()}
        # Advance past the terminal window; the spent dart is dropped promptly.
        for _ in range(5):
            cs.tick(0.1, {"r": rover, "t": tgt})
        assert proj.id not in {p["id"] for p in cs.get_active_projectiles()}
        assert proj.id not in cs._projectiles
