# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.combat — weapons, combat, squads."""

import math
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from tritium_lib.sim_engine.combat import (
    CombatSystem,
    HIT_RADIUS,
    MISS_OVERSHOOT,
    Projectile,
    Squad,
    SquadManager,
    SQUAD_RADIUS,
    FORMATION_SPACING,
    Weapon,
    WeaponSystem,
    WEAPON_CATALOG,
)


# ---------------------------------------------------------------------------
# Lightweight SimulationTarget stand-in for tests.
# We avoid importing the real SimulationTarget because its __post_init__
# pulls in heavy optional dependencies (inventory, movement controller,
# engine.units) that are not available in the lib test environment.
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
    speed: float = 5.0
    health: float = 100.0
    max_health: float = 100.0
    weapon_range: float = 30.0
    weapon_cooldown: float = 0.0  # instant fire for testing
    weapon_damage: float = 10.0
    last_fired: float = 0.0
    kills: int = 0
    is_combatant: bool = True
    status: str = "active"
    ammo_count: int = -1  # unlimited
    squad_id: str | None = None
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    _waypoint_index: int = 0
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


def _make_event_bus():
    """Return a MagicMock that records publish() calls."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


# ===================================================================
# Weapon dataclass tests
# ===================================================================


class TestWeapon:
    def test_default_values(self):
        w = Weapon()
        assert w.name == "nerf_blaster"
        assert w.damage == 10.0
        assert w.weapon_range == 15.0
        assert w.cooldown == 2.0
        assert w.accuracy == 0.85
        assert w.ammo == 30
        assert w.max_ammo == 30
        assert w.weapon_class == "ballistic"
        assert w.blast_radius == 0.0

    def test_custom_weapon(self):
        w = Weapon(name="railgun", damage=100.0, weapon_range=50.0,
                   weapon_class="beam", accuracy=1.0, ammo=5, max_ammo=5)
        assert w.name == "railgun"
        assert w.damage == 100.0
        assert w.weapon_class == "beam"

    def test_weapon_catalog_populated(self):
        assert len(WEAPON_CATALOG) >= 4
        assert "nerf_rifle" in WEAPON_CATALOG
        assert "nerf_shotgun" in WEAPON_CATALOG
        assert "nerf_rpg" in WEAPON_CATALOG
        assert "nerf_smg" in WEAPON_CATALOG

    def test_catalog_rpg_is_missile_class(self):
        rpg = WEAPON_CATALOG["nerf_rpg"]
        assert rpg.weapon_class == "missile"
        assert rpg.damage == 60.0


# ===================================================================
# WeaponSystem tests
# ===================================================================


class TestWeaponSystem:
    def test_equip_known_type(self):
        ws = WeaponSystem()
        ws.equip("u1", "turret")
        w = ws.get_weapon("u1")
        assert w is not None
        assert w.name == "nerf_turret_gun"
        assert w.damage == 15.0

    def test_equip_unknown_type_gets_default(self):
        ws = WeaponSystem()
        ws.equip("u2", "unknown_thing")
        w = ws.get_weapon("u2")
        assert w is not None
        # Should get a generic Weapon()
        assert w.name == "nerf_blaster"

    def test_equip_copies_weapon(self):
        """Two units of the same type should NOT share weapon state."""
        ws = WeaponSystem()
        ws.equip("a", "turret")
        ws.equip("b", "turret")
        wa = ws.get_weapon("a")
        wb = ws.get_weapon("b")
        assert wa is not wb
        wa.ammo = 0
        assert wb.ammo > 0

    def test_consume_ammo(self):
        ws = WeaponSystem()
        ws.equip("u1", "drone")
        initial = ws.get_ammo("u1")
        assert initial == 20
        assert ws.consume_ammo("u1") is True
        assert ws.get_ammo("u1") == 19

    def test_consume_ammo_until_empty(self):
        ws = WeaponSystem()
        w = Weapon(name="tiny", ammo=2, max_ammo=2, damage=1.0)
        ws.assign_weapon("u1", w)
        assert ws.consume_ammo("u1") is True   # 2 -> 1
        assert ws.consume_ammo("u1") is True   # 1 -> 0
        assert ws.consume_ammo("u1") is False  # empty

    def test_consume_ammo_fires_event_on_depleted(self):
        bus = _make_event_bus()
        ws = WeaponSystem(event_bus=bus)
        w = Weapon(name="tiny", ammo=1, max_ammo=1, damage=1.0)
        ws.assign_weapon("u1", w)
        ws.consume_ammo("u1")
        bus.publish.assert_called_once()
        args = bus.publish.call_args
        assert args[0][0] == "ammo_depleted"

    def test_consume_ammo_fires_ammo_low_event(self):
        bus = _make_event_bus()
        ws = WeaponSystem(event_bus=bus)
        w = Weapon(name="test", ammo=3, max_ammo=20, damage=1.0)
        ws.assign_weapon("u1", w)
        ws.consume_ammo("u1")  # 3 -> 2, 2/20 = 0.1 < 0.2 => ammo_low
        bus.publish.assert_called_once()
        assert bus.publish.call_args[0][0] == "ammo_low"

    def test_no_weapon_consume_returns_true(self):
        """Legacy behavior: no weapon = infinite ammo."""
        ws = WeaponSystem()
        assert ws.consume_ammo("nonexistent") is True

    def test_get_ammo_pct(self):
        ws = WeaponSystem()
        ws.equip("u1", "turret")  # 100/100
        assert ws.get_ammo_pct("u1") == 1.0
        ws.consume_ammo("u1")
        assert ws.get_ammo_pct("u1") == pytest.approx(0.99)

    def test_get_ammo_pct_no_weapon(self):
        ws = WeaponSystem()
        assert ws.get_ammo_pct("none") == 1.0

    def test_reload_tick(self):
        ws = WeaponSystem()
        w = Weapon(name="tiny", ammo=1, max_ammo=10, damage=1.0)
        ws.assign_weapon("u1", w)
        ws.consume_ammo("u1")  # now 0

        # Tick should start reload
        ws.tick(0.1)
        assert ws.is_reloading("u1") is True

        # Tick through the reload duration (default 3.0s)
        ws.tick(3.0)
        assert ws.is_reloading("u1") is False
        assert ws.get_ammo("u1") == 10  # fully reloaded

    def test_assign_default_weapon_person_hostile(self):
        ws = WeaponSystem()
        ws.assign_default_weapon("u1", "person", alliance="hostile")
        w = ws.get_weapon("u1")
        assert w is not None
        assert w.name == "nerf_pistol"

    def test_remove_unit(self):
        ws = WeaponSystem()
        ws.equip("u1", "turret")
        ws.remove_unit("u1")
        assert ws.get_weapon("u1") is None

    def test_reset(self):
        ws = WeaponSystem()
        ws.equip("u1", "turret")
        ws.equip("u2", "drone")
        ws.reset()
        assert ws.get_weapon("u1") is None
        assert ws.get_weapon("u2") is None


# ===================================================================
# Projectile tests
# ===================================================================


class TestProjectile:
    def test_z_height_flat_projectile(self):
        p = Projectile(
            id="p1", source_id="s", source_name="S", target_id="t",
            position=(0.0, 0.0), target_pos=(10.0, 0.0),
        )
        assert p.z_height == 0.0  # not a mortar

    def test_z_height_mortar_at_midpoint(self):
        p = Projectile(
            id="p1", source_id="s", source_name="S", target_id="t",
            position=(5.0, 0.0), target_pos=(10.0, 0.0),
            is_mortar=True, arc_peak=20.0,
            flight_progress=0.5,
        )
        # At midpoint: z = 4 * 20 * 0.5 * 0.5 = 20.0
        assert p.z_height == pytest.approx(20.0)

    def test_z_height_mortar_at_endpoints(self):
        p = Projectile(
            id="p1", source_id="s", source_name="S", target_id="t",
            position=(0.0, 0.0), target_pos=(10.0, 0.0),
            is_mortar=True, arc_peak=15.0,
        )
        p.flight_progress = 0.0
        assert p.z_height == pytest.approx(0.0)
        p.flight_progress = 1.0
        assert p.z_height == pytest.approx(0.0)

    def test_to_dict_basic(self):
        p = Projectile(
            id="p1", source_id="s", source_name="Shooter", target_id="t",
            position=(1.0, 2.0), target_pos=(10.0, 20.0),
            damage=15.0,
        )
        d = p.to_dict()
        assert d["id"] == "p1"
        assert d["position"] == {"x": 1.0, "y": 2.0}
        assert d["target_pos"] == {"x": 10.0, "y": 20.0}
        assert d["damage"] == 15.0
        assert "is_mortar" not in d  # non-mortar doesn't include mortar fields

    def test_to_dict_mortar(self):
        p = Projectile(
            id="p1", source_id="s", source_name="Shooter", target_id="t",
            position=(0.0, 0.0), target_pos=(10.0, 0.0),
            is_mortar=True, arc_peak=12.0, flight_progress=0.25,
        )
        d = p.to_dict()
        assert d["is_mortar"] is True
        assert "z_height" in d
        assert "arc_peak" in d
        assert "flight_progress" in d


# ===================================================================
# CombatSystem tests
# ===================================================================


class TestCombatSystem:
    def test_fire_creates_projectile(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t1", position=(10.0, 0.0), alliance="hostile")

        proj = cs.fire(src, tgt)
        assert proj is not None
        assert proj.source_id == "s1"
        assert proj.target_id == "t1"
        assert cs.projectile_count == 1

    def test_fire_publishes_event(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t1", position=(10.0, 0.0), alliance="hostile")

        cs.fire(src, tgt)
        bus.publish.assert_called()
        event_name = bus.publish.call_args_list[0][0][0]
        assert event_name == "projectile_fired"

    def test_fire_respects_range(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0), weapon_range=5.0)
        tgt = _FakeTarget(target_id="t1", position=(100.0, 0.0), alliance="hostile")

        proj = cs.fire(src, tgt)
        assert proj is None  # Out of range

    def test_fire_respects_can_fire(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", status="eliminated")
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0))

        proj = cs.fire(src, tgt)
        assert proj is None

    def test_fire_respects_ammo(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0), ammo_count=0)
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0))

        proj = cs.fire(src, tgt)
        assert proj is None

    def test_fire_decrements_ammo(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0), ammo_count=5)
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0))

        cs.fire(src, tgt)
        assert src.ammo_count == 4

    def test_tick_hit_detection(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0), alliance="hostile",
                          health=100.0)

        proj = cs.fire(src, tgt)
        assert proj is not None

        targets = {"s1": src, "t1": tgt}
        # Move projectile to target's position
        proj.position = tgt.position
        cs.tick(0.0, targets)

        # Projectile should have hit
        assert proj.hit is True
        # Target should have taken damage
        assert tgt.health < 100.0

    def test_tick_elimination(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0), weapon_damage=150.0)
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0), alliance="hostile",
                          health=50.0)

        proj = cs.fire(src, tgt)
        assert proj is not None

        targets = {"s1": src, "t1": tgt}
        proj.position = tgt.position
        cs.tick(0.0, targets)

        assert tgt.status == "eliminated"
        assert src.kills == 1

        # Check target_eliminated event was published
        event_names = [call[0][0] for call in bus.publish.call_args_list]
        assert "target_eliminated" in event_names

    def test_elimination_streak(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0), weapon_damage=200.0)

        targets = {"s1": src}

        # Kill 3 targets for a streak
        for i in range(3):
            tid = f"t{i}"
            tgt = _FakeTarget(target_id=tid, position=(3.0, 0.0), alliance="hostile",
                              health=10.0)
            targets[tid] = tgt
            proj = cs.fire(src, tgt)
            assert proj is not None
            proj.position = tgt.position
            cs.tick(0.0, targets)

        # Check for elimination_streak event
        event_names = [call[0][0] for call in bus.publish.call_args_list]
        assert "elimination_streak" in event_names

    def test_streak_name_thresholds(self):
        assert CombatSystem._get_streak_name(3) == "ON A STREAK"
        assert CombatSystem._get_streak_name(5) == "RAMPAGE"
        assert CombatSystem._get_streak_name(7) == "DOMINATING"
        assert CombatSystem._get_streak_name(10) == "GODLIKE"
        assert CombatSystem._get_streak_name(4) is None
        assert CombatSystem._get_streak_name(1) is None

    def test_reset_streaks(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)
        cs._elimination_streaks["s1"] = 5
        cs.reset_streaks()
        assert len(cs._elimination_streaks) == 0

    def test_reset_streak_single(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)
        cs._elimination_streaks["s1"] = 3
        cs._elimination_streaks["s2"] = 7
        cs.reset_streak("s1")
        assert "s1" not in cs._elimination_streaks
        assert "s2" in cs._elimination_streaks

    def test_get_active_projectiles(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t1", position=(20.0, 0.0), alliance="hostile")

        cs.fire(src, tgt)
        active = cs.get_active_projectiles()
        assert len(active) == 1
        assert active[0]["source_id"] == "s1"

    def test_clear_projectiles(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0))
        tgt = _FakeTarget(target_id="t1", position=(20.0, 0.0), alliance="hostile")

        cs.fire(src, tgt)
        assert cs.projectile_count == 1
        cs.clear()
        assert cs.projectile_count == 0

    def test_detonate_bomber(self):
        bus = _make_event_bus()
        cs = CombatSystem(event_bus=bus)

        bomber = _FakeTarget(target_id="b1", position=(10.0, 10.0),
                             weapon_damage=50.0, asset_type="drone")
        nearby = _FakeTarget(target_id="t1", position=(12.0, 10.0),
                             health=100.0, alliance="hostile")
        far_away = _FakeTarget(target_id="t2", position=(100.0, 100.0),
                               health=100.0, alliance="hostile")

        targets = {"b1": bomber, "t1": nearby, "t2": far_away}
        damaged = cs.detonate_bomber(bomber, targets, radius=5.0)

        assert "t1" in damaged
        assert "t2" not in damaged
        assert nearby.health < 100.0
        assert far_away.health == 100.0
        assert bomber.status == "eliminated"
        assert bomber.health == 0

    def test_weapon_system_integration(self):
        """CombatSystem should use WeaponSystem damage when available."""
        bus = _make_event_bus()
        ws = WeaponSystem()
        cs = CombatSystem(event_bus=bus, weapon_system=ws)

        # Assign a high-damage weapon
        big_gun = Weapon(name="big_gun", damage=99.0, accuracy=1.0,
                         weapon_range=50.0, ammo=10, max_ammo=10)
        ws.assign_weapon("s1", big_gun)

        src = _FakeTarget(target_id="s1", position=(0.0, 0.0),
                          weapon_damage=10.0)  # base damage is low
        tgt = _FakeTarget(target_id="t1", position=(5.0, 0.0), alliance="hostile")

        proj = cs.fire(src, tgt)
        assert proj is not None
        # Projectile should use weapon system damage, not base
        assert proj.damage == 99.0


# ===================================================================
# Squad tests
# ===================================================================


class TestSquad:
    def test_wedge_formation_offsets(self):
        s = Squad(squad_id="sq1", member_ids=["l", "f1", "f2"], leader_id="l",
                  formation="wedge")
        offsets = s.get_formation_offsets()
        assert offsets["l"] == (0.0, 0.0)
        assert "f1" in offsets
        assert "f2" in offsets
        # Followers should be behind (negative y)
        assert offsets["f1"][1] < 0
        assert offsets["f2"][1] < 0

    def test_line_formation_offsets(self):
        s = Squad(squad_id="sq1", member_ids=["l", "f1", "f2"], leader_id="l",
                  formation="line")
        offsets = s.get_formation_offsets()
        assert offsets["l"] == (0.0, 0.0)
        # Line formation: all on same y, spread left/right
        assert offsets["f1"][1] == 0.0
        assert offsets["f2"][1] == 0.0
        # Left and right spread
        assert offsets["f1"][0] != offsets["f2"][0]

    def test_column_formation_offsets(self):
        s = Squad(squad_id="sq1", member_ids=["l", "f1", "f2"], leader_id="l",
                  formation="column")
        offsets = s.get_formation_offsets()
        # Column: all at x=0, spaced behind
        assert offsets["f1"] == (0.0, -FORMATION_SPACING * 1)
        assert offsets["f2"] == (0.0, -FORMATION_SPACING * 2)

    def test_circle_formation_offsets(self):
        s = Squad(squad_id="sq1", member_ids=["l", "f1", "f2", "f3", "f4"],
                  leader_id="l", formation="circle")
        offsets = s.get_formation_offsets()
        # Followers evenly distributed around a circle
        assert len(offsets) == 5  # leader + 4 followers
        # All follower offsets should be at FORMATION_SPACING distance from origin
        for fid in ["f1", "f2", "f3", "f4"]:
            dx, dy = offsets[fid]
            dist = math.hypot(dx, dy)
            assert dist == pytest.approx(FORMATION_SPACING, abs=0.01)

    def test_empty_squad_offsets(self):
        s = Squad(squad_id="sq1", member_ids=[], leader_id=None)
        offsets = s.get_formation_offsets()
        assert offsets == {}


class TestSquadManager:
    def test_get_squad_none(self):
        sm = SquadManager()
        assert sm.get_squad(None) is None
        assert sm.get_squad("nonexistent") is None

    def test_issue_order(self):
        sm = SquadManager()
        squad = Squad(squad_id="sq1", member_ids=["a", "b"], leader_id="a")
        sm._squads["sq1"] = squad

        sm.issue_order("sq1", "hold")
        assert squad.last_order == "hold"

    def test_on_leader_eliminated(self):
        sm = SquadManager()
        squad = Squad(squad_id="sq1", member_ids=["a", "b"], leader_id="a",
                      cohesion=1.0)
        sm._squads["sq1"] = squad

        sm.on_leader_eliminated("sq1")
        assert squad.cohesion == pytest.approx(0.3)
        assert squad.last_order == "retreat"

    def test_promote_new_leader(self):
        sm = SquadManager()
        squad = Squad(squad_id="sq1", member_ids=["a", "b", "c"], leader_id="a")
        sm._squads["sq1"] = squad

        t_b = _FakeTarget(target_id="b", position=(1.0, 0.0), status="active")
        t_c = _FakeTarget(target_id="c", position=(100.0, 100.0), status="active")
        targets = {"b": t_b, "c": t_c}

        # Old leader was at (0, 0), b is closer
        sm.promote_new_leader("sq1", (0.0, 0.0), targets)
        assert squad.leader_id == "b"

    def test_is_leader(self):
        sm = SquadManager()
        squad = Squad(squad_id="sq1", member_ids=["a", "b"], leader_id="a")
        sm._squads["sq1"] = squad

        assert sm.is_leader("a") is True
        assert sm.is_leader("b") is False

    def test_clear(self):
        sm = SquadManager()
        squad = Squad(squad_id="sq1", member_ids=["a", "b"], leader_id="a")
        sm._squads["sq1"] = squad

        t_a = _FakeTarget(target_id="a", squad_id="sq1")
        t_b = _FakeTarget(target_id="b", squad_id="sq1")
        targets = {"a": t_a, "b": t_b}

        sm.clear(targets)
        assert len(sm._squads) == 0
        assert t_a.squad_id is None
        assert t_b.squad_id is None
