# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the naval combat module.

Covers: ShipClass, ShipState, NavalPhysics, Torpedo, NavalCombatEngine,
SHIP_TEMPLATES, NavalFormation, Three.js export, and combat resolution.
"""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.naval import (
    ShipClass,
    ShipState,
    Torpedo,
    ShellProjectile,
    CombatEffect,
    NavalPhysics,
    NavalCombatEngine,
    NavalFormation,
    FormationType,
    SHIP_TEMPLATES,
    create_ship,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ===========================================================================
# ShipClass enum
# ===========================================================================

class TestShipClass:

    def test_all_8_classes_exist(self):
        expected = {
            "PATROL_BOAT", "FRIGATE", "DESTROYER", "CRUISER",
            "CARRIER", "SUBMARINE", "SPEEDBOAT", "CARGO",
        }
        assert {c.name for c in ShipClass} == expected

    def test_values_are_snake_case(self):
        for c in ShipClass:
            assert c.value == c.name.lower()

    def test_from_string(self):
        assert ShipClass("destroyer") == ShipClass.DESTROYER

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError):
            ShipClass("battleship")


# ===========================================================================
# SHIP_TEMPLATES
# ===========================================================================

class TestShipTemplates:

    def test_all_8_classes_have_templates(self):
        for cls in ShipClass:
            assert cls in SHIP_TEMPLATES, f"Missing template for {cls}"

    def test_template_keys(self):
        required = {"max_speed", "turn_rate", "max_health", "armor", "weapons",
                     "radar_range", "sonar_range", "crew"}
        for cls, tmpl in SHIP_TEMPLATES.items():
            assert required.issubset(tmpl.keys()), f"{cls} missing keys: {required - tmpl.keys()}"

    def test_carrier_is_toughest(self):
        carrier = SHIP_TEMPLATES[ShipClass.CARRIER]
        for cls, tmpl in SHIP_TEMPLATES.items():
            if cls != ShipClass.CARRIER:
                assert tmpl["max_health"] <= carrier["max_health"]

    def test_speedboat_is_fastest(self):
        speedboat = SHIP_TEMPLATES[ShipClass.SPEEDBOAT]
        for tmpl in SHIP_TEMPLATES.values():
            assert tmpl["max_speed"] <= speedboat["max_speed"]

    def test_submarine_has_best_sonar(self):
        sub = SHIP_TEMPLATES[ShipClass.SUBMARINE]
        for cls, tmpl in SHIP_TEMPLATES.items():
            if cls != ShipClass.SUBMARINE:
                assert tmpl["sonar_range"] <= sub["sonar_range"]

    def test_cargo_has_no_weapons(self):
        assert SHIP_TEMPLATES[ShipClass.CARGO]["weapons"] == []


# ===========================================================================
# create_ship factory
# ===========================================================================

class TestCreateShip:

    def test_creates_destroyer(self):
        ship = create_ship(ShipClass.DESTROYER, "USS Test", "blue")
        assert ship.name == "USS Test"
        assert ship.alliance == "blue"
        assert ship.ship_class == ShipClass.DESTROYER
        assert ship.speed == 0.0
        assert ship.health == ship.max_health

    def test_custom_position_and_heading(self):
        ship = create_ship(ShipClass.FRIGATE, "F1", "red", position=(100, 200), heading=1.5)
        assert ship.position == (100, 200)
        assert ship.heading == 1.5

    def test_custom_id(self):
        ship = create_ship(ShipClass.CRUISER, "C1", "blue", ship_id="my_cruiser")
        assert ship.ship_id == "my_cruiser"

    def test_auto_id_prefix(self):
        ship = create_ship(ShipClass.CARRIER, "CV1", "blue")
        assert ship.ship_id.startswith("ship_")

    def test_health_matches_template(self):
        for cls in ShipClass:
            ship = create_ship(cls, "test", "x")
            assert ship.max_health == SHIP_TEMPLATES[cls]["max_health"]
            assert ship.health == ship.max_health


# ===========================================================================
# ShipState
# ===========================================================================

class TestShipState:

    def test_is_alive_positive_health(self):
        ship = create_ship(ShipClass.PATROL_BOAT, "PB1", "blue")
        assert ship.is_alive()

    def test_is_alive_zero_health(self):
        ship = create_ship(ShipClass.PATROL_BOAT, "PB1", "blue")
        ship.health = 0
        assert not ship.is_alive()

    def test_health_pct(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue")
        assert ship.health_pct() == pytest.approx(1.0)
        ship.health = ship.max_health / 2
        assert ship.health_pct() == pytest.approx(0.5)

    def test_health_pct_zero_max(self):
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue")
        ship.max_health = 0
        assert ship.health_pct() == 0.0

    def test_default_depth(self):
        ship = create_ship(ShipClass.SUBMARINE, "Sub1", "red")
        assert ship.depth == 0.0
        assert not ship.is_submerged


# ===========================================================================
# NavalPhysics
# ===========================================================================

class TestNavalPhysics:

    def test_stationary_ship_no_movement(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", position=(0, 0))
        NavalPhysics.update(ship, throttle=0.0, rudder=0.0, dt=1.0)
        # Ship stays near origin (may drift slightly from initial speed=0)
        assert distance(ship.position, (0, 0)) < 1.0

    def test_full_throttle_accelerates(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", position=(0, 0), heading=0.0)
        for _ in range(10):
            NavalPhysics.update(ship, throttle=1.0, rudder=0.0, dt=1.0)
        assert ship.speed > 0
        assert ship.position[0] > 0  # moved in +x direction

    def test_reverse_throttle(self):
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue", position=(0, 0), heading=0.0)
        for _ in range(20):
            NavalPhysics.update(ship, throttle=-0.5, rudder=0.0, dt=1.0)
        assert ship.speed < 0
        assert ship.position[0] < 0

    def test_rudder_turns_ship(self):
        ship = create_ship(ShipClass.FRIGATE, "F1", "blue", position=(0, 0), heading=0.0)
        # Get up to speed first
        for _ in range(20):
            NavalPhysics.update(ship, throttle=1.0, rudder=0.0, dt=1.0)
        initial_heading = ship.heading
        # Turn right
        for _ in range(5):
            NavalPhysics.update(ship, throttle=1.0, rudder=1.0, dt=1.0)
        # Heading should have changed
        assert ship.heading != initial_heading

    def test_no_turn_when_stationary(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", heading=0.5)
        ship.speed = 0.0
        NavalPhysics.update(ship, throttle=0.0, rudder=1.0, dt=1.0)
        assert ship.heading == pytest.approx(0.5, abs=0.01)

    def test_wake_intensity_increases_with_speed(self):
        ship = create_ship(ShipClass.CRUISER, "C1", "blue")
        assert ship.wake_intensity == 0.0
        for _ in range(30):
            NavalPhysics.update(ship, throttle=1.0, rudder=0.0, dt=1.0)
        assert ship.wake_intensity > 0.5

    def test_throttle_clamped(self):
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue")
        NavalPhysics.update(ship, throttle=5.0, rudder=0.0, dt=1.0)
        assert ship._throttle == 1.0

    def test_heading_normalized(self):
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue", heading=6.0)
        ship.speed = 10.0
        for _ in range(100):
            NavalPhysics.update(ship, throttle=1.0, rudder=1.0, dt=0.5)
        assert 0.0 <= ship.heading < 2 * math.pi

    def test_momentum_gradual_acceleration(self):
        ship = create_ship(ShipClass.CARRIER, "CV1", "blue")
        NavalPhysics.update(ship, throttle=1.0, rudder=0.0, dt=1.0)
        # Should not instantly reach max speed
        assert ship.speed < ship.max_speed
        assert ship.speed > 0

    def test_deceleration(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue")
        # Accelerate
        for _ in range(50):
            NavalPhysics.update(ship, throttle=1.0, rudder=0.0, dt=1.0)
        fast_speed = ship.speed
        # Cut throttle
        NavalPhysics.update(ship, throttle=0.0, rudder=0.0, dt=1.0)
        assert ship.speed < fast_speed

    def test_calculate_wake_stationary(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue")
        wake = NavalPhysics.calculate_wake(ship)
        assert wake["length"] == pytest.approx(0.0)
        assert wake["foam_intensity"] == pytest.approx(0.0)

    def test_calculate_wake_moving(self):
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue")
        ship.speed = ship.max_speed
        ship.wake_intensity = 1.0
        wake = NavalPhysics.calculate_wake(ship)
        assert wake["length"] > 0
        assert wake["foam_intensity"] > 0
        assert "heading" in wake

    def test_wave_effect_calm(self):
        dx, dy = NavalPhysics.wave_effect((100, 200), sea_state=0.0)
        assert dx == 0.0
        assert dy == 0.0

    def test_wave_effect_storm(self):
        dx, dy = NavalPhysics.wave_effect((100, 200), sea_state=1.0, time=1.0)
        # Should produce non-zero displacement
        assert abs(dx) > 0 or abs(dy) > 0


# ===========================================================================
# Torpedo
# ===========================================================================

class TestTorpedo:

    def test_torpedo_dataclass(self):
        t = Torpedo(
            torpedo_id="t1",
            position=(0, 0),
            heading=0.0,
            speed=25.0,
            target_id="ship_1",
            damage=800.0,
            range_remaining=10000.0,
        )
        assert t.is_active
        assert t.target_id == "ship_1"
        assert t._trail == []


# ===========================================================================
# CombatEffect
# ===========================================================================

class TestCombatEffect:

    def test_to_dict(self):
        e = CombatEffect("explosion", (10.5, 20.3), radius=15.0, duration=2.0, intensity=0.8)
        d = e.to_dict()
        assert d["type"] == "explosion"
        assert d["x"] == 10.5
        assert d["y"] == 20.3
        assert d["radius"] == 15.0


# ===========================================================================
# NavalFormation
# ===========================================================================

class TestNavalFormation:

    def test_line_ahead_count_3(self):
        offsets = NavalFormation.get_offsets(FormationType.LINE_AHEAD, 3, spacing=100)
        assert len(offsets) == 3
        assert offsets[0] == (0.0, 0.0)
        # Others should be behind (negative x)
        assert offsets[1][0] < 0
        assert offsets[2][0] < offsets[1][0]

    def test_line_abreast_spread(self):
        offsets = NavalFormation.get_offsets(FormationType.LINE_ABREAST, 5, spacing=100)
        assert len(offsets) == 5
        # Leader at origin
        assert offsets[0] == (0.0, 0.0)
        # Others spread laterally (x ~= 0)
        for off in offsets[1:]:
            assert off[0] == pytest.approx(0.0)
            assert off[1] != 0.0

    def test_diamond_count_4(self):
        offsets = NavalFormation.get_offsets(FormationType.DIAMOND, 4, spacing=200)
        assert len(offsets) == 4
        assert offsets[0] == (0.0, 0.0)

    def test_screen_semicircle(self):
        offsets = NavalFormation.get_offsets(FormationType.SCREEN, 4, spacing=300)
        assert len(offsets) == 4
        # Leader at origin
        assert offsets[0] == (0.0, 0.0)
        # Escorts should be ahead of origin (positive x component in forward arc)
        for off in offsets[1:]:
            assert off[0] > 0, f"Screen escort should be ahead of leader, got x={off[0]}"

    def test_wedge_v_shape(self):
        offsets = NavalFormation.get_offsets(FormationType.WEDGE, 5, spacing=150)
        assert len(offsets) == 5
        assert offsets[0] == (0.0, 0.0)
        # Check V shape: later offsets are further back and wider
        for off in offsets[1:]:
            assert off[0] < 0  # behind leader

    def test_empty_formation(self):
        offsets = NavalFormation.get_offsets(FormationType.LINE_AHEAD, 0)
        assert offsets == []

    def test_single_ship(self):
        offsets = NavalFormation.get_offsets(FormationType.DIAMOND, 1)
        assert offsets == [(0.0, 0.0)]

    def test_world_positions_heading_zero(self):
        positions = NavalFormation.world_positions(
            leader_pos=(1000, 2000),
            leader_heading=0.0,
            formation=FormationType.LINE_AHEAD,
            count=3,
            spacing=100,
        )
        assert len(positions) == 3
        # Leader at (1000, 2000)
        assert positions[0] == pytest.approx((1000, 2000), abs=0.1)
        # Followers behind at heading 0 means negative x
        assert positions[1][0] < positions[0][0]

    def test_world_positions_heading_rotated(self):
        # Heading pi/2 means "up" (+y). Line ahead should have followers
        # in -y direction (behind the leader in local frame = -x, rotated to -y)
        positions = NavalFormation.world_positions(
            leader_pos=(0, 0),
            leader_heading=math.pi / 2,
            formation=FormationType.LINE_AHEAD,
            count=2,
            spacing=100,
        )
        assert len(positions) == 2
        # Follower should be below (negative y) the leader
        assert positions[1][1] < positions[0][1] - 50

    def test_diamond_overflow(self):
        # More ships than diamond slots
        offsets = NavalFormation.get_offsets(FormationType.DIAMOND, 12, spacing=200)
        assert len(offsets) == 12


# ===========================================================================
# NavalCombatEngine — setup and ship management
# ===========================================================================

class TestNavalCombatEngineSetup:

    def test_init_defaults(self):
        engine = NavalCombatEngine()
        assert engine.sea_state == 0.3
        assert engine.ships == []
        assert engine.torpedoes == []

    def test_custom_sea_state(self):
        engine = NavalCombatEngine(sea_state=0.7)
        assert engine.sea_state == 0.7

    def test_sea_state_clamped(self):
        engine = NavalCombatEngine(sea_state=5.0)
        assert engine.sea_state == 1.0
        engine2 = NavalCombatEngine(sea_state=-1.0)
        assert engine2.sea_state == 0.0

    def test_add_and_get_ship(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1")
        engine.add_ship(ship)
        assert engine.get_ship("d1") is ship

    def test_get_nonexistent_ship(self):
        engine = NavalCombatEngine()
        assert engine.get_ship("nope") is None

    def test_remove_ship(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.FRIGATE, "F1", "red", ship_id="f1")
        engine.add_ship(ship)
        assert engine.remove_ship("f1")
        assert engine.get_ship("f1") is None

    def test_remove_nonexistent(self):
        engine = NavalCombatEngine()
        assert not engine.remove_ship("nope")

    def test_ships_by_alliance(self):
        engine = NavalCombatEngine()
        engine.add_ship(create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1"))
        engine.add_ship(create_ship(ShipClass.FRIGATE, "F1", "red", ship_id="f1"))
        engine.add_ship(create_ship(ShipClass.CRUISER, "C1", "blue", ship_id="c1"))
        blue = engine.ships_by_alliance("blue")
        assert len(blue) == 2
        assert all(s.alliance == "blue" for s in blue)

    def test_set_ship_controls(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue", ship_id="s1")
        engine.add_ship(ship)
        assert engine.set_ship_controls("s1", throttle=0.8, rudder=-0.5)
        assert ship._throttle == pytest.approx(0.8)
        assert ship._rudder == pytest.approx(-0.5)

    def test_set_controls_nonexistent(self):
        engine = NavalCombatEngine()
        assert not engine.set_ship_controls("nope", throttle=1.0)


# ===========================================================================
# NavalCombatEngine — submarine operations
# ===========================================================================

class TestSubmarineOps:

    def test_submerge_submarine(self):
        engine = NavalCombatEngine()
        sub = create_ship(ShipClass.SUBMARINE, "Sub1", "red", ship_id="sub1")
        engine.add_ship(sub)
        assert engine.submerge("sub1", depth=-100.0)
        assert sub.is_submerged
        assert sub.depth == -100.0

    def test_surface_submarine(self):
        engine = NavalCombatEngine()
        sub = create_ship(ShipClass.SUBMARINE, "Sub1", "red", ship_id="sub1")
        engine.add_ship(sub)
        engine.submerge("sub1")
        assert engine.surface("sub1")
        assert not sub.is_submerged
        assert sub.depth == 0.0

    def test_cannot_submerge_destroyer(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1")
        engine.add_ship(ship)
        assert not engine.submerge("d1")

    def test_cannot_surface_nonexistent(self):
        engine = NavalCombatEngine()
        assert not engine.surface("nope")


# ===========================================================================
# NavalCombatEngine — torpedoes
# ===========================================================================

class TestTorpedoFiring:

    def test_fire_torpedo_homing(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.SUBMARINE, "Sub1", "blue", ship_id="sub1")
        engine.add_ship(ship)
        torp = engine.fire_torpedo("sub1", target_id="enemy1")
        assert torp is not None
        assert torp.target_id == "enemy1"
        assert torp.is_active
        assert len(engine.torpedoes) == 1

    def test_fire_torpedo_dumbfire(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.FRIGATE, "F1", "blue", ship_id="f1")
        engine.add_ship(ship)
        torp = engine.fire_torpedo("f1", heading=1.0)
        assert torp is not None
        assert torp.target_id is None
        assert torp.heading == pytest.approx(1.0)

    def test_fire_torpedo_no_tubes(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.CARGO, "Cargo1", "blue", ship_id="cargo1")
        engine.add_ship(ship)
        torp = engine.fire_torpedo("cargo1")
        assert torp is None

    def test_fire_torpedo_dead_ship(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.SUBMARINE, "Sub1", "blue", ship_id="sub1")
        ship.health = 0
        engine.add_ship(ship)
        assert engine.fire_torpedo("sub1") is None

    def test_torpedo_homing_tracks_target(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        sub = create_ship(ShipClass.SUBMARINE, "Sub1", "blue",
                          ship_id="sub1", position=(0, 0), heading=0.0)
        target = create_ship(ShipClass.DESTROYER, "D1", "red",
                             ship_id="d1", position=(0, 500))
        engine.add_ship(sub)
        engine.add_ship(target)
        torp = engine.fire_torpedo("sub1", target_id="d1")

        # Tick several times — torpedo should turn toward target
        for _ in range(10):
            engine.tick(1.0)

        # Torpedo should be heading roughly toward target (positive y)
        remaining = [t for t in engine.torpedoes if t.torpedo_id == torp.torpedo_id]
        if remaining:
            # Heading should be pointing upward (toward y=500)
            heading = remaining[0].heading
            assert math.sin(heading) > 0  # positive y component


# ===========================================================================
# NavalCombatEngine — guns
# ===========================================================================

class TestGunFire:

    def test_fire_guns_creates_shells(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue",
                           ship_id="d1", position=(0, 0))
        engine.add_ship(ship)
        shells = engine.fire_guns("d1", target_pos=(1000, 0))
        assert len(shells) > 0
        assert all(isinstance(s, ShellProjectile) for s in shells)

    def test_fire_guns_out_of_range(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.PATROL_BOAT, "PB1", "blue",
                           ship_id="pb1", position=(0, 0))
        engine.add_ship(ship)
        # Target very far away — beyond all weapons
        shells = engine.fire_guns("pb1", target_pos=(1_000_000, 0))
        assert len(shells) == 0

    def test_fire_guns_dead_ship(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1")
        ship.health = 0
        engine.add_ship(ship)
        assert engine.fire_guns("d1", (100, 0)) == []

    def test_gun_cooldown(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue",
                           ship_id="d1", position=(0, 0))
        engine.add_ship(ship)
        shells1 = engine.fire_guns("d1", target_pos=(1000, 0))
        # Immediately fire again — should be on cooldown
        shells2 = engine.fire_guns("d1", target_pos=(1000, 0))
        assert len(shells2) < len(shells1)


# ===========================================================================
# NavalCombatEngine — tick
# ===========================================================================

class TestTick:

    def test_basic_tick_returns_dict(self):
        engine = NavalCombatEngine()
        result = engine.tick(1.0)
        assert "time" in result
        assert "torpedo_hits" in result
        assert "shell_hits" in result
        assert "sunk" in result
        assert "effects" in result

    def test_tick_moves_ships(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue",
                           ship_id="d1", position=(0, 0), heading=0.0)
        engine.add_ship(ship)
        engine.set_ship_controls("d1", throttle=1.0)
        for _ in range(20):
            engine.tick(1.0)
        assert ship.position[0] > 0

    def test_torpedo_hit_damages_ship(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        sub = create_ship(ShipClass.SUBMARINE, "Sub1", "blue",
                          ship_id="sub1", position=(0, 0), heading=0.0)
        target = create_ship(ShipClass.DESTROYER, "D1", "red",
                             ship_id="d1", position=(200, 0))
        engine.add_ship(sub)
        engine.add_ship(target)
        initial_health = target.health

        # Fire torpedo directly at target
        engine.fire_torpedo("sub1", target_id="d1")

        # Tick with small dt so torpedo doesn't overshoot
        hits_found = False
        for _ in range(200):
            result = engine.tick(0.1)
            if result["torpedo_hits"]:
                hits_found = True
                break

        assert hits_found, "Torpedo should hit the target"
        assert target.health < initial_health

    def test_shell_hit_damages_ship(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        attacker = create_ship(ShipClass.DESTROYER, "D1", "blue",
                               ship_id="d1", position=(0, 0))
        target = create_ship(ShipClass.CARGO, "Cargo", "red",
                             ship_id="cargo1", position=(200, 0))
        engine.add_ship(attacker)
        engine.add_ship(target)

        initial_health = target.health
        engine.fire_guns("d1", target_pos=(200, 0))

        # Tick with small dt so shells don't overshoot
        hit_found = False
        for _ in range(100):
            result = engine.tick(0.01)
            if result["shell_hits"]:
                hit_found = True
                break

        assert hit_found, "Shells should hit the target"
        assert target.health < initial_health

    def test_sunk_ship_reported(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "red", ship_id="s1")
        ship.health = 1  # nearly dead
        engine.add_ship(ship)

        # Damage it below 0
        ship.health = -10
        result = engine.tick(0.1)
        assert "s1" in result["sunk"]

    def test_tick_cleans_up_inactive_projectiles(self):
        engine = NavalCombatEngine()
        torp = Torpedo(
            torpedo_id="t1", position=(0, 0), heading=0.0,
            speed=25.0, target_id=None, damage=800, range_remaining=5.0,
        )
        engine.torpedoes.append(torp)
        # Tick long enough for torpedo to exhaust range
        engine.tick(1.0)
        # Should be removed
        assert len(engine.torpedoes) == 0

    def test_time_accumulates(self):
        engine = NavalCombatEngine()
        engine.tick(1.0)
        engine.tick(2.0)
        assert engine._time == pytest.approx(3.0)


# ===========================================================================
# NavalCombatEngine — Three.js export
# ===========================================================================

class TestThreeJsExport:

    def test_to_three_js_structure(self):
        engine = NavalCombatEngine(sea_state=0.5)
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1")
        engine.add_ship(ship)
        data = engine.to_three_js()

        assert "ships" in data
        assert "torpedoes" in data
        assert "shells" in data
        assert "effects" in data
        assert "sea_state" in data
        assert "time" in data
        assert data["sea_state"] == 0.5

    def test_ship_export_fields(self):
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.CRUISER, "C1", "red",
                           ship_id="c1", position=(100, 200), heading=1.5)
        ship.speed = 10.0
        ship.wake_intensity = 0.7
        engine.add_ship(ship)

        data = engine.to_three_js()
        s = data["ships"][0]

        assert s["id"] == "c1"
        assert s["name"] == "C1"
        assert s["x"] == 100.0
        assert s["y"] == 200.0
        assert s["heading"] == pytest.approx(1.5, abs=0.01)
        assert s["class"] == "cruiser"
        assert s["alliance"] == "red"
        assert s["health_pct"] == pytest.approx(1.0)
        assert s["wake"] == pytest.approx(0.7)
        assert s["is_alive"] is True
        assert "wake_data" in s
        assert "turret_angles" in s

    def test_torpedo_export_with_trail(self):
        engine = NavalCombatEngine()
        torp = Torpedo(
            torpedo_id="t1", position=(50, 60), heading=0.5,
            speed=25.0, target_id=None, damage=800, range_remaining=5000,
        )
        torp._trail = [(10, 20), (30, 40), (50, 60)]
        engine.torpedoes.append(torp)

        data = engine.to_three_js()
        t = data["torpedoes"][0]
        assert t["id"] == "t1"
        assert t["x"] == 50.0
        assert t["y"] == 60.0
        assert len(t["trail"]) == 3

    def test_effects_in_export(self):
        engine = NavalCombatEngine()
        engine.effects.append(CombatEffect("splash", (10, 20)))
        data = engine.to_three_js()
        assert len(data["effects"]) == 1
        assert data["effects"][0]["type"] == "splash"

    def test_json_serializable(self):
        """Ensure to_three_js output is JSON-serializable."""
        import json
        engine = NavalCombatEngine()
        ship = create_ship(ShipClass.CARRIER, "CV1", "blue", ship_id="cv1")
        engine.add_ship(ship)
        engine.set_ship_controls("cv1", throttle=0.5)
        engine.tick(1.0)
        data = engine.to_three_js()
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0


# ===========================================================================
# NavalCombatEngine — detect_targets
# ===========================================================================

class TestDetectTargets:

    def test_detect_surface_target_in_range(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue",
                           ship_id="d1", position=(0, 0))
        enemy = create_ship(ShipClass.FRIGATE, "F1", "red",
                            ship_id="f1", position=(1000, 0))
        engine.add_ship(ship)
        engine.add_ship(enemy)

        targets = engine.detect_targets("d1")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "f1"
        assert targets[0]["sensor"] == "radar"

    def test_detect_nothing_out_of_range(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.SPEEDBOAT, "S1", "blue",
                           ship_id="s1", position=(0, 0))
        enemy = create_ship(ShipClass.CARGO, "Cargo", "red",
                            ship_id="c1", position=(100000, 0))
        engine.add_ship(ship)
        engine.add_ship(enemy)

        targets = engine.detect_targets("s1")
        assert len(targets) == 0

    def test_detect_does_not_find_self(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue", ship_id="d1")
        engine.add_ship(ship)
        targets = engine.detect_targets("d1")
        assert len(targets) == 0

    def test_detect_dead_ship_excluded(self):
        engine = NavalCombatEngine(rng=random.Random(42))
        ship = create_ship(ShipClass.DESTROYER, "D1", "blue",
                           ship_id="d1", position=(0, 0))
        dead = create_ship(ShipClass.FRIGATE, "F1", "red",
                           ship_id="f1", position=(100, 0))
        dead.health = 0
        engine.add_ship(ship)
        engine.add_ship(dead)

        targets = engine.detect_targets("d1")
        assert len(targets) == 0


# ===========================================================================
# Integration: multi-ship engagement
# ===========================================================================

class TestIntegrationEngagement:

    def test_two_ship_duel(self):
        """Two destroyers exchange fire over several ticks."""
        engine = NavalCombatEngine(sea_state=0.2, rng=random.Random(123))
        d1 = create_ship(ShipClass.DESTROYER, "Alpha", "blue",
                         ship_id="alpha", position=(0, 0), heading=0.0)
        d2 = create_ship(ShipClass.DESTROYER, "Bravo", "red",
                         ship_id="bravo", position=(500, 0), heading=math.pi)
        engine.add_ship(d1)
        engine.add_ship(d2)

        engine.set_ship_controls("alpha", throttle=0.5)
        engine.set_ship_controls("bravo", throttle=0.5)

        total_hits = 0
        # Small dt so shells don't overshoot targets
        for i in range(600):
            if i % 10 == 0:
                engine.fire_guns("alpha", d2.position)
                engine.fire_guns("bravo", d1.position)
            result = engine.tick(0.01)
            total_hits += len(result["shell_hits"])

        assert total_hits > 0

    def test_torpedo_sinks_ship(self):
        """A submarine torpedo should be able to sink a patrol boat."""
        engine = NavalCombatEngine(rng=random.Random(99))
        sub = create_ship(ShipClass.SUBMARINE, "Sub", "blue",
                          ship_id="sub1", position=(0, 0), heading=0.0)
        boat = create_ship(ShipClass.PATROL_BOAT, "PB", "red",
                           ship_id="pb1", position=(200, 0))
        engine.add_ship(sub)
        engine.add_ship(boat)

        engine.fire_torpedo("sub1", target_id="pb1")

        sunk = False
        for _ in range(500):
            result = engine.tick(0.05)
            if "pb1" in result["sunk"]:
                sunk = True
                break

        assert sunk, "Torpedo should sink the patrol boat"
