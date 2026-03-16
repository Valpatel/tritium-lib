# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 -- see LICENSE for details.
"""Tests for the asymmetric warfare / IED / guerrilla module."""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.asymmetric import (
    CELL_BEHAVIORS,
    TRAP_TEMPLATES,
    AsymmetricEngine,
    GuerrillaCell,
    Trap,
    TrapType,
)
from tritium_lib.sim_engine.ai.steering import distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> AsymmetricEngine:
    return AsymmetricEngine(rng=random.Random(42))


@pytest.fixture
def seeded_engine() -> AsymmetricEngine:
    """Engine with deterministic seed for reproducibility."""
    return AsymmetricEngine(rng=random.Random(12345))


# ---------------------------------------------------------------------------
# TrapType enum
# ---------------------------------------------------------------------------

class TestTrapType:
    def test_all_members_exist(self):
        expected = {
            "IED_ROADSIDE", "IED_VEHICLE", "BOOBY_TRAP", "TRIP_WIRE",
            "SNARE", "DECOY", "AMBUSH_POINT",
        }
        assert {m.name for m in TrapType} == expected

    def test_values_are_lowercase(self):
        for member in TrapType:
            assert member.value == member.name.lower()


# ---------------------------------------------------------------------------
# Trap dataclass
# ---------------------------------------------------------------------------

class TestTrap:
    def test_defaults(self):
        t = Trap(
            trap_id="t1", trap_type=TrapType.IED_ROADSIDE,
            position=(10.0, 20.0), facing=0.0, damage=60.0,
            blast_radius=5.0, trigger_type="proximity", trigger_radius=3.0,
        )
        assert t.is_armed is True
        assert t.is_hidden is True
        assert t.detection_difficulty == 0.5
        assert t.placer_alliance == "hostile"
        assert t.timer_remaining is None

    def test_custom_values(self):
        t = Trap(
            trap_id="t2", trap_type=TrapType.IED_VEHICLE,
            position=(0.0, 0.0), facing=1.57, damage=300.0,
            blast_radius=20.0, trigger_type="remote", trigger_radius=8.0,
            is_armed=False, is_hidden=False, detection_difficulty=0.3,
            placer_alliance="friendly", timer_remaining=30.0,
        )
        assert t.is_armed is False
        assert t.placer_alliance == "friendly"
        assert t.timer_remaining == 30.0


# ---------------------------------------------------------------------------
# GuerrillaCell dataclass
# ---------------------------------------------------------------------------

class TestGuerrillaCell:
    def test_defaults(self):
        c = GuerrillaCell(cell_id="c1")
        assert c.morale == 0.8
        assert c.aggression == 0.5
        assert c.state == "hiding"
        assert c.member_count == 0

    def test_member_count(self):
        c = GuerrillaCell(cell_id="c1", members=["a", "b", "c"])
        assert c.member_count == 3


# ---------------------------------------------------------------------------
# TRAP_TEMPLATES
# ---------------------------------------------------------------------------

class TestTrapTemplates:
    def test_all_templates_present(self):
        expected = {"ied_small", "ied_large", "vbied", "booby_trap", "decoy", "ambush_point"}
        assert set(TRAP_TEMPLATES.keys()) == expected

    def test_ied_small_values(self):
        t = TRAP_TEMPLATES["ied_small"]
        assert t["damage"] == 60.0
        assert t["blast_radius"] == 5.0
        assert t["trigger_radius"] == 3.0
        assert t["detection_difficulty"] == 0.6

    def test_ied_large_values(self):
        t = TRAP_TEMPLATES["ied_large"]
        assert t["damage"] == 150.0
        assert t["blast_radius"] == 10.0

    def test_vbied_values(self):
        t = TRAP_TEMPLATES["vbied"]
        assert t["damage"] == 300.0
        assert t["blast_radius"] == 20.0
        assert t["trigger_type"] == "remote"

    def test_booby_trap_high_detection_difficulty(self):
        assert TRAP_TEMPLATES["booby_trap"]["detection_difficulty"] == 0.8

    def test_decoy_zero_damage(self):
        t = TRAP_TEMPLATES["decoy"]
        assert t["damage"] == 0.0
        assert t["blast_radius"] == 0.0

    def test_ambush_point_no_damage(self):
        t = TRAP_TEMPLATES["ambush_point"]
        assert t["damage"] == 0.0

    def test_all_templates_have_required_keys(self):
        required = {"trap_type", "damage", "blast_radius", "trigger_type",
                     "trigger_radius", "detection_difficulty"}
        for name, tmpl in TRAP_TEMPLATES.items():
            assert required.issubset(tmpl.keys()), f"{name} missing keys"


# ---------------------------------------------------------------------------
# CELL_BEHAVIORS
# ---------------------------------------------------------------------------

class TestCellBehaviors:
    def test_all_behaviors_present(self):
        expected = {"hit_and_run", "ied_ambush", "sniper_harassment", "mob_attack"}
        assert set(CELL_BEHAVIORS.keys()) == expected

    def test_hit_and_run_duration(self):
        assert CELL_BEHAVIORS["hit_and_run"]["attack_duration"] == 10.0

    def test_mob_attack_min_morale(self):
        assert CELL_BEHAVIORS["mob_attack"]["min_morale"] == 0.8


# ---------------------------------------------------------------------------
# AsymmetricEngine — placement
# ---------------------------------------------------------------------------

class TestPlaceTrap:
    def test_place_creates_trap(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (50.0, 50.0), "hostile")
        assert trap.trap_id in engine.traps
        assert trap.is_armed is True
        assert trap.position == (50.0, 50.0)

    def test_place_custom_params(self, engine: AsymmetricEngine):
        trap = engine.place_trap(
            TrapType.BOOBY_TRAP, (10.0, 20.0), "friendly",
            trigger_type="tripwire", facing=1.0,
            damage=30.0, blast_radius=2.0, trigger_radius=1.0,
            detection_difficulty=0.9,
        )
        assert trap.trigger_type == "tripwire"
        assert trap.damage == 30.0
        assert trap.placer_alliance == "friendly"

    def test_place_generates_unique_ids(self, engine: AsymmetricEngine):
        ids = set()
        for _ in range(50):
            t = engine.place_trap(TrapType.SNARE, (0.0, 0.0), "hostile")
            ids.add(t.trap_id)
        assert len(ids) == 50

    def test_place_with_timer(self, engine: AsymmetricEngine):
        trap = engine.place_trap(
            TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile",
            trigger_type="timer", timer_remaining=15.0,
        )
        assert trap.trigger_type == "timer"
        assert trap.timer_remaining == 15.0


class TestPlaceFromTemplate:
    def test_ied_small(self, engine: AsymmetricEngine):
        trap = engine.place_from_template("ied_small", (10.0, 10.0), "hostile")
        assert trap.damage == 60.0
        assert trap.blast_radius == 5.0
        assert trap.trap_type == TrapType.IED_ROADSIDE

    def test_vbied(self, engine: AsymmetricEngine):
        trap = engine.place_from_template("vbied", (0.0, 0.0), "hostile")
        assert trap.damage == 300.0
        assert trap.trigger_type == "remote"

    def test_unknown_template_raises(self, engine: AsymmetricEngine):
        with pytest.raises(KeyError):
            engine.place_from_template("nonexistent", (0.0, 0.0), "hostile")


class TestPlaceIEDPattern:
    def test_places_correct_count(self, engine: AsymmetricEngine):
        route = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        traps = engine.place_ied_pattern(route, 5, "hostile")
        assert len(traps) == 5

    def test_empty_route_returns_empty(self, engine: AsymmetricEngine):
        assert engine.place_ied_pattern([], 3, "hostile") == []

    def test_zero_count_returns_empty(self, engine: AsymmetricEngine):
        assert engine.place_ied_pattern([(0.0, 0.0), (10.0, 0.0)], 0, "hostile") == []

    def test_ieds_near_route(self, engine: AsymmetricEngine):
        route = [(0.0, 0.0), (100.0, 0.0)]
        traps = engine.place_ied_pattern(route, 10, "hostile")
        for trap in traps:
            # Should be within ~3m of the route line (lateral offset)
            assert -5.0 <= trap.position[1] <= 5.0
            assert -5.0 <= trap.position[0] <= 105.0

    def test_single_point_route_returns_empty(self, engine: AsymmetricEngine):
        # Single point = no segments
        assert engine.place_ied_pattern([(5.0, 5.0)], 3, "hostile") == []

    def test_all_placed_are_armed(self, engine: AsymmetricEngine):
        route = [(0.0, 0.0), (50.0, 0.0)]
        traps = engine.place_ied_pattern(route, 4, "hostile")
        assert all(t.is_armed for t in traps)


# ---------------------------------------------------------------------------
# AsymmetricEngine — sweep / detection
# ---------------------------------------------------------------------------

class TestSweepArea:
    def test_high_skill_finds_easy_traps(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.DECOY, (10.0, 10.0), "hostile",
            detection_difficulty=0.0,  # trivial to find
        )
        found = engine.sweep_area((10.0, 10.0), radius=20.0, skill=1.0)
        assert len(found) == 1

    def test_zero_skill_finds_nothing(self, engine: AsymmetricEngine):
        engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile")
        found = engine.sweep_area((10.0, 10.0), radius=20.0, skill=0.0)
        assert len(found) == 0

    def test_out_of_range_not_detected(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (100.0, 100.0), "hostile",
            detection_difficulty=0.0,
        )
        found = engine.sweep_area((0.0, 0.0), radius=5.0, skill=1.0)
        assert len(found) == 0

    def test_already_detected_not_returned(self, engine: AsymmetricEngine):
        trap = engine.place_trap(
            TrapType.IED_ROADSIDE, (5.0, 5.0), "hostile",
            detection_difficulty=0.0,
        )
        engine.detected_traps.add(trap.trap_id)
        found = engine.sweep_area((5.0, 5.0), radius=20.0, skill=1.0)
        assert len(found) == 0

    def test_disarmed_trap_not_swept(self, engine: AsymmetricEngine):
        trap = engine.place_trap(
            TrapType.IED_ROADSIDE, (5.0, 5.0), "hostile",
            detection_difficulty=0.0,
        )
        trap.is_armed = False
        found = engine.sweep_area((5.0, 5.0), radius=20.0, skill=1.0)
        assert len(found) == 0

    def test_detection_difficulty_affects_chance(self):
        """High detection_difficulty should reduce find rate statistically."""
        rng = random.Random(99)
        eng = AsymmetricEngine(rng=rng)
        easy_found = 0
        hard_found = 0
        trials = 200
        for i in range(trials):
            eng2 = AsymmetricEngine(rng=random.Random(i))
            eng2.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile",
                            detection_difficulty=0.1)
            if eng2.sweep_area((0.0, 0.0), 10.0, skill=0.5):
                easy_found += 1
            eng3 = AsymmetricEngine(rng=random.Random(i + 10000))
            eng3.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile",
                            detection_difficulty=0.9)
            if eng3.sweep_area((0.0, 0.0), 10.0, skill=0.5):
                hard_found += 1
        assert easy_found > hard_found


# ---------------------------------------------------------------------------
# AsymmetricEngine — disarm
# ---------------------------------------------------------------------------

class TestDisarm:
    def test_successful_disarm(self, engine: AsymmetricEngine):
        # Skill 1.0 always succeeds
        engine._rng = random.Random(0)
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile")
        result = engine.disarm_trap(trap.trap_id, engineer_skill=1.0)
        assert result is True
        assert trap.is_armed is False

    def test_failed_disarm_detonates(self, engine: AsymmetricEngine):
        # Skill 0.0 always fails
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile")
        result = engine.disarm_trap(trap.trap_id, engineer_skill=0.0)
        assert result is False
        assert trap.is_armed is False  # detonated

    def test_disarm_nonexistent_returns_true(self, engine: AsymmetricEngine):
        assert engine.disarm_trap("fake_id") is True

    def test_disarm_already_disarmed(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile")
        trap.is_armed = False
        assert engine.disarm_trap(trap.trap_id) is True


# ---------------------------------------------------------------------------
# AsymmetricEngine — remote detonation
# ---------------------------------------------------------------------------

class TestDetonateRemote:
    def test_successful_detonation(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_VEHICLE, (50.0, 50.0), "hostile",
                                 trigger_type="remote")
        result = engine.detonate_remote(trap.trap_id)
        assert result["detonated"] is True
        assert result["trap_type"] == "ied_vehicle"
        assert result["damage"] == trap.damage
        assert trap.is_armed is False

    def test_detonate_missing_trap(self, engine: AsymmetricEngine):
        result = engine.detonate_remote("nonexistent")
        assert result["detonated"] is False

    def test_detonate_already_disarmed(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile")
        trap.is_armed = False
        result = engine.detonate_remote(trap.trap_id)
        assert result["detonated"] is False


# ---------------------------------------------------------------------------
# AsymmetricEngine — cell management
# ---------------------------------------------------------------------------

class TestCreateCell:
    def test_creates_cell(self, engine: AsymmetricEngine):
        cell = engine.create_cell((100.0, 100.0), 5, 50.0)
        assert cell.cell_id in engine.cells
        assert cell.member_count == 5
        assert cell.operating_radius == 50.0
        assert cell.state == "hiding"

    def test_members_have_unique_ids(self, engine: AsymmetricEngine):
        cell = engine.create_cell((0.0, 0.0), 10, 100.0)
        assert len(set(cell.members)) == 10

    def test_weapons_cache_at_base(self, engine: AsymmetricEngine):
        cell = engine.create_cell((30.0, 40.0), 3, 20.0)
        assert cell.weapons_cache == (30.0, 40.0)


# ---------------------------------------------------------------------------
# AsymmetricEngine — tick: proximity triggers
# ---------------------------------------------------------------------------

class TestTickProximity:
    def test_proximity_trigger_fires(self, engine: AsymmetricEngine):
        trap = engine.place_trap(
            TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile",
            trigger_type="proximity", trigger_radius=5.0,
        )
        units = {"u1": ((10.0, 10.0), "friendly")}
        events = engine.tick(1.0, units)
        triggered = [e for e in events if e.get("event") == "trap_triggered"]
        assert len(triggered) == 1
        assert triggered[0]["victim_id"] == "u1"
        assert trap.is_armed is False

    def test_friendly_does_not_trigger_own_trap(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile",
            trigger_type="proximity", trigger_radius=5.0,
        )
        units = {"u1": ((10.0, 10.0), "hostile")}  # same alliance
        events = engine.tick(1.0, units)
        triggered = [e for e in events if e.get("event") == "trap_triggered"]
        assert len(triggered) == 0

    def test_out_of_range_no_trigger(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile",
            trigger_type="proximity", trigger_radius=2.0,
        )
        units = {"u1": ((100.0, 100.0), "friendly")}
        events = engine.tick(1.0, units)
        triggered = [e for e in events if e.get("event") == "trap_triggered"]
        assert len(triggered) == 0

    def test_remote_trigger_not_proximity(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile",
            trigger_type="remote", trigger_radius=5.0,
        )
        units = {"u1": ((10.0, 10.0), "friendly")}
        events = engine.tick(1.0, units)
        triggered = [e for e in events if e.get("event") == "trap_triggered"]
        assert len(triggered) == 0


# ---------------------------------------------------------------------------
# AsymmetricEngine — tick: timer triggers
# ---------------------------------------------------------------------------

class TestTickTimer:
    def test_timer_detonates_when_expired(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile",
            trigger_type="timer", timer_remaining=5.0,
        )
        # Tick 3 seconds — not yet
        events = engine.tick(3.0, {})
        assert len([e for e in events if e.get("event") == "trap_triggered"]) == 0

        # Tick 3 more — fires
        events = engine.tick(3.0, {})
        assert len([e for e in events if e.get("event") == "trap_triggered"]) == 1

    def test_timer_none_does_not_fire(self, engine: AsymmetricEngine):
        engine.place_trap(
            TrapType.IED_ROADSIDE, (0.0, 0.0), "hostile",
            trigger_type="timer", timer_remaining=None,
        )
        events = engine.tick(100.0, {})
        assert len([e for e in events if e.get("event") == "trap_triggered"]) == 0


# ---------------------------------------------------------------------------
# AsymmetricEngine — tick: cell AI
# ---------------------------------------------------------------------------

class TestTickCellAI:
    def test_cell_transitions_to_preparing(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.aggression = 0.8
        cell.morale = 0.9
        units = {"u1": ((60.0, 50.0), "friendly")}
        events = engine.tick(1.0, units)
        state_changes = [e for e in events if e.get("event") == "cell_state_change"]
        assert any(e["new_state"] == "preparing" for e in state_changes)

    def test_cell_disbands_low_morale(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.morale = 0.1
        events = engine.tick(1.0, {})
        assert any(e.get("event") == "cell_disbanded" for e in events)
        assert cell.state == "disbanded"

    def test_cell_disbands_no_members(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 0, 100.0)
        events = engine.tick(1.0, {})
        assert any(e.get("event") == "cell_disbanded" for e in events)

    def test_cell_stays_hiding_no_enemies(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        events = engine.tick(1.0, {})
        assert cell.state == "hiding"

    def test_cell_attacks_from_preparing(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.state = "preparing"
        cell.morale = 0.9
        units = {"u1": ((55.0, 50.0), "friendly")}
        events = engine.tick(1.0, units)
        assert any(e.get("event") == "cell_attack" for e in events)

    def test_cell_flees_after_attacking(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.state = "attacking"
        events = engine.tick(1.0, {})
        assert cell.state == "fleeing"

    def test_cell_hides_after_fleeing(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.state = "fleeing"
        engine.tick(1.0, {})
        assert cell.state == "hiding"

    def test_patrol_routes_stored_as_knowledge(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        routes = [[(0.0, 0.0), (100.0, 0.0)]]
        engine.tick(1.0, {}, patrol_routes=routes)
        assert cell.knowledge.get("patrol_routes") == routes

    def test_disbanded_cell_ignored(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.state = "disbanded"
        events = engine.tick(1.0, {"u1": ((50.0, 50.0), "friendly")})
        # No events from disbanded cell
        assert not any(e.get("cell_id") == cell.cell_id for e in events)


# ---------------------------------------------------------------------------
# AsymmetricEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJS:
    def test_own_traps_visible(self, engine: AsymmetricEngine):
        engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "friendly")
        data = engine.to_three_js("friendly")
        assert len(data["traps"]) == 1

    def test_enemy_hidden_traps_invisible(self, engine: AsymmetricEngine):
        engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile")
        data = engine.to_three_js("friendly")
        assert len(data["traps"]) == 0

    def test_detected_enemy_traps_visible(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile")
        engine.detected_traps.add(trap.trap_id)
        data = engine.to_three_js("friendly")
        assert len(data["traps"]) == 1
        assert data["traps"][0]["detected"] is True

    def test_disarmed_traps_not_shown(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "friendly")
        trap.is_armed = False
        data = engine.to_three_js("friendly")
        assert len(data["traps"]) == 0

    def test_cells_in_output(self, engine: AsymmetricEngine):
        engine.create_cell((50.0, 50.0), 5, 100.0)
        data = engine.to_three_js("hostile")
        assert len(data["cells"]) == 1
        assert data["cells"][0]["members"] == 5

    def test_disbanded_cells_hidden(self, engine: AsymmetricEngine):
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.state = "disbanded"
        data = engine.to_three_js("hostile")
        assert len(data["cells"]) == 0

    def test_effects_after_detonation(self, engine: AsymmetricEngine):
        trap = engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 10.0), "hostile")
        units = {"u1": ((10.0, 10.0), "friendly")}
        engine.tick(1.0, units)
        data = engine.to_three_js("friendly")
        assert len(data["effects"]) == 1
        assert data["effects"][0]["type"] == "ied_explosion"

    def test_output_structure(self, engine: AsymmetricEngine):
        data = engine.to_three_js("friendly")
        assert "traps" in data
        assert "cells" in data
        assert "sweep_areas" in data
        assert "effects" in data

    def test_trap_json_fields(self, engine: AsymmetricEngine):
        engine.place_trap(TrapType.IED_ROADSIDE, (10.0, 20.0), "friendly")
        trap_data = engine.to_three_js("friendly")["traps"][0]
        assert "id" in trap_data
        assert trap_data["x"] == 10.0
        assert trap_data["y"] == 20.0
        assert trap_data["type"] == "ied_roadside"
        assert "armed" in trap_data
        assert "radius" in trap_data


# ---------------------------------------------------------------------------
# Integration / multi-step scenarios
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    def test_full_ied_ambush_scenario(self, engine: AsymmetricEngine):
        """Place IEDs, patrol walks through, traps trigger."""
        route = [(0.0, 0.0), (100.0, 0.0)]
        traps = engine.place_ied_pattern(route, 3, "hostile")
        assert len(traps) == 3

        # Patrol walks down the road
        all_events: list[dict] = []
        for x in range(0, 110, 5):
            units = {"patrol1": ((float(x), 0.0), "friendly")}
            events = engine.tick(1.0, units)
            all_events.extend(events)

        # At least one trap should have triggered
        triggered = [e for e in all_events if e.get("event") == "trap_triggered"]
        assert len(triggered) >= 1

    def test_sweep_then_disarm(self, engine: AsymmetricEngine):
        """Sweep finds trap, engineer disarms it."""
        trap = engine.place_trap(
            TrapType.BOOBY_TRAP, (20.0, 20.0), "hostile",
            detection_difficulty=0.0,
        )
        found = engine.sweep_area((20.0, 20.0), 30.0, skill=1.0)
        assert len(found) == 1
        success = engine.disarm_trap(found[0].trap_id, engineer_skill=1.0)
        assert success is True
        assert trap.is_armed is False

    def test_cell_lifecycle(self, engine: AsymmetricEngine):
        """Cell goes through hiding -> preparing -> attacking -> fleeing -> hiding."""
        cell = engine.create_cell((50.0, 50.0), 5, 100.0)
        cell.aggression = 0.8
        cell.morale = 0.9
        enemy_units = {"e1": ((55.0, 50.0), "friendly")}

        # hiding -> preparing
        engine.tick(1.0, enemy_units)
        assert cell.state == "preparing"

        # preparing -> attacking
        engine.tick(1.0, enemy_units)
        assert cell.state == "attacking"

        # attacking -> fleeing
        engine.tick(1.0, {})
        assert cell.state == "fleeing"

        # fleeing -> hiding
        engine.tick(1.0, {})
        assert cell.state == "hiding"

    def test_timer_ied_with_cell(self, engine: AsymmetricEngine):
        """Cell places timed IED, it detonates on schedule."""
        engine.create_cell((50.0, 50.0), 3, 100.0)
        engine.place_trap(
            TrapType.IED_ROADSIDE, (50.0, 50.0), "hostile",
            trigger_type="timer", timer_remaining=2.0,
        )
        events1 = engine.tick(1.0, {})
        timer_events = [e for e in events1 if e.get("cause") == "timer"]
        assert len(timer_events) == 0

        events2 = engine.tick(1.5, {})
        timer_events = [e for e in events2 if e.get("cause") == "timer"]
        assert len(timer_events) == 1

    def test_multiple_traps_multiple_units(self, engine: AsymmetricEngine):
        """Multiple traps, multiple units — each trap triggers at most once."""
        for x in range(0, 50, 10):
            engine.place_trap(
                TrapType.IED_ROADSIDE, (float(x), 0.0), "hostile",
                trigger_type="proximity", trigger_radius=3.0,
            )
        units = {
            f"u{i}": ((float(i * 10), 0.0), "friendly")
            for i in range(5)
        }
        events = engine.tick(1.0, units)
        triggered = [e for e in events if e.get("event") == "trap_triggered"]
        assert len(triggered) == 5  # one trap per unit position
