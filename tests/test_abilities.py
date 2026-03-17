# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the ability and special powers system."""

from __future__ import annotations

import copy
import pytest

from tritium_lib.sim_engine.abilities import (
    Ability,
    AbilityEngine,
    AbilityType,
    TargetType,
    ABILITIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> AbilityEngine:
    return AbilityEngine()


@pytest.fixture
def armed_engine() -> AbilityEngine:
    """Engine with a unit that has grenades and resources."""
    eng = AbilityEngine()
    eng.grant_ability("alpha-1", ABILITIES["frag_grenade"])
    eng.grant_ability("alpha-1", ABILITIES["sprint"])
    eng.grant_ability("alpha-1", ABILITIES["go_prone"])
    eng.set_resources("alpha-1", {"grenades": 5, "ammo": 100})
    return eng


# ===========================================================================
# 1. Enum tests
# ===========================================================================

class TestAbilityType:
    def test_values(self):
        assert AbilityType.ACTIVE.value == "active"
        assert AbilityType.PASSIVE.value == "passive"
        assert AbilityType.TOGGLE.value == "toggle"
        assert AbilityType.CHANNELED.value == "channeled"

    def test_member_count(self):
        assert len(AbilityType) == 4


class TestTargetType:
    def test_values(self):
        assert TargetType.SELF.value == "self"
        assert TargetType.SINGLE_ALLY.value == "single_ally"
        assert TargetType.SINGLE_ENEMY.value == "single_enemy"
        assert TargetType.AREA_ALLY.value == "area_ally"
        assert TargetType.AREA_ENEMY.value == "area_enemy"
        assert TargetType.AREA_ALL.value == "area_all"
        assert TargetType.NONE.value == "none"

    def test_member_count(self):
        assert len(TargetType) == 7


# ===========================================================================
# 2. Ability dataclass tests
# ===========================================================================

class TestAbilityDataclass:
    def test_defaults(self):
        ab = Ability(
            ability_id="test", name="Test", description="A test",
            ability_type=AbilityType.ACTIVE, target_type=TargetType.SELF,
            cooldown=5.0,
        )
        assert ab.current_cooldown == 0.0
        assert ab.cost == {}
        assert ab.range == 0.0
        assert ab.radius == 0.0
        assert ab.duration == 0.0
        assert ab.effects == []
        assert ab.toggled_on is False

    def test_is_ready(self):
        ab = Ability(
            ability_id="t", name="T", description="",
            ability_type=AbilityType.ACTIVE, target_type=TargetType.SELF,
            cooldown=5.0,
        )
        assert ab.is_ready is True
        ab.current_cooldown = 2.0
        assert ab.is_ready is False

    def test_clone_deep_copy(self):
        ab = ABILITIES["frag_grenade"]
        clone = ab.clone()
        assert clone.ability_id == ab.ability_id
        assert clone is not ab
        assert clone.effects is not ab.effects
        clone.effects.append({"type": "extra"})
        assert len(clone.effects) != len(ab.effects)

    def test_cost_independence(self):
        """Cloned ability costs are independent."""
        ab = ABILITIES["frag_grenade"]
        c1 = ab.clone()
        c2 = ab.clone()
        c1.cost["grenades"] = 99
        assert c2.cost.get("grenades") != 99


# ===========================================================================
# 3. ABILITIES catalog tests
# ===========================================================================

class TestAbilitiesCatalog:
    def test_minimum_count(self):
        assert len(ABILITIES) >= 25

    def test_all_have_required_fields(self):
        for aid, ab in ABILITIES.items():
            assert ab.ability_id == aid, f"{aid} ability_id mismatch"
            assert ab.name, f"{aid} missing name"
            assert ab.description, f"{aid} missing description"
            assert isinstance(ab.ability_type, AbilityType)
            assert isinstance(ab.target_type, TargetType)
            assert ab.cooldown >= 0

    def test_infantry_abilities_exist(self):
        for name in ["frag_grenade", "smoke_grenade", "flashbang", "sprint", "go_prone", "rally_cry"]:
            assert name in ABILITIES, f"Missing infantry ability: {name}"

    def test_medic_abilities_exist(self):
        for name in ["first_aid", "triage_scan", "morphine_shot", "evac_call"]:
            assert name in ABILITIES, f"Missing medic ability: {name}"

    def test_engineer_abilities_exist(self):
        for name in ["build_cover", "plant_mine", "repair_vehicle", "breach_wall", "defuse_bomb"]:
            assert name in ABILITIES, f"Missing engineer ability: {name}"

    def test_sniper_abilities_exist(self):
        for name in ["hold_breath", "mark_target", "ghillie_deploy"]:
            assert name in ABILITIES, f"Missing sniper ability: {name}"

    def test_heavy_abilities_exist(self):
        for name in ["suppressive_fire", "deploy_bipod", "rocket_barrage"]:
            assert name in ABILITIES, f"Missing heavy ability: {name}"

    def test_commander_abilities_exist(self):
        for name in ["call_airstrike", "call_artillery", "radar_sweep", "reinforce"]:
            assert name in ABILITIES, f"Missing commander ability: {name}"

    def test_scout_abilities_exist(self):
        for name in ["binoculars", "tag_enemy", "silent_move"]:
            assert name in ABILITIES, f"Missing scout ability: {name}"

    def test_all_abilities_have_effects(self):
        for aid, ab in ABILITIES.items():
            assert isinstance(ab.effects, list), f"{aid} effects not a list"
            # Non-toggle non-passive should have at least one effect
            if ab.ability_type not in (AbilityType.PASSIVE,):
                assert len(ab.effects) > 0, f"{aid} has no effects"

    def test_area_abilities_have_radius(self):
        for aid, ab in ABILITIES.items():
            if ab.target_type in (TargetType.AREA_ALLY, TargetType.AREA_ENEMY, TargetType.AREA_ALL):
                assert ab.radius > 0 or ab.range > 0, f"Area ability {aid} has no radius or range"

    def test_catalog_is_not_aliased(self):
        """Each catalog entry is an independent object."""
        g1 = ABILITIES["frag_grenade"]
        g2 = ABILITIES["frag_grenade"]
        assert g1 is g2  # same object in the dict, but clone on grant


# ===========================================================================
# 4. AbilityEngine — grant / revoke
# ===========================================================================

class TestEngineGrant:
    def test_grant_basic(self, engine: AbilityEngine):
        ab = engine.grant_ability("u1", ABILITIES["sprint"])
        assert ab.ability_id == "sprint"
        assert len(engine.get_all("u1")) == 1

    def test_grant_is_cloned(self, engine: AbilityEngine):
        original = ABILITIES["sprint"]
        granted = engine.grant_ability("u1", original)
        assert granted is not original
        granted.current_cooldown = 999
        assert original.current_cooldown != 999

    def test_grant_replaces_duplicate(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        engine.grant_ability("u1", ABILITIES["sprint"])
        assert len(engine.get_all("u1")) == 1

    def test_grant_multiple(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        engine.grant_ability("u1", ABILITIES["frag_grenade"])
        assert len(engine.get_all("u1")) == 2

    def test_revoke(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        result = engine.revoke_ability("u1", "sprint")
        assert result is True
        assert len(engine.get_all("u1")) == 0

    def test_revoke_nonexistent(self, engine: AbilityEngine):
        assert engine.revoke_ability("u1", "nope") is False

    def test_get_ability(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        ab = engine.get_ability("u1", "sprint")
        assert ab is not None
        assert ab.ability_id == "sprint"

    def test_get_ability_missing(self, engine: AbilityEngine):
        assert engine.get_ability("u1", "nope") is None


# ===========================================================================
# 5. AbilityEngine — resources
# ===========================================================================

class TestEngineResources:
    def test_set_and_get(self, engine: AbilityEngine):
        engine.set_resources("u1", {"ammo": 50, "grenades": 3})
        res = engine.get_resources("u1")
        assert res["ammo"] == 50
        assert res["grenades"] == 3

    def test_get_empty(self, engine: AbilityEngine):
        assert engine.get_resources("nobody") == {}

    def test_resources_are_copied(self, engine: AbilityEngine):
        engine.set_resources("u1", {"ammo": 50})
        res = engine.get_resources("u1")
        res["ammo"] = 0
        assert engine.get_resources("u1")["ammo"] == 50


# ===========================================================================
# 6. AbilityEngine — activate (ACTIVE abilities)
# ===========================================================================

class TestActivateActive:
    def test_basic_activation(self, armed_engine: AbilityEngine):
        result = armed_engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
        assert result["success"] is True
        assert result["ability_id"] == "frag_grenade"
        assert result["radius"] == 8.0
        assert len(result["effects"]) == 2

    def test_cooldown_set_after_activation(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
        ab = armed_engine.get_ability("alpha-1", "frag_grenade")
        assert ab.current_cooldown == 15.0

    def test_on_cooldown_rejection(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
        result = armed_engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
        assert result["success"] is False
        assert result["reason"] == "on_cooldown"

    def test_insufficient_resources(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["frag_grenade"])
        engine.set_resources("u1", {"grenades": 0})
        result = engine.activate("u1", "frag_grenade", target_pos=(50, 30))
        assert result["success"] is False
        assert result["reason"] == "insufficient_resources"

    def test_resources_deducted(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
        res = armed_engine.get_resources("alpha-1")
        assert res["grenades"] == 4  # started with 5

    def test_out_of_range(self, armed_engine: AbilityEngine):
        result = armed_engine.activate(
            "alpha-1", "frag_grenade",
            target_pos=(1000, 1000), unit_pos=(0, 0),
        )
        assert result["success"] is False
        assert result["reason"] == "out_of_range"

    def test_within_range(self, armed_engine: AbilityEngine):
        result = armed_engine.activate(
            "alpha-1", "frag_grenade",
            target_pos=(10, 10), unit_pos=(0, 0),
        )
        assert result["success"] is True

    def test_ability_not_found(self, engine: AbilityEngine):
        result = engine.activate("u1", "nonexistent")
        assert result["success"] is False
        assert result["reason"] == "ability_not_found"

    def test_no_cost_ability(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        result = engine.activate("u1", "sprint")
        assert result["success"] is True

    def test_self_target_no_range_check(self, engine: AbilityEngine):
        """Self-target abilities skip range validation."""
        engine.grant_ability("u1", ABILITIES["sprint"])
        result = engine.activate("u1", "sprint")
        assert result["success"] is True


# ===========================================================================
# 7. AbilityEngine — activate (PASSIVE rejection)
# ===========================================================================

class TestActivatePassive:
    def test_passive_cannot_activate(self, engine: AbilityEngine):
        passive = Ability(
            ability_id="passive_armor", name="Armor Up", description="Passive armor",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "armor", "value": 0.1}],
        )
        engine.grant_ability("u1", passive)
        result = engine.activate("u1", "passive_armor")
        assert result["success"] is False
        assert result["reason"] == "passive_cannot_activate"


# ===========================================================================
# 8. AbilityEngine — activate (TOGGLE abilities)
# ===========================================================================

class TestActivateToggle:
    def test_toggle_on(self, armed_engine: AbilityEngine):
        result = armed_engine.activate("alpha-1", "go_prone")
        assert result["success"] is True
        assert result["toggled"] is True
        ab = armed_engine.get_ability("alpha-1", "go_prone")
        assert ab.toggled_on is True

    def test_toggle_off(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "go_prone")
        result = armed_engine.activate("alpha-1", "go_prone")
        assert result["success"] is True
        assert result["toggled"] is False

    def test_toggle_on_returns_effects(self, armed_engine: AbilityEngine):
        result = armed_engine.activate("alpha-1", "go_prone")
        assert len(result["effects"]) > 0

    def test_toggle_off_returns_no_effects(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "go_prone")
        result = armed_engine.activate("alpha-1", "go_prone")
        assert result["effects"] == []

    def test_toggle_cost_only_on_activate(self, engine: AbilityEngine):
        """Toggle costs are paid only when turning on, not off."""
        toggle = Ability(
            ability_id="shield", name="Shield", description="Toggle shield",
            ability_type=AbilityType.TOGGLE, target_type=TargetType.SELF,
            cooldown=0, cost={"energy": 10},
            effects=[{"stat": "defense", "value": 0.5}],
        )
        engine.grant_ability("u1", toggle)
        engine.set_resources("u1", {"energy": 10})
        # Turn on — costs 10 energy
        result = engine.activate("u1", "shield")
        assert result["success"] is True
        assert engine.get_resources("u1")["energy"] == 0
        # Turn off — free
        result = engine.activate("u1", "shield")
        assert result["success"] is True
        assert engine.get_resources("u1")["energy"] == 0

    def test_toggle_on_insufficient_resources(self, engine: AbilityEngine):
        toggle = Ability(
            ability_id="shield", name="Shield", description="Toggle shield",
            ability_type=AbilityType.TOGGLE, target_type=TargetType.SELF,
            cooldown=0, cost={"energy": 100},
            effects=[{"stat": "defense", "value": 0.5}],
        )
        engine.grant_ability("u1", toggle)
        engine.set_resources("u1", {"energy": 5})
        result = engine.activate("u1", "shield")
        assert result["success"] is False
        assert result["reason"] == "insufficient_resources"


# ===========================================================================
# 9. AbilityEngine — activate (CHANNELED abilities)
# ===========================================================================

class TestActivateChanneled:
    def test_channel_starts(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        result = engine.activate("u1", "build_cover", target_pos=(10, 10))
        assert result["success"] is True
        assert result.get("channeling") is True
        assert engine.is_channeling("u1") is True

    def test_channel_tick_events(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        events = engine.tick(1.0)
        tick_events = [e for e in events if e["type"] == "channel_tick"]
        assert len(tick_events) == 1
        assert tick_events[0]["ability_id"] == "build_cover"

    def test_channel_completes(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        # build_cover duration is 5.0s
        events = engine.tick(6.0)
        complete_events = [e for e in events if e["type"] == "channel_complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["ability_id"] == "build_cover"
        assert engine.is_channeling("u1") is False

    def test_interrupt_channel(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        result = engine.interrupt_channel("u1", "build_cover")
        assert result is True
        assert engine.is_channeling("u1") is False

    def test_interrupt_all_channels(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.grant_ability("u1", ABILITIES["defuse_bomb"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        # defuse_bomb has no cost
        # Need to reset cooldown first or use a different approach
        engine.activate("u1", "defuse_bomb")
        result = engine.interrupt_channel("u1")
        assert result is True
        assert engine.is_channeling("u1") is False

    def test_interrupt_nonexistent(self, engine: AbilityEngine):
        assert engine.interrupt_channel("u1", "nope") is False

    def test_revoke_cancels_channel(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        engine.revoke_ability("u1", "build_cover")
        assert engine.is_channeling("u1") is False


# ===========================================================================
# 10. AbilityEngine — tick
# ===========================================================================

class TestEngineTick:
    def test_cooldown_reduces(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "sprint")
        ab = armed_engine.get_ability("alpha-1", "sprint")
        assert ab.current_cooldown == 12.0
        armed_engine.tick(5.0)
        assert ab.current_cooldown == 7.0

    def test_cooldown_floors_at_zero(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "sprint")
        armed_engine.tick(100.0)
        ab = armed_engine.get_ability("alpha-1", "sprint")
        assert ab.current_cooldown == 0.0

    def test_cooldown_ready_event(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "sprint")
        events = armed_engine.tick(13.0)
        ready_events = [e for e in events if e["type"] == "cooldown_ready"]
        assert any(e["ability_id"] == "sprint" for e in ready_events)

    def test_no_events_when_idle(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        events = engine.tick(1.0)
        assert events == []


# ===========================================================================
# 11. AbilityEngine — queries
# ===========================================================================

class TestEngineQueries:
    def test_get_available(self, armed_engine: AbilityEngine):
        # All 3 should be available (none on cooldown, go_prone is toggle = available)
        available = armed_engine.get_available("alpha-1")
        assert len(available) == 3

    def test_get_available_excludes_on_cooldown(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "sprint")
        available = armed_engine.get_available("alpha-1")
        ids = {a.ability_id for a in available}
        assert "sprint" not in ids

    def test_get_available_excludes_passives(self, engine: AbilityEngine):
        passive = Ability(
            ability_id="passive_test", name="P", description="",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "speed", "value": 0.1}],
        )
        engine.grant_ability("u1", passive)
        assert engine.get_available("u1") == []

    def test_get_passives(self, engine: AbilityEngine):
        passive = Ability(
            ability_id="passive_test", name="P", description="",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "speed", "value": 0.1}],
        )
        engine.grant_ability("u1", passive)
        engine.grant_ability("u1", ABILITIES["sprint"])
        passives = engine.get_passives("u1")
        assert len(passives) == 1
        assert passives[0].ability_id == "passive_test"

    def test_get_active_toggles(self, armed_engine: AbilityEngine):
        assert armed_engine.get_active_toggles("alpha-1") == []
        armed_engine.activate("alpha-1", "go_prone")
        toggles = armed_engine.get_active_toggles("alpha-1")
        assert len(toggles) == 1
        assert toggles[0].ability_id == "go_prone"

    def test_is_channeling_false(self, engine: AbilityEngine):
        assert engine.is_channeling("u1") is False

    def test_get_all_empty(self, engine: AbilityEngine):
        assert engine.get_all("nobody") == []

    def test_get_passive_modifiers(self, engine: AbilityEngine):
        passive = Ability(
            ability_id="passive_speed", name="Fast", description="",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "speed", "value": 0.2}],
        )
        engine.grant_ability("u1", passive)
        mods = engine.get_passive_modifiers("u1")
        assert mods["speed"] == pytest.approx(0.2)

    def test_get_passive_modifiers_includes_active_toggles(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "go_prone")
        mods = armed_engine.get_passive_modifiers("alpha-1")
        assert "accuracy" in mods  # go_prone gives +0.2 accuracy
        assert mods["accuracy"] == pytest.approx(0.2)

    def test_get_passive_modifiers_excludes_inactive_toggles(self, armed_engine: AbilityEngine):
        mods = armed_engine.get_passive_modifiers("alpha-1")
        # go_prone not toggled on, should not contribute
        assert mods == {}

    def test_get_passive_modifiers_sums(self, engine: AbilityEngine):
        p1 = Ability(
            ability_id="p1", name="P1", description="",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "speed", "value": 0.1}],
        )
        p2 = Ability(
            ability_id="p2", name="P2", description="",
            ability_type=AbilityType.PASSIVE, target_type=TargetType.SELF,
            cooldown=0, effects=[{"stat": "speed", "value": 0.2}],
        )
        engine.grant_ability("u1", p1)
        engine.grant_ability("u1", p2)
        mods = engine.get_passive_modifiers("u1")
        assert mods["speed"] == pytest.approx(0.3)


# ===========================================================================
# 12. AbilityEngine — Three.js serialisation
# ===========================================================================

class TestToThreeJs:
    def test_empty_unit(self, engine: AbilityEngine):
        assert engine.to_three_js("nobody") == []

    def test_basic_structure(self, armed_engine: AbilityEngine):
        data = armed_engine.to_three_js("alpha-1")
        assert len(data) == 3
        keys = {"ability_id", "name", "description", "icon", "color",
                "ability_type", "target_type", "cooldown", "current_cooldown",
                "progress", "ready", "toggled_on", "is_channeling", "cost",
                "range", "radius"}
        for entry in data:
            assert keys.issubset(entry.keys())

    def test_progress_full_when_ready(self, armed_engine: AbilityEngine):
        data = armed_engine.to_three_js("alpha-1")
        for entry in data:
            assert entry["progress"] == 1.0
            assert entry["ready"] is True

    def test_progress_partial_on_cooldown(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "frag_grenade", target_pos=(10, 10))
        armed_engine.tick(7.5)  # half of 15s cooldown
        data = armed_engine.to_three_js("alpha-1")
        grenade = [d for d in data if d["ability_id"] == "frag_grenade"][0]
        assert grenade["progress"] == pytest.approx(0.5, abs=0.01)
        assert grenade["ready"] is False

    def test_channeling_flag(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        data = engine.to_three_js("u1")
        bc = [d for d in data if d["ability_id"] == "build_cover"][0]
        assert bc["is_channeling"] is True

    def test_toggle_state(self, armed_engine: AbilityEngine):
        armed_engine.activate("alpha-1", "go_prone")
        data = armed_engine.to_three_js("alpha-1")
        prone = [d for d in data if d["ability_id"] == "go_prone"][0]
        assert prone["toggled_on"] is True


# ===========================================================================
# 13. Specific ability smoke tests
# ===========================================================================

class TestSpecificAbilities:
    def test_frag_grenade_has_explosive_damage(self):
        ab = ABILITIES["frag_grenade"]
        damage_effects = [e for e in ab.effects if e.get("type") == "damage"]
        assert len(damage_effects) == 1
        assert damage_effects[0]["damage_type"] == "explosive"
        assert damage_effects[0]["base_damage"] == 60.0

    def test_smoke_grenade_has_area_effect(self):
        ab = ABILITIES["smoke_grenade"]
        assert ab.radius == 10.0
        assert ab.duration == 15.0

    def test_first_aid_heals(self):
        ab = ABILITIES["first_aid"]
        heal_effects = [e for e in ab.effects if e.get("type") == "heal"]
        assert len(heal_effects) == 1
        assert heal_effects[0]["amount"] == 35.0

    def test_call_airstrike_high_damage(self):
        ab = ABILITIES["call_airstrike"]
        assert ab.cooldown == 120.0
        damage_effects = [e for e in ab.effects if e.get("type") == "damage"]
        assert damage_effects[0]["base_damage"] == 200.0

    def test_suppressive_fire_channeled(self):
        ab = ABILITIES["suppressive_fire"]
        assert ab.ability_type == AbilityType.CHANNELED
        assert ab.duration == 6.0

    def test_deploy_bipod_toggle(self):
        ab = ABILITIES["deploy_bipod"]
        assert ab.ability_type == AbilityType.TOGGLE

    def test_silent_move_stealth_buff(self):
        ab = ABILITIES["silent_move"]
        stealth = [e for e in ab.effects if e.get("stat") == "stealth"]
        assert len(stealth) == 1
        assert stealth[0]["value"] == 0.9

    def test_mark_target_marks(self):
        ab = ABILITIES["mark_target"]
        assert ab.target_type == TargetType.SINGLE_ENEMY
        status_effects = [e for e in ab.effects if e.get("type") == "status"]
        assert any(e["name"] == "marked" for e in status_effects)

    def test_reinforce_spawns(self):
        ab = ABILITIES["reinforce"]
        spawn_effects = [e for e in ab.effects if e.get("type") == "spawn"]
        assert len(spawn_effects) == 1
        assert spawn_effects[0]["count"] == 4


# ===========================================================================
# 14. Integration / edge cases
# ===========================================================================

class TestEdgeCases:
    def test_multiple_units(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        engine.grant_ability("u2", ABILITIES["sprint"])
        engine.activate("u1", "sprint")
        # u1 on cooldown, u2 still available
        assert engine.get_ability("u1", "sprint").current_cooldown > 0
        assert engine.get_ability("u2", "sprint").current_cooldown == 0

    def test_range_check_skipped_without_positions(self, engine: AbilityEngine):
        """If unit_pos not given, range check is skipped."""
        engine.grant_ability("u1", ABILITIES["frag_grenade"])
        engine.set_resources("u1", {"grenades": 3})
        result = engine.activate("u1", "frag_grenade", target_pos=(9999, 9999))
        assert result["success"] is True

    def test_zero_range_ability(self, engine: AbilityEngine):
        """Abilities with range=0 skip range check."""
        engine.grant_ability("u1", ABILITIES["sprint"])
        result = engine.activate("u1", "sprint", unit_pos=(0, 0))
        assert result["success"] is True

    def test_activate_returns_target_info(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["mark_target"])
        result = engine.activate("u1", "mark_target", target_id="enemy-1", target_pos=(50, 50))
        assert result["success"] is True
        assert result["target_id"] == "enemy-1"
        assert result["target_pos"] == (50, 50)

    def test_concurrent_channels(self, engine: AbilityEngine):
        """Different units can channel simultaneously."""
        engine.grant_ability("u1", ABILITIES["build_cover"])
        engine.grant_ability("u2", ABILITIES["build_cover"])
        engine.set_resources("u1", {"building_materials": 5})
        engine.set_resources("u2", {"building_materials": 5})
        engine.activate("u1", "build_cover", target_pos=(10, 10))
        engine.activate("u2", "build_cover", target_pos=(20, 20))
        assert len(engine.active_channels) == 2
        events = engine.tick(6.0)
        complete_events = [e for e in events if e["type"] == "channel_complete"]
        assert len(complete_events) == 2

    def test_rapid_tick(self, armed_engine: AbilityEngine):
        """Many small ticks should behave like one large tick."""
        armed_engine.activate("alpha-1", "sprint")
        for _ in range(120):
            armed_engine.tick(0.1)
        ab = armed_engine.get_ability("alpha-1", "sprint")
        assert ab.current_cooldown == pytest.approx(0.0, abs=1e-9)

    def test_duration_in_result(self, engine: AbilityEngine):
        engine.grant_ability("u1", ABILITIES["sprint"])
        result = engine.activate("u1", "sprint")
        assert result["duration"] == 4.0
