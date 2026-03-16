# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.status_effects — buffs, debuffs, DOT/HOT, crowd control."""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.status_effects import (
    EFFECTS_CATALOG,
    EffectType,
    StatusEffect,
    StatusEffectEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_buff(name: str = "test_buff", duration: float = 10.0, **kw) -> StatusEffect:
    return StatusEffect(
        effect_id=f"test_{name}",
        name=name,
        effect_type=kw.pop("effect_type", EffectType.BUFF),
        duration=duration,
        remaining=duration if duration > 0 else float("inf"),
        **kw,
    )


# ---------------------------------------------------------------------------
# EffectType enum
# ---------------------------------------------------------------------------

class TestEffectType:
    def test_values(self):
        assert EffectType.BUFF.value == "buff"
        assert EffectType.DEBUFF.value == "debuff"
        assert EffectType.DOT.value == "dot"
        assert EffectType.HOT.value == "hot"
        assert EffectType.CROWD_CONTROL.value == "crowd_control"

    def test_count(self):
        assert len(EffectType) == 5


# ---------------------------------------------------------------------------
# StatusEffect dataclass
# ---------------------------------------------------------------------------

class TestStatusEffect:
    def test_basic_creation(self):
        e = _simple_buff()
        assert e.name == "test_buff"
        assert e.stacks == 1
        assert e.max_stacks == 1
        assert e.damage_per_second == 0.0
        assert e.heal_per_second == 0.0

    def test_stat_modifiers(self):
        e = _simple_buff(stat_modifiers={"speed": -0.3, "accuracy": 0.1})
        assert e.stat_modifiers["speed"] == -0.3
        assert e.stat_modifiers["accuracy"] == 0.1

    def test_clone_is_independent(self):
        original = _simple_buff(stat_modifiers={"speed": 0.5})
        cloned = original.clone()
        cloned.stacks = 5
        cloned.stat_modifiers["speed"] = 9.9
        assert original.stacks == 1
        assert original.stat_modifiers["speed"] == 0.5

    def test_default_color_and_icon(self):
        e = _simple_buff()
        assert e.color == "#ffffff"
        assert e.icon == ""

    def test_permanent_duration(self):
        e = _simple_buff(duration=-1)
        assert e.duration == -1
        assert e.remaining == float("inf")


# ---------------------------------------------------------------------------
# StatusEffectEngine — apply
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_new_effect(self):
        eng = StatusEffectEngine()
        eff = _simple_buff()
        result = eng.apply("u1", eff)
        assert result.name == "test_buff"
        assert eng.has_effect("u1", "test_buff")

    def test_apply_creates_deep_copy(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(stat_modifiers={"x": 1.0})
        eng.apply("u1", eff)
        eff.stat_modifiers["x"] = 999.0
        assert eng.get_effects("u1")[0].stat_modifiers["x"] == 1.0

    def test_apply_stacking(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(max_stacks=3)
        eng.apply("u1", eff)
        eng.apply("u1", eff)
        eng.apply("u1", eff)
        effects = eng.get_effects("u1")
        assert len(effects) == 1
        assert effects[0].stacks == 3

    def test_apply_at_max_stacks_does_not_exceed(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(max_stacks=2)
        eng.apply("u1", eff)
        eng.apply("u1", eff)
        eng.apply("u1", eff)  # should not exceed 2
        assert eng.get_effects("u1")[0].stacks == 2

    def test_apply_refreshes_duration(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(duration=10.0)
        eng.apply("u1", eff)
        # Simulate some time passing
        eng.active_effects["u1"][0].remaining = 3.0
        eng.apply("u1", _simple_buff(duration=10.0))
        assert eng.active_effects["u1"][0].remaining == 10.0

    def test_apply_different_effects_coexist(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("buff_a"))
        eng.apply("u1", _simple_buff("buff_b"))
        assert len(eng.get_effects("u1")) == 2

    def test_apply_permanent_effect(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(duration=-1)
        result = eng.apply("u1", eff)
        assert result.remaining == float("inf")

    def test_apply_to_multiple_units(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        eng.apply("u2", _simple_buff("y"))
        assert eng.has_effect("u1", "x")
        assert eng.has_effect("u2", "y")
        assert not eng.has_effect("u1", "y")


# ---------------------------------------------------------------------------
# StatusEffectEngine — remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_existing(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        assert eng.remove("u1", "x") is True
        assert not eng.has_effect("u1", "x")

    def test_remove_nonexistent_name(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        assert eng.remove("u1", "y") is False

    def test_remove_nonexistent_unit(self):
        eng = StatusEffectEngine()
        assert eng.remove("ghost", "x") is False

    def test_remove_cleans_up_empty_unit(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        eng.remove("u1", "x")
        assert "u1" not in eng.active_effects

    def test_remove_preserves_other_effects(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a"))
        eng.apply("u1", _simple_buff("b"))
        eng.remove("u1", "a")
        assert not eng.has_effect("u1", "a")
        assert eng.has_effect("u1", "b")


# ---------------------------------------------------------------------------
# StatusEffectEngine — clear_all
# ---------------------------------------------------------------------------

class TestClearAll:
    def test_clear_all(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a"))
        eng.apply("u1", _simple_buff("b"))
        count = eng.clear_all("u1")
        assert count == 2
        assert "u1" not in eng.active_effects

    def test_clear_all_empty_unit(self):
        eng = StatusEffectEngine()
        assert eng.clear_all("ghost") == 0

    def test_clear_all_does_not_affect_other_units(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a"))
        eng.apply("u2", _simple_buff("b"))
        eng.clear_all("u1")
        assert eng.has_effect("u2", "b")


# ---------------------------------------------------------------------------
# StatusEffectEngine — has_effect
# ---------------------------------------------------------------------------

class TestHasEffect:
    def test_has_effect_true(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        assert eng.has_effect("u1", "x") is True

    def test_has_effect_false(self):
        eng = StatusEffectEngine()
        assert eng.has_effect("u1", "x") is False

    def test_has_effect_after_remove(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("x"))
        eng.remove("u1", "x")
        assert eng.has_effect("u1", "x") is False


# ---------------------------------------------------------------------------
# StatusEffectEngine — get_modifier
# ---------------------------------------------------------------------------

class TestGetModifier:
    def test_no_effects(self):
        eng = StatusEffectEngine()
        assert eng.get_modifier("u1", "speed") == 0.0

    def test_single_effect(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(stat_modifiers={"speed": -0.3}))
        assert eng.get_modifier("u1", "speed") == pytest.approx(-0.3)

    def test_multiple_effects_additive(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a", stat_modifiers={"speed": -0.3}))
        eng.apply("u1", _simple_buff("b", stat_modifiers={"speed": 0.5}))
        assert eng.get_modifier("u1", "speed") == pytest.approx(0.2)

    def test_stacks_multiply(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(max_stacks=3, stat_modifiers={"speed": -0.1}))
        eng.apply("u1", _simple_buff(max_stacks=3, stat_modifiers={"speed": -0.1}))
        eng.apply("u1", _simple_buff(max_stacks=3, stat_modifiers={"speed": -0.1}))
        assert eng.get_modifier("u1", "speed") == pytest.approx(-0.3)

    def test_unrelated_stat_zero(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(stat_modifiers={"speed": -0.3}))
        assert eng.get_modifier("u1", "accuracy") == 0.0

    def test_modifier_across_units_independent(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(stat_modifiers={"speed": -0.5}))
        eng.apply("u2", _simple_buff(stat_modifiers={"speed": 0.2}))
        assert eng.get_modifier("u1", "speed") == pytest.approx(-0.5)
        assert eng.get_modifier("u2", "speed") == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# StatusEffectEngine — tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_reduces_remaining(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=10.0))
        eng.tick(3.0)
        assert eng.get_effects("u1")[0].remaining == pytest.approx(7.0)

    def test_tick_expires_effect(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=2.0))
        events = eng.tick(3.0)
        assert not eng.has_effect("u1", "test_buff")
        expired = [e for e in events if e["type"] == "effect_expired"]
        assert len(expired) == 1
        assert expired[0]["effect"] == "test_buff"

    def test_tick_dot_damage(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(
            "burn", duration=10.0,
            effect_type=EffectType.DOT, damage_per_second=5.0,
        )
        eng.apply("u1", eff)
        events = eng.tick(1.0)
        dots = [e for e in events if e["type"] == "dot_tick"]
        assert len(dots) == 1
        assert dots[0]["damage"] == pytest.approx(5.0)

    def test_tick_hot_healing(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(
            "regen", duration=10.0,
            effect_type=EffectType.HOT, heal_per_second=3.0,
        )
        eng.apply("u1", eff)
        events = eng.tick(2.0)
        hots = [e for e in events if e["type"] == "hot_tick"]
        assert len(hots) == 1
        assert hots[0]["healing"] == pytest.approx(6.0)

    def test_tick_dot_scales_with_stacks(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(
            "bleed", duration=30.0,
            effect_type=EffectType.DOT, damage_per_second=2.0, max_stacks=3,
        )
        eng.apply("u1", eff)
        eng.apply("u1", eff)
        eng.apply("u1", eff)
        events = eng.tick(1.0)
        dots = [e for e in events if e["type"] == "dot_tick"]
        assert dots[0]["damage"] == pytest.approx(6.0)  # 2.0 * 3 stacks * 1s

    def test_tick_permanent_effect_never_expires(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=-1))
        events = eng.tick(1000.0)
        assert eng.has_effect("u1", "test_buff")
        expired = [e for e in events if e["type"] == "effect_expired"]
        assert len(expired) == 0

    def test_tick_multiple_effects_on_same_unit(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a", duration=5.0))
        eng.apply("u1", _simple_buff("b", duration=3.0))
        events = eng.tick(4.0)
        assert eng.has_effect("u1", "a")
        assert not eng.has_effect("u1", "b")

    def test_tick_zero_dt(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=5.0))
        events = eng.tick(0.0)
        assert eng.has_effect("u1", "test_buff")
        assert len(events) == 0

    def test_tick_cleans_up_empty_unit(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=1.0))
        eng.tick(2.0)
        assert "u1" not in eng.active_effects

    def test_tick_fractional_dt(self):
        eng = StatusEffectEngine()
        eff = _simple_buff(
            "dot", duration=10.0,
            effect_type=EffectType.DOT, damage_per_second=10.0,
        )
        eng.apply("u1", eff)
        events = eng.tick(0.1)
        dots = [e for e in events if e["type"] == "dot_tick"]
        assert dots[0]["damage"] == pytest.approx(1.0)

    def test_tick_multiple_units(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a", duration=2.0))
        eng.apply("u2", _simple_buff("b", duration=5.0))
        events = eng.tick(3.0)
        assert not eng.has_effect("u1", "a")
        assert eng.has_effect("u2", "b")


# ---------------------------------------------------------------------------
# StatusEffectEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_empty_unit(self):
        eng = StatusEffectEngine()
        assert eng.to_three_js("ghost") == []

    def test_basic_output_shape(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(
            duration=10.0, icon="star", color="#00ff00",
        ))
        result = eng.to_three_js("u1")
        assert len(result) == 1
        r = result[0]
        assert r["name"] == "test_buff"
        assert r["icon"] == "star"
        assert r["color"] == "#00ff00"
        assert r["effect_type"] == "buff"
        assert r["stacks"] == 1
        assert r["max_stacks"] == 1
        assert r["duration"] == 10.0
        assert 0.0 <= r["progress"] <= 1.0

    def test_progress_decreases_with_time(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=10.0))
        eng.tick(5.0)
        result = eng.to_three_js("u1")
        assert result[0]["progress"] == pytest.approx(0.5, abs=0.01)

    def test_permanent_effect_progress_is_one(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff(duration=-1))
        result = eng.to_three_js("u1")
        assert result[0]["progress"] == 1.0
        assert result[0]["remaining"] == -1

    def test_multiple_effects_listed(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff("a", duration=10.0))
        eng.apply("u1", _simple_buff("b", duration=5.0))
        result = eng.to_three_js("u1")
        names = {r["name"] for r in result}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# EFFECTS_CATALOG completeness
# ---------------------------------------------------------------------------

class TestEffectsCatalog:
    def test_catalog_has_at_least_30_entries(self):
        assert len(EFFECTS_CATALOG) >= 30

    def test_all_entries_are_status_effects(self):
        for name, eff in EFFECTS_CATALOG.items():
            assert isinstance(eff, StatusEffect), f"{name} is not a StatusEffect"

    def test_all_entries_have_valid_type(self):
        for name, eff in EFFECTS_CATALOG.items():
            assert isinstance(eff.effect_type, EffectType), f"{name} bad type"

    def test_combat_effects_present(self):
        for name in ["suppressed", "pinned", "flanked", "entrenched", "overwatch", "adrenaline"]:
            assert name in EFFECTS_CATALOG, f"missing combat effect: {name}"

    def test_medical_effects_present(self):
        for name in ["bleeding", "concussed", "morphine", "bandaged", "tourniquet"]:
            assert name in EFFECTS_CATALOG, f"missing medical effect: {name}"

    def test_environmental_effects_present(self):
        for name in ["burning", "frozen", "soaked", "blinded", "deafened", "irradiated"]:
            assert name in EFFECTS_CATALOG, f"missing env effect: {name}"

    def test_tactical_effects_present(self):
        for name in ["camouflaged", "spotted", "marked", "jammed", "hacked"]:
            assert name in EFFECTS_CATALOG, f"missing tactical effect: {name}"

    def test_morale_effects_present(self):
        for name in ["inspired", "terrified", "berserk", "shell_shocked", "rallied"]:
            assert name in EFFECTS_CATALOG, f"missing morale effect: {name}"

    def test_vehicle_effects_present(self):
        for name in ["engine_damage", "flat_tire", "fuel_leak", "turret_jammed"]:
            assert name in EFFECTS_CATALOG, f"missing vehicle effect: {name}"

    def test_catalog_effects_are_independent_copies_when_applied(self):
        """Applying a catalog effect should not mutate the catalog."""
        eng = StatusEffectEngine()
        orig_stacks = EFFECTS_CATALOG["suppressed"].stacks
        eng.apply("u1", EFFECTS_CATALOG["suppressed"])
        eng.apply("u1", EFFECTS_CATALOG["suppressed"])
        assert EFFECTS_CATALOG["suppressed"].stacks == orig_stacks

    def test_dot_effects_have_damage(self):
        dot_effects = [e for e in EFFECTS_CATALOG.values() if e.effect_type == EffectType.DOT]
        assert len(dot_effects) >= 3
        for eff in dot_effects:
            assert eff.damage_per_second > 0, f"{eff.name} DOT has no damage"

    def test_hot_effects_have_healing(self):
        hot_effects = [e for e in EFFECTS_CATALOG.values() if e.effect_type == EffectType.HOT]
        assert len(hot_effects) >= 1
        for eff in hot_effects:
            assert eff.heal_per_second > 0, f"{eff.name} HOT has no healing"

    def test_all_effects_have_color(self):
        for name, eff in EFFECTS_CATALOG.items():
            assert eff.color.startswith("#"), f"{name} bad color: {eff.color}"

    def test_all_effects_have_icon(self):
        for name, eff in EFFECTS_CATALOG.items():
            assert len(eff.icon) > 0, f"{name} missing icon"


# ---------------------------------------------------------------------------
# Integration: catalog effects through the engine
# ---------------------------------------------------------------------------

class TestCatalogIntegration:
    def test_suppressed_reduces_accuracy(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["suppressed"])
        assert eng.get_modifier("u1", "accuracy") < 0

    def test_bleeding_does_dot_damage(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["bleeding"])
        events = eng.tick(1.0)
        dots = [e for e in events if e["type"] == "dot_tick"]
        assert len(dots) == 1
        assert dots[0]["damage"] > 0

    def test_bandaged_heals_over_time(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["bandaged"])
        events = eng.tick(1.0)
        hots = [e for e in events if e["type"] == "hot_tick"]
        assert len(hots) == 1
        assert hots[0]["healing"] > 0

    def test_entrenched_is_permanent(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["entrenched"])
        eng.tick(9999.0)
        assert eng.has_effect("u1", "entrenched")

    def test_bleeding_stacks_three_times(self):
        eng = StatusEffectEngine()
        for _ in range(5):
            eng.apply("u1", EFFECTS_CATALOG["bleeding"])
        assert eng.get_effects("u1")[0].stacks == 3

    def test_engine_damage_permanent_and_stackable(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["engine_damage"])
        eng.apply("u1", EFFECTS_CATALOG["engine_damage"])
        eff = eng.get_effects("u1")[0]
        assert eff.stacks == 2
        speed_mod = eng.get_modifier("u1", "speed")
        assert speed_mod == pytest.approx(-1.0)  # -0.5 * 2 stacks

    def test_morphine_heals_and_debuffs(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["morphine"])
        events = eng.tick(1.0)
        # Should have HOT tick
        hots = [e for e in events if e["type"] == "hot_tick"]
        assert len(hots) == 1
        # Should reduce accuracy
        assert eng.get_modifier("u1", "accuracy") < 0

    def test_combo_buff_debuff_on_same_stat(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["overwatch"])   # accuracy +0.2
        eng.apply("u1", EFFECTS_CATALOG["concussed"])    # accuracy -0.5
        mod = eng.get_modifier("u1", "accuracy")
        assert mod == pytest.approx(-0.3)

    def test_to_three_js_with_catalog_effects(self):
        eng = StatusEffectEngine()
        eng.apply("u1", EFFECTS_CATALOG["burning"])
        eng.apply("u1", EFFECTS_CATALOG["inspired"])
        result = eng.to_three_js("u1")
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "burning" in names
        assert "inspired" in names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_effects_returns_copy(self):
        eng = StatusEffectEngine()
        eng.apply("u1", _simple_buff())
        effects = eng.get_effects("u1")
        effects.clear()
        assert len(eng.get_effects("u1")) == 1

    def test_apply_same_name_different_type(self):
        """Same name = same effect regardless of type differences in the template."""
        eng = StatusEffectEngine()
        a = _simple_buff("test", effect_type=EffectType.BUFF, max_stacks=2)
        b = _simple_buff("test", effect_type=EffectType.DEBUFF, max_stacks=2)
        eng.apply("u1", a)
        eng.apply("u1", b)
        assert len(eng.get_effects("u1")) == 1
        assert eng.get_effects("u1")[0].stacks == 2

    def test_large_dt_expires_everything(self):
        eng = StatusEffectEngine()
        for i in range(10):
            eng.apply("u1", _simple_buff(f"e{i}", duration=float(i + 1)))
        events = eng.tick(100.0)
        assert "u1" not in eng.active_effects
        expired = [e for e in events if e["type"] == "effect_expired"]
        assert len(expired) == 10

    def test_many_units(self):
        eng = StatusEffectEngine()
        for i in range(100):
            eng.apply(f"unit_{i}", _simple_buff(duration=5.0))
        eng.tick(6.0)
        assert len(eng.active_effects) == 0

    def test_tick_with_no_effects(self):
        eng = StatusEffectEngine()
        events = eng.tick(1.0)
        assert events == []
