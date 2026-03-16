"""Tests for tritium_lib.sim_engine.units — 50+ test cases.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

import math

import pytest

from tritium_lib.sim_engine.units import (
    Alliance,
    Unit,
    UnitState,
    UnitStats,
    UnitType,
    UNIT_TEMPLATES,
    create_unit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit(
    template: str = "infantry",
    alliance: Alliance = Alliance.FRIENDLY,
    position: tuple[float, float] = (0.0, 0.0),
    **overrides,
) -> Unit:
    return create_unit(template, "u1", "Test", alliance, position)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_unit_type_values(self):
        assert UnitType.INFANTRY.value == "infantry"
        assert UnitType.DRONE.value == "drone"

    def test_alliance_values(self):
        assert Alliance.FRIENDLY.value == "friendly"
        assert Alliance.HOSTILE.value == "hostile"
        assert Alliance.NEUTRAL.value == "neutral"
        assert Alliance.UNKNOWN.value == "unknown"

    def test_all_unit_types_exist(self):
        expected = {
            "infantry", "sniper", "heavy", "medic", "engineer",
            "scout", "vehicle", "drone", "turret", "civilian",
        }
        assert {t.value for t in UnitType} == expected


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestTemplates:
    @pytest.mark.parametrize("name", list(UNIT_TEMPLATES.keys()))
    def test_template_creates_unit(self, name):
        u = create_unit(name, f"id_{name}", name.title(), Alliance.FRIENDLY, (0, 0))
        assert u.is_alive()
        assert u.state.health == u.stats.max_health

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            create_unit("unknown_type", "x", "X", Alliance.FRIENDLY, (0, 0))

    def test_infantry_stats(self):
        u = _make_unit("infantry")
        assert u.stats.max_health == 100.0
        assert u.stats.armor == pytest.approx(0.1)
        assert u.stats.speed == 5.0

    def test_sniper_stats(self):
        u = _make_unit("sniper")
        assert u.stats.max_health == 60.0
        assert u.stats.attack_range == 80.0
        assert u.stats.accuracy == 0.9
        assert u.stats.attack_cooldown == 3.0

    def test_heavy_stats(self):
        u = _make_unit("heavy")
        assert u.stats.max_health == 200.0
        assert u.stats.armor == pytest.approx(0.4)

    def test_turret_stats(self):
        u = _make_unit("turret")
        assert u.stats.max_health == 500.0
        assert u.stats.speed == 0.0
        assert u.stats.armor == pytest.approx(0.6)

    def test_civilian_no_attack(self):
        u = _make_unit("civilian")
        assert u.stats.attack_damage == 0.0
        assert u.stats.accuracy == 0.0

    def test_drone_fast(self):
        u = _make_unit("drone")
        assert u.stats.speed == 12.0

    def test_scout_detection(self):
        u = _make_unit("scout")
        assert u.stats.detection_range == 80.0

    def test_medic_low_damage(self):
        u = _make_unit("medic")
        assert u.stats.attack_damage == 5.0


# ---------------------------------------------------------------------------
# Damage tests
# ---------------------------------------------------------------------------


class TestDamage:
    def test_basic_damage(self):
        u = _make_unit("scout")  # 0 armor
        actual = u.take_damage(20.0)
        assert actual == pytest.approx(20.0)
        assert u.state.health == pytest.approx(50.0)

    def test_armor_reduces_damage(self):
        u = _make_unit("infantry")  # 0.1 armor
        actual = u.take_damage(100.0)
        assert actual == pytest.approx(90.0)
        assert u.state.health == pytest.approx(10.0)

    def test_heavy_armor(self):
        u = _make_unit("heavy")  # 0.4 armor
        actual = u.take_damage(50.0)
        assert actual == pytest.approx(30.0)

    def test_turret_armor(self):
        u = _make_unit("turret")  # 0.6 armor
        actual = u.take_damage(100.0)
        assert actual == pytest.approx(40.0)

    def test_death_on_zero_health(self):
        u = _make_unit("civilian")  # 50hp, 0 armor
        u.take_damage(50.0)
        assert not u.is_alive()
        assert u.state.status == "dead"
        assert u.state.health == 0.0

    def test_death_on_overkill(self):
        u = _make_unit("civilian")
        u.take_damage(9999.0)
        assert not u.is_alive()
        assert u.state.health == 0.0

    def test_no_damage_to_dead(self):
        u = _make_unit("civilian")
        u.take_damage(9999.0)
        actual = u.take_damage(10.0)
        assert actual == 0.0

    def test_zero_damage(self):
        u = _make_unit("infantry")
        actual = u.take_damage(0.0)
        assert actual == 0.0
        assert u.state.health == 100.0

    def test_negative_damage(self):
        u = _make_unit("infantry")
        actual = u.take_damage(-5.0)
        assert actual == 0.0

    def test_damage_taken_tracked(self):
        u = _make_unit("scout")
        u.take_damage(10.0)
        u.take_damage(15.0)
        assert u.state.damage_taken == pytest.approx(25.0)

    def test_source_dir_accepted(self):
        u = _make_unit("infantry")
        actual = u.take_damage(10.0, source_dir=(1.0, 0.0))
        assert actual > 0


# ---------------------------------------------------------------------------
# Healing tests
# ---------------------------------------------------------------------------


class TestHealing:
    def test_basic_heal(self):
        u = _make_unit("infantry")
        u.take_damage(30.0)  # 0.1 armor -> 27 actual
        healed = u.heal(10.0)
        assert healed == pytest.approx(10.0)

    def test_heal_caps_at_max(self):
        u = _make_unit("infantry")
        u.take_damage(10.0)  # small damage
        healed = u.heal(9999.0)
        assert u.state.health == pytest.approx(u.stats.max_health)

    def test_heal_full_health(self):
        u = _make_unit("infantry")
        healed = u.heal(50.0)
        assert healed == 0.0

    def test_no_heal_when_dead(self):
        u = _make_unit("civilian")
        u.take_damage(9999.0)
        healed = u.heal(50.0)
        assert healed == 0.0
        assert not u.is_alive()

    def test_negative_heal(self):
        u = _make_unit("infantry")
        u.take_damage(20.0)
        healed = u.heal(-5.0)
        assert healed == 0.0


# ---------------------------------------------------------------------------
# Suppression tests
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_apply_suppression(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.5)
        assert u.state.suppression == pytest.approx(0.5)

    def test_suppression_clamps_high(self):
        u = _make_unit("infantry")
        u.apply_suppression(1.5)
        assert u.state.suppression == pytest.approx(1.0)

    def test_suppression_stacks(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.3)
        u.apply_suppression(0.4)
        assert u.state.suppression == pytest.approx(0.7)

    def test_recover_suppression(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.9)
        u.recover_suppression(dt=1.0, rate=0.3)
        assert u.state.suppression == pytest.approx(0.6)

    def test_recover_suppression_clamps_zero(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.1)
        u.recover_suppression(dt=10.0, rate=0.3)
        assert u.state.suppression == pytest.approx(0.0)

    def test_suppression_blocks_attack(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.95)
        assert not u.can_attack(sim_time=999.0)

    def test_suppression_reduces_accuracy(self):
        u = _make_unit("infantry")
        base = u.effective_accuracy()
        u.apply_suppression(0.5)
        assert u.effective_accuracy() == pytest.approx(base * 0.5)


# ---------------------------------------------------------------------------
# Morale tests
# ---------------------------------------------------------------------------


class TestMorale:
    def test_full_morale_full_accuracy(self):
        u = _make_unit("infantry")
        assert u.effective_accuracy() == pytest.approx(0.7)

    def test_low_morale_reduces_accuracy(self):
        u = _make_unit("infantry")
        u.state.morale = 0.5
        assert u.effective_accuracy() == pytest.approx(0.7 * 0.5)

    def test_zero_morale(self):
        u = _make_unit("infantry")
        u.state.morale = 0.0
        assert u.effective_accuracy() == pytest.approx(0.0)

    def test_morale_affects_speed(self):
        u = _make_unit("infantry")
        full = u.effective_speed()
        u.state.morale = 0.0
        half = u.effective_speed()
        assert half == pytest.approx(full * 0.5)

    def test_half_morale_speed(self):
        u = _make_unit("infantry")
        u.state.morale = 0.5
        # factor = 0.5 + 0.5*0.5 = 0.75
        assert u.effective_speed() == pytest.approx(5.0 * 0.75)


# ---------------------------------------------------------------------------
# Ammo tests
# ---------------------------------------------------------------------------


class TestAmmo:
    def test_unlimited_ammo_default(self):
        u = _make_unit("infantry")
        assert u.state.ammo == -1
        assert u.can_attack(sim_time=999.0)

    def test_zero_ammo_blocks_attack(self):
        u = _make_unit("infantry")
        u.state.ammo = 0
        assert not u.can_attack(sim_time=999.0)

    def test_positive_ammo_allows_attack(self):
        u = _make_unit("infantry")
        u.state.ammo = 5
        assert u.can_attack(sim_time=999.0)


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_can_attack_after_cooldown(self):
        u = _make_unit("infantry")  # 1.0s cooldown
        u.state.last_attack_time = 0.0
        assert u.can_attack(sim_time=1.0)

    def test_cannot_attack_during_cooldown(self):
        u = _make_unit("infantry")
        u.state.last_attack_time = 0.0
        assert not u.can_attack(sim_time=0.5)

    def test_sniper_long_cooldown(self):
        u = _make_unit("sniper")  # 3.0s cooldown
        u.state.last_attack_time = 0.0
        assert not u.can_attack(sim_time=2.0)
        assert u.can_attack(sim_time=3.0)


# ---------------------------------------------------------------------------
# Vision / distance tests
# ---------------------------------------------------------------------------


class TestVisionDistance:
    def test_distance_to(self):
        a = create_unit("infantry", "a", "A", Alliance.FRIENDLY, (0, 0))
        b = create_unit("infantry", "b", "B", Alliance.HOSTILE, (3, 4))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_can_see_in_range(self):
        a = create_unit("infantry", "a", "A", Alliance.FRIENDLY, (0, 0))
        b = create_unit("infantry", "b", "B", Alliance.HOSTILE, (10, 0))
        assert a.can_see(b)

    def test_cannot_see_out_of_range(self):
        a = create_unit("infantry", "a", "A", Alliance.FRIENDLY, (0, 0))
        b = create_unit("infantry", "b", "B", Alliance.HOSTILE, (100, 0))
        assert not a.can_see(b)

    def test_cannot_see_invisible(self):
        a = create_unit("infantry", "a", "A", Alliance.FRIENDLY, (0, 0))
        b = create_unit("infantry", "b", "B", Alliance.HOSTILE, (10, 0))
        b.state.is_visible = False
        assert not a.can_see(b)

    def test_sniper_sees_far(self):
        s = create_unit("sniper", "s", "S", Alliance.FRIENDLY, (0, 0))
        t = create_unit("infantry", "t", "T", Alliance.HOSTILE, (90, 0))
        assert s.can_see(t)

    def test_civilian_short_sight(self):
        c = create_unit("civilian", "c", "C", Alliance.NEUTRAL, (0, 0))
        t = create_unit("infantry", "t", "T", Alliance.HOSTILE, (25, 0))
        assert not c.can_see(t)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_with_correct_type(self):
        u = create_unit("sniper", "s1", "Sniper1", Alliance.HOSTILE, (5, 10))
        assert u.unit_type == UnitType.SNIPER
        assert u.alliance == Alliance.HOSTILE
        assert u.position == (5, 10)
        assert u.name == "Sniper1"
        assert u.unit_id == "s1"

    def test_health_matches_template(self):
        for name in UNIT_TEMPLATES:
            u = create_unit(name, "x", "X", Alliance.FRIENDLY, (0, 0))
            assert u.state.health == u.stats.max_health

    def test_default_weapon(self):
        u = create_unit("infantry", "x", "X", Alliance.FRIENDLY, (0, 0))
        assert u.weapon == "rifle"

    def test_default_heading(self):
        u = create_unit("infantry", "x", "X", Alliance.FRIENDLY, (0, 0))
        assert u.heading == 0.0

    def test_squad_id_none_by_default(self):
        u = create_unit("infantry", "x", "X", Alliance.FRIENDLY, (0, 0))
        assert u.squad_id is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_dead_unit_cannot_attack(self):
        u = _make_unit("infantry")
        u.state.is_alive = False
        assert not u.can_attack(sim_time=999.0)

    def test_dead_unit_cannot_heal(self):
        u = _make_unit("infantry")
        u.state.is_alive = False
        assert u.heal(50.0) == 0.0

    def test_turret_zero_speed(self):
        u = _make_unit("turret")
        assert u.effective_speed() == 0.0

    def test_combined_morale_suppression(self):
        u = _make_unit("infantry")  # accuracy 0.7
        u.state.morale = 0.5
        u.apply_suppression(0.5)
        # 0.7 * 0.5 * 0.5 = 0.175
        assert u.effective_accuracy() == pytest.approx(0.175)

    def test_multiple_damage_events(self):
        u = _make_unit("scout")  # 70hp, 0 armor
        u.take_damage(20.0)
        u.take_damage(20.0)
        u.take_damage(20.0)
        assert u.state.health == pytest.approx(10.0)
        assert u.is_alive()
        u.take_damage(10.0)
        assert not u.is_alive()

    def test_exact_lethal_damage(self):
        u = _make_unit("civilian")  # 50hp, 0 armor
        u.take_damage(50.0)
        assert not u.is_alive()
        assert u.state.health == 0.0

    def test_suppression_negative_apply(self):
        u = _make_unit("infantry")
        u.apply_suppression(0.5)
        u.apply_suppression(-0.3)
        assert u.state.suppression == pytest.approx(0.2)

    def test_distance_to_self(self):
        u = _make_unit("infantry")
        assert u.distance_to(u) == pytest.approx(0.0)
