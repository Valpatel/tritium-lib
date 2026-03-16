"""Tests for sim_engine.damage — ballistics, explosions, burst fire, tracking.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.damage import (
    DamageTracker,
    DamageType,
    HitResult,
    resolve_attack,
    resolve_burst,
    resolve_explosion,
    _range_modifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ORIGIN: tuple[float, float] = (0.0, 0.0)
NEAR: tuple[float, float] = (1.0, 0.0)       # 1 m away
FAR: tuple[float, float] = (500.0, 0.0)       # 500 m away
VERY_FAR: tuple[float, float] = (2000.0, 0.0) # 2 km away


def _seeded(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ===== DamageType enum =====================================================

class TestDamageType:
    def test_all_members(self):
        assert len(DamageType) == 5

    def test_values(self):
        assert DamageType.KINETIC.value == "kinetic"
        assert DamageType.EXPLOSIVE.value == "explosive"
        assert DamageType.FIRE.value == "fire"
        assert DamageType.ENERGY.value == "energy"
        assert DamageType.MELEE.value == "melee"


# ===== HitResult dataclass =================================================

class TestHitResult:
    def test_defaults(self):
        hr = HitResult(hit=True, damage=10.0, damage_type=DamageType.KINETIC)
        assert hr.critical is False
        assert hr.headshot is False
        assert hr.armor_absorbed == 0.0
        assert hr.suppression_caused == 0.0
        assert hr.source_id == ""
        assert hr.target_id == ""
        assert hr.range_m == 0.0

    def test_full_construction(self):
        hr = HitResult(
            hit=True, damage=50.0, damage_type=DamageType.FIRE,
            critical=True, headshot=True, armor_absorbed=5.0,
            suppression_caused=0.4, source_id="a", target_id="b", range_m=12.5,
        )
        assert hr.damage == 50.0
        assert hr.source_id == "a"


# ===== _range_modifier =====================================================

class TestRangeModifier:
    def test_within_falloff_start(self):
        assert _range_modifier(10.0, 50.0, 200.0) == 1.0

    def test_at_falloff_start(self):
        assert _range_modifier(50.0, 50.0, 200.0) == 1.0

    def test_at_falloff_end(self):
        assert _range_modifier(200.0, 50.0, 200.0) == pytest.approx(0.1)

    def test_beyond_falloff_end(self):
        assert _range_modifier(999.0, 50.0, 200.0) == pytest.approx(0.1)

    def test_midpoint(self):
        mid = (50.0 + 200.0) / 2.0  # 125
        val = _range_modifier(mid, 50.0, 200.0)
        assert 0.1 < val < 1.0
        assert val == pytest.approx(0.55)

    def test_equal_start_end(self):
        assert _range_modifier(50.0, 100.0, 100.0) == 1.0

    def test_zero_distance(self):
        assert _range_modifier(0.0, 50.0, 200.0) == 1.0


# ===== resolve_attack =======================================================

class TestResolveAttack:
    def test_point_blank_always_hits(self):
        """Accuracy 1.0 at range 0 should always hit."""
        rng = _seeded(1)
        for _ in range(50):
            r = resolve_attack(
                ORIGIN, ORIGIN, accuracy=1.0, damage=25.0,
                damage_type=DamageType.KINETIC, armor=0.0,
                range_falloff_start=100.0, range_falloff_end=200.0,
                critical_chance=0.0, headshot_chance=0.0, rng=rng,
            )
            assert r.hit is True

    def test_max_range_rarely_hits(self):
        """At extreme range with low accuracy, very few hits expected."""
        rng = _seeded(99)
        hits = 0
        for _ in range(200):
            r = resolve_attack(
                ORIGIN, VERY_FAR, accuracy=0.3, damage=25.0,
                damage_type=DamageType.KINETIC, armor=0.0,
                range_falloff_start=50.0, range_falloff_end=500.0, rng=rng,
            )
            if r.hit:
                hits += 1
        assert hits < 30  # <15% hit rate expected

    def test_zero_accuracy_never_hits(self):
        rng = _seeded(7)
        for _ in range(50):
            r = resolve_attack(
                ORIGIN, NEAR, accuracy=0.0, damage=50.0,
                damage_type=DamageType.ENERGY, armor=0.0,
                range_falloff_start=100.0, range_falloff_end=200.0, rng=rng,
            )
            assert r.hit is False

    def test_armor_reduces_damage(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=40.0,
            damage_type=DamageType.KINETIC, armor=0.5,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.hit is True
        assert r.damage == pytest.approx(20.0)
        assert r.armor_absorbed == pytest.approx(20.0)

    def test_full_armor_blocks_all(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=40.0,
            damage_type=DamageType.KINETIC, armor=1.0,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.hit is True
        assert r.damage == pytest.approx(0.0)

    def test_headshot_bypasses_armor(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=30.0,
            damage_type=DamageType.KINETIC, armor=0.8,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=1.0, rng=rng,
        )
        assert r.hit is True
        assert r.headshot is True
        assert r.armor_absorbed == 0.0
        # Headshot = 3x damage, no armor
        assert r.damage == pytest.approx(90.0)

    def test_critical_doubles_damage(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=30.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=1.0, headshot_chance=0.0, rng=rng,
        )
        assert r.hit is True
        assert r.critical is True
        assert r.damage == pytest.approx(60.0)

    def test_critical_and_headshot_stack(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=10.0,
            damage_type=DamageType.KINETIC, armor=0.5,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=1.0, headshot_chance=1.0, rng=rng,
        )
        assert r.critical is True
        assert r.headshot is True
        # 10 * 2 (crit) * 3 (headshot) = 60, armor bypassed
        assert r.damage == pytest.approx(60.0)

    def test_suppression_on_miss(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=0.0, damage=50.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            range_falloff_start=100.0, range_falloff_end=200.0, rng=rng,
        )
        assert r.hit is False
        assert r.suppression_caused > 0.0

    def test_suppression_on_hit(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=50.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.hit is True
        assert r.suppression_caused == pytest.approx(0.1 + 0.3 * (50.0 / 50.0))

    def test_range_recorded(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, FAR, accuracy=1.0, damage=10.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            range_falloff_start=1000.0, range_falloff_end=2000.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.range_m == pytest.approx(500.0)

    def test_source_target_ids(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=10.0,
            damage_type=DamageType.MELEE, armor=0.0,
            range_falloff_start=5.0, range_falloff_end=10.0,
            critical_chance=0.0, headshot_chance=0.0,
            rng=rng, source_id="unit_a", target_id="unit_b",
        )
        assert r.source_id == "unit_a"
        assert r.target_id == "unit_b"

    def test_zero_damage(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=0.0,
            damage_type=DamageType.KINETIC, armor=0.5,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.hit is True
        assert r.damage == 0.0

    def test_zero_armor(self):
        rng = _seeded(0)
        r = resolve_attack(
            ORIGIN, NEAR, accuracy=1.0, damage=30.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            range_falloff_start=100.0, range_falloff_end=200.0,
            critical_chance=0.0, headshot_chance=0.0, rng=rng,
        )
        assert r.armor_absorbed == 0.0
        assert r.damage == pytest.approx(30.0)

    def test_damage_type_preserved(self):
        rng = _seeded(0)
        for dt in DamageType:
            r = resolve_attack(
                ORIGIN, NEAR, accuracy=1.0, damage=10.0,
                damage_type=dt, armor=0.0,
                range_falloff_start=100.0, range_falloff_end=200.0,
                critical_chance=0.0, headshot_chance=0.0, rng=rng,
            )
            assert r.damage_type is dt


# ===== resolve_explosion ====================================================

class TestResolveExplosion:
    def test_targets_in_radius_hit(self):
        targets = [(ORIGIN, "a"), (NEAR, "b"), ((5.0, 0.0), "c")]
        results = resolve_explosion(ORIGIN, 10.0, targets, base_damage=100.0)
        assert len(results) == 3
        assert all(r.hit for r in results)

    def test_targets_outside_radius_not_hit(self):
        targets = [(FAR, "far_away")]
        results = resolve_explosion(ORIGIN, 10.0, targets, base_damage=100.0)
        assert len(results) == 0

    def test_epicenter_full_damage(self):
        targets = [(ORIGIN, "a")]
        results = resolve_explosion(ORIGIN, 50.0, targets, base_damage=100.0)
        assert len(results) == 1
        assert results[0].damage == pytest.approx(100.0)

    def test_linear_falloff(self):
        targets = [((25.0, 0.0), "half")]
        results = resolve_explosion(ORIGIN, 50.0, targets, base_damage=100.0, damage_falloff="linear")
        assert len(results) == 1
        assert results[0].damage == pytest.approx(50.0)

    def test_quadratic_falloff(self):
        targets = [((25.0, 0.0), "half")]
        results = resolve_explosion(ORIGIN, 50.0, targets, base_damage=100.0, damage_falloff="quadratic")
        assert len(results) == 1
        # 1 - (25/50)^2 = 1 - 0.25 = 0.75
        assert results[0].damage == pytest.approx(75.0)

    def test_edge_of_radius(self):
        targets = [((50.0, 0.0), "edge")]
        results = resolve_explosion(ORIGIN, 50.0, targets, base_damage=100.0)
        assert len(results) == 1
        assert results[0].damage == pytest.approx(0.0)

    def test_just_outside_radius(self):
        targets = [((50.1, 0.0), "outside")]
        results = resolve_explosion(ORIGIN, 50.0, targets, base_damage=100.0)
        assert len(results) == 0

    def test_explosion_damage_type_is_explosive(self):
        targets = [(NEAR, "a")]
        results = resolve_explosion(ORIGIN, 10.0, targets, base_damage=50.0)
        assert results[0].damage_type is DamageType.EXPLOSIVE

    def test_explosion_suppression(self):
        targets = [(NEAR, "a")]
        results = resolve_explosion(ORIGIN, 10.0, targets, base_damage=50.0)
        assert results[0].suppression_caused > 0.0

    def test_multiple_targets_varied_distances(self):
        targets = [
            ((0.0, 0.0), "center"),
            ((10.0, 0.0), "mid"),
            ((19.0, 0.0), "far"),
        ]
        results = resolve_explosion(ORIGIN, 20.0, targets, base_damage=100.0)
        assert len(results) == 3
        damages = [r.damage for r in results]
        # center > mid > far
        assert damages[0] > damages[1] > damages[2]

    def test_empty_targets(self):
        results = resolve_explosion(ORIGIN, 50.0, [], base_damage=100.0)
        assert results == []


# ===== resolve_burst ========================================================

class TestResolveBurst:
    def test_correct_number_of_rounds(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=10, accuracy=1.0,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.0, rng=_seeded(0),
        )
        assert len(results) == 10

    def test_zero_spread_all_hit_at_close_range(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=20, accuracy=1.0,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.0, rng=_seeded(0),
        )
        assert all(r.hit for r in results)

    def test_high_spread_reduces_hits(self):
        rng = _seeded(42)
        tight = resolve_burst(
            ORIGIN, NEAR, rounds=100, accuracy=0.8,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.0, rng=_seeded(42),
        )
        loose = resolve_burst(
            ORIGIN, NEAR, rounds=100, accuracy=0.8,
            damage_per_round=10.0, spread_deg=45.0,
            damage_type=DamageType.KINETIC, armor=0.0, rng=_seeded(42),
        )
        tight_hits = sum(1 for r in tight if r.hit)
        loose_hits = sum(1 for r in loose if r.hit)
        assert tight_hits >= loose_hits

    def test_burst_preserves_damage_type(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=3, accuracy=1.0,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.ENERGY, armor=0.0, rng=_seeded(0),
        )
        assert all(r.damage_type is DamageType.ENERGY for r in results)

    def test_burst_armor_applied(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=5, accuracy=1.0,
            damage_per_round=20.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.5,
            rng=_seeded(0),
        )
        for r in results:
            if r.hit and not r.headshot:
                assert r.damage <= 20.0

    def test_burst_source_target_ids(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=2, accuracy=1.0,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.0,
            rng=_seeded(0), source_id="s", target_id="t",
        )
        for r in results:
            assert r.source_id == "s"
            assert r.target_id == "t"

    def test_zero_rounds(self):
        results = resolve_burst(
            ORIGIN, NEAR, rounds=0, accuracy=1.0,
            damage_per_round=10.0, spread_deg=0.0,
            damage_type=DamageType.KINETIC, armor=0.0, rng=_seeded(0),
        )
        assert results == []


# ===== DamageTracker ========================================================

class TestDamageTracker:
    def _make_tracker(self) -> DamageTracker:
        """Build a tracker with some canned data."""
        dt = DamageTracker()
        dt.record(HitResult(hit=True, damage=50.0, damage_type=DamageType.KINETIC,
                            source_id="alpha", target_id="bravo"))
        dt.record(HitResult(hit=False, damage=0.0, damage_type=DamageType.KINETIC,
                            source_id="alpha", target_id="bravo"))
        dt.record(HitResult(hit=True, damage=120.0, damage_type=DamageType.EXPLOSIVE,
                            source_id="alpha", target_id="charlie"))
        dt.record(HitResult(hit=True, damage=30.0, damage_type=DamageType.KINETIC,
                            source_id="bravo", target_id="alpha"))
        return dt

    def test_total_damage_dealt(self):
        dt = self._make_tracker()
        assert dt.total_damage_dealt("alpha") == pytest.approx(170.0)

    def test_total_damage_taken(self):
        dt = self._make_tracker()
        assert dt.total_damage_taken("alpha") == pytest.approx(30.0)
        assert dt.total_damage_taken("bravo") == pytest.approx(50.0)

    def test_kill_count(self):
        dt = self._make_tracker()
        # Only the 120 dmg hit qualifies as a kill (>= 100)
        assert dt.kill_count("alpha") == 1
        assert dt.kill_count("bravo") == 0

    def test_accuracy_rate(self):
        dt = self._make_tracker()
        # alpha: 2 hits / 3 attempts
        assert dt.accuracy_rate("alpha") == pytest.approx(2 / 3)
        # bravo: 1 hit / 1 attempt
        assert dt.accuracy_rate("bravo") == pytest.approx(1.0)

    def test_accuracy_unknown_unit(self):
        dt = self._make_tracker()
        assert dt.accuracy_rate("nobody") == 0.0

    def test_mvp(self):
        dt = self._make_tracker()
        assert dt.mvp() == "alpha"

    def test_mvp_empty(self):
        dt = DamageTracker()
        assert dt.mvp() == ""

    def test_summary_keys(self):
        dt = self._make_tracker()
        s = dt.summary()
        assert "total_attacks" in s
        assert "total_hits" in s
        assert "total_misses" in s
        assert "total_damage" in s
        assert "total_criticals" in s
        assert "total_headshots" in s
        assert "mvp" in s
        assert "per_unit" in s

    def test_summary_totals(self):
        dt = self._make_tracker()
        s = dt.summary()
        assert s["total_attacks"] == 4
        assert s["total_hits"] == 3
        assert s["total_misses"] == 1
        assert s["total_damage"] == pytest.approx(200.0)

    def test_record_many(self):
        dt = DamageTracker()
        results = [
            HitResult(hit=True, damage=10.0, damage_type=DamageType.KINETIC, source_id="x"),
            HitResult(hit=True, damage=20.0, damage_type=DamageType.KINETIC, source_id="x"),
        ]
        dt.record_many(results)
        assert dt.total_damage_dealt("x") == pytest.approx(30.0)

    def test_damage_dealt_unknown(self):
        dt = self._make_tracker()
        assert dt.total_damage_dealt("zzz") == 0.0

    def test_damage_taken_unknown(self):
        dt = self._make_tracker()
        assert dt.total_damage_taken("zzz") == 0.0

    def test_per_unit_in_summary(self):
        dt = self._make_tracker()
        s = dt.summary()
        assert "alpha" in s["per_unit"]
        assert "bravo" in s["per_unit"]
        assert s["per_unit"]["alpha"]["damage_dealt"] == pytest.approx(170.0)
        assert s["per_unit"]["alpha"]["kills"] == 1
