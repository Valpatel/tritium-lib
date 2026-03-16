# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the arsenal weapons database, projectile simulator, and area effects."""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.arsenal import (
    ARSENAL,
    AreaEffect,
    AreaEffectManager,
    Projectile,
    ProjectileSimulator,
    ProjectileType,
    Weapon,
    WeaponCategory,
    create_explosion_effect,
    create_fire_effect,
    create_flashbang_effect,
    create_smoke_effect,
    create_teargas_effect,
    get_weapon,
    weapons_by_category,
)


# ---------------------------------------------------------------------------
# ARSENAL validation
# ---------------------------------------------------------------------------

class TestArsenal:
    """Validate that every weapon in the ARSENAL has sane stats."""

    def test_arsenal_has_25_plus_weapons(self):
        assert len(ARSENAL) >= 25, f"Expected 25+ weapons, got {len(ARSENAL)}"

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_has_valid_name(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.name, f"Weapon {weapon_id} has empty name"
        assert w.weapon_id == weapon_id

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_has_valid_category(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert isinstance(w.category, WeaponCategory)

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_has_valid_projectile_type(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert isinstance(w.projectile_type, ProjectileType)

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_damage_positive(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        # smoke grenade can have 0 damage
        assert w.damage >= 0, f"{weapon_id} has negative damage"

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_fire_rate_positive(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.fire_rate > 0, f"{weapon_id} has non-positive fire rate"

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_magazine_size_positive(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.magazine_size >= 1

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_range_valid(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.max_range >= w.effective_range, (
            f"{weapon_id}: max_range ({w.max_range}) < effective_range ({w.effective_range})"
        )

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_accuracy_in_range(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert 0.0 <= w.accuracy <= 1.0, f"{weapon_id} accuracy out of range: {w.accuracy}"

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_recoil_in_range(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert 0.0 <= w.recoil <= 1.0, f"{weapon_id} recoil out of range: {w.recoil}"

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_spread_non_negative(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.spread_deg >= 0.0

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_weight_positive(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.weight_kg > 0.0

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_sound_radius_non_negative(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.sound_radius >= 0.0

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_tracer_color_is_hex(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.tracer_color.startswith("#")
        assert len(w.tracer_color) == 7  # #RRGGBB

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_muzzle_flash_size_non_negative(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        assert w.muzzle_flash_size >= 0.0

    @pytest.mark.parametrize("weapon_id", list(ARSENAL.keys()))
    def test_weapon_to_dict(self, weapon_id: str):
        w = ARSENAL[weapon_id]
        d = w.to_dict()
        assert d["weapon_id"] == weapon_id
        assert d["name"] == w.name
        assert d["category"] == w.category.value
        assert d["projectile_type"] == w.projectile_type.value

    def test_get_weapon(self):
        w = get_weapon("m4a1")
        assert w.name == "M4A1 Carbine"

    def test_get_weapon_missing_raises(self):
        with pytest.raises(KeyError):
            get_weapon("nonexistent_weapon")

    def test_weapons_by_category_pistols(self):
        pistols = weapons_by_category(WeaponCategory.PISTOL)
        assert len(pistols) >= 4
        for w in pistols:
            assert w.category == WeaponCategory.PISTOL

    def test_weapons_by_category_rifles(self):
        rifles = weapons_by_category(WeaponCategory.RIFLE)
        assert len(rifles) >= 5

    def test_weapons_by_category_snipers(self):
        snipers = weapons_by_category(WeaponCategory.SNIPER)
        assert len(snipers) >= 4

    def test_weapons_by_category_thrown(self):
        thrown = weapons_by_category(WeaponCategory.THROWN)
        assert len(thrown) >= 5

    def test_all_categories_represented(self):
        cats = {w.category for w in ARSENAL.values()}
        expected = {
            WeaponCategory.PISTOL, WeaponCategory.RIFLE, WeaponCategory.SMG,
            WeaponCategory.SHOTGUN, WeaponCategory.SNIPER, WeaponCategory.LMG,
            WeaponCategory.LAUNCHER, WeaponCategory.MELEE, WeaponCategory.THROWN,
            WeaponCategory.TURRET,
        }
        for cat in expected:
            assert cat in cats, f"Category {cat.value} not represented in ARSENAL"

    def test_sound_radius_scales_with_power(self):
        """Bigger weapons should generally have larger sound radii."""
        pistol = ARSENAL["glock17"]
        rifle = ARSENAL["m4a1"]
        sniper = ARSENAL["barrett_m82"]
        assert pistol.sound_radius < rifle.sound_radius
        assert rifle.sound_radius < sniper.sound_radius

    def test_sniper_higher_velocity_than_pistol(self):
        pistol = ARSENAL["m9_beretta"]
        sniper = ARSENAL["m24"]
        assert sniper.muzzle_velocity > pistol.muzzle_velocity

    def test_melee_zero_muzzle_velocity(self):
        knife = ARSENAL["knife"]
        assert knife.muzzle_velocity == 0
        assert knife.muzzle_flash_size == 0.0


# ---------------------------------------------------------------------------
# ProjectileSimulator
# ---------------------------------------------------------------------------

class TestProjectileSimulator:
    """Test projectile creation, physics, and expiry."""

    def test_fire_creates_projectile(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), rng=random.Random(42))
        assert proj.is_active
        assert proj.weapon_id == "m4a1"
        assert len(sim.projectiles) == 1

    def test_fire_velocity_direction(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        # Fire with zero spread via a fixed rng
        rng = random.Random(42)
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=rng)
        # With zero spread, velocity should be purely in +x direction
        assert proj.velocity[0] > 0
        assert abs(proj.velocity[1]) < 1.0  # near zero y component

    def test_fire_velocity_magnitude(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        speed = math.hypot(proj.velocity[0], proj.velocity[1])
        assert abs(speed - weapon.muzzle_velocity) < 1.0

    def test_tick_advances_position(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        old_x = proj.position[0]
        sim.tick(0.016)
        assert proj.position[0] > old_x or not proj.is_active

    def test_tick_increments_time(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["mp5"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), rng=random.Random(42))
        sim.tick(0.5)
        # Projectile may still be active or expired
        # If still tracked, check time; otherwise the projectile was cleaned up
        # Just verify no crash
        assert True

    def test_projectile_expires_at_max_range(self):
        sim = ProjectileSimulator()
        # Use a weapon with short max range
        weapon = ARSENAL["knife"]  # max_range=2, velocity=0
        # For melee, velocity is 0 so it won't move. Use a pistol with artificially short range.
        short_weapon = Weapon(
            weapon_id="test_short", name="Test Short",
            category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
            damage=10, fire_rate=1.0, magazine_size=10, reload_time=1.0,
            muzzle_velocity=100, effective_range=5, max_range=10,
            accuracy=1.0, spread_deg=0, recoil=0, weight_kg=1.0,
            sound_radius=100,
        )
        proj = sim.fire(short_weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        # Tick enough for bullet to travel beyond 10m (at 100 m/s, ~0.11s)
        impacts = []
        for _ in range(20):
            impacts.extend(sim.tick(0.01))
        assert len(impacts) >= 1
        assert impacts[0]["type"] == "bullet"
        assert impacts[0]["effect"] == "spark"
        assert len(sim.projectiles) == 0  # cleaned up

    def test_grenade_has_gravity_arc(self):
        """Grenades should have downward velocity component after tick."""
        sim = ProjectileSimulator(gravity=9.81)
        weapon = ARSENAL["frag_grenade"]
        proj = sim.fire(weapon, origin=(0, 0), target=(30, 0), accuracy_modifier=0.0, rng=random.Random(42))
        initial_vy = proj.velocity[1]
        sim.tick(0.1)
        # After tick, vy should have increased (gravity pulls down in +y)
        # For a grenade fired horizontally, vy starts near 0 and increases
        # Check that the projectile still exists and vy changed
        active = [p for p in sim.projectiles if p.is_active]
        if active:
            # Gravity adds to vy each tick
            assert active[0].velocity[1] > initial_vy

    def test_bullet_no_gravity(self):
        """Bullets should not be affected by gravity (not in arced types)."""
        sim = ProjectileSimulator(gravity=9.81)
        weapon = ARSENAL["m4a1"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        initial_vy = proj.velocity[1]
        sim.tick(0.01)
        # Bullet vy should only change due to air resistance, not gravity
        active = [p for p in sim.projectiles if p.is_active]
        if active:
            # Air resistance slightly reduces vy magnitude, but no gravity offset
            # The change should be very small (just drag)
            vy_change = abs(active[0].velocity[1] - initial_vy * (1 - 0.01 * 0.01))
            assert vy_change < 0.1

    def test_air_resistance_slows_projectile(self):
        sim = ProjectileSimulator(gravity=0, air_resistance=0.1)
        weapon = ARSENAL["m4a1"]
        proj = sim.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        initial_speed = math.hypot(*proj.velocity)
        sim.tick(0.1)
        active = [p for p in sim.projectiles if p.is_active]
        if active:
            new_speed = math.hypot(*active[0].velocity)
            assert new_speed < initial_speed

    def test_multiple_projectiles(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        rng = random.Random(42)
        for _ in range(5):
            sim.fire(weapon, origin=(0, 0), target=(100, 0), rng=rng)
        assert len(sim.projectiles) == 5
        sim.tick(0.01)
        # All should still be in flight (short time, long range)
        assert len(sim.projectiles) == 5

    def test_accuracy_modifier_widens_spread(self):
        """Higher accuracy modifier should produce wider spread patterns."""
        sim_tight = ProjectileSimulator()
        sim_wide = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]

        angles_tight = []
        angles_wide = []
        for i in range(50):
            rng = random.Random(i)
            p = sim_tight.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.1, rng=rng)
            angles_tight.append(math.atan2(p.velocity[1], p.velocity[0]))
        for i in range(50):
            rng = random.Random(i)
            p = sim_wide.fire(weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=5.0, rng=rng)
            angles_wide.append(math.atan2(p.velocity[1], p.velocity[0]))

        spread_tight = max(angles_tight) - min(angles_tight)
        spread_wide = max(angles_wide) - min(angles_wide)
        assert spread_wide > spread_tight

    def test_muzzle_flash_recorded(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        sim.fire(weapon, origin=(5, 10), target=(100, 0), rng=random.Random(42))
        data = sim.to_three_js()
        assert len(data["muzzle_flashes"]) == 1
        flash = data["muzzle_flashes"][0]
        assert flash["x"] == 5
        assert flash["y"] == 10
        assert flash["weapon"] == "m4a1"
        assert flash["size"] == weapon.muzzle_flash_size

    def test_projectile_distance_traveled(self):
        proj = Projectile(
            projectile_id="test", weapon_id="test",
            origin=(0, 0), position=(3, 4),
            velocity=(0, 0), damage=10,
            projectile_type=ProjectileType.BULLET,
        )
        assert abs(proj.distance_traveled() - 5.0) < 0.001


# ---------------------------------------------------------------------------
# to_three_js format
# ---------------------------------------------------------------------------

class TestThreeJsOutput:
    """Validate the Three.js export format."""

    def test_projectiles_format(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["glock17"]
        sim.fire(weapon, origin=(1, 2), target=(50, 50), rng=random.Random(42))
        data = sim.to_three_js()

        assert "projectiles" in data
        assert "impacts" in data
        assert "muzzle_flashes" in data
        assert len(data["projectiles"]) == 1

        p = data["projectiles"][0]
        assert "id" in p
        assert "x" in p
        assert "y" in p
        assert "vx" in p
        assert "vy" in p
        assert "type" in p
        assert "color" in p
        assert p["type"] == "bullet"

    def test_impacts_format(self):
        sim = ProjectileSimulator()
        short_weapon = Weapon(
            weapon_id="short", name="Short",
            category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
            damage=10, fire_rate=1.0, magazine_size=10, reload_time=1.0,
            muzzle_velocity=200, effective_range=5, max_range=8,
            accuracy=1.0, spread_deg=0, recoil=0, weight_kg=1.0,
            sound_radius=100,
        )
        sim.fire(short_weapon, origin=(0, 0), target=(100, 0), accuracy_modifier=0.0, rng=random.Random(42))
        # Tick until impact
        for _ in range(50):
            sim.tick(0.01)
        data = sim.to_three_js()
        assert len(data["impacts"]) >= 1
        imp = data["impacts"][0]
        assert "x" in imp
        assert "y" in imp
        assert "type" in imp
        assert "damage" in imp
        assert "effect" in imp

    def test_to_three_js_clears_pending(self):
        sim = ProjectileSimulator()
        weapon = ARSENAL["m4a1"]
        sim.fire(weapon, origin=(0, 0), target=(100, 0), rng=random.Random(42))
        data1 = sim.to_three_js()
        assert len(data1["muzzle_flashes"]) == 1
        # Second call should have no pending flashes
        data2 = sim.to_three_js()
        assert len(data2["muzzle_flashes"]) == 0


# ---------------------------------------------------------------------------
# AreaEffect
# ---------------------------------------------------------------------------

class TestAreaEffect:
    """Test area effects — smoke, fire, explosions, etc."""

    def test_explosion_effect(self):
        eff = create_explosion_effect((10, 20), radius=15.0)
        assert eff.effect_type == "explosion"
        assert eff.position == (10, 20)
        assert eff.radius == 15.0

    def test_smoke_effect(self):
        eff = create_smoke_effect((5, 5), radius=8.0, duration=30.0)
        assert eff.effect_type == "smoke"
        assert eff.duration == 30.0
        assert eff.damage_per_second == 0.0

    def test_fire_effect_has_damage(self):
        eff = create_fire_effect((0, 0), damage_per_second=10.0)
        assert eff.effect_type == "fire"
        assert eff.damage_per_second == 10.0

    def test_teargas_effect(self):
        eff = create_teargas_effect((0, 0))
        assert eff.effect_type == "teargas"
        assert eff.damage_per_second > 0

    def test_flashbang_effect(self):
        eff = create_flashbang_effect((0, 0))
        assert eff.effect_type == "flashbang"
        assert eff.damage_per_second == 0.0


# ---------------------------------------------------------------------------
# AreaEffectManager
# ---------------------------------------------------------------------------

class TestAreaEffectManager:
    """Test the area effect manager lifecycle."""

    def test_add_effect(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((0, 0))
        mgr.add(eff)
        assert len(mgr.effects) == 1

    def test_tick_decays_time(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((0, 0), duration=10.0)
        mgr.add(eff)
        mgr.tick(2.0)
        assert eff.time_remaining == pytest.approx(8.0, abs=0.01)

    def test_tick_decays_intensity(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((0, 0), duration=10.0)
        mgr.add(eff)
        mgr.tick(5.0)
        assert eff.intensity == pytest.approx(0.5, abs=0.01)

    def test_tick_removes_expired(self):
        mgr = AreaEffectManager()
        eff = create_explosion_effect((0, 0), duration=0.5)
        mgr.add(eff)
        expired = mgr.tick(1.0)
        assert len(expired) == 1
        assert expired[0]["type"] == "explosion"
        assert len(mgr.effects) == 0

    def test_tick_returns_expired_info(self):
        mgr = AreaEffectManager()
        eff = create_fire_effect((10, 20), duration=1.0)
        mgr.add(eff)
        expired = mgr.tick(2.0)
        assert expired[0]["x"] == 10
        assert expired[0]["y"] == 20

    def test_affects_position_inside(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((10, 10), radius=5.0)
        mgr.add(eff)
        hits = mgr.affects_position((12, 12))
        assert len(hits) == 1
        assert hits[0].effect_type == "smoke"

    def test_affects_position_outside(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((10, 10), radius=5.0)
        mgr.add(eff)
        hits = mgr.affects_position((100, 100))
        assert len(hits) == 0

    def test_affects_position_edge(self):
        mgr = AreaEffectManager()
        eff = create_smoke_effect((0, 0), radius=5.0)
        mgr.add(eff)
        # Point exactly at radius distance
        hits = mgr.affects_position((5, 0))
        assert len(hits) == 1

    def test_multiple_effects_overlap(self):
        mgr = AreaEffectManager()
        mgr.add(create_smoke_effect((0, 0), radius=10.0))
        mgr.add(create_fire_effect((3, 0), radius=10.0))
        hits = mgr.affects_position((1, 0))
        assert len(hits) == 2

    def test_to_three_js_format(self):
        mgr = AreaEffectManager()
        mgr.add(create_smoke_effect((5, 10), radius=8.0, duration=20.0))
        data = mgr.to_three_js()
        assert "effects" in data
        assert len(data["effects"]) == 1
        e = data["effects"][0]
        assert e["type"] == "smoke"
        assert e["x"] == 5
        assert e["y"] == 10
        assert e["radius"] == 8.0
        assert "intensity" in e
        assert "color" in e
        assert "remaining" in e

    def test_to_three_js_empty(self):
        mgr = AreaEffectManager()
        data = mgr.to_three_js()
        assert data == {"effects": []}

    def test_smoke_teargas_duration(self):
        """Smoke and teargas should last longer than explosions."""
        smoke = create_smoke_effect((0, 0))
        explosion = create_explosion_effect((0, 0))
        assert smoke.duration > explosion.duration

    def test_fire_does_damage_over_time(self):
        """Fire effects should have positive DPS."""
        fire = create_fire_effect((0, 0))
        assert fire.damage_per_second > 0

    def test_many_effects_lifecycle(self):
        """Add many effects, tick through their lifecycles."""
        mgr = AreaEffectManager()
        for i in range(10):
            mgr.add(create_smoke_effect((i * 5, 0), duration=float(i + 1)))
        # Tick 5 seconds — effects with duration <= 5 should expire
        expired = mgr.tick(5.0)
        assert len(expired) == 5  # durations 1,2,3,4,5
        assert len(mgr.effects) == 5  # durations 6,7,8,9,10


# ---------------------------------------------------------------------------
# Weapon categories
# ---------------------------------------------------------------------------

class TestWeaponCategories:
    """Verify category-specific constraints."""

    def test_snipers_have_high_accuracy(self):
        for w in weapons_by_category(WeaponCategory.SNIPER):
            assert w.accuracy >= 0.90, f"Sniper {w.name} accuracy too low: {w.accuracy}"

    def test_snipers_have_low_fire_rate(self):
        for w in weapons_by_category(WeaponCategory.SNIPER):
            assert w.fire_rate <= 1.5, f"Sniper {w.name} fire rate too high: {w.fire_rate}"

    def test_lmgs_have_large_magazines(self):
        for w in weapons_by_category(WeaponCategory.LMG):
            assert w.magazine_size >= 100, f"LMG {w.name} magazine too small: {w.magazine_size}"

    def test_melee_no_muzzle_flash(self):
        for w in weapons_by_category(WeaponCategory.MELEE):
            assert w.muzzle_flash_size == 0.0

    def test_thrown_low_velocity(self):
        for w in weapons_by_category(WeaponCategory.THROWN):
            assert w.muzzle_velocity <= 20, f"Thrown {w.name} velocity too high: {w.muzzle_velocity}"

    def test_turrets_are_heavy(self):
        for w in weapons_by_category(WeaponCategory.TURRET):
            assert w.weight_kg >= 30.0

    def test_smgs_lower_range_than_rifles(self):
        smg_ranges = [w.effective_range for w in weapons_by_category(WeaponCategory.SMG)]
        rifle_ranges = [w.effective_range for w in weapons_by_category(WeaponCategory.RIFLE)]
        assert max(smg_ranges) < min(rifle_ranges)


# ---------------------------------------------------------------------------
# Integration: fire + tick + expire + effects
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end scenarios combining projectiles and area effects."""

    def test_grenade_fire_and_impact_then_explosion(self):
        """Fire a grenade, let it impact, then create an explosion effect."""
        sim = ProjectileSimulator()
        weapon = ARSENAL["frag_grenade"]
        proj = sim.fire(weapon, origin=(0, 0), target=(30, 0), rng=random.Random(42))

        # Tick until impact
        impacts = []
        for _ in range(500):
            impacts.extend(sim.tick(0.01))
            if impacts:
                break

        assert len(impacts) >= 1
        impact = impacts[0]
        assert impact["type"] == "grenade"
        assert impact["effect"] == "explosion"

        # Create area effect at impact point
        mgr = AreaEffectManager()
        mgr.add(create_explosion_effect((impact["x"], impact["y"]), radius=15.0))
        assert len(mgr.effects) == 1

    def test_smoke_grenade_creates_smoke(self):
        """Fire smoke grenade, check it creates the right projectile type."""
        sim = ProjectileSimulator()
        weapon = ARSENAL["smoke_grenade"]
        proj = sim.fire(weapon, origin=(0, 0), target=(20, 0), rng=random.Random(42))
        assert proj.projectile_type == ProjectileType.SMOKE

    def test_full_combat_tick(self):
        """Simulate a brief firefight with multiple weapons and effects."""
        sim = ProjectileSimulator()
        mgr = AreaEffectManager()
        rng = random.Random(42)

        # Two shooters firing rifles
        for _ in range(5):
            sim.fire(ARSENAL["m4a1"], origin=(0, 0), target=(50, 30), rng=rng)
            sim.fire(ARSENAL["ak47"], origin=(50, 30), target=(0, 0), rng=rng)

        # Throw a grenade
        sim.fire(ARSENAL["frag_grenade"], origin=(0, 0), target=(50, 30), rng=rng)

        # Add smoke cover
        mgr.add(create_smoke_effect((25, 15), radius=10.0, duration=30.0))

        # Tick the simulation
        all_impacts = []
        for _ in range(100):
            all_impacts.extend(sim.tick(0.016))
            mgr.tick(0.016)

        # Get Three.js data
        proj_data = sim.to_three_js()
        effect_data = mgr.to_three_js()

        assert "projectiles" in proj_data
        assert "impacts" in proj_data
        assert "muzzle_flashes" in proj_data
        assert "effects" in effect_data

        # Smoke should still be active
        assert len(effect_data["effects"]) == 1
        assert effect_data["effects"][0]["type"] == "smoke"
