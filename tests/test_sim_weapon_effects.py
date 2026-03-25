# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.effects.weapons — WeaponFirer, WeaponProfile, WEAPONS."""

import math
import pytest

from tritium_lib.sim_engine.effects.weapons import (
    WeaponProfile,
    WeaponFirer,
    FiredRound,
    FireMode,
    WEAPONS,
    create_firer,
)


class TestFireMode:
    def test_all_modes_exist(self):
        assert FireMode.SEMI.value == "semi"
        assert FireMode.BURST.value == "burst"
        assert FireMode.AUTO.value == "auto"
        assert FireMode.BOLT.value == "bolt"
        assert FireMode.PUMP.value == "pump"
        assert FireMode.MELEE.value == "melee"
        assert FireMode.THROWN.value == "thrown"


class TestWeaponProfile:
    def test_seconds_per_round(self):
        wp = WeaponProfile("Test", FireMode.AUTO, rpm=600)
        assert abs(wp.seconds_per_round - 0.1) < 0.001

    def test_seconds_per_round_zero_rpm(self):
        wp = WeaponProfile("Test", FireMode.AUTO, rpm=0)
        assert wp.seconds_per_round == 1.0

    def test_default_values(self):
        wp = WeaponProfile("Test", FireMode.SEMI, rpm=120)
        assert wp.damage == 25.0
        assert wp.spread_deg == 2.0
        assert wp.projectile_speed == 300.0


class TestWeaponsRegistry:
    def test_m4_exists(self):
        assert "m4" in WEAPONS
        assert WEAPONS["m4"].fire_mode == FireMode.AUTO

    def test_pistol_exists(self):
        assert "pistol_9mm" in WEAPONS
        assert WEAPONS["pistol_9mm"].fire_mode == FireMode.SEMI

    def test_shotgun_has_pellets(self):
        assert "shotgun" in WEAPONS
        assert WEAPONS["shotgun"].pellet_count == 8

    def test_sniper_exists(self):
        assert "sniper" in WEAPONS
        assert WEAPONS["sniper"].fire_mode == FireMode.BOLT
        assert WEAPONS["sniper"].effective_range == 800

    def test_knife_is_melee(self):
        assert "knife" in WEAPONS
        assert WEAPONS["knife"].fire_mode == FireMode.MELEE
        assert WEAPONS["knife"].projectile_speed == 0

    def test_grenade_is_thrown(self):
        assert "grenade" in WEAPONS
        assert WEAPONS["grenade"].fire_mode == FireMode.THROWN

    def test_all_weapons_have_name(self):
        for wid, wp in WEAPONS.items():
            assert wp.name, f"Weapon {wid} has no name"
            assert wp.rpm >= 0, f"Weapon {wid} has negative RPM"

    def test_weapon_count(self):
        assert len(WEAPONS) >= 15  # We counted 20+ in the source


class TestWeaponFirer:
    def test_creation(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        assert firer.ammo == 30
        assert firer.max_ammo == 30
        assert not firer.is_firing

    def test_pull_trigger(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        assert firer.pull_trigger()
        assert firer.is_firing

    def test_pull_trigger_no_ammo(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=0)
        assert not firer.pull_trigger()
        assert not firer.is_firing

    def test_release_trigger(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()
        firer.release_trigger()
        assert not firer.is_firing

    def test_auto_fire_produces_rounds(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()
        rounds = firer.tick(0.5)  # 0.5s at 700 RPM = ~5-6 rounds
        assert len(rounds) > 0
        assert firer.ammo < 30

    def test_semi_fires_once_per_pull(self):
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=15)
        firer.pull_trigger()
        rounds = firer.tick(1.0)
        assert len(rounds) == 1
        # Holding trigger should not fire again
        rounds2 = firer.tick(1.0)
        assert len(rounds2) == 0

    def test_burst_fires_correct_count(self):
        firer = WeaponFirer(WEAPONS["m16_burst"], ammo=30)
        firer.pull_trigger()
        total_rounds = []
        for _ in range(20):
            total_rounds.extend(firer.tick(0.05))
        assert len(total_rounds) == 3  # 3-round burst

    def test_ammo_depletes(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=5)
        firer.pull_trigger()
        for _ in range(100):
            firer.tick(0.1)
        assert firer.ammo == 0

    def test_reload(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()
        for _ in range(50):
            firer.tick(0.1)
        old_ammo = firer.ammo
        firer.reload()
        assert firer.ammo == 30

    def test_reload_custom_ammo(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.reload(ammo=10)
        assert firer.ammo == 10

    def test_can_fire_property(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        assert firer.can_fire
        firer.ammo = 0
        assert not firer.can_fire

    def test_spread_bloom_increases(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        initial_spread = firer.current_spread_deg
        firer.pull_trigger()
        # Use longer ticks to ensure multiple rounds fire (M4 = 700 RPM = ~0.086s/round)
        for _ in range(30):
            firer.tick(0.1)
        # Bloom should have increased after sustained fire
        assert firer.total_rounds_fired > 1, "Should have fired multiple rounds"
        assert firer.current_spread_deg > initial_spread

    def test_spread_bloom_recovers(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()
        for _ in range(30):
            firer.tick(0.1)
        firer.release_trigger()
        bloomed = firer.current_spread_deg
        assert bloomed > WEAPONS["m4"].spread_deg, "Should have bloomed before recovery"
        # Let it recover (no firing, bloom decreases over time)
        for _ in range(100):
            firer.tick(0.1)
        assert firer.current_spread_deg < bloomed

    def test_to_dict(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        d = firer.to_dict()
        assert d["weapon"] == "M4 Carbine"
        assert d["ammo"] == 30
        assert "fire_mode" in d

    def test_to_three_js(self):
        firer = WeaponFirer(WEAPONS["m4"], ammo=30, position=(5.0, 10.0))
        d = firer.to_three_js()
        assert d["type"] == "weapon_firer"
        assert d["position"] == [5.0, 0.0, 10.0]
        assert d["weapon"] == "M4 Carbine"


class TestFiredRound:
    def test_to_dict(self):
        from tritium_lib.sim_engine.audio.spatial import SoundEvent
        from tritium_lib.sim_engine.effects.particles import muzzle_flash
        sound = SoundEvent("rifle_auto", (0.0, 0.0))
        flash = muzzle_flash((0.0, 0.0), 0.0)
        r = FiredRound(
            position=(0.0, 0.0), heading=0.5, speed=300.0,
            spread_angle=0.01, is_tracer=True, damage=25.0,
            sound_event=sound, muzzle_flash_emitter=flash,
            weapon_name="M4 Carbine", round_number=1,
        )
        d = r.to_dict()
        assert d["weapon"] == "M4 Carbine"
        assert d["is_tracer"]
        assert d["speed"] == 300.0
        assert "sound" in d


class TestCreateFirer:
    def test_create_known_weapon(self):
        firer = create_firer("m4")
        assert firer.weapon.name == "M4 Carbine"
        assert firer.ammo == 30

    def test_create_with_custom_ammo(self):
        firer = create_firer("pistol_9mm", ammo=20)
        assert firer.ammo == 20

    def test_create_unknown_weapon_raises(self):
        with pytest.raises(KeyError):
            create_firer("nonexistent_weapon")

    def test_create_bolt_action_default_ammo(self):
        firer = create_firer("sniper")
        assert firer.ammo == 5  # Bolt action default

    def test_create_melee_weapon(self):
        firer = create_firer("knife")
        assert firer.weapon.fire_mode == FireMode.MELEE
