# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the weapons system — fire modes, RPM, tracers, spread, ammo."""

import math

import pytest

from tritium_lib.sim_engine.effects.weapons import (
    WEAPONS,
    FireMode,
    FiredRound,
    WeaponFirer,
    WeaponProfile,
    create_firer,
)


# ---------------------------------------------------------------------------
# WeaponProfile basics
# ---------------------------------------------------------------------------

class TestWeaponProfile:
    def test_seconds_per_round_m4(self):
        """M4 at 700 RPM = ~0.0857s per round."""
        m4 = WEAPONS["m4"]
        assert abs(m4.seconds_per_round - 60.0 / 700.0) < 1e-6

    def test_seconds_per_round_sniper(self):
        """Sniper at 20 RPM = 3.0s per round."""
        sniper = WEAPONS["sniper"]
        assert abs(sniper.seconds_per_round - 3.0) < 1e-6

    def test_seconds_per_round_minigun(self):
        """Minigun at 3000 RPM = 0.02s per round."""
        mg = WEAPONS["minigun"]
        assert abs(mg.seconds_per_round - 0.02) < 1e-6

    def test_all_weapons_have_positive_rpm(self):
        for wid, w in WEAPONS.items():
            assert w.rpm > 0, f"{wid} has non-positive RPM"
            assert w.seconds_per_round > 0

    def test_all_weapons_have_valid_fire_mode(self):
        for wid, w in WEAPONS.items():
            assert isinstance(w.fire_mode, FireMode), f"{wid} bad fire_mode"

    def test_weapon_count(self):
        """We should have 12 pre-built weapon profiles."""
        assert len(WEAPONS) >= 20  # 23 weapons across all categories


# ---------------------------------------------------------------------------
# Semi-auto firing (pistol: pow... pow... pow...)
# ---------------------------------------------------------------------------

class TestSemiAuto:
    def test_one_shot_per_pull(self):
        """Semi-auto fires exactly one round per trigger pull."""
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=10)
        firer.pull_trigger()
        rounds = firer.tick(1.0)  # Long tick — only 1 round
        assert len(rounds) == 1
        assert firer.ammo == 9

    def test_must_release_and_repull(self):
        """Semi-auto won't fire again without releasing trigger."""
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=10)
        firer.pull_trigger()
        firer.tick(0.1)
        # Still holding trigger, tick again after cooldown
        rounds = firer.tick(2.0)
        assert len(rounds) == 0  # No fire — trigger not released

    def test_rapid_pulls(self):
        """Releasing and re-pulling fires again after cooldown."""
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=10)

        # First shot
        firer.pull_trigger()
        r1 = firer.tick(0.01)
        assert len(r1) == 1
        firer.release_trigger()

        # Wait for cooldown (120 RPM = 0.5s/round)
        firer.tick(0.6)

        # Second shot
        firer.pull_trigger()
        r2 = firer.tick(0.01)
        assert len(r2) == 1
        assert firer.ammo == 8

    def test_deagle_fires_semi(self):
        firer = WeaponFirer(WEAPONS["deagle"], ammo=7)
        firer.pull_trigger()
        rounds = firer.tick(5.0)
        assert len(rounds) == 1


# ---------------------------------------------------------------------------
# Burst fire (M16: taptaptap... taptaptap...)
# ---------------------------------------------------------------------------

class TestBurstFire:
    def test_three_round_burst(self):
        """M16 burst fires exactly 3 rounds then stops."""
        firer = WeaponFirer(WEAPONS["m16_burst"], ammo=30)
        firer.pull_trigger()

        all_rounds = []
        # Tick enough times to allow all 3 rounds
        for _ in range(50):
            all_rounds.extend(firer.tick(0.02))

        assert len(all_rounds) == 3
        assert firer.ammo == 27

    def test_burst_pauses_between_bursts(self):
        """After a burst completes, trigger must be re-pulled for next burst."""
        firer = WeaponFirer(WEAPONS["m16_burst"], ammo=30)

        # First burst
        firer.pull_trigger()
        r1 = []
        for _ in range(50):
            r1.extend(firer.tick(0.02))
        assert len(r1) == 3

        # Still holding trigger — no more rounds
        r2 = []
        for _ in range(20):
            r2.extend(firer.tick(0.02))
        assert len(r2) == 0

        # Release and re-pull
        firer.release_trigger()
        firer.pull_trigger()
        r3 = []
        for _ in range(50):
            r3.extend(firer.tick(0.02))
        assert len(r3) == 3
        assert firer.ammo == 24

    def test_burst_with_insufficient_ammo(self):
        """Burst stops early if ammo runs out mid-burst."""
        firer = WeaponFirer(WEAPONS["m16_burst"], ammo=2)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(50):
            all_rounds.extend(firer.tick(0.02))
        assert len(all_rounds) == 2
        assert firer.ammo == 0


# ---------------------------------------------------------------------------
# Full auto (M4: brrrrrrt, minigun: BRRRRRRRRRT)
# ---------------------------------------------------------------------------

class TestFullAuto:
    def test_auto_fires_continuously(self):
        """Full auto fires multiple rounds while trigger held."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=100)
        firer.pull_trigger()

        all_rounds = []
        # 1 second at 700 RPM should yield ~11-12 rounds
        for _ in range(100):
            all_rounds.extend(firer.tick(0.01))

        # 700 RPM = 11.67 rounds/sec, in 1 second ~11-12
        assert 10 <= len(all_rounds) <= 14

    def test_auto_stops_on_release(self):
        """Releasing trigger stops auto fire."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=100)
        firer.pull_trigger()

        r1 = []
        for _ in range(10):
            r1.extend(firer.tick(0.01))
        firer.release_trigger()

        r2 = []
        for _ in range(100):
            r2.extend(firer.tick(0.01))

        assert len(r1) > 0
        assert len(r2) == 0

    def test_minigun_high_rpm(self):
        """Minigun at 3000 RPM fires ~50 rounds/sec."""
        firer = WeaponFirer(WEAPONS["minigun"], ammo=500)
        firer.pull_trigger()

        all_rounds = []
        # 1 second total
        for _ in range(100):
            all_rounds.extend(firer.tick(0.01))

        # 3000 RPM = 50/sec. Allow tolerance.
        assert 40 <= len(all_rounds) <= 60

    def test_ak47_fires_auto(self):
        firer = WeaponFirer(WEAPONS["ak47"], ammo=30)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(50):
            all_rounds.extend(firer.tick(0.01))
        # 600 RPM = 10/sec, 0.5s => ~5
        assert 4 <= len(all_rounds) <= 7

    def test_mp5_fires_auto(self):
        firer = WeaponFirer(WEAPONS["mp5"], ammo=30)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(100):
            all_rounds.extend(firer.tick(0.01))
        # 800 RPM = ~13/sec
        assert 10 <= len(all_rounds) <= 16

    def test_uzi_high_rate(self):
        firer = WeaponFirer(WEAPONS["uzi"], ammo=50)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(100):
            all_rounds.extend(firer.tick(0.01))
        # 950 RPM = ~15.8/sec
        assert 13 <= len(all_rounds) <= 19


# ---------------------------------------------------------------------------
# Bolt / Pump (sniper: CRACK..., shotgun: BOOM... chk-chk...)
# ---------------------------------------------------------------------------

class TestBoltPump:
    def test_bolt_single_shot(self):
        """Bolt action fires one round per pull."""
        firer = WeaponFirer(WEAPONS["sniper"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(5.0)
        assert len(rounds) == 1

    def test_bolt_requires_repull(self):
        """Must release and re-pull for next bolt action shot."""
        firer = WeaponFirer(WEAPONS["sniper"], ammo=5)
        firer.pull_trigger()
        firer.tick(0.1)
        firer.release_trigger()
        firer.tick(4.0)  # Wait for long cooldown (20 RPM = 3s)
        firer.pull_trigger()
        r = firer.tick(0.1)
        assert len(r) == 1
        assert firer.ammo == 3

    def test_pump_shotgun_single(self):
        """Pump fires one shot per pull."""
        firer = WeaponFirer(WEAPONS["shotgun"], ammo=8)
        firer.pull_trigger()
        rounds = firer.tick(2.0)
        assert len(rounds) == 1

    def test_shotgun_pellet_damage(self):
        """Shotgun damage is split across pellets."""
        firer = WeaponFirer(WEAPONS["shotgun"], ammo=8)
        firer.pull_trigger()
        rounds = firer.tick(0.1)
        r = rounds[0]
        # 80 damage / 8 pellets = 10 per pellet
        assert abs(r.damage - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Ammo and reload
# ---------------------------------------------------------------------------

class TestAmmo:
    def test_ammo_depletes(self):
        """Firing reduces ammo count."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=5)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(200):
            all_rounds.extend(firer.tick(0.01))
        assert firer.ammo == 0
        assert len(all_rounds) == 5

    def test_empty_gun_wont_fire(self):
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=0)
        result = firer.pull_trigger()
        assert result is False
        rounds = firer.tick(1.0)
        assert len(rounds) == 0

    def test_reload_default(self):
        """Reload restores to max_ammo."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()
        for _ in range(100):
            firer.tick(0.01)
        assert firer.ammo < 30
        firer.release_trigger()
        firer.reload()
        assert firer.ammo == 30

    def test_reload_partial(self):
        """Reload with specific ammo count."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.reload(15)
        assert firer.ammo == 15

    def test_cant_fire_after_empty(self):
        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=1)
        firer.pull_trigger()
        firer.tick(0.1)
        assert firer.ammo == 0
        firer.release_trigger()
        firer.pull_trigger()
        rounds = firer.tick(1.0)
        assert len(rounds) == 0


# ---------------------------------------------------------------------------
# Tracers
# ---------------------------------------------------------------------------

class TestTracers:
    def test_tracer_interval(self):
        """Tracers appear at the correct interval (every Nth round)."""
        m4 = WEAPONS["m4"]  # tracer_every=5
        firer = WeaponFirer(m4, ammo=100)
        firer.pull_trigger()

        all_rounds = []
        for _ in range(500):
            all_rounds.extend(firer.tick(0.01))

        tracers = [r for r in all_rounds if r.is_tracer]
        non_tracers = [r for r in all_rounds if not r.is_tracer]

        # Every 5th round is tracer
        assert len(tracers) > 0
        for r in tracers:
            assert r.round_number % 5 == 0

    def test_m249_tracer_every_4(self):
        """M249 has tracers every 4th round."""
        firer = WeaponFirer(WEAPONS["m249"], ammo=100)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(500):
            all_rounds.extend(firer.tick(0.01))

        tracers = [r for r in all_rounds if r.is_tracer]
        assert len(tracers) > 0
        for r in tracers:
            assert r.round_number % 4 == 0

    def test_minigun_tracer_every_3(self):
        firer = WeaponFirer(WEAPONS["minigun"], ammo=200)
        firer.pull_trigger()
        all_rounds = []
        for _ in range(200):
            all_rounds.extend(firer.tick(0.01))

        tracers = [r for r in all_rounds if r.is_tracer]
        assert len(tracers) > 0
        for r in tracers:
            assert r.round_number % 3 == 0


# ---------------------------------------------------------------------------
# Spread / bloom
# ---------------------------------------------------------------------------

class TestSpread:
    def test_spread_increases_during_sustained_fire(self):
        """Sustained auto fire increases spread bloom."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=100)
        initial_spread = firer.current_spread_deg

        firer.pull_trigger()
        for _ in range(100):
            firer.tick(0.01)

        assert firer.current_spread_deg > initial_spread

    def test_spread_recovers_when_not_firing(self):
        """Spread bloom recovers over time when not firing."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=100)
        firer.pull_trigger()
        for _ in range(50):
            firer.tick(0.01)
        bloomed = firer.current_spread_deg
        firer.release_trigger()

        # Let spread recover
        for _ in range(200):
            firer.tick(0.01)

        assert firer.current_spread_deg < bloomed

    def test_spread_capped_at_max(self):
        """Spread never exceeds spread_max_deg."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=1000)
        firer.pull_trigger()
        for _ in range(1000):
            firer.tick(0.01)

        assert firer.current_spread_deg <= WEAPONS["m4"].spread_max_deg

    def test_sniper_no_bloom(self):
        """Sniper has zero bloom per shot."""
        firer = WeaponFirer(WEAPONS["sniper"], ammo=5)
        initial = firer.current_spread_deg
        firer.pull_trigger()
        firer.tick(0.1)
        assert firer.current_spread_deg == initial


# ---------------------------------------------------------------------------
# Sound events
# ---------------------------------------------------------------------------

class TestSoundEvents:
    def test_sound_has_correct_id(self):
        """Each round's sound matches the weapon's sound_id."""
        for wid, wp in WEAPONS.items():
            firer = WeaponFirer(wp, ammo=5)
            firer.pull_trigger()
            rounds = firer.tick(0.01)
            if rounds:
                assert rounds[0].sound_event.sound_id == wp.sound_id, (
                    f"{wid} sound mismatch"
                )

    def test_sound_pitch_near_base(self):
        """Sound pitch is near the weapon's base pitch."""
        firer = WeaponFirer(WEAPONS["ak47"], ammo=10)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        r = rounds[0]
        base = WEAPONS["ak47"].sound_pitch_base
        variance = WEAPONS["ak47"].sound_pitch_variance
        assert abs(r.sound_event.pitch - base) <= variance + 1e-6

    def test_sound_volume_matches(self):
        """Nerf blaster has lower volume."""
        firer = WeaponFirer(WEAPONS["nerf"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        assert rounds[0].sound_event.volume == 0.3

    def test_weapon_category_is_weapon(self):
        """Sound events use category 'weapon'."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        assert rounds[0].sound_event.category == "weapon"


# ---------------------------------------------------------------------------
# Muzzle flash
# ---------------------------------------------------------------------------

class TestMuzzleFlash:
    def test_muzzle_flash_present(self):
        """Normal weapons produce muzzle flash particles."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        assert rounds[0].muzzle_flash_emitter is not None

    def test_nerf_no_muzzle_flash(self):
        """Nerf has zero muzzle flash size — no emitter."""
        firer = WeaponFirer(WEAPONS["nerf"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        assert rounds[0].muzzle_flash_emitter is None

    def test_shotgun_large_flash(self):
        """Shotgun has larger muzzle flash than pistol."""
        sg = WEAPONS["shotgun"]
        p9 = WEAPONS["pistol_9mm"]
        assert sg.muzzle_flash_size > p9.muzzle_flash_size


# ---------------------------------------------------------------------------
# FiredRound serialization
# ---------------------------------------------------------------------------

class TestFiredRoundSerialization:
    def test_to_dict(self):
        """FiredRound.to_dict() produces valid JSON-ready dict."""
        firer = WeaponFirer(WEAPONS["m4"], ammo=5)
        firer.pull_trigger()
        rounds = firer.tick(0.01)
        d = rounds[0].to_dict()
        assert "position" in d
        assert "heading" in d
        assert "speed" in d
        assert "is_tracer" in d
        assert "damage" in d
        assert "sound" in d
        assert "weapon" in d
        assert d["weapon"] == "M4 Carbine"

    def test_firer_to_dict(self):
        """WeaponFirer.to_dict() exports HUD state."""
        firer = create_firer("m4")
        d = firer.to_dict()
        assert d["weapon"] == "M4 Carbine"
        assert d["fire_mode"] == "auto"
        assert d["ammo"] == 30
        assert d["is_firing"] is False


# ---------------------------------------------------------------------------
# create_firer factory
# ---------------------------------------------------------------------------

class TestCreateFirer:
    def test_create_by_id(self):
        firer = create_firer("ak47")
        assert firer.weapon.name == "AK-47"
        assert firer.ammo == 30  # AUTO default

    def test_create_sniper_default_ammo(self):
        firer = create_firer("sniper")
        assert firer.ammo == 5  # BOLT default

    def test_create_shotgun_default_ammo(self):
        firer = create_firer("shotgun")
        assert firer.ammo == 8  # PUMP default

    def test_create_with_custom_ammo(self):
        firer = create_firer("m4", ammo=100)
        assert firer.ammo == 100

    def test_create_with_position(self):
        firer = create_firer("pistol_9mm", position=(10.0, 20.0), heading=1.5)
        assert firer.position == (10.0, 20.0)
        assert firer.heading == 1.5

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError):
            create_firer("phaser")


# ---------------------------------------------------------------------------
# RPM accuracy across weapon types
# ---------------------------------------------------------------------------

class TestRPMAccuracy:
    """Verify that each auto/burst weapon fires at approximately the right rate."""

    @pytest.mark.parametrize("weapon_id,expected_rps", [
        ("m4", 700 / 60),
        ("ak47", 600 / 60),
        ("mp5", 800 / 60),
        ("uzi", 950 / 60),
        ("m249", 850 / 60),
        ("minigun", 3000 / 60),
    ])
    def test_auto_rpm(self, weapon_id, expected_rps):
        """Auto weapons fire at their rated RPM."""
        firer = WeaponFirer(WEAPONS[weapon_id], ammo=9999)
        firer.pull_trigger()

        all_rounds = []
        duration = 2.0  # 2 seconds for accuracy
        steps = 200
        dt = duration / steps
        for _ in range(steps):
            all_rounds.extend(firer.tick(dt))

        actual_rps = len(all_rounds) / duration
        # Allow 15% tolerance
        assert abs(actual_rps - expected_rps) / expected_rps < 0.15, (
            f"{weapon_id}: expected {expected_rps:.1f} rps, got {actual_rps:.1f}"
        )
