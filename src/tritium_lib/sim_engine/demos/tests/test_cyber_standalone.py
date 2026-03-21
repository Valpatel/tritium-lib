# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone unit tests for cyber.py — cyber warfare engine.

Tests CyberWarfareEngine, attack types (jamming, GPS spoofing, drone
hijack, intercept, malware, DoS, spoofing, decoy), effect lifecycle,
and to_three_js serialization.
"""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.cyber import (
    CyberWarfareEngine,
    CyberAsset,
    CyberCapability,
    CyberEffect,
    CyberAttackType,
    create_asset_from_preset,
    CYBER_PRESETS,
    _opposing_alliance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jammer(asset_id: str = "jam1", pos: tuple = (0.0, 0.0),
                 alliance: str = "friendly") -> CyberAsset:
    return CyberAsset(
        asset_id=asset_id,
        position=pos,
        alliance=alliance,
        capabilities=[CyberCapability(
            capability_id=f"{asset_id}_jam",
            attack_type=CyberAttackType.JAMMING,
            power=0.8,
            range_m=200.0,
            duration=60.0,
            cooldown=10.0,
        )],
    )


def _make_gps_spoofer(asset_id: str = "gps1", pos: tuple = (0.0, 0.0),
                       alliance: str = "hostile") -> CyberAsset:
    return CyberAsset(
        asset_id=asset_id,
        position=pos,
        alliance=alliance,
        capabilities=[CyberCapability(
            capability_id=f"{asset_id}_spoof",
            attack_type=CyberAttackType.GPS_SPOOFING,
            power=0.8,
            range_m=300.0,
            duration=90.0,
            cooldown=20.0,
        )],
    )


# ---------------------------------------------------------------------------
# Basic engine lifecycle
# ---------------------------------------------------------------------------

class TestCyberEngineLifecycle:
    def test_deploy_and_retrieve(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        asset = _make_jammer()
        eng.deploy_asset(asset)
        assert "jam1" in eng.assets

    def test_remove_asset(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.remove_asset("jam1")
        assert "jam1" not in eng.assets

    def test_remove_asset_cleans_effects(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        asset = _make_jammer()
        eng.deploy_asset(asset)
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert len(eng.active_effects) == 1
        eng.remove_asset("jam1")
        assert len(eng.active_effects) == 0

    def test_remove_nonexistent_asset_no_error(self) -> None:
        eng = CyberWarfareEngine()
        eng.remove_asset("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Launch attack
# ---------------------------------------------------------------------------

class TestLaunchAttack:
    def test_successful_launch(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        effect = eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert effect is not None
        assert effect.attack_type == CyberAttackType.JAMMING
        assert len(eng.active_effects) == 1

    def test_launch_starts_cooldown(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        cap = eng.assets["jam1"].capabilities[0]
        assert cap.cooldown_remaining == 10.0

    def test_launch_fails_during_cooldown(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        # Second launch should fail (cooldown)
        effect2 = eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert effect2 is None

    def test_launch_fails_out_of_range(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())  # range 200m
        effect = eng.launch_attack("jam1", "jam1_jam", (500.0, 0.0))
        assert effect is None

    def test_launch_fails_no_power(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        asset = _make_jammer()
        asset.power_level = 0.0
        eng.deploy_asset(asset)
        effect = eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert effect is None

    def test_launch_drains_power(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        initial_power = eng.assets["jam1"].power_level
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert eng.assets["jam1"].power_level < initial_power

    def test_launch_unknown_asset(self) -> None:
        eng = CyberWarfareEngine()
        assert eng.launch_attack("nonexistent", "cap1", (0.0, 0.0)) is None

    def test_launch_unknown_capability(self) -> None:
        eng = CyberWarfareEngine()
        eng.deploy_asset(_make_jammer())
        assert eng.launch_attack("jam1", "nonexistent_cap", (50.0, 0.0)) is None

    def test_gps_spoofing_generates_fake_position(self) -> None:
        eng = CyberWarfareEngine(rng_seed=42)
        eng.deploy_asset(_make_gps_spoofer())
        effect = eng.launch_attack("gps1", "gps1_spoof", (50.0, 0.0))
        assert effect is not None
        assert effect.fake_position is not None
        assert effect.fake_position != (50.0, 0.0)


# ---------------------------------------------------------------------------
# Tick — jamming
# ---------------------------------------------------------------------------

class TestJammingTick:
    def test_jammed_unit_gets_comms_degraded(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        # Unit within jamming radius (50m from effect center, radius=100)
        events = eng.tick(1.0, {"u1": (60.0, 0.0)})
        degraded = [e for e in events if e["type"] == "comms_degraded"]
        assert len(degraded) >= 1
        assert degraded[0]["unit_id"] == "u1"

    def test_is_jammed_query(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert eng.is_jammed((50.0, 0.0)) is True
        assert eng.is_jammed((500.0, 500.0)) is False


# ---------------------------------------------------------------------------
# Tick — GPS spoofing
# ---------------------------------------------------------------------------

class TestGPSSpoofingTick:
    def test_drone_gets_gps_spoofed(self) -> None:
        eng = CyberWarfareEngine(rng_seed=42)
        eng.deploy_asset(_make_gps_spoofer())
        eng.launch_attack("gps1", "gps1_spoof", (50.0, 0.0))
        events = eng.tick(1.0, {}, {"drone1": (55.0, 0.0)})
        spoofed = [e for e in events if e["type"] == "gps_spoofed"]
        assert len(spoofed) >= 1
        assert spoofed[0]["drone_id"] == "drone1"

    def test_ground_unit_gets_gps_degraded(self) -> None:
        eng = CyberWarfareEngine(rng_seed=42)
        eng.deploy_asset(_make_gps_spoofer())
        eng.launch_attack("gps1", "gps1_spoof", (50.0, 0.0))
        events = eng.tick(1.0, {"u1": (55.0, 0.0)}, {})
        degraded = [e for e in events if e["type"] == "gps_degraded"]
        assert len(degraded) >= 1

    def test_is_gps_spoofed_query(self) -> None:
        eng = CyberWarfareEngine(rng_seed=42)
        eng.deploy_asset(_make_gps_spoofer())
        eng.launch_attack("gps1", "gps1_spoof", (50.0, 0.0))
        spoofed, fake_pos = eng.is_gps_spoofed((50.0, 0.0))
        assert spoofed is True
        assert fake_pos != (50.0, 0.0)

    def test_not_gps_spoofed_far_away(self) -> None:
        eng = CyberWarfareEngine(rng_seed=42)
        eng.deploy_asset(_make_gps_spoofer())
        eng.launch_attack("gps1", "gps1_spoof", (50.0, 0.0))
        spoofed, pos = eng.is_gps_spoofed((999.0, 999.0))
        assert spoofed is False
        assert pos == (999.0, 999.0)


# ---------------------------------------------------------------------------
# Tick — effect expiry
# ---------------------------------------------------------------------------

class TestEffectExpiry:
    def test_effect_expires_after_duration(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        effect = eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert effect is not None
        # Tick past the duration (60s)
        events = eng.tick(61.0, {})
        expired = [e for e in events if e["type"] == "effect_expired"]
        assert len(expired) == 1
        assert len(eng.active_effects) == 0

    def test_cooldown_ticks_down(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        cap = eng.assets["jam1"].capabilities[0]
        assert cap.cooldown_remaining == 10.0
        eng.tick(5.0, {})
        assert cap.cooldown_remaining == 5.0
        eng.tick(6.0, {})
        assert cap.cooldown_remaining == 0.0
        assert cap.is_ready is True


# ---------------------------------------------------------------------------
# Tick — denial of service
# ---------------------------------------------------------------------------

class TestDenialOfService:
    def test_dos_produces_network_denied(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        asset = CyberAsset(
            asset_id="dos1", position=(0.0, 0.0), alliance="hostile",
            capabilities=[CyberCapability(
                capability_id="dos1_dos",
                attack_type=CyberAttackType.DENIAL_OF_SERVICE,
                power=0.8, range_m=200.0, duration=30.0, cooldown=10.0,
            )],
        )
        eng.deploy_asset(asset)
        eng.launch_attack("dos1", "dos1_dos", (50.0, 0.0))
        events = eng.tick(1.0, {"u1": (50.0, 0.0)})
        denied = [e for e in events if e["type"] == "network_denied"]
        assert len(denied) >= 1


# ---------------------------------------------------------------------------
# Tick — spoofing (sensor)
# ---------------------------------------------------------------------------

class TestSpoofingTick:
    def test_spoofing_produces_sensor_spoofed(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        asset = CyberAsset(
            asset_id="spoof1", position=(0.0, 0.0), alliance="hostile",
            capabilities=[CyberCapability(
                capability_id="spoof1_spf",
                attack_type=CyberAttackType.SPOOFING,
                power=0.8, range_m=200.0, duration=40.0, cooldown=10.0,
            )],
        )
        eng.deploy_asset(asset)
        eng.launch_attack("spoof1", "spoof1_spf", (50.0, 0.0))
        events = eng.tick(1.0, {"u1": (50.0, 0.0)})
        spoofed = [e for e in events if e["type"] == "sensor_spoofed"]
        assert len(spoofed) >= 1


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestCyberQueries:
    def test_get_active_effects_at(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        effects = eng.get_active_effects_at((50.0, 0.0))
        assert len(effects) == 1

    def test_get_active_effects_at_out_of_range(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        effects = eng.get_active_effects_at((999.0, 999.0))
        assert len(effects) == 0

    def test_get_effects_by_type(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        jamming = eng.get_effects_by_type(CyberAttackType.JAMMING)
        assert len(jamming) == 1
        gps = eng.get_effects_by_type(CyberAttackType.GPS_SPOOFING)
        assert len(gps) == 0

    def test_get_affected_units(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        effect = eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        assert effect is not None
        affected = eng.get_affected_units(effect, {
            "u1": (50.0, 0.0),       # in range
            "u2": (999.0, 999.0),     # out of range
        })
        assert "u1" in affected
        assert "u2" not in affected


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestCyberPresets:
    def test_all_presets_create_valid_assets(self) -> None:
        for name in CYBER_PRESETS:
            asset = create_asset_from_preset(name, f"test_{name}", (0.0, 0.0), "friendly")
            assert len(asset.capabilities) >= 1
            assert asset.asset_id == f"test_{name}"

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            create_asset_from_preset("nonexistent", "x", (0.0, 0.0), "friendly")

    def test_preset_capabilities_have_independent_cooldowns(self) -> None:
        a1 = create_asset_from_preset("jammer_truck", "a1", (0.0, 0.0), "friendly")
        a2 = create_asset_from_preset("jammer_truck", "a2", (0.0, 0.0), "friendly")
        a1.capabilities[0].cooldown_remaining = 999.0
        assert a2.capabilities[0].cooldown_remaining == 0.0


# ---------------------------------------------------------------------------
# to_three_js
# ---------------------------------------------------------------------------

class TestCyberToThreeJs:
    def test_structure(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        data = eng.to_three_js()
        assert "assets" in data
        assert "effects" in data
        assert "attack_lines" in data

    def test_asset_serialization(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        data = eng.to_three_js()
        assert len(data["assets"]) == 1
        asset_data = data["assets"][0]
        assert asset_data["id"] == "jam1"
        assert "position" in asset_data
        assert len(asset_data["position"]) == 3  # [x, y, z]

    def test_effect_serialization_with_attack(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        data = eng.to_three_js()
        assert len(data["effects"]) == 1
        assert len(data["attack_lines"]) == 1
        assert data["effects"][0]["type"] == "jamming"

    def test_empty_engine_serializes(self) -> None:
        eng = CyberWarfareEngine()
        data = eng.to_three_js()
        assert data["assets"] == []
        assert data["effects"] == []
        assert data["attack_lines"] == []


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_drain_returns_and_clears(self) -> None:
        eng = CyberWarfareEngine(rng_seed=0)
        eng.deploy_asset(_make_jammer())
        eng.launch_attack("jam1", "jam1_jam", (50.0, 0.0))
        log = eng.drain_event_log()
        assert len(log) >= 1
        assert log[0]["type"] == "cyber_attack_launched"
        # Second drain is empty
        assert eng.drain_event_log() == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_opposing_alliance(self) -> None:
        assert _opposing_alliance("friendly") == "hostile"
        assert _opposing_alliance("hostile") == "friendly"
        assert _opposing_alliance("neutral") == "unknown"
