# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the cyber warfare engine (sim_engine.cyber)."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.cyber import (
    CyberAttackType,
    CyberAsset,
    CyberCapability,
    CyberEffect,
    CyberWarfareEngine,
    CYBER_PRESETS,
    create_asset_from_preset,
    _opposing_alliance,
    _alliance_color,
    _effect_color,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> CyberWarfareEngine:
    return CyberWarfareEngine(rng_seed=42)


def _jammer_cap(cap_id: str = "jam1", **kw) -> CyberCapability:
    defaults = dict(
        capability_id=cap_id,
        attack_type=CyberAttackType.JAMMING,
        power=0.8,
        range_m=200.0,
        duration=60.0,
        cooldown=10.0,
    )
    defaults.update(kw)
    return CyberCapability(**defaults)


def _gps_cap(cap_id: str = "gps1", **kw) -> CyberCapability:
    defaults = dict(
        capability_id=cap_id,
        attack_type=CyberAttackType.GPS_SPOOFING,
        power=0.7,
        range_m=300.0,
        duration=90.0,
        cooldown=20.0,
    )
    defaults.update(kw)
    return CyberCapability(**defaults)


def _make_asset(
    asset_id: str = "asset1",
    pos: tuple[float, float] = (100.0, 100.0),
    alliance: str = "friendly",
    caps: list[CyberCapability] | None = None,
) -> CyberAsset:
    if caps is None:
        caps = [_jammer_cap()]
    return CyberAsset(
        asset_id=asset_id,
        position=pos,
        alliance=alliance,
        capabilities=caps,
    )


# ---------------------------------------------------------------------------
# CyberAttackType enum
# ---------------------------------------------------------------------------

class TestCyberAttackType:
    def test_all_types_exist(self):
        assert len(CyberAttackType) == 8

    def test_jamming(self):
        assert CyberAttackType.JAMMING.value == "jamming"

    def test_spoofing(self):
        assert CyberAttackType.SPOOFING.value == "spoofing"

    def test_dos(self):
        assert CyberAttackType.DENIAL_OF_SERVICE.value == "denial_of_service"

    def test_malware(self):
        assert CyberAttackType.MALWARE.value == "malware"

    def test_intercept(self):
        assert CyberAttackType.INTERCEPT.value == "intercept"

    def test_decoy(self):
        assert CyberAttackType.DECOY.value == "decoy"

    def test_gps_spoofing(self):
        assert CyberAttackType.GPS_SPOOFING.value == "gps_spoofing"

    def test_drone_hijack(self):
        assert CyberAttackType.DRONE_HIJACK.value == "drone_hijack"


# ---------------------------------------------------------------------------
# CyberCapability
# ---------------------------------------------------------------------------

class TestCyberCapability:
    def test_creation(self):
        cap = _jammer_cap()
        assert cap.capability_id == "jam1"
        assert cap.attack_type == CyberAttackType.JAMMING
        assert cap.power == 0.8
        assert cap.range_m == 200.0

    def test_is_ready_default(self):
        cap = _jammer_cap()
        assert cap.is_ready is True

    def test_is_ready_on_cooldown(self):
        cap = _jammer_cap()
        cap.cooldown_remaining = 5.0
        assert cap.is_ready is False

    def test_requires_los_default(self):
        cap = _jammer_cap()
        assert cap.requires_los is False

    def test_requires_los_set(self):
        cap = _jammer_cap(requires_los=True)
        assert cap.requires_los is True


# ---------------------------------------------------------------------------
# CyberAsset
# ---------------------------------------------------------------------------

class TestCyberAsset:
    def test_creation(self):
        asset = _make_asset()
        assert asset.asset_id == "asset1"
        assert asset.alliance == "friendly"
        assert len(asset.capabilities) == 1

    def test_default_power_level(self):
        asset = _make_asset()
        assert asset.power_level == 1.0

    def test_detected_default(self):
        asset = _make_asset()
        assert asset.detected is False

    def test_active_attacks_empty(self):
        asset = _make_asset()
        assert asset.active_attacks == []


# ---------------------------------------------------------------------------
# CyberEffect
# ---------------------------------------------------------------------------

class TestCyberEffect:
    def test_creation(self):
        effect = CyberEffect(
            effect_id="e1",
            attack_type=CyberAttackType.JAMMING,
            position=(50.0, 50.0),
            radius=100.0,
            intensity=0.8,
            remaining=30.0,
            source_id="asset1",
            target_alliance="hostile",
        )
        assert effect.effect_id == "e1"
        assert effect.remaining == 30.0

    def test_gps_spoofing_fake_position(self):
        effect = CyberEffect(
            effect_id="e2",
            attack_type=CyberAttackType.GPS_SPOOFING,
            position=(50.0, 50.0),
            radius=100.0,
            intensity=0.7,
            remaining=60.0,
            source_id="asset1",
            target_alliance="hostile",
            fake_position=(999.0, 999.0),
        )
        assert effect.fake_position == (999.0, 999.0)

    def test_intercepted_messages_default(self):
        effect = CyberEffect(
            effect_id="e3",
            attack_type=CyberAttackType.INTERCEPT,
            position=(0.0, 0.0),
            radius=50.0,
            intensity=0.5,
            remaining=10.0,
            source_id="s1",
            target_alliance="hostile",
        )
        assert effect.intercepted_messages == []


# ---------------------------------------------------------------------------
# CyberWarfareEngine — deploy / remove
# ---------------------------------------------------------------------------

class TestEngineDeployRemove:
    def test_deploy_asset(self, engine: CyberWarfareEngine):
        asset = _make_asset()
        engine.deploy_asset(asset)
        assert "asset1" in engine.assets

    def test_remove_asset(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.remove_asset("asset1")
        assert "asset1" not in engine.assets

    def test_remove_nonexistent(self, engine: CyberWarfareEngine):
        engine.remove_asset("nope")  # should not raise

    def test_remove_cleans_effects(self, engine: CyberWarfareEngine):
        asset = _make_asset()
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert len(engine.active_effects) == 1
        engine.remove_asset("asset1")
        assert len(engine.active_effects) == 0


# ---------------------------------------------------------------------------
# CyberWarfareEngine — launch_attack
# ---------------------------------------------------------------------------

class TestLaunchAttack:
    def test_basic_launch(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        effect = engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert effect is not None
        assert effect.attack_type == CyberAttackType.JAMMING
        assert len(engine.active_effects) == 1

    def test_launch_nonexistent_asset(self, engine: CyberWarfareEngine):
        result = engine.launch_attack("nope", "jam1", (0.0, 0.0))
        assert result is None

    def test_launch_nonexistent_capability(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        result = engine.launch_attack("asset1", "nope", (110.0, 100.0))
        assert result is None

    def test_launch_out_of_range(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        result = engine.launch_attack("asset1", "jam1", (9999.0, 9999.0))
        assert result is None

    def test_launch_on_cooldown(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        # Capability now on cooldown
        result = engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert result is None

    def test_launch_no_power(self, engine: CyberWarfareEngine):
        asset = _make_asset()
        asset.power_level = 0.0
        engine.deploy_asset(asset)
        result = engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert result is None

    def test_launch_drains_power(self, engine: CyberWarfareEngine):
        asset = _make_asset()
        engine.deploy_asset(asset)
        before = asset.power_level
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert asset.power_level < before

    def test_launch_sets_cooldown(self, engine: CyberWarfareEngine):
        asset = _make_asset()
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert asset.capabilities[0].cooldown_remaining == 10.0

    def test_launch_records_active_attack(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert len(engine.assets["asset1"].active_attacks) == 1

    def test_launch_gps_spoofing_creates_fake_position(self, engine: CyberWarfareEngine):
        asset = _make_asset(caps=[_gps_cap()])
        engine.deploy_asset(asset)
        effect = engine.launch_attack("asset1", "gps1", (150.0, 100.0))
        assert effect is not None
        assert effect.fake_position is not None
        assert effect.fake_position != (150.0, 100.0)

    def test_event_log_populated(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        log = engine.drain_event_log()
        assert len(log) == 1
        assert log[0]["type"] == "cyber_attack_launched"


# ---------------------------------------------------------------------------
# CyberWarfareEngine — tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_expires_effects(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset(caps=[_jammer_cap(duration=1.0)]))
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert len(engine.active_effects) == 1
        events = engine.tick(2.0)
        assert len(engine.active_effects) == 0
        expired = [e for e in events if e["type"] == "effect_expired"]
        assert len(expired) == 1

    def test_tick_cooldown_decreases(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        cap = engine.assets["asset1"].capabilities[0]
        assert cap.cooldown_remaining == 10.0
        engine.tick(5.0)
        assert cap.cooldown_remaining == 5.0

    def test_tick_jamming_degrades_comms(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (110.0, 100.0)})
        degraded = [e for e in events if e["type"] == "comms_degraded"]
        assert len(degraded) == 1
        assert degraded[0]["unit_id"] == "u1"

    def test_tick_jamming_ignores_distant_units(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (9999.0, 9999.0)})
        degraded = [e for e in events if e["type"] == "comms_degraded"]
        assert len(degraded) == 0

    def test_tick_gps_spoofing(self, engine: CyberWarfareEngine):
        asset = _make_asset(caps=[_gps_cap()])
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "gps1", (150.0, 100.0))
        events = engine.tick(1.0, drone_positions={"d1": (150.0, 100.0)})
        gps_events = [e for e in events if e["type"] == "gps_spoofed"]
        assert len(gps_events) == 1
        assert gps_events[0]["drone_id"] == "d1"

    def test_tick_gps_spoofing_ground_unit(self, engine: CyberWarfareEngine):
        asset = _make_asset(caps=[_gps_cap()])
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "gps1", (150.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (150.0, 100.0)})
        gps_events = [e for e in events if e["type"] == "gps_degraded"]
        assert len(gps_events) == 1

    def test_tick_dos(self, engine: CyberWarfareEngine):
        cap = CyberCapability(
            capability_id="dos1",
            attack_type=CyberAttackType.DENIAL_OF_SERVICE,
            power=0.7, range_m=200.0, duration=30.0, cooldown=10.0,
        )
        engine.deploy_asset(_make_asset(caps=[cap]))
        engine.launch_attack("asset1", "dos1", (110.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (110.0, 100.0)})
        dos_events = [e for e in events if e["type"] == "network_denied"]
        assert len(dos_events) == 1

    def test_tick_spoofing(self, engine: CyberWarfareEngine):
        cap = CyberCapability(
            capability_id="spoof1",
            attack_type=CyberAttackType.SPOOFING,
            power=0.6, range_m=200.0, duration=40.0, cooldown=10.0,
        )
        engine.deploy_asset(_make_asset(caps=[cap]))
        engine.launch_attack("asset1", "spoof1", (110.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (110.0, 100.0)})
        spoof_events = [e for e in events if e["type"] == "sensor_spoofed"]
        assert len(spoof_events) == 1

    def test_tick_decoy(self, engine: CyberWarfareEngine):
        cap = CyberCapability(
            capability_id="decoy1",
            attack_type=CyberAttackType.DECOY,
            power=0.7, range_m=300.0, duration=50.0, cooldown=10.0,
        )
        engine.deploy_asset(_make_asset(caps=[cap]))
        engine.launch_attack("asset1", "decoy1", (110.0, 100.0))
        events = engine.tick(1.0, unit_positions={"u1": (110.0, 100.0)})
        decoy_events = [e for e in events if e["type"] == "decoy_active"]
        assert len(decoy_events) == 1

    def test_tick_empty(self, engine: CyberWarfareEngine):
        events = engine.tick(1.0)
        assert events == []

    def test_tick_cleans_active_attacks_on_expiry(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset(caps=[_jammer_cap(duration=1.0)]))
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert len(engine.assets["asset1"].active_attacks) == 1
        engine.tick(2.0)
        assert len(engine.assets["asset1"].active_attacks) == 0


# ---------------------------------------------------------------------------
# CyberWarfareEngine — queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_is_jammed_true(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert engine.is_jammed((110.0, 100.0)) is True

    def test_is_jammed_false(self, engine: CyberWarfareEngine):
        assert engine.is_jammed((0.0, 0.0)) is False

    def test_is_jammed_outside_radius(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert engine.is_jammed((9999.0, 9999.0)) is False

    def test_is_gps_spoofed_true(self, engine: CyberWarfareEngine):
        asset = _make_asset(caps=[_gps_cap()])
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "gps1", (150.0, 100.0))
        spoofed, fake_pos = engine.is_gps_spoofed((150.0, 100.0))
        assert spoofed is True
        assert fake_pos != (150.0, 100.0)

    def test_is_gps_spoofed_false(self, engine: CyberWarfareEngine):
        spoofed, pos = engine.is_gps_spoofed((0.0, 0.0))
        assert spoofed is False
        assert pos == (0.0, 0.0)

    def test_get_active_effects_at(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        effects = engine.get_active_effects_at((110.0, 100.0))
        assert len(effects) == 1

    def test_get_active_effects_at_empty(self, engine: CyberWarfareEngine):
        effects = engine.get_active_effects_at((0.0, 0.0))
        assert effects == []

    def test_get_effects_by_type(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        jamming = engine.get_effects_by_type(CyberAttackType.JAMMING)
        assert len(jamming) == 1
        gps = engine.get_effects_by_type(CyberAttackType.GPS_SPOOFING)
        assert len(gps) == 0

    def test_get_affected_units(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        effect = engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert effect is not None
        affected = engine.get_affected_units(
            effect, {"u1": (110.0, 100.0), "u2": (9999.0, 0.0)}
        )
        assert "u1" in affected
        assert "u2" not in affected

    def test_get_intercepted_messages(self, engine: CyberWarfareEngine):
        cap = CyberCapability(
            capability_id="int1",
            attack_type=CyberAttackType.INTERCEPT,
            power=1.0, range_m=500.0, duration=300.0, cooldown=5.0,
        )
        engine.deploy_asset(_make_asset(caps=[cap]))
        engine.launch_attack("asset1", "int1", (110.0, 100.0))
        # Run many ticks to get intercepts (probabilistic)
        for _ in range(200):
            engine.tick(1.0, unit_positions={"u1": (110.0, 100.0)})
        msgs = engine.get_intercepted_messages("asset1")
        # With power=1.0 and 200 ticks, we should have intercepted something
        assert len(msgs) > 0


# ---------------------------------------------------------------------------
# CyberWarfareEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_empty(self, engine: CyberWarfareEngine):
        viz = engine.to_three_js()
        assert viz["assets"] == []
        assert viz["effects"] == []
        assert viz["attack_lines"] == []

    def test_with_asset(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        viz = engine.to_three_js()
        assert len(viz["assets"]) == 1
        assert viz["assets"][0]["id"] == "asset1"
        assert viz["assets"][0]["alliance"] == "friendly"

    def test_with_effect(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        viz = engine.to_three_js()
        assert len(viz["effects"]) == 1
        assert viz["effects"][0]["type"] == "jamming"
        assert "radius" in viz["effects"][0]

    def test_attack_lines(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        viz = engine.to_three_js()
        assert len(viz["attack_lines"]) == 1
        assert viz["attack_lines"][0]["asset_id"] == "asset1"

    def test_gps_spoof_fake_position_in_viz(self, engine: CyberWarfareEngine):
        asset = _make_asset(caps=[_gps_cap()])
        engine.deploy_asset(asset)
        engine.launch_attack("asset1", "gps1", (150.0, 100.0))
        viz = engine.to_three_js()
        assert "fake_position" in viz["effects"][0]

    def test_capabilities_in_asset(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        viz = engine.to_three_js()
        assert len(viz["assets"][0]["capabilities"]) == 1
        assert viz["assets"][0]["capabilities"][0]["type"] == "jamming"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_all_presets_exist(self):
        expected = {"jammer_truck", "sigint_post", "gps_spoofer",
                    "drone_controller", "cyber_team"}
        assert set(CYBER_PRESETS.keys()) == expected

    def test_jammer_truck(self):
        caps = CYBER_PRESETS["jammer_truck"]["capabilities"]
        assert len(caps) == 1
        assert caps[0].attack_type == CyberAttackType.JAMMING

    def test_sigint_post(self):
        caps = CYBER_PRESETS["sigint_post"]["capabilities"]
        assert caps[0].attack_type == CyberAttackType.INTERCEPT

    def test_gps_spoofer(self):
        caps = CYBER_PRESETS["gps_spoofer"]["capabilities"]
        types = {c.attack_type for c in caps}
        assert CyberAttackType.GPS_SPOOFING in types

    def test_drone_controller(self):
        caps = CYBER_PRESETS["drone_controller"]["capabilities"]
        assert caps[0].attack_type == CyberAttackType.DRONE_HIJACK
        assert caps[0].requires_los is True

    def test_cyber_team_has_multiple_caps(self):
        caps = CYBER_PRESETS["cyber_team"]["capabilities"]
        assert len(caps) == 5


# ---------------------------------------------------------------------------
# create_asset_from_preset
# ---------------------------------------------------------------------------

class TestCreateAssetFromPreset:
    def test_basic(self):
        asset = create_asset_from_preset("jammer_truck", "jt1", (0.0, 0.0), "friendly")
        assert asset.asset_id == "jt1"
        assert len(asset.capabilities) == 1

    def test_caps_have_prefixed_ids(self):
        asset = create_asset_from_preset("jammer_truck", "jt1", (0.0, 0.0), "friendly")
        assert asset.capabilities[0].capability_id.startswith("jt1_")

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            create_asset_from_preset("nope", "x", (0.0, 0.0), "friendly")

    def test_independent_cooldowns(self):
        a1 = create_asset_from_preset("jammer_truck", "a1", (0.0, 0.0), "friendly")
        a2 = create_asset_from_preset("jammer_truck", "a2", (0.0, 0.0), "friendly")
        a1.capabilities[0].cooldown_remaining = 99.0
        assert a2.capabilities[0].cooldown_remaining == 0.0

    def test_cyber_team_preset(self):
        asset = create_asset_from_preset("cyber_team", "ct1", (50.0, 50.0), "hostile")
        assert asset.alliance == "hostile"
        assert len(asset.capabilities) == 5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_opposing_alliance_friendly(self):
        assert _opposing_alliance("friendly") == "hostile"

    def test_opposing_alliance_hostile(self):
        assert _opposing_alliance("hostile") == "friendly"

    def test_opposing_alliance_unknown(self):
        assert _opposing_alliance("neutral") == "unknown"

    def test_alliance_color(self):
        assert _alliance_color("friendly") == "#05ffa1"
        assert _alliance_color("hostile") == "#ff2a6d"
        assert _alliance_color("neutral") == "#fcee0a"
        assert _alliance_color("other") == "#888888"

    def test_effect_color(self):
        assert _effect_color(CyberAttackType.JAMMING) == "#ff2a6d"
        assert _effect_color(CyberAttackType.INTERCEPT) == "#00f0ff"
        assert _effect_color(CyberAttackType.DECOY) == "#fcee0a"


# ---------------------------------------------------------------------------
# drain_event_log
# ---------------------------------------------------------------------------

class TestDrainEventLog:
    def test_drain_returns_and_clears(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset())
        engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        log = engine.drain_event_log()
        assert len(log) == 1
        assert engine.drain_event_log() == []


# ---------------------------------------------------------------------------
# Integration: multi-asset, multi-effect
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_multiple_assets(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset("a1", (0.0, 0.0)))
        engine.deploy_asset(_make_asset("a2", (500.0, 500.0)))
        assert len(engine.assets) == 2

    def test_multiple_effects_overlap(self, engine: CyberWarfareEngine):
        engine.deploy_asset(_make_asset("a1", (100.0, 100.0)))
        engine.deploy_asset(_make_asset("a2", (120.0, 100.0)))
        engine.launch_attack("a1", "jam1", (110.0, 100.0))
        engine.launch_attack("a2", "jam1", (110.0, 100.0))
        assert len(engine.active_effects) == 2
        assert engine.is_jammed((110.0, 100.0))

    def test_full_lifecycle(self, engine: CyberWarfareEngine):
        """Deploy, attack, tick until expired, verify cleanup."""
        asset = _make_asset(caps=[_jammer_cap(duration=3.0, cooldown=1.0)])
        engine.deploy_asset(asset)
        effect = engine.launch_attack("asset1", "jam1", (110.0, 100.0))
        assert effect is not None
        assert engine.is_jammed((110.0, 100.0))

        # Tick 2s — still active
        engine.tick(2.0)
        assert len(engine.active_effects) == 1

        # Tick 2s more — expired
        engine.tick(2.0)
        assert len(engine.active_effects) == 0
        assert engine.is_jammed((110.0, 100.0)) is False

        # Cooldown should have elapsed (1s cooldown, 4s total)
        cap = engine.assets["asset1"].capabilities[0]
        assert cap.is_ready

    def test_mixed_attack_types(self, engine: CyberWarfareEngine):
        asset = create_asset_from_preset("cyber_team", "ct", (100.0, 100.0), "friendly")
        engine.deploy_asset(asset)
        # Launch each capability
        launched = 0
        for cap in asset.capabilities:
            result = engine.launch_attack("ct", cap.capability_id, (110.0, 100.0))
            if result is not None:
                launched += 1
        assert launched == 5
        assert len(engine.active_effects) == 5
