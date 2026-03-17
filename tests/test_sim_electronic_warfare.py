# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the electronic warfare simulation module."""

import json

import pytest

from tritium_lib.sim_engine.electronic_warfare import (
    CyberAttack,
    CyberAttackType,
    EMPEvent,
    EMPScale,
    EMP_PRESETS,
    EWEngine,
    EWJammer,
    JammerType,
    SpoofContact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> EWEngine:
    """A fresh EW engine with deterministic RNG."""
    return EWEngine(rng_seed=42)


@pytest.fixture
def jammer() -> EWJammer:
    return EWJammer(
        jammer_id="j1",
        position=(200.0, 200.0),
        radius=100.0,
        jammer_type=JammerType.BROADBAND,
        alliance="friendly",
    )


# ---------------------------------------------------------------------------
# Jammer tests
# ---------------------------------------------------------------------------

class TestJammers:
    def test_place_and_query(self, engine: EWEngine, jammer: EWJammer) -> None:
        engine.place_jammer(jammer)
        assert "j1" in engine.jammers

    def test_remove_jammer(self, engine: EWEngine, jammer: EWJammer) -> None:
        engine.place_jammer(jammer)
        engine.remove_jammer("j1")
        assert "j1" not in engine.jammers

    def test_position_jammed_inside(self, engine: EWEngine, jammer: EWJammer) -> None:
        engine.place_jammer(jammer)
        assert engine.is_position_jammed((200.0, 200.0)) is True
        assert engine.is_position_jammed((250.0, 200.0)) is True

    def test_position_not_jammed_outside(self, engine: EWEngine, jammer: EWJammer) -> None:
        engine.place_jammer(jammer)
        assert engine.is_position_jammed((500.0, 500.0)) is False

    def test_inactive_jammer_no_effect(self, engine: EWEngine, jammer: EWJammer) -> None:
        jammer.is_active = False
        engine.place_jammer(jammer)
        assert engine.is_position_jammed((200.0, 200.0)) is False

    def test_activate_deactivate(self, engine: EWEngine, jammer: EWJammer) -> None:
        jammer.is_active = False
        engine.place_jammer(jammer)
        assert engine.activate_jammer("j1") is True
        assert engine.jammers["j1"].is_active is True
        assert engine.deactivate_jammer("j1") is True
        assert engine.jammers["j1"].is_active is False

    def test_get_jammers_affecting(self, engine: EWEngine, jammer: EWJammer) -> None:
        engine.place_jammer(jammer)
        affecting = engine.get_jammers_affecting((210.0, 200.0))
        assert len(affecting) == 1
        assert affecting[0].jammer_id == "j1"

    def test_battery_drain(self, engine: EWEngine, jammer: EWJammer) -> None:
        jammer.battery = 0.01
        jammer.drain_rate = 0.01
        engine.place_jammer(jammer)
        result = engine.tick(1.0)
        assert engine.jammers["j1"].battery == 0.0
        assert engine.jammers["j1"].is_active is False
        events = result["events"]
        assert any(e["type"] == "jammer_battery_dead" for e in events)

    def test_no_battery_jammer(self, engine: EWEngine, jammer: EWJammer) -> None:
        jammer.battery = 0.0
        engine.place_jammer(jammer)
        assert engine.is_position_jammed((200.0, 200.0)) is False

    def test_activate_no_battery_fails(self, engine: EWEngine, jammer: EWJammer) -> None:
        jammer.battery = 0.0
        jammer.is_active = False
        engine.place_jammer(jammer)
        assert engine.activate_jammer("j1") is False


# ---------------------------------------------------------------------------
# Cyber attack tests
# ---------------------------------------------------------------------------

class TestCyberAttacks:
    def test_launch_cyber_attack_success(self, engine: EWEngine) -> None:
        attack = CyberAttack(
            attack_id="c1",
            target_system="radar",
            target_id="sensor_1",
            duration=10.0,
            success_probability=1.0,  # guaranteed success
        )
        engine.launch_cyber_attack(attack)
        assert attack.is_active is True
        assert attack.succeeded is True
        assert engine.is_system_disrupted("sensor_1") is True

    def test_launch_cyber_attack_failure(self, engine: EWEngine) -> None:
        attack = CyberAttack(
            attack_id="c2",
            target_system="comms",
            target_id="radio_1",
            duration=10.0,
            success_probability=0.0,  # guaranteed failure
        )
        engine.launch_cyber_attack(attack)
        assert attack.succeeded is False
        assert engine.is_system_disrupted("radio_1") is False

    def test_cyber_attack_expires(self, engine: EWEngine) -> None:
        attack = CyberAttack(
            attack_id="c1",
            target_system="radar",
            target_id="sensor_1",
            duration=5.0,
            success_probability=1.0,
        )
        engine.launch_cyber_attack(attack)
        # Tick past duration
        for _ in range(60):
            engine.tick(0.1)
        assert engine.is_system_disrupted("sensor_1") is False

    def test_system_restored_event(self, engine: EWEngine) -> None:
        attack = CyberAttack(
            attack_id="c1",
            target_system="radar",
            target_id="sensor_1",
            duration=1.0,
            success_probability=1.0,
        )
        engine.launch_cyber_attack(attack)
        result = engine.tick(1.5)
        events = result["events"]
        assert any(e["type"] == "system_restored" for e in events)


# ---------------------------------------------------------------------------
# EMP tests
# ---------------------------------------------------------------------------

class TestEMP:
    def test_detonate_emp(self, engine: EWEngine) -> None:
        emp = engine.detonate_emp((100.0, 100.0), preset="tactical", emp_id="emp_1")
        assert emp.emp_id == "emp_1"
        assert emp.is_active is True
        assert emp.radius == EMP_PRESETS["tactical"]["radius"]

    def test_position_emp_affected(self, engine: EWEngine) -> None:
        engine.detonate_emp((100.0, 100.0), preset="tactical")
        assert engine.is_position_emp_affected((110.0, 100.0)) is True
        assert engine.is_position_emp_affected((500.0, 500.0)) is False

    def test_emp_severity_falloff(self, engine: EWEngine) -> None:
        engine.detonate_emp((100.0, 100.0), preset="tactical")
        center = engine.get_emp_severity((100.0, 100.0))
        edge = engine.get_emp_severity((140.0, 100.0))
        assert center > edge
        assert center > 0.0
        assert edge >= 0.0

    def test_emp_expires(self, engine: EWEngine) -> None:
        engine.detonate_emp((100.0, 100.0), preset="tactical")
        # Tick past duration
        for _ in range(100):
            engine.tick(0.1)
        assert engine.is_position_emp_affected((100.0, 100.0)) is False

    def test_emp_auto_id(self, engine: EWEngine) -> None:
        emp = engine.detonate_emp((0.0, 0.0))
        assert emp.emp_id.startswith("emp_")

    def test_theater_emp_larger(self, engine: EWEngine) -> None:
        emp = engine.detonate_emp((0.0, 0.0), preset="theater")
        assert emp.radius == 200.0


# ---------------------------------------------------------------------------
# Spoof tests
# ---------------------------------------------------------------------------

class TestSpoofing:
    def test_create_spoof(self, engine: EWEngine) -> None:
        sc = engine.create_spoof((300.0, 300.0), target_alliance="hostile")
        assert sc.contact_id.startswith("spoof_")
        assert sc.alliance == "hostile"

    def test_get_spoofs_for_alliance(self, engine: EWEngine) -> None:
        engine.create_spoof((300.0, 300.0), target_alliance="hostile")
        engine.create_spoof((400.0, 400.0), target_alliance="friendly")
        hostile_spoofs = engine.get_spoofs_for_alliance("hostile")
        assert len(hostile_spoofs) == 1

    def test_spoof_moves(self, engine: EWEngine) -> None:
        sc = engine.create_spoof(
            (100.0, 100.0),
            velocity=(10.0, 5.0),
            target_alliance="hostile",
        )
        engine.tick(1.0)
        remaining = engine.get_spoofs_for_alliance("hostile")
        assert len(remaining) == 1
        assert remaining[0].position[0] == pytest.approx(110.0, abs=0.1)
        assert remaining[0].position[1] == pytest.approx(105.0, abs=0.1)

    def test_spoof_expires(self, engine: EWEngine) -> None:
        engine.create_spoof((100.0, 100.0), duration=2.0, target_alliance="hostile")
        for _ in range(30):
            engine.tick(0.1)
        assert len(engine.get_spoofs_for_alliance("hostile")) == 0

    def test_spoof_with_classification(self, engine: EWEngine) -> None:
        sc = engine.create_spoof(
            (100.0, 100.0),
            classification="tank",
            target_alliance="hostile",
        )
        assert sc.classification == "tank"


# ---------------------------------------------------------------------------
# Disruption summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_disruption_summary(self, engine: EWEngine) -> None:
        engine.place_jammer(EWJammer(
            jammer_id="j1", position=(0.0, 0.0), radius=50.0,
        ))
        engine.detonate_emp((100.0, 100.0))
        engine.create_spoof((200.0, 200.0), target_alliance="hostile")
        summary = engine.get_disruption_summary()
        assert summary["active_jammers"] == 1
        assert summary["active_emps"] == 1
        assert summary["active_spoofs"] == 1


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_drain_event_log(self, engine: EWEngine) -> None:
        engine.detonate_emp((0.0, 0.0))
        log = engine.drain_event_log()
        assert len(log) > 0
        assert log[0]["type"] == "emp_detonation"
        # Second drain should be empty
        assert len(engine.drain_event_log()) == 0


# ---------------------------------------------------------------------------
# Three.js visualization
# ---------------------------------------------------------------------------

class TestThreeJS:
    def test_to_three_js_structure(self, engine: EWEngine) -> None:
        engine.place_jammer(EWJammer(
            jammer_id="j1", position=(100.0, 100.0), radius=50.0,
        ))
        engine.detonate_emp((200.0, 200.0))
        engine.create_spoof((300.0, 300.0), target_alliance="hostile")

        attack = CyberAttack(
            attack_id="c1", target_system="radar", target_id="s1",
            duration=10.0, success_probability=1.0,
        )
        engine.launch_cyber_attack(attack)

        viz = engine.to_three_js()
        assert "jammers" in viz
        assert "emp_effects" in viz
        assert "spoof_contacts" in viz
        assert "disrupted_systems" in viz
        assert len(viz["jammers"]) == 1
        assert len(viz["emp_effects"]) == 1

    def test_to_three_js_serializable(self, engine: EWEngine) -> None:
        engine.place_jammer(EWJammer(
            jammer_id="j1", position=(100.0, 100.0), radius=50.0,
        ))
        engine.detonate_emp((200.0, 200.0))
        viz = engine.to_three_js()
        serialized = json.dumps(viz)
        assert len(serialized) > 0

    def test_jammer_fields(self, engine: EWEngine) -> None:
        engine.place_jammer(EWJammer(
            jammer_id="j1", position=(100.0, 100.0), radius=50.0,
            jammer_type=JammerType.GPS, alliance="friendly",
        ))
        viz = engine.to_three_js()
        j = viz["jammers"][0]
        assert j["id"] == "j1"
        assert j["type"] == "gps"
        assert j["alliance"] == "friendly"
        assert "color" in j
        assert "radius" in j
