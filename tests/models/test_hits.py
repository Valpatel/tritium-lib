# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for hit-feedback models (damage/health wire contract).

Covers:
  - RegisterHitCommand wire form ("command": "register_hit" so robot.py can
    key on it), defaults, damage >= 0, source Literal enforcement.
  - HitReport field validation + source Literal enforcement.
  - HealthStatus computed ``alive``, to_health_state EXACT snapshot keys.
  - Wire round-trip (model_dump_json -> model_validate_json) for all three.
  - HealthTracker: drain, clamp at 0, the mobility ladder
    1.0 -> LIMP_MOBILITY -> 0.0 (including the exact fraction ==
    LIMP_THRESHOLD boundary), hit_id uniqueness + echo, dead-body hits,
    HitReport contents from apply_hit.
"""

import pytest
from pydantic import ValidationError

from tritium_lib.models.hits import (
    DEFAULT_HP,
    HIT_SOURCES,
    LIMP_MOBILITY,
    LIMP_THRESHOLD,
    HealthStatus,
    HealthTracker,
    HitReport,
    RegisterHitCommand,
)


# --- RegisterHitCommand — adjudicator -> robot -----------------------------


def test_register_hit_command_defaults():
    cmd = RegisterHitCommand()
    assert cmd.command == "register_hit"
    assert cmd.damage == 10.0
    assert cmd.source == "referee"
    assert cmd.shooter_id is None
    assert cmd.location is None
    assert len(cmd.hit_id) == 12
    assert cmd.timestamp  # ISO default populated


def test_register_hit_command_wire_form_keys_on_command():
    """robot.py handle_command keys on the "command" field — it must be on
    the wire as the literal string "register_hit"."""
    cmd = RegisterHitCommand(shooter_id="dog_a", damage=8.0, location="chassis")
    wire = cmd.model_dump()
    assert wire["command"] == "register_hit"
    assert '"command":"register_hit"' in cmd.model_dump_json().replace(" ", "")


def test_register_hit_command_hit_ids_unique():
    assert RegisterHitCommand().hit_id != RegisterHitCommand().hit_id


def test_register_hit_command_rejects_negative_damage():
    with pytest.raises(ValidationError):
        RegisterHitCommand(damage=-1.0)


def test_register_hit_command_source_literal_enforced():
    with pytest.raises(ValidationError):
        RegisterHitCommand(source="laser_tag")
    for source in HIT_SOURCES:
        assert RegisterHitCommand(source=source).source == source


# --- HitReport — robot -> world --------------------------------------------


def _report(**overrides) -> HitReport:
    base = dict(
        hit_id="abc123def456", target_id="dog_b", shooter_id="dog_a",
        damage=8.0, hp_after=32.0, max_hp=40.0, alive=True,
        location="chassis",
    )
    base.update(overrides)
    return HitReport(**base)


def test_hit_report_fields_and_defaults():
    rep = _report()
    assert rep.source == "referee"
    assert rep.ts  # ISO default populated
    assert rep.alive is True


def test_hit_report_rejects_negative_damage_and_hp():
    with pytest.raises(ValidationError):
        _report(damage=-0.5)
    with pytest.raises(ValidationError):
        _report(hp_after=-1.0)
    with pytest.raises(ValidationError):
        _report(max_hp=0.0)  # ge=1


def test_hit_report_source_literal_enforced():
    with pytest.raises(ValidationError):
        _report(source="rumor")
    for source in HIT_SOURCES:
        assert _report(source=source).source == source


# --- HealthStatus — telemetry "health" block --------------------------------


def test_health_status_alive_computed():
    assert HealthStatus(device_id="dog_b", hp=12.0, max_hp=40.0).alive is True
    assert HealthStatus(device_id="dog_b", hp=0.0, max_hp=40.0).alive is False


def test_health_status_alive_on_the_wire():
    """Computed field serialises — a subscriber never derives it itself."""
    dumped = HealthStatus(device_id="dog_b", hp=0.0, max_hp=40.0).model_dump()
    assert dumped["alive"] is False


def test_health_status_validation_bounds():
    with pytest.raises(ValidationError):
        HealthStatus(device_id="d", hp=-1.0, max_hp=40.0)
    with pytest.raises(ValidationError):
        HealthStatus(device_id="d", hp=1.0, max_hp=0.5)  # max_hp ge 1
    with pytest.raises(ValidationError):
        HealthStatus(device_id="d", hp=1.0, max_hp=40.0, hits_taken=-1)
    with pytest.raises(ValidationError):
        HealthStatus(device_id="d", hp=1.0, max_hp=40.0, mobility=1.5)


def test_to_health_state_exact_keys():
    status = HealthStatus(
        device_id="dog_b", hp=12.0, max_hp=40.0, hits_taken=4, mobility=0.45,
    )
    state = status.to_health_state()
    assert set(state) == {"hp", "max_hp", "alive", "hits_taken", "mobility"}
    assert state == {
        "hp": 12.0, "max_hp": 40.0, "alive": True,
        "hits_taken": 4, "mobility": 0.45,
    }


# --- Wire round-trips — the whole point of a wire contract ------------------


def test_wire_round_trip_all_models():
    cmd = RegisterHitCommand(shooter_id="dog_a", damage=8.0, location="turret")
    rep = _report(source="hit_sensor")
    status = HealthStatus(
        device_id="dog_b", hp=14.0, max_hp=40.0, hits_taken=3,
        mobility=LIMP_MOBILITY, last_hit_ts="2026-07-11T00:00:00+00:00",
    )
    assert RegisterHitCommand.model_validate_json(cmd.model_dump_json()) == cmd
    assert HitReport.model_validate_json(rep.model_dump_json()) == rep
    assert HealthStatus.model_validate_json(status.model_dump_json()) == status


# --- HealthTracker — the dog's own book -------------------------------------


def test_tracker_initial_state():
    t = HealthTracker("dog_b")
    assert t.hp == DEFAULT_HP
    assert t.max_hp == DEFAULT_HP
    assert t.alive is True
    assert t.hp_fraction == 1.0
    assert t.mobility_factor() == 1.0
    status = t.status()
    assert status.device_id == "dog_b"
    assert status.hits_taken == 0
    assert status.last_hit_ts is None
    assert status.mobility == 1.0


def test_tracker_hit_drains_and_reports():
    t = HealthTracker("dog_b", max_hp=40.0)
    rep = t.apply_hit(10.0, shooter_id="dog_a", location="chassis")
    assert t.hp == 30.0
    assert isinstance(rep, HitReport)
    assert rep.target_id == "dog_b"
    assert rep.shooter_id == "dog_a"
    assert rep.damage == 10.0
    assert rep.hp_after == 30.0
    assert rep.max_hp == 40.0
    assert rep.alive is True
    assert rep.location == "chassis"
    assert rep.source == "referee"
    assert rep.ts
    status = t.status()
    assert status.hits_taken == 1
    assert status.last_hit_ts == rep.ts


def test_tracker_clamps_hp_at_zero():
    t = HealthTracker("dog_b", max_hp=40.0)
    rep = t.apply_hit(1000.0)
    assert t.hp == 0.0
    assert t.alive is False
    assert rep.hp_after == 0.0
    assert rep.alive is False


def test_tracker_mobility_ladder():
    """1.0 healthy -> LIMP_MOBILITY at/below the threshold -> 0.0 dead."""
    t = HealthTracker("dog_b", max_hp=40.0)
    assert t.mobility_factor() == 1.0
    t.apply_hit(20.0)  # hp 20, fraction 0.5 > threshold
    assert t.mobility_factor() == 1.0
    t.apply_hit(10.0)  # hp 10, fraction 0.25 <= threshold
    assert t.mobility_factor() == LIMP_MOBILITY
    t.apply_hit(10.0)  # hp 0 — dead
    assert t.mobility_factor() == 0.0
    assert t.status().mobility == 0.0


def test_tracker_mobility_exact_threshold_boundary_limps():
    """fraction == LIMP_THRESHOLD is inclusive — the boundary dog limps."""
    t = HealthTracker("dog_b", max_hp=40.0)
    t.apply_hit(26.0)  # hp 14 -> 14/40 == 0.35 exactly
    assert t.hp_fraction == LIMP_THRESHOLD
    assert t.mobility_factor() == LIMP_MOBILITY
    assert t.status().mobility == LIMP_MOBILITY


def test_tracker_hit_id_generated_unique_and_echoed():
    t = HealthTracker("dog_b")
    a = t.apply_hit(1.0)
    b = t.apply_hit(1.0)
    assert a.hit_id != b.hit_id  # fresh ids when none supplied
    echoed = t.apply_hit(1.0, hit_id="cmd_echo_0001")
    assert echoed.hit_id == "cmd_echo_0001"  # RegisterHitCommand provenance


def test_tracker_dead_body_still_takes_hits():
    """Foam keeps flying after a KO: hp pinned at 0, hits still counted."""
    t = HealthTracker("dog_b", max_hp=40.0)
    t.apply_hit(40.0)
    assert t.alive is False
    rep = t.apply_hit(8.0, source="hit_sensor")
    assert t.hp == 0.0
    assert t.alive is False
    assert rep.hp_after == 0.0
    assert rep.alive is False
    assert rep.source == "hit_sensor"
    assert t.status().hits_taken == 2


def test_tracker_negative_damage_clamped():
    """A hit never heals — negative damage books as a zero-damage impact."""
    t = HealthTracker("dog_b", max_hp=40.0)
    rep = t.apply_hit(-5.0)
    assert t.hp == 40.0
    assert rep.damage == 0.0
    assert t.status().hits_taken == 1
