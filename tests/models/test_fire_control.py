# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for fire-control command models (turret actuation wire contract)."""

from math import atan, degrees

import pytest
from pydantic import ValidationError

from tritium_lib.models.fire_control import (
    PAN_MAX_DEG,
    PAN_MIN_DEG,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    FireCommand,
    FireSolution,
    TurretAimCommand,
    WeaponStatus,
    compute_fire_solution,
)


# --- Validation clamps mirror the physical servo limits -------------------


def test_turret_aim_accepts_in_range():
    cmd = TurretAimCommand(pan=45.0, tilt=-10.0)
    assert cmd.command == "turret_aim"
    assert cmd.pan == 45.0
    assert cmd.tilt == -10.0
    assert cmd.target_id is None
    assert cmd.timestamp  # ISO default populated


def test_turret_aim_rejects_pan_out_of_range():
    with pytest.raises(ValidationError):
        TurretAimCommand(pan=100.0, tilt=0.0)
    with pytest.raises(ValidationError):
        TurretAimCommand(pan=-100.0, tilt=0.0)


def test_turret_aim_rejects_tilt_out_of_range():
    # -45 is below the -30 floor; 61 is above the 60 ceiling.
    with pytest.raises(ValidationError):
        TurretAimCommand(pan=0.0, tilt=-45.0)
    with pytest.raises(ValidationError):
        TurretAimCommand(pan=0.0, tilt=61.0)


def test_fire_command_burst_bounds():
    assert FireCommand().burst == 1
    FireCommand(burst=10)  # upper bound OK
    with pytest.raises(ValidationError):
        FireCommand(burst=0)
    with pytest.raises(ValidationError):
        FireCommand(burst=11)


# --- Wire dicts match the robot-template contract keys --------------------


def test_turret_aim_wire_keys():
    """model_dump(exclude_none=True) must produce the exact keys robot.py parses."""
    cmd = TurretAimCommand(pan=45.0, tilt=-10.0)
    wire = cmd.model_dump(exclude_none=True)
    # command/pan/tilt are the keys examples/robot-template/robot.py reads.
    assert wire["command"] == "turret_aim"
    assert wire["pan"] == 45.0
    assert wire["tilt"] == -10.0
    # target_id dropped when unset; timestamp always present for ACK correlation.
    assert "target_id" not in wire
    assert "timestamp" in wire
    # Round-trips back through the model.
    assert TurretAimCommand(**wire).pan == 45.0


def test_turret_aim_wire_keeps_target_id_when_set():
    wire = TurretAimCommand(pan=0.0, tilt=0.0, target_id="det_person_3").model_dump(
        exclude_none=True
    )
    assert wire["target_id"] == "det_person_3"


def test_fire_command_wire_keys():
    cmd = FireCommand(target_id="det_person_3")
    wire = cmd.model_dump(exclude_none=True)
    assert wire["command"] == "fire"
    assert wire["target_id"] == "det_person_3"
    assert wire["burst"] == 1
    assert FireCommand(**wire).target_id == "det_person_3"


def test_fire_command_wire_drops_none_target():
    wire = FireCommand().model_dump(exclude_none=True)
    assert wire["command"] == "fire"
    assert "target_id" not in wire


# --- compute_fire_solution: bearing convention (0 = +y north, clockwise) --


def test_solution_due_north_pan_zero():
    sol = compute_fire_solution((0.0, 0.0), (0.0, 10.0))
    assert isinstance(sol, FireSolution)
    assert sol.pan == pytest.approx(0.0)
    assert sol.distance == pytest.approx(10.0)
    assert sol.tilt == 0.0  # flat shot


def test_solution_due_east_pan_ninety():
    sol = compute_fire_solution((0.0, 0.0), (10.0, 0.0))
    assert sol.pan == pytest.approx(90.0)
    assert sol.distance == pytest.approx(10.0)


def test_solution_behind_left_clamps_to_minus_ninety():
    # South-west target: true bearing -135 deg, clamped to the -90 pan limit.
    sol = compute_fire_solution((0.0, 0.0), (-10.0, -10.0))
    assert sol.pan == pytest.approx(-90.0)


# --- compute_fire_solution: mortar arc elevation --------------------------


def test_solution_mortar_arc_tilt():
    # arc_peak 10 over dist 25 -> atan(4*10/25) ~= 57.99 deg, within [-30,60].
    sol = compute_fire_solution((0.0, 0.0), (0.0, 25.0), arc_peak=10.0)
    expected = degrees(atan(40.0 / 25.0))
    assert sol.tilt == pytest.approx(expected, abs=1e-6)
    assert expected < 60.0  # not clamped


def test_solution_steep_arc_clamps_tilt_to_sixty():
    # arc_peak 20 over dist 25 -> atan(80/25) ~= 72.6 deg, clamped to 60.
    sol = compute_fire_solution((0.0, 0.0), (0.0, 25.0), arc_peak=20.0)
    assert sol.tilt == pytest.approx(60.0)


def test_solution_flat_shot_tilt_zero():
    sol = compute_fire_solution((5.0, 5.0), (5.0, 30.0), arc_peak=0.0)
    assert sol.tilt == 0.0


# --- WeaponStatus telemetry (robot -> SC reverse direction) ---------------


def test_weapon_status_construction_and_defaults():
    ws = WeaponStatus(
        device_id="turret_alpha", ammo=6, max_ammo=12,
        pan_deg=10.0, tilt_deg=5.0,
    )
    assert ws.device_id == "turret_alpha"
    assert ws.weapon_id == "primary"     # default
    assert ws.ammo == 6
    assert ws.max_ammo == 12
    assert ws.reloading is False          # default
    assert ws.reload_remaining_s == 0.0   # default
    assert ws.fault is None               # default
    assert ws.ts                          # ISO default populated
    assert ws.pan_deg == 10.0
    assert ws.tilt_deg == 5.0


def test_weapon_status_ammo_pct():
    full = WeaponStatus(device_id="t", ammo=12, max_ammo=12, pan_deg=0.0, tilt_deg=0.0)
    assert full.ammo_pct == pytest.approx(1.0)
    half = WeaponStatus(device_id="t", ammo=6, max_ammo=12, pan_deg=0.0, tilt_deg=0.0)
    assert half.ammo_pct == pytest.approx(0.5)
    empty = WeaponStatus(device_id="t", ammo=0, max_ammo=12, pan_deg=0.0, tilt_deg=0.0)
    assert empty.ammo_pct == pytest.approx(0.0)


def test_weapon_status_rejects_bad_bounds():
    with pytest.raises(ValidationError):
        WeaponStatus(device_id="t", ammo=-1, max_ammo=12, pan_deg=0.0, tilt_deg=0.0)
    with pytest.raises(ValidationError):
        WeaponStatus(device_id="t", ammo=0, max_ammo=0, pan_deg=0.0, tilt_deg=0.0)
    with pytest.raises(ValidationError):
        WeaponStatus(
            device_id="t", ammo=0, max_ammo=1, reload_remaining_s=-1.0,
            pan_deg=0.0, tilt_deg=0.0,
        )


def test_weapon_status_servo_clamp_matches_command_limits():
    """Actual servo readings are clamped to the SAME limits the command
    direction enforces (pan +/-90, tilt -30..+60)."""
    over = WeaponStatus(device_id="t", ammo=1, max_ammo=1, pan_deg=200.0, tilt_deg=200.0)
    assert over.pan_deg == PAN_MAX_DEG
    assert over.tilt_deg == TILT_MAX_DEG
    under = WeaponStatus(device_id="t", ammo=1, max_ammo=1, pan_deg=-200.0, tilt_deg=-200.0)
    assert under.pan_deg == PAN_MIN_DEG
    assert under.tilt_deg == TILT_MIN_DEG
    # An in-range reading a command could also produce is preserved unchanged.
    ok = WeaponStatus(device_id="t", ammo=1, max_ammo=1, pan_deg=45.0, tilt_deg=-10.0)
    assert ok.pan_deg == 45.0
    assert ok.tilt_deg == -10.0
    # Same servo window bounds the command model (which rejects out-of-range).
    assert TurretAimCommand(pan=PAN_MAX_DEG, tilt=TILT_MAX_DEG).pan == PAN_MAX_DEG
    with pytest.raises(ValidationError):
        TurretAimCommand(pan=200.0, tilt=0.0)


def test_weapon_status_fault_round_trip():
    ws = WeaponStatus(
        device_id="t", ammo=0, max_ammo=6, reloading=True,
        reload_remaining_s=1.5, pan_deg=0.0, tilt_deg=0.0, fault="jam",
    )
    assert ws.fault == "jam"
    assert ws.reloading is True
    assert ws.reload_remaining_s == 1.5


def test_weapon_status_json_round_trip():
    ws = WeaponStatus(
        device_id="turret_alpha", weapon_id="secondary",
        ammo=3, max_ammo=6, reloading=True, reload_remaining_s=0.8,
        pan_deg=30.0, tilt_deg=-5.0, fault="servo_stall",
    )
    dumped = ws.model_dump()
    assert dumped["device_id"] == "turret_alpha"
    assert dumped["ammo_pct"] == pytest.approx(0.5)  # computed field serialized
    # dict round-trip (computed ammo_pct in the payload is ignored on validate).
    restored = WeaponStatus.model_validate(dumped)
    assert restored.device_id == ws.device_id
    assert restored.weapon_id == "secondary"
    assert restored.ammo == 3
    assert restored.fault == "servo_stall"
    assert restored.pan_deg == 30.0
    assert restored.ammo_pct == pytest.approx(0.5)
    # JSON string round-trip too.
    restored2 = WeaponStatus.model_validate_json(ws.model_dump_json())
    assert restored2.reload_remaining_s == pytest.approx(0.8)
    assert restored2.tilt_deg == -5.0
