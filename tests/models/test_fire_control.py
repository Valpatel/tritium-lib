# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for fire-control command models (turret actuation wire contract)."""

from math import atan, degrees

import pytest
from pydantic import ValidationError

from tritium_lib.models.fire_control import (
    FireCommand,
    FireSolution,
    TurretAimCommand,
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
