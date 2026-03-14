# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for drone/UAV integration models."""

import pytest
from tritium_lib.models.drone import (
    DroneCommand,
    DroneMission,
    DroneRegistration,
    DroneState,
    DroneTelemetry,
    DroneType,
    Waypoint,
)


class TestDroneState:
    def test_enum_values(self):
        assert DroneState.IDLE == "idle"
        assert DroneState.FLYING == "flying"
        assert DroneState.RTL == "rtl"
        assert DroneState.EMERGENCY == "emergency"

    def test_from_string(self):
        assert DroneState("armed") == DroneState.ARMED


class TestDroneTelemetry:
    def test_defaults(self):
        t = DroneTelemetry(drone_id="drone-01")
        assert t.drone_id == "drone-01"
        assert t.state == DroneState.OFFLINE
        assert t.battery_percent == 0.0

    def test_with_position(self):
        t = DroneTelemetry(
            drone_id="drone-01",
            state=DroneState.FLYING,
            latitude=37.7159,
            longitude=-121.8960,
            altitude_agl=30.0,
            battery_percent=75.0,
            gps_fix=3,
            satellites=12,
        )
        assert t.state == DroneState.FLYING
        assert t.altitude_agl == 30.0

    def test_json_roundtrip(self):
        t = DroneTelemetry(drone_id="test", state=DroneState.HOVERING)
        j = t.model_dump_json()
        t2 = DroneTelemetry.model_validate_json(j)
        assert t2.drone_id == "test"
        assert t2.state == DroneState.HOVERING


class TestDroneCommand:
    def test_simple_command(self):
        cmd = DroneCommand(command="arm")
        assert cmd.command == "arm"
        assert cmd.params == {}

    def test_goto_command(self):
        cmd = DroneCommand(
            command="goto",
            params={"lat": 37.7, "lng": -121.9, "alt": 30.0},
        )
        assert cmd.params["lat"] == 37.7


class TestDroneMission:
    def test_empty_mission(self):
        m = DroneMission(mission_id="patrol_01")
        assert m.waypoints == []

    def test_mission_with_waypoints(self):
        m = DroneMission(
            mission_id="survey_01",
            name="Area Survey",
            waypoints=[
                Waypoint(seq=0, latitude=37.71, longitude=-121.89),
                Waypoint(seq=1, latitude=37.72, longitude=-121.88, hold_seconds=5.0),
                Waypoint(seq=2, latitude=37.71, longitude=-121.88, action="photo"),
            ],
        )
        assert len(m.waypoints) == 3
        assert m.waypoints[2].action == "photo"


class TestDroneRegistration:
    def test_defaults(self):
        r = DroneRegistration(drone_id="d01")
        assert r.drone_type == DroneType.MULTIROTOR
        assert r.has_camera is True

    def test_fixed_wing(self):
        r = DroneRegistration(
            drone_id="fw01",
            drone_type=DroneType.FIXED_WING,
            max_flight_time_min=120.0,
        )
        assert r.drone_type == DroneType.FIXED_WING
