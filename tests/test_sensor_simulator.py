# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SensorSimulator — virtual sensor network with triggers and debounce."""

import time

import pytest

from tritium_lib.sim_engine.world.sensors import SensorDevice, SensorSimulator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeEventBus:
    def __init__(self):
        self.events = []

    def publish(self, event_type: str, data: dict):
        self.events.append((event_type, data))


class _FakeTarget:
    def __init__(self, target_id: str, name: str, position: tuple,
                 status: str = "active"):
        self.target_id = target_id
        self.name = name
        self.position = position
        self.status = status


# ---------------------------------------------------------------------------
# SensorDevice
# ---------------------------------------------------------------------------

class TestSensorDevice:
    def test_creation(self):
        sd = SensorDevice(
            sensor_id="s1",
            name="Front Door",
            sensor_type="motion",
            position=(10.0, 20.0),
            radius=5.0,
        )
        assert sd.sensor_id == "s1"
        assert sd.name == "Front Door"
        assert sd.sensor_type == "motion"
        assert sd.position == (10.0, 20.0)
        assert sd.radius == 5.0
        assert sd.active is False
        assert sd.last_triggered == 0.0
        assert sd.triggered_by == ""


# ---------------------------------------------------------------------------
# SensorSimulator — basic operations
# ---------------------------------------------------------------------------

class TestSensorSimulatorBasic:
    def test_construction(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        assert len(sim.sensors) == 0

    def test_add_sensor(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Gate", "motion", (0, 0), 10.0)
        assert len(sim.sensors) == 1
        assert sim.sensors[0].sensor_id == "s1"

    def test_add_multiple_sensors(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "North", "motion", (0, 50), 10.0)
        sim.add_sensor("s2", "South", "door", (0, -50), 5.0)
        sim.add_sensor("s3", "East", "tripwire", (50, 0), 3.0)
        assert len(sim.sensors) == 3

    def test_sensors_returns_copy(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Test", "motion", (0, 0), 5.0)
        sensors = sim.sensors
        sensors.clear()  # Should not affect internal list
        assert len(sim.sensors) == 1


# ---------------------------------------------------------------------------
# SensorSimulator — tick and triggering
# ---------------------------------------------------------------------------

class TestSensorTick:
    def test_trigger_on_nearby_target(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (10.0, 10.0), 5.0)
        target = _FakeTarget("t1", "Intruder", position=(10.0, 10.0))
        sim.tick(0.1, [target])
        # Should trigger
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) >= 1
        assert triggered[0][1]["sensor_id"] == "s1"
        assert triggered[0][1]["triggered_by"] == "Intruder"

    def test_no_trigger_on_distant_target(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 5.0)
        target = _FakeTarget("t1", "Far Away", position=(100, 100))
        sim.tick(0.1, [target])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 0

    def test_cleared_when_target_leaves(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 5.0)
        target = _FakeTarget("t1", "Walker", position=(0, 0))
        sim.tick(0.1, [target])
        # Now target moves away
        target.position = (100, 100)
        sim.tick(0.1, [target])
        cleared = [e for e in bus.events if e[0] == "sensor_cleared"]
        assert len(cleared) >= 1

    def test_eliminated_targets_ignored(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 10.0)
        target = _FakeTarget("t1", "Dead Guy", position=(0, 0), status="destroyed")
        sim.tick(0.1, [target])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 0

    def test_despawned_targets_ignored(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 10.0)
        target = _FakeTarget("t1", "Gone", position=(0, 0), status="despawned")
        sim.tick(0.1, [target])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 0

    def test_debounce_prevents_rapid_retrigger(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 10.0)
        target = _FakeTarget("t1", "Walker", position=(0, 0))

        # First trigger
        sim.tick(0.1, [target])
        triggered_count_1 = len([e for e in bus.events if e[0] == "sensor_triggered"])
        assert triggered_count_1 == 1

        # Target leaves and comes back quickly
        target.position = (100, 100)
        sim.tick(0.1, [target])  # Clears
        target.position = (0, 0)
        sim.tick(0.1, [target])  # Should NOT re-trigger within debounce window

        triggered_count_2 = len([e for e in bus.events if e[0] == "sensor_triggered"])
        # Should still be just 1 trigger due to debounce
        assert triggered_count_2 == 1

    def test_multiple_targets_nearest_triggers(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Zone", "motion", (0, 0), 20.0)
        t1 = _FakeTarget("t1", "Alpha", position=(5, 0))
        t2 = _FakeTarget("t2", "Bravo", position=(10, 0))
        sim.tick(0.1, [t1, t2])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 1
        # First target in list that's nearby triggers it
        assert triggered[0][1]["triggered_by"] == "Alpha"

    def test_multiple_sensors_independent(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "North", "motion", (0, 50), 10.0)
        sim.add_sensor("s2", "South", "motion", (0, -50), 10.0)
        t = _FakeTarget("t1", "Walker", position=(0, 50))
        sim.tick(0.1, [t])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        # Only north sensor should trigger
        assert len(triggered) == 1
        assert triggered[0][1]["sensor_id"] == "s1"


# ---------------------------------------------------------------------------
# SensorSimulator — event data format
# ---------------------------------------------------------------------------

class TestSensorEventFormat:
    def test_trigger_event_has_required_fields(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Gate", "motion", (10, 20), 5.0)
        t = _FakeTarget("t1", "Person A", position=(10, 20))
        sim.tick(0.1, [t])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 1
        data = triggered[0][1]
        assert data["sensor_id"] == "s1"
        assert data["name"] == "Gate"
        assert data["type"] == "motion"
        assert data["triggered_by"] == "Person A"
        assert data["target_id"] == "t1"
        assert "position" in data
        assert data["position"]["x"] == 10
        assert data["position"]["z"] == 20

    def test_clear_event_has_required_fields(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Gate", "door", (5, 5), 3.0)
        t = _FakeTarget("t1", "X", position=(5, 5))
        sim.tick(0.1, [t])
        t.position = (100, 100)
        sim.tick(0.1, [t])
        cleared = [e for e in bus.events if e[0] == "sensor_cleared"]
        assert len(cleared) == 1
        data = cleared[0][1]
        assert data["sensor_id"] == "s1"
        assert data["name"] == "Gate"
        assert data["type"] == "door"
        assert "position" in data


# ---------------------------------------------------------------------------
# SensorSimulator — set_event_bus
# ---------------------------------------------------------------------------

class TestSensorSetEventBus:
    def test_set_event_bus(self):
        old_bus = _FakeEventBus()
        new_bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=old_bus)
        sim.set_event_bus(new_bus)
        sim.add_sensor("s1", "Door", "motion", (0, 0), 5.0)
        t = _FakeTarget("t1", "X", position=(0, 0))
        sim.tick(0.1, [t])
        # Events should go to new bus
        assert len(new_bus.events) > 0
        assert len(old_bus.events) == 0


# ---------------------------------------------------------------------------
# SensorSimulator — edge distances
# ---------------------------------------------------------------------------

class TestSensorEdgeDistances:
    def test_exactly_at_radius_triggers(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Test", "motion", (0, 0), 10.0)
        t = _FakeTarget("t1", "Edge", position=(10.0, 0.0))
        sim.tick(0.1, [t])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 1

    def test_just_outside_radius_no_trigger(self):
        bus = _FakeEventBus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Test", "motion", (0, 0), 10.0)
        t = _FakeTarget("t1", "Outside", position=(10.001, 0.0))
        sim.tick(0.1, [t])
        triggered = [e for e in bus.events if e[0] == "sensor_triggered"]
        assert len(triggered) == 0
