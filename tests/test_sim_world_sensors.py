# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.world.sensors — SensorSimulator."""

import time
import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.world.sensors import SensorSimulator, SensorDevice


def _make_event_bus():
    """Create a minimal event bus mock that records published events."""
    bus = SimpleNamespace()
    bus._events = []
    bus.publish = lambda topic, data: bus._events.append((topic, data))
    return bus


def _make_target(name, tid, x, y, status="active"):
    return SimpleNamespace(
        name=name,
        target_id=tid,
        position=(x, y),
        status=status,
    )


class TestSensorDeviceConstruction:
    def test_basic_creation(self):
        d = SensorDevice(
            sensor_id="s1", name="Motion 1", sensor_type="motion",
            position=(10.0, 20.0), radius=5.0,
        )
        assert d.sensor_id == "s1"
        assert d.sensor_type == "motion"
        assert d.active is False
        assert d.last_triggered == 0.0
        assert d.triggered_by == ""


class TestSensorSimulatorConstruction:
    def test_empty_simulator(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        assert sim.sensors == []

    def test_add_sensor(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion Sensor", "motion", (10.0, 20.0), 5.0)
        assert len(sim.sensors) == 1
        assert sim.sensors[0].sensor_id == "s1"

    def test_add_multiple_sensors(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "A", "motion", (0.0, 0.0), 5.0)
        sim.add_sensor("s2", "B", "door", (10.0, 10.0), 3.0)
        sim.add_sensor("s3", "C", "tripwire", (20.0, 20.0), 2.0)
        assert len(sim.sensors) == 3


class TestSensorSimulatorTick:
    def test_sensor_triggers_on_nearby_target(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)
        targets = [_make_target("Person", "p1", 11.0, 11.0)]
        sim.tick(0.1, targets)
        assert any(topic == "sensor_triggered" for topic, _ in bus._events)

    def test_sensor_not_triggered_by_distant_target(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)
        targets = [_make_target("Person", "p1", 100.0, 100.0)]
        sim.tick(0.1, targets)
        assert not any(topic == "sensor_triggered" for topic, _ in bus._events)

    def test_sensor_ignores_destroyed_targets(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 50.0)
        targets = [_make_target("Person", "p1", 11.0, 11.0, status="destroyed")]
        sim.tick(0.1, targets)
        assert not any(topic == "sensor_triggered" for topic, _ in bus._events)

    def test_sensor_ignores_eliminated_targets(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 50.0)
        targets = [_make_target("Person", "p1", 11.0, 11.0, status="eliminated")]
        sim.tick(0.1, targets)
        assert not any(topic == "sensor_triggered" for topic, _ in bus._events)

    def test_sensor_clears_when_target_leaves(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)

        # Target enters range
        targets = [_make_target("Person", "p1", 11.0, 11.0)]
        sim.tick(0.1, targets)
        assert sim.sensors[0].active

        # Target leaves range
        targets = [_make_target("Person", "p1", 500.0, 500.0)]
        sim.tick(0.1, targets)
        assert not sim.sensors[0].active
        assert any(topic == "sensor_cleared" for topic, _ in bus._events)

    def test_debounce_prevents_retrigger(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)

        # First trigger
        targets = [_make_target("Person", "p1", 11.0, 11.0)]
        sim.tick(0.1, targets)
        trigger_count_1 = sum(1 for t, _ in bus._events if t == "sensor_triggered")

        # Target leaves, then re-enters immediately (within debounce)
        sim.tick(0.1, [_make_target("Person", "p1", 500.0, 500.0)])
        bus._events.clear()
        sim.tick(0.1, targets)
        trigger_count_2 = sum(1 for t, _ in bus._events if t == "sensor_triggered")
        # Should not re-trigger within debounce window
        assert trigger_count_2 == 0

    def test_trigger_records_target_info(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)
        targets = [_make_target("John", "p1", 11.0, 11.0)]
        sim.tick(0.1, targets)
        triggered_events = [(t, d) for t, d in bus._events if t == "sensor_triggered"]
        assert len(triggered_events) >= 1
        data = triggered_events[0][1]
        assert data["triggered_by"] == "John"
        assert data["target_id"] == "p1"

    def test_multiple_sensors_independent(self):
        bus = _make_event_bus()
        sim = SensorSimulator(event_bus=bus)
        sim.add_sensor("s1", "Sensor A", "motion", (0.0, 0.0), 5.0)
        sim.add_sensor("s2", "Sensor B", "motion", (100.0, 100.0), 5.0)
        targets = [_make_target("Person", "p1", 1.0, 1.0)]
        sim.tick(0.1, targets)
        # Only s1 should trigger
        assert sim.sensors[0].active
        assert not sim.sensors[1].active


class TestSensorSimulatorSetEventBus:
    def test_set_event_bus(self):
        bus1 = _make_event_bus()
        bus2 = _make_event_bus()
        sim = SensorSimulator(event_bus=bus1)
        sim.set_event_bus(bus2)
        sim.add_sensor("s1", "Motion", "motion", (10.0, 10.0), 5.0)
        targets = [_make_target("Person", "p1", 11.0, 11.0)]
        sim.tick(0.1, targets)
        # Events should go to bus2
        assert len(bus2._events) > 0
        assert len(bus1._events) == 0
