# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.synthetic.data_generators."""

import pytest
from tritium_lib.geo import approx_distance_m
from tritium_lib.synthetic.data_generators import (
    BLEScanGenerator,
    MeshtasticNodeGenerator,
    CameraDetectionGenerator,
    TrilaterationDemoGenerator,
)


class FakeEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, data):
        self.events.append((topic, data))


# --- BLEScanGenerator ---

def test_ble_gen_not_running_initially():
    gen = BLEScanGenerator()
    assert gen.running is False


def test_ble_gen_tick_publishes():
    bus = FakeEventBus()
    gen = BLEScanGenerator(node_lat=37.7749, node_lon=-122.4194)
    gen._event_bus = bus
    gen._running = True
    gen._tick()
    assert len(bus.events) == 1
    assert bus.events[0][0] == "fleet.ble_presence"
    payload = bus.events[0][1]
    assert "node_id" in payload
    assert "devices" in payload
    assert "node_lat" in payload


def test_ble_gen_devices_have_required_fields():
    bus = FakeEventBus()
    gen = BLEScanGenerator(node_lat=37.0, node_lon=-122.0)
    gen._event_bus = bus
    gen._running = True
    gen._tick()
    devices = bus.events[0][1]["devices"]
    for d in devices:
        assert "mac" in d
        assert "rssi" in d
        assert "type" in d


def test_ble_gen_max_devices_respected():
    bus = FakeEventBus()
    gen = BLEScanGenerator(max_devices=5)
    gen._event_bus = bus
    gen._running = True
    gen._tick()
    devices = bus.events[0][1]["devices"]
    assert len(devices) <= 5


def test_ble_gen_building_position_assignment():
    gen = BLEScanGenerator()
    pos1 = gen._assign_building("AA:BB:CC:11:22:0A")
    pos2 = gen._assign_building("AA:BB:CC:11:22:0A")
    assert pos1 == pos2  # same MAC, same building


# --- MeshtasticNodeGenerator ---

def test_mesh_gen_not_running_initially():
    gen = MeshtasticNodeGenerator()
    assert gen.running is False


def test_mesh_gen_start_initializes_nodes():
    bus = FakeEventBus()
    gen = MeshtasticNodeGenerator(node_count=3)
    gen.start(bus)
    try:
        assert len(gen._nodes) == 3
    finally:
        gen.stop()


def test_mesh_gen_tick_publishes():
    bus = FakeEventBus()
    gen = MeshtasticNodeGenerator(node_count=3)
    gen.start(bus)
    try:
        gen.stop()  # stop the background thread
        bus.events.clear()  # clear any events from background tick
        gen._running = True
        gen._event_bus = bus
        gen._tick()
        assert len(bus.events) == 1
        assert bus.events[0][0] == "meshtastic:nodes_updated"
        payload = bus.events[0][1]
        assert payload["count"] == 3
    finally:
        gen._running = False


def test_mesh_gen_nodes_have_position():
    bus = FakeEventBus()
    gen = MeshtasticNodeGenerator(node_count=2)
    gen.start(bus)
    try:
        gen.stop()
        gen._running = True
        gen._event_bus = bus
        gen._tick()
        nodes = bus.events[0][1]["nodes"]
        for n in nodes:
            assert "position" in n
            assert "lat" in n["position"]
            assert "battery" in n
    finally:
        gen._running = False


# --- CameraDetectionGenerator ---

def test_cam_gen_not_running_initially():
    gen = CameraDetectionGenerator()
    assert gen.running is False


def test_cam_gen_tick_publishes():
    bus = FakeEventBus()
    gen = CameraDetectionGenerator(camera_id="test-cam")
    gen._event_bus = bus
    gen._running = True
    gen._tick()
    assert len(bus.events) == 1
    assert bus.events[0][0] == "detection:camera"
    payload = bus.events[0][1]
    assert payload["camera_id"] == "test-cam"


def test_cam_gen_spawns_objects():
    bus = FakeEventBus()
    gen = CameraDetectionGenerator()
    gen._event_bus = bus
    gen._running = True
    # Run many ticks to get some spawns
    for _ in range(20):
        gen._tick()
    # At least some detections should exist
    last_payload = bus.events[-1][1]
    # May or may not have detections depending on random, but no crash
    assert "detections" in last_payload


def test_cam_gen_detection_format():
    bus = FakeEventBus()
    gen = CameraDetectionGenerator()
    gen._event_bus = bus
    gen._running = True
    gen._spawn()
    gen._tick()
    detections = bus.events[0][1]["detections"]
    if detections:
        d = detections[0]
        assert "id" in d
        assert "label" in d
        assert "bbox" in d
        assert "x" in d["bbox"]


# --- TrilaterationDemoGenerator ---

def test_trilat_gen_not_running_initially():
    gen = TrilaterationDemoGenerator()
    assert gen.running is False


def test_trilat_gen_start_initializes_targets():
    bus = FakeEventBus()
    gen = TrilaterationDemoGenerator()
    gen.start(bus)
    try:
        assert len(gen._targets) == 3
    finally:
        gen.stop()


def test_trilat_gen_tick_publishes_per_node():
    bus = FakeEventBus()
    gen = TrilaterationDemoGenerator()
    gen.start(bus)
    try:
        gen.stop()
        bus.events.clear()
        gen._running = True
        gen._event_bus = bus
        gen._tick()
        # 3 nodes * fleet.ble_presence + 1 trilat:position_update = 4 events
        topics = [e[0] for e in bus.events]
        assert topics.count("fleet.ble_presence") == 3
        assert topics.count("trilat:position_update") == 1
    finally:
        gen._running = False


def test_trilat_gen_haversine():
    dist = approx_distance_m(37.0, -122.0, 37.001, -122.0)
    assert 100 < dist < 120  # ~111m per 0.001 degree latitude
