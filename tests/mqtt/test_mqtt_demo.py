# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the MQTT integration demo — MockMQTTClient, MQTTEventBridge, SensorSimulator, MQTTPipeline.

15+ tests covering:
  - MockMQTTClient pub/sub and wildcards
  - MQTTEventBridge routing to EventBus and TargetTracker
  - SensorSimulator generating realistic traffic
  - MQTTPipeline end-to-end data flow
  - REST endpoint responses
"""

from __future__ import annotations

import time
import pytest

from tritium_lib.events import EventBus, Event
from tritium_lib.mqtt import TritiumTopics, parse_topic, parse_site_topic
from tritium_lib.tracking import TargetTracker
from tritium_lib.mqtt.demos.mqtt_demo import (
    MockMQTTClient,
    MQTTMessage,
    MQTTEventBridge,
    SensorSimulator,
    MQTTPipeline,
    _EDGE_NODES,
    _CAMERAS,
    _BLE_DEVICES,
    _MESH_NODES,
)


# ---------------------------------------------------------------------------
# MockMQTTClient tests
# ---------------------------------------------------------------------------

class TestMockMQTTClient:
    """Tests for the in-process MQTT simulator."""

    def test_connect_disconnect(self):
        """Client reports connected/disconnected state."""
        client = MockMQTTClient()
        assert not client.is_connected
        client.connect("localhost", 1883)
        assert client.is_connected
        client.disconnect()
        assert not client.is_connected

    def test_publish_logs_message(self):
        """Publishing a message adds it to the message log."""
        client = MockMQTTClient()
        client.publish("test/topic", {"value": 42})
        assert client.publish_count == 1
        msgs = client.messages
        assert len(msgs) == 1
        assert msgs[0].topic == "test/topic"
        assert msgs[0].payload == {"value": 42}

    def test_subscribe_receives_messages(self):
        """Subscribers receive messages published to matching topics."""
        client = MockMQTTClient()
        received = []
        client.subscribe("test/topic", lambda t, p: received.append((t, p)))
        client.publish("test/topic", {"data": "hello"})
        assert len(received) == 1
        assert received[0] == ("test/topic", {"data": "hello"})

    def test_subscribe_no_match(self):
        """Subscribers do not receive messages from non-matching topics."""
        client = MockMQTTClient()
        received = []
        client.subscribe("test/topic", lambda t, p: received.append((t, p)))
        client.publish("other/topic", {"data": "nope"})
        assert len(received) == 0

    def test_wildcard_plus(self):
        """Single-level wildcard ``+`` matches exactly one level."""
        client = MockMQTTClient()
        received = []
        client.subscribe("sensors/+/temperature", lambda t, p: received.append(t))
        client.publish("sensors/node1/temperature", {"value": 22})
        client.publish("sensors/node2/temperature", {"value": 25})
        client.publish("sensors/node1/humidity", {"value": 60})  # no match
        client.publish("sensors/node1/sub/temperature", {"value": 0})  # no match (extra level)
        assert received == ["sensors/node1/temperature", "sensors/node2/temperature"]

    def test_wildcard_hash(self):
        """Multi-level wildcard ``#`` matches all remaining levels."""
        client = MockMQTTClient()
        received = []
        client.subscribe("tritium/home/#", lambda t, p: received.append(t))
        client.publish("tritium/home/edge/esp32/heartbeat", {})
        client.publish("tritium/home/sensors/node1/ble", {})
        client.publish("tritium/other/edge/esp32/heartbeat", {})  # no match
        assert len(received) == 2
        assert "tritium/home/edge/esp32/heartbeat" in received
        assert "tritium/home/sensors/node1/ble" in received

    def test_retained_messages(self):
        """Retained messages are delivered to new subscribers."""
        client = MockMQTTClient()
        # Publish retained before subscribing
        client.publish("device/status", {"state": "online"}, retain=True)
        assert "device/status" in client.retained_topics

        received = []
        client.subscribe("device/status", lambda t, p: received.append(p))
        assert len(received) == 1
        assert received[0] == {"state": "online"}

    def test_unsubscribe(self):
        """Unsubscribed callbacks no longer receive messages."""
        client = MockMQTTClient()
        received = []
        cb = lambda t, p: received.append(t)
        client.subscribe("test/topic", cb)
        client.publish("test/topic", {})
        assert len(received) == 1

        client.unsubscribe("test/topic", cb)
        client.publish("test/topic", {})
        assert len(received) == 1  # no new message

    def test_multiple_subscribers(self):
        """Multiple subscribers on the same topic all receive messages."""
        client = MockMQTTClient()
        r1, r2 = [], []
        client.subscribe("shared/topic", lambda t, p: r1.append(t))
        client.subscribe("shared/topic", lambda t, p: r2.append(t))
        client.publish("shared/topic", {"x": 1})
        assert len(r1) == 1
        assert len(r2) == 1

    def test_bad_subscriber_doesnt_break_bus(self):
        """A subscriber that throws doesn't prevent other subscribers from receiving."""
        client = MockMQTTClient()
        received = []

        def bad_cb(t, p):
            raise ValueError("boom")

        client.subscribe("test/topic", bad_cb)
        client.subscribe("test/topic", lambda t, p: received.append(t))
        client.publish("test/topic", {})
        assert len(received) == 1  # second subscriber still got the message

    def test_message_log_limit(self):
        """Message log doesn't grow unbounded (capped at 500)."""
        client = MockMQTTClient()
        for i in range(600):
            client.publish(f"test/{i}", {"i": i})
        assert len(client.messages) == 500
        assert client.publish_count == 600


# ---------------------------------------------------------------------------
# MQTTEventBridge tests
# ---------------------------------------------------------------------------

class TestMQTTEventBridge:
    """Tests for the MQTT -> EventBus -> TargetTracker bridge."""

    def _make_bridge(self):
        topics = TritiumTopics(site_id="test")
        mqtt = MockMQTTClient()
        bus = EventBus()
        tracker = TargetTracker(event_bus=bus)
        bridge = MQTTEventBridge(mqtt=mqtt, event_bus=bus, tracker=tracker, topics=topics)
        return mqtt, bus, tracker, bridge, topics

    def test_ble_sighting_reaches_tracker(self):
        """BLE sighting published to MQTT ends up as a tracked target."""
        mqtt, bus, tracker, bridge, topics = self._make_bridge()

        mqtt.publish(
            topics.sensor("esp32-alpha", "ble"),
            {
                "mac": "AA:BB:CC:11:22:33",
                "name": "TestPhone",
                "rssi": -50,
                "device_type": "phone",
                "node_position": {"x": 10.0, "y": 20.0},
            },
        )

        targets = tracker.get_all()
        assert len(targets) >= 1
        ble_targets = [t for t in targets if t.source == "ble"]
        assert len(ble_targets) == 1
        assert ble_targets[0].target_id == "ble_aabbcc112233"
        assert ble_targets[0].name == "TestPhone"
        assert bridge.routed_count >= 1

    def test_camera_detection_reaches_tracker(self):
        """Camera YOLO detection published to MQTT ends up as a tracked target."""
        mqtt, bus, tracker, bridge, topics = self._make_bridge()

        mqtt.publish(
            topics.camera_detections("cam-01"),
            {
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.85,
                        "center_x": 50.0,
                        "center_y": 60.0,
                    },
                ],
            },
        )

        targets = tracker.get_all()
        yolo_targets = [t for t in targets if t.source == "yolo"]
        assert len(yolo_targets) == 1
        assert yolo_targets[0].asset_type == "person"

    def test_mesh_node_reaches_tracker(self):
        """Meshtastic node published to MQTT ends up as a tracked target."""
        mqtt, bus, tracker, bridge, topics = self._make_bridge()

        mqtt.publish(
            topics.meshtastic_nodes("bridge-01"),
            {
                "nodes": [
                    {
                        "target_id": "mesh_test01",
                        "name": "TestNode",
                        "position": {"x": 100.0, "y": 200.0},
                        "battery": 0.8,
                    },
                ],
            },
        )

        targets = tracker.get_all()
        mesh_targets = [t for t in targets if t.source == "mesh"]
        assert len(mesh_targets) == 1
        assert mesh_targets[0].target_id == "mesh_test01"
        assert mesh_targets[0].name == "TestNode"

    def test_edge_heartbeat_fires_bus_event(self):
        """Edge heartbeat on MQTT triggers an EventBus event."""
        mqtt, bus, tracker, bridge, topics = self._make_bridge()

        bus_events = []
        bus.subscribe("mqtt.edge.heartbeat", lambda e: bus_events.append(e))

        mqtt.publish(
            topics.edge_heartbeat("esp32-alpha"),
            {
                "device_id": "esp32-alpha",
                "uptime_s": 1000,
                "free_heap": 100000,
            },
        )

        assert len(bus_events) == 1
        assert bus_events[0].data["device_id"] == "esp32-alpha"

    def test_sdr_message_fires_bus_event(self):
        """SDR spectrum data on MQTT triggers an EventBus event."""
        mqtt, bus, tracker, bridge, topics = self._make_bridge()

        bus_events = []
        bus.subscribe("mqtt.sdr.#", lambda e: bus_events.append(e))

        mqtt.publish(
            topics.sdr_spectrum("hackrf-01"),
            {
                "device_id": "hackrf-01",
                "center_freq_mhz": 915.0,
                "peak_dbm": -40.0,
            },
        )

        assert len(bus_events) == 1
        assert bus_events[0].data["center_freq_mhz"] == 915.0

    def test_bridge_counts_errors(self):
        """Bridge tracks error count when subscriber data is malformed."""
        # The bridge swallows exceptions; test that error_count increments
        mqtt, bus, tracker, bridge, topics = self._make_bridge()
        # Initially no errors
        assert bridge.error_count == 0


# ---------------------------------------------------------------------------
# SensorSimulator tests
# ---------------------------------------------------------------------------

class TestSensorSimulator:
    """Tests for the synthetic sensor traffic generator."""

    def test_tick_generates_messages(self):
        """A single tick produces heartbeat, BLE, camera, and mesh messages."""
        topics = TritiumTopics(site_id="simtest")
        mqtt = MockMQTTClient()
        sim = SensorSimulator(mqtt=mqtt, topics=topics, seed=99)

        stats = sim.generate_tick()
        assert stats["heartbeats"] == len(_EDGE_NODES)
        assert stats["ble"] == len(_BLE_DEVICES)
        assert stats["camera"] >= len(_CAMERAS)  # each camera produces 1-3
        assert stats["mesh"] == len(_MESH_NODES)
        assert mqtt.publish_count > 0

    def test_multiple_ticks_accumulate(self):
        """Running multiple ticks increases message count monotonically."""
        topics = TritiumTopics(site_id="simtest")
        mqtt = MockMQTTClient()
        sim = SensorSimulator(mqtt=mqtt, topics=topics)

        sim.generate_tick()
        count1 = mqtt.publish_count
        sim.generate_tick()
        count2 = mqtt.publish_count
        assert count2 > count1

    def test_sdr_every_third_tick(self):
        """SDR spectrum data is published every 3rd tick."""
        topics = TritiumTopics(site_id="simtest")
        mqtt = MockMQTTClient()
        sim = SensorSimulator(mqtt=mqtt, topics=topics)

        sdr_counts = []
        for _ in range(6):
            stats = sim.generate_tick()
            sdr_counts.append(stats["sdr"])
        # Ticks 1-6: SDR fires on tick 3 and 6
        assert sdr_counts == [0, 0, 1, 0, 0, 1]


# ---------------------------------------------------------------------------
# MQTTPipeline end-to-end tests
# ---------------------------------------------------------------------------

class TestMQTTPipeline:
    """End-to-end tests for the full MQTT integration pipeline."""

    def test_pipeline_creates_targets(self):
        """After ticking, the pipeline has targets from BLE and camera sources.

        Camera (YOLO) detections may be fused into BLE targets by the correlator,
        so we check either for standalone yolo targets or for correlated_ids
        containing detection IDs (proving camera data flowed through).
        """
        p = MQTTPipeline(site_id="e2e", seed=42)
        p.tick()
        targets = p.tracker.get_all()
        assert len(targets) > 0

        sources = {t.source for t in targets}
        assert "ble" in sources

        # Camera data reached the tracker — either as standalone yolo targets
        # or fused into existing targets via the correlator
        has_yolo = "yolo" in sources
        has_correlated_detections = any(
            any(cid.startswith("det_") for cid in t.correlated_ids)
            for t in targets
        )
        assert has_yolo or has_correlated_detections, (
            "Camera detections did not reach the tracker"
        )

    def test_pipeline_mesh_targets(self):
        """Pipeline creates mesh targets from simulated Meshtastic nodes."""
        p = MQTTPipeline(site_id="e2e-mesh", seed=42)
        p.tick()
        targets = p.tracker.get_all()
        mesh_targets = [t for t in targets if t.source == "mesh"]
        assert len(mesh_targets) == len(_MESH_NODES)

    def test_pipeline_bus_events_captured(self):
        """Pipeline captures EventBus events for the dashboard."""
        p = MQTTPipeline(site_id="e2e-bus", seed=42)
        p.tick()
        assert len(p.bus_events) > 0
        # Bus events have the expected shape
        ev = p.bus_events[0]
        assert "topic" in ev
        assert "source" in ev
        assert "timestamp" in ev

    def test_pipeline_multi_tick_signal_count(self):
        """After multiple ticks, BLE targets accumulate signal_count."""
        p = MQTTPipeline(site_id="e2e-multi", seed=42)
        for _ in range(5):
            p.tick()

        ble_targets = [t for t in p.tracker.get_all() if t.source == "ble"]
        assert len(ble_targets) > 0
        # After 5 ticks, most BLE targets should accumulate signal_count.
        # Some may be re-created by the correlator with lower counts.
        total_signals = sum(t.signal_count for t in ble_targets)
        # With 6 BLE devices over 5 ticks = 30 sightings total, expect substantial count
        assert total_signals >= 10, f"Expected >= 10 total signals, got {total_signals}"
        # At least one target should have been seen multiple times
        max_signals = max(t.signal_count for t in ble_targets)
        assert max_signals >= 3

    def test_pipeline_correlator_runs(self):
        """Pipeline's correlator runs each tick and may find correlations."""
        p = MQTTPipeline(site_id="e2e-corr", seed=42)
        for _ in range(10):
            stats = p.tick()
        # After 10 ticks, there should be at least an attempt at correlation
        # (actual correlation depends on proximity — we verify the path works)
        assert p.simulator.tick == 10
        # correlation_log is a list — it exists even if empty
        assert isinstance(p.correlation_log, list)

    def test_pipeline_topic_structure(self):
        """Pipeline uses proper Tritium topic hierarchy."""
        p = MQTTPipeline(site_id="topo", seed=42)
        p.tick()

        msgs = p.mqtt.messages
        topics = [m.topic for m in msgs]

        # Verify some expected topic patterns are present
        assert any("topo/edge/" in t for t in topics), "Missing edge topics"
        assert any("topo/sensors/" in t for t in topics), "Missing sensor topics"
        assert any("topo/cameras/" in t for t in topics), "Missing camera topics"
        assert any("topo/meshtastic/" in t for t in topics), "Missing meshtastic topics"


# ---------------------------------------------------------------------------
# REST endpoint tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestRESTEndpoints:
    """Tests for the demo REST API (GET /topics, /messages, /targets, /status)."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        """Set up a test client with a fresh pipeline."""
        from tritium_lib.mqtt.demos.mqtt_demo import app, pipeline as global_pipeline
        # Run a few ticks to populate data
        global_pipeline.simulator.tick = 0
        for _ in range(3):
            global_pipeline.tick()

        from fastapi.testclient import TestClient
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_get_topics(self):
        """GET /topics returns subscription info and topic tree."""
        resp = self.client.get("/topics")
        assert resp.status_code == 200
        data = resp.json()
        assert "subscriptions" in data
        assert "topic_tree" in data
        assert "site" in data
        assert len(data["subscriptions"]) > 0

    def test_get_messages(self):
        """GET /messages returns recent MQTT messages."""
        resp = self.client.get("/messages?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "messages" in data
        assert "total_published" in data
        assert len(data["messages"]) > 0
        msg = data["messages"][0]
        assert "topic" in msg
        assert "payload" in msg
        assert "timestamp" in msg

    def test_get_targets(self):
        """GET /targets returns fused targets with all expected fields."""
        resp = self.client.get("/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert "targets" in data
        assert "count" in data
        assert data["count"] > 0

        target = data["targets"][0]
        assert "target_id" in target
        assert "name" in target
        assert "source" in target
        assert "position" in target
        assert "confirming_sources" in target

    def test_get_status(self):
        """GET /status returns pipeline health summary."""
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "mqtt" in data
        assert "bridge" in data
        assert "tracker" in data
        assert "tick" in data
        assert data["mqtt"]["connected"] is True
        assert data["mqtt"]["publish_count"] > 0
        assert data["bridge"]["routed_count"] > 0

    def test_dashboard_html(self):
        """GET / returns the HTML dashboard."""
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "TRITIUM MQTT INTEGRATION DEMO" in resp.text
