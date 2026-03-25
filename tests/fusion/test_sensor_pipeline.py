# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.fusion.pipeline — SensorPipeline event-bus bridge."""

import time

import pytest

from tritium_lib.events.bus import EventBus
from tritium_lib.fusion.engine import FusionEngine
from tritium_lib.fusion.pipeline import SensorPipeline, _TOPIC_MAP
from tritium_lib.tracking.geofence import GeoZone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup():
    """Pipeline wired to a FusionEngine and EventBus."""
    bus = EventBus(history_size=100)
    engine = FusionEngine(event_bus=bus)
    pipeline = SensorPipeline(
        engine, bus,
        correlation_interval=0,  # disable background loop for unit tests
        snapshot_interval=0,
    )
    yield engine, bus, pipeline
    pipeline.stop()
    engine.shutdown()


@pytest.fixture
def running_setup():
    """Pipeline with background loop running."""
    bus = EventBus(history_size=100)
    engine = FusionEngine(event_bus=bus)
    pipeline = SensorPipeline(
        engine, bus,
        correlation_interval=0.5,
        snapshot_interval=1.0,
    )
    pipeline.start()
    yield engine, bus, pipeline
    pipeline.stop()
    engine.shutdown()


# ---------------------------------------------------------------------------
# Subscription and routing
# ---------------------------------------------------------------------------

class TestSubscription:
    def test_start_subscribes_to_all_topics(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        # Verify all sensor topics are in the map
        assert len(_TOPIC_MAP) == 7
        for topic in _TOPIC_MAP:
            assert topic.startswith("sensor.")

    def test_stop_is_idempotent(self, setup):
        _, _, pipeline = setup
        pipeline.start()
        pipeline.stop()
        pipeline.stop()  # Should not error

    def test_start_is_idempotent(self, setup):
        _, _, pipeline = setup
        pipeline.start()
        pipeline.start()  # Should not error


# ---------------------------------------------------------------------------
# Event routing — BLE
# ---------------------------------------------------------------------------

class TestBLERouting:
    def test_ble_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
        })
        targets = engine.get_fused_targets()
        assert len(targets) == 1
        assert targets[0].target_id == "ble_aabbccddeeff"

    def test_ble_event_increments_ingest_count(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
        })
        status = pipeline.get_status()
        assert status["ingested_count"] == 1


# ---------------------------------------------------------------------------
# Event routing — WiFi
# ---------------------------------------------------------------------------

class TestWiFiRouting:
    def test_wifi_event_feeds_network_analyzer(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.wifi.probe", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "ssid": "TestNetwork",
            "rssi": -60,
        })
        profile = engine.network_analyzer.get_device_profile("AA:BB:CC:DD:EE:FF")
        assert profile is not None
        assert "TestNetwork" in profile["ssids"]


# ---------------------------------------------------------------------------
# Event routing — Camera
# ---------------------------------------------------------------------------

class TestCameraRouting:
    def test_camera_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.camera.detection", {
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        targets = engine.get_fused_targets()
        assert len(targets) == 1

    def test_camera_low_confidence_rejected(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.camera.detection", {
            "class_name": "person",
            "confidence": 0.2,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        targets = engine.get_fused_targets()
        assert len(targets) == 0


# ---------------------------------------------------------------------------
# Event routing — Acoustic
# ---------------------------------------------------------------------------

class TestAcousticRouting:
    def test_acoustic_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.acoustic.event", {
            "event_type": "gunshot",
            "confidence": 0.8,
            "sensor_id": "mic_01",
            "position": {"x": 15.0, "y": 10.0},
        })
        target = engine.tracker.get_target("acoustic_mic_01_gunshot")
        assert target is not None


# ---------------------------------------------------------------------------
# Event routing — Mesh
# ---------------------------------------------------------------------------

class TestMeshRouting:
    def test_mesh_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.mesh.node", {
            "target_id": "mesh_node_42",
            "name": "Relay Alpha",
            "position": {"x": 100.0, "y": 200.0},
        })
        target = engine.tracker.get_target("mesh_node_42")
        assert target is not None


# ---------------------------------------------------------------------------
# Event routing — ADS-B
# ---------------------------------------------------------------------------

class TestADSBRouting:
    def test_adsb_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.adsb.detection", {
            "target_id": "adsb_ABC123",
            "name": "N12345",
        })
        target = engine.tracker.get_target("adsb_ABC123")
        assert target is not None


# ---------------------------------------------------------------------------
# Event routing — RF Motion
# ---------------------------------------------------------------------------

class TestRFMotionRouting:
    def test_rf_motion_event_creates_target(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.rf_motion.event", {
            "target_id": "rf_pair_01",
            "position": {"x": 10.0, "y": 20.0},
            "confidence": 0.6,
        })
        target = engine.tracker.get_target("rf_pair_01")
        assert target is not None


# ---------------------------------------------------------------------------
# Output events
# ---------------------------------------------------------------------------

class TestOutputEvents:
    def test_fusion_target_updated_published(self, setup):
        engine, bus, pipeline = setup
        updated_events = []
        bus.subscribe("fusion.target.updated", lambda e: updated_events.append(e))
        pipeline.start()
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
        })
        assert len(updated_events) == 1
        assert updated_events[0].data["target_id"] == "ble_aabbccddeeff"

    def test_geofence_events_republished(self, setup):
        engine, bus, pipeline = setup
        entered_events = []
        bus.subscribe("fusion.zone.entered", lambda e: entered_events.append(e))
        pipeline.start()

        # Add a zone
        zone = GeoZone(
            zone_id="zone1",
            name="Test Zone",
            polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
        )
        engine.add_zone(zone)

        # Ingest a target inside the zone
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 10.0, "y": 10.0},
        })
        assert len(entered_events) >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_non_dict_data_ignored(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        bus.publish("sensor.ble.sighting", "not a dict")
        status = pipeline.get_status()
        assert status["ingested_count"] == 0

    def test_invalid_data_increments_error_or_ignores(self, setup):
        engine, bus, pipeline = setup
        pipeline.start()
        # Empty dict for BLE should return None (no mac)
        bus.publish("sensor.ble.sighting", {})
        # Still counted as ingested (the method was called, returned None)
        status = pipeline.get_status()
        assert status["ingested_count"] == 1


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_fields(self, setup):
        _, _, pipeline = setup
        pipeline.start()
        status = pipeline.get_status()
        assert "running" in status
        assert "subscribed" in status
        assert "ingested_count" in status
        assert "error_count" in status
        assert "topics" in status
        assert len(status["topics"]) == 7

    def test_is_running_property(self, setup):
        _, _, pipeline = setup
        assert pipeline.is_running is False
        # With interval=0, background loop does not start
        pipeline.start()
        assert pipeline.is_running is False  # intervals are 0


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

class TestBackgroundLoop:
    def test_background_loop_runs_correlation(self, running_setup):
        engine, bus, pipeline = running_setup
        # Ingest some data
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 10.0, "y": 5.0},
        })
        # Wait for at least one loop cycle
        time.sleep(1.2)
        assert pipeline.is_running is True

    def test_background_loop_publishes_snapshot(self, running_setup):
        engine, bus, pipeline = running_setup
        snapshots = []
        bus.subscribe("fusion.snapshot", lambda e: snapshots.append(e))
        bus.publish("sensor.ble.sighting", {
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
        })
        time.sleep(2.0)
        assert len(snapshots) >= 1
        assert "total_targets" in snapshots[0].data
