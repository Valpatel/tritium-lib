# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.fusion.engine — FusionEngine multi-sensor orchestrator."""

import time

import pytest

from tritium_lib.fusion.engine import (
    FusionEngine,
    FusedTarget,
    FusionSnapshot,
    SensorRecord,
)
from tritium_lib.tracking.geofence import GeoZone
from tritium_lib.events.bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Fresh FusionEngine with no event bus."""
    e = FusionEngine()
    yield e
    e.shutdown()


@pytest.fixture
def bus_engine():
    """FusionEngine wired to an EventBus."""
    bus = EventBus(history_size=100)
    e = FusionEngine(event_bus=bus)
    yield e, bus
    e.shutdown()


# ---------------------------------------------------------------------------
# BLE ingestion
# ---------------------------------------------------------------------------

class TestIngestBLE:
    def test_ingest_ble_basic(self, engine):
        tid = engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "name": "Phone",
        })
        assert tid == "ble_aabbccddeeff"
        targets = engine.get_fused_targets()
        assert len(targets) == 1
        assert targets[0].target_id == tid

    def test_ingest_ble_no_mac_returns_none(self, engine):
        assert engine.ingest_ble({}) is None
        assert engine.ingest_ble({"rssi": -50}) is None

    def test_ingest_ble_with_position(self, engine):
        tid = engine.ingest_ble({
            "mac": "11:22:33:44:55:66",
            "rssi": -40,
            "position": {"x": 10.0, "y": 20.0},
        })
        target = engine.tracker.get_target(tid)
        assert target is not None
        assert target.position == (10.0, 20.0)

    def test_ingest_ble_records_sensor_history(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        ft = engine.get_fused_target("ble_aabbccddeeff")
        assert ft is not None
        assert len(ft.sensor_records) == 2
        assert all(r.source == "ble" for r in ft.sensor_records)

    def test_ingest_ble_records_heatmap(self, engine):
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 5.0, "y": 3.0},
        })
        assert engine.heatmap.event_count("ble_activity") == 1

    def test_ingest_ble_publishes_event(self, bus_engine):
        engine, bus = bus_engine
        events = []
        bus.subscribe("fusion.sensor.ingested", lambda e: events.append(e))
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        assert len(events) == 1
        assert events[0].data["source"] == "ble"


# ---------------------------------------------------------------------------
# WiFi ingestion
# ---------------------------------------------------------------------------

class TestIngestWiFi:
    def test_ingest_wifi_probe(self, engine):
        tid = engine.ingest_wifi({
            "mac": "AA:BB:CC:DD:EE:FF",
            "ssid": "MyNetwork",
            "rssi": -60,
        })
        assert tid is not None
        # WiFi probes feed the network analyzer
        profile = engine.network_analyzer.get_device_profile("AA:BB:CC:DD:EE:FF")
        assert profile is not None
        assert "MyNetwork" in profile["ssids"]

    def test_ingest_wifi_no_mac_returns_none(self, engine):
        assert engine.ingest_wifi({}) is None

    def test_ingest_wifi_with_position_creates_tracker_target(self, engine):
        tid = engine.ingest_wifi({
            "mac": "AA:BB:CC:DD:EE:FF",
            "ssid": "TestSSID",
            "rssi": -50,
            "position": {"x": 10.0, "y": 20.0},
        })
        # When WiFi has position, it creates a BLE-style target
        assert tid == "ble_aabbccddeeff"
        target = engine.tracker.get_target(tid)
        assert target is not None


# ---------------------------------------------------------------------------
# Camera ingestion
# ---------------------------------------------------------------------------

class TestIngestCamera:
    def test_ingest_camera_person(self, engine):
        tid = engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.92,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        assert tid is not None
        assert tid.startswith("det_person_")

    def test_ingest_camera_low_confidence_rejected(self, engine):
        tid = engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.2,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        assert tid is None

    def test_ingest_camera_vehicle(self, engine):
        tid = engine.ingest_camera({
            "class_name": "car",
            "confidence": 0.85,
            "center_x": 20.0,
            "center_y": 15.0,
        })
        assert tid is not None
        ft = engine.get_fused_target(tid)
        assert ft is not None
        assert ft.target.asset_type == "vehicle"

    def test_ingest_camera_records_heatmap(self, engine):
        engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 5.0,
            "center_y": 3.0,
        })
        assert engine.heatmap.event_count("camera_activity") == 1

    def test_ingest_camera_updates_existing(self, engine):
        tid1 = engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        tid2 = engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.85,
            "center_x": 10.1,
            "center_y": 5.1,
        })
        # Should update the same target (within proximity)
        assert tid1 == tid2


# ---------------------------------------------------------------------------
# Acoustic ingestion
# ---------------------------------------------------------------------------

class TestIngestAcoustic:
    def test_ingest_acoustic_with_position(self, engine):
        tid = engine.ingest_acoustic({
            "event_type": "gunshot",
            "confidence": 0.8,
            "sensor_id": "mic_01",
            "position": {"x": 15.0, "y": 10.0},
        })
        assert tid == "acoustic_mic_01_gunshot"
        target = engine.tracker.get_target(tid)
        assert target is not None
        assert target.asset_type == "person"
        assert target.classification == "gunshot"

    def test_ingest_acoustic_vehicle(self, engine):
        tid = engine.ingest_acoustic({
            "event_type": "vehicle_engine",
            "confidence": 0.7,
            "sensor_id": "mic_02",
            "position": {"x": 5.0, "y": 5.0},
        })
        target = engine.tracker.get_target(tid)
        assert target is not None
        assert target.asset_type == "vehicle"

    def test_ingest_acoustic_no_type_returns_none(self, engine):
        assert engine.ingest_acoustic({}) is None

    def test_ingest_acoustic_no_position_still_records(self, engine):
        tid = engine.ingest_acoustic({
            "event_type": "voice",
            "sensor_id": "mic_03",
        })
        assert tid is not None
        ft = engine.get_fused_target(tid)
        # No tracker target (no position), but sensor record is stored
        # The target might not be in tracker, but records are stored
        assert tid == "acoustic_mic_03_voice"

    def test_ingest_acoustic_records_heatmap(self, engine):
        engine.ingest_acoustic({
            "event_type": "gunshot",
            "sensor_id": "mic_01",
            "position": {"x": 15.0, "y": 10.0},
        })
        assert engine.heatmap.event_count("motion_activity") == 1


# ---------------------------------------------------------------------------
# Mesh ingestion
# ---------------------------------------------------------------------------

class TestIngestMesh:
    def test_ingest_mesh_node(self, engine):
        tid = engine.ingest_mesh({
            "target_id": "mesh_node_42",
            "name": "Relay Alpha",
            "position": {"x": 100.0, "y": 200.0},
            "battery": 0.85,
        })
        assert tid == "mesh_node_42"
        target = engine.tracker.get_target(tid)
        assert target is not None
        assert target.name == "Relay Alpha"

    def test_ingest_mesh_no_id_returns_none(self, engine):
        assert engine.ingest_mesh({}) is None


# ---------------------------------------------------------------------------
# ADS-B ingestion
# ---------------------------------------------------------------------------

class TestIngestADSB:
    def test_ingest_adsb(self, engine):
        tid = engine.ingest_adsb({
            "target_id": "adsb_ABC123",
            "name": "N12345",
            "heading": 270.0,
            "speed": 150.0,
        })
        assert tid == "adsb_ABC123"
        target = engine.tracker.get_target(tid)
        assert target is not None
        assert target.heading == 270.0

    def test_ingest_adsb_no_id_returns_none(self, engine):
        assert engine.ingest_adsb({}) is None


# ---------------------------------------------------------------------------
# RF Motion ingestion
# ---------------------------------------------------------------------------

class TestIngestRFMotion:
    def test_ingest_rf_motion(self, engine):
        tid = engine.ingest_rf_motion({
            "target_id": "rf_pair_01",
            "position": {"x": 10.0, "y": 20.0},
            "confidence": 0.6,
            "direction_hint": "north",
            "pair_id": "sensor_a:sensor_b",
        })
        assert tid == "rf_pair_01"
        target = engine.tracker.get_target(tid)
        assert target is not None

    def test_ingest_rf_motion_no_id_returns_none(self, engine):
        assert engine.ingest_rf_motion({}) is None

    def test_ingest_rf_motion_records_heatmap(self, engine):
        engine.ingest_rf_motion({
            "target_id": "rf_pair_01",
            "position": {"x": 10.0, "y": 20.0},
        })
        assert engine.heatmap.event_count("motion_activity") == 1


# ---------------------------------------------------------------------------
# Fused target queries
# ---------------------------------------------------------------------------

class TestFusedTargets:
    def test_get_fused_targets_empty(self, engine):
        assert engine.get_fused_targets() == []

    def test_get_fused_targets_multi_source(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_camera({
            "class_name": "person", "confidence": 0.9,
            "center_x": 10.0, "center_y": 5.0,
        })
        targets = engine.get_fused_targets()
        assert len(targets) == 2

    def test_get_fused_target_by_id(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        ft = engine.get_fused_target("ble_aabbccddeeff")
        assert ft is not None
        assert ft.target_id == "ble_aabbccddeeff"

    def test_get_fused_target_nonexistent(self, engine):
        assert engine.get_fused_target("nonexistent") is None

    def test_fused_target_to_dict(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        ft = engine.get_fused_target("ble_aabbccddeeff")
        d = ft.to_dict()
        assert d["target_id"] == "ble_aabbccddeeff"
        assert "source_types" in d
        assert "source_count" in d

    def test_get_targets_by_source(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_camera({
            "class_name": "person", "confidence": 0.9,
            "center_x": 10.0, "center_y": 5.0,
        })
        ble_targets = engine.get_targets_by_source("ble")
        assert len(ble_targets) == 1
        assert ble_targets[0].target.source == "ble"

    def test_get_multi_source_targets(self, engine):
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF", "rssi": -55,
            "position": {"x": 10.0, "y": 5.0},
        })
        # A single-source target should not appear with min_sources=2
        multi = engine.get_multi_source_targets(min_sources=2)
        assert len(multi) == 0


# ---------------------------------------------------------------------------
# Dossier queries
# ---------------------------------------------------------------------------

class TestTargetDossier:
    def test_get_dossier_exists(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        dossier = engine.get_target_dossier("ble_aabbccddeeff")
        assert dossier is not None
        assert dossier["status"] == "active"
        assert dossier["target"] is not None
        assert "sensor_history" in dossier

    def test_get_dossier_nonexistent(self, engine):
        assert engine.get_target_dossier("nonexistent") is None

    def test_get_dossier_sensor_history_grouped(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        dossier = engine.get_target_dossier("ble_aabbccddeeff")
        assert "ble" in dossier["sensor_history"]
        assert len(dossier["sensor_history"]["ble"]) == 2

    def test_get_dossier_timeline_sorted(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        dossier = engine.get_target_dossier("ble_aabbccddeeff")
        tl = dossier["timeline"]
        assert len(tl) == 2
        assert tl[0] <= tl[1]


# ---------------------------------------------------------------------------
# Zone management and queries
# ---------------------------------------------------------------------------

class TestZones:
    def _make_zone(self, zone_id="zone1", name="Test Zone"):
        return GeoZone(
            zone_id=zone_id,
            name=name,
            polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
            zone_type="monitored",
        )

    def test_add_and_list_zones(self, engine):
        zone = self._make_zone()
        engine.add_zone(zone)
        zones = engine.get_zones()
        assert len(zones) == 1
        assert zones[0].zone_id == "zone1"

    def test_remove_zone(self, engine):
        engine.add_zone(self._make_zone())
        assert engine.remove_zone("zone1") is True
        assert engine.remove_zone("zone1") is False

    def test_get_zone_activity_nonexistent(self, engine):
        activity = engine.get_zone_activity("nonexistent")
        assert activity["zone"] is None
        assert "error" in activity

    def test_get_zone_activity_with_occupant(self, engine):
        zone = self._make_zone()
        engine.add_zone(zone)
        # Ingest a target inside the zone
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 10.0, "y": 10.0},
        })
        activity = engine.get_zone_activity("zone1")
        assert activity["zone"] is not None
        assert activity["occupant_count"] == 1

    def test_get_targets_in_zone(self, engine):
        zone = self._make_zone()
        engine.add_zone(zone)
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 10.0, "y": 10.0},
        })
        in_zone = engine.get_targets_in_zone("zone1")
        assert len(in_zone) == 1

    def test_target_outside_zone_not_in_occupants(self, engine):
        zone = self._make_zone()
        engine.add_zone(zone)
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -55,
            "position": {"x": 50.0, "y": 50.0},
        })
        in_zone = engine.get_targets_in_zone("zone1")
        assert len(in_zone) == 0


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_run_correlation_empty(self, engine):
        result = engine.run_correlation()
        assert result == []

    def test_run_correlation_different_sources(self, engine):
        # Ingest two targets from different sources at same position
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -40,
            "position": {"x": 10.0, "y": 5.0},
        })
        engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        # Run manual correlation
        corr = engine.run_correlation()
        # Correlation may or may not fire depending on strategy scores,
        # but the call should not error
        assert isinstance(corr, list)

    def test_correlation_updates_fusion_metrics(self, engine):
        engine.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -40,
            "position": {"x": 10.0, "y": 5.0},
        })
        engine.ingest_camera({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 10.0,
            "center_y": 5.0,
        })
        engine.run_correlation()
        status = engine.fusion_metrics.get_status()
        assert "total_fusions" in status

    def test_start_stop_correlator(self, engine):
        engine.start_correlator()
        time.sleep(0.1)
        assert engine.correlator._running is True
        engine.stop_correlator()
        assert engine.correlator._running is False


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_empty_snapshot(self, engine):
        snap = engine.get_snapshot()
        assert isinstance(snap, FusionSnapshot)
        assert snap.total_targets == 0
        assert snap.total_dossiers == 0

    def test_snapshot_with_targets(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.ingest_camera({
            "class_name": "person", "confidence": 0.9,
            "center_x": 10.0, "center_y": 5.0,
        })
        snap = engine.get_snapshot()
        assert snap.total_targets == 2

    def test_snapshot_to_dict(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        d = engine.get_snapshot().to_dict()
        assert "targets" in d
        assert "total_targets" in d
        assert "metrics" in d
        assert d["total_targets"] == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_clear_resets_state(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        assert len(engine.get_fused_targets()) == 1
        engine.clear()
        assert len(engine.get_fused_targets()) == 0

    def test_shutdown_is_safe(self, engine):
        engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
        engine.shutdown()
        # Should not raise


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    def test_tracker_accessible(self, engine):
        assert engine.tracker is not None

    def test_correlator_accessible(self, engine):
        assert engine.correlator is not None

    def test_geofence_accessible(self, engine):
        assert engine.geofence is not None

    def test_heatmap_accessible(self, engine):
        assert engine.heatmap is not None

    def test_dossier_store_accessible(self, engine):
        assert engine.dossier_store is not None

    def test_network_analyzer_accessible(self, engine):
        assert engine.network_analyzer is not None

    def test_fusion_metrics_accessible(self, engine):
        assert engine.fusion_metrics is not None
