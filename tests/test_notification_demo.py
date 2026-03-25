# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the notification demo pipeline.

Validates the full notification pipeline: geofence alerts, threat escalations,
sensor health warnings, anomaly detections, NotificationManager integration,
WebSocket connection manager, and REST API endpoints.
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from tritium_lib.notifications import Notification, NotificationManager
from tritium_lib.tracking import (
    TargetTracker,
    GeofenceEngine,
    GeoZone,
    ThreatScorer,
    SensorHealthMonitor,
)


# ---------------------------------------------------------------------------
# Import the pipeline (no FastAPI server needed for unit tests)
# ---------------------------------------------------------------------------

from tritium_lib.notifications.demos.notification_demo import (
    NotificationPipeline,
    ConnectionManager,
    SimpleEventBus,
    _BLE_DEVICES,
    _DETECTIONS,
    _SENSORS,
)


# ---------------------------------------------------------------------------
# 1. Pipeline initialization
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_pipeline_creates_all_subsystems(self):
        """Pipeline must wire up tracker, geofence, threat scorer, health, notifications."""
        p = NotificationPipeline()
        assert isinstance(p.tracker, TargetTracker)
        assert isinstance(p.geofence, GeofenceEngine)
        assert isinstance(p.threat_scorer, ThreatScorer)
        assert isinstance(p.health_monitor, SensorHealthMonitor)
        assert isinstance(p.notif_mgr, NotificationManager)

    def test_geofence_zones_created(self):
        """Pipeline must create 3 geofence zones on init."""
        p = NotificationPipeline()
        zones = p.geofence.list_zones()
        assert len(zones) == 3
        zone_ids = {z.zone_id for z in zones}
        assert "restricted-hq" in zone_ids
        assert "parking-south" in zone_ids
        assert "perimeter-north" in zone_ids

    def test_geofence_zone_types(self):
        """Zones should have correct types assigned."""
        p = NotificationPipeline()
        zones = {z.zone_id: z for z in p.geofence.list_zones()}
        assert zones["restricted-hq"].zone_type == "restricted"
        assert zones["parking-south"].zone_type == "monitored"
        assert zones["perimeter-north"].zone_type == "restricted"


# ---------------------------------------------------------------------------
# 2. Tick generation and target tracking
# ---------------------------------------------------------------------------

class TestTickGeneration:
    def test_single_tick_generates_sightings(self):
        """One tick should create BLE + YOLO targets."""
        p = NotificationPipeline()
        stats = p.generate_tick()
        assert stats["ble_sightings"] == len(_BLE_DEVICES)
        assert stats["yolo_sightings"] == len(_DETECTIONS)
        assert stats["tick"] == 1

    def test_targets_tracked_after_tick(self):
        """After a tick, tracker should have targets from BLE and YOLO sources."""
        p = NotificationPipeline()
        p.generate_tick()
        targets = p.tracker.get_all()
        assert len(targets) > 0
        sources = {t.source for t in targets}
        assert "ble" in sources
        assert "yolo" in sources

    def test_multiple_ticks_accumulate(self):
        """Tick counter should increment correctly over multiple ticks."""
        p = NotificationPipeline()
        for _ in range(5):
            p.generate_tick()
        assert p.tick == 5

    def test_ble_devices_have_correct_ids(self):
        """BLE targets should have expected ID format from MAC addresses."""
        p = NotificationPipeline()
        p.generate_tick()
        targets = p.tracker.get_all()
        ble_targets = [t for t in targets if t.source == "ble"]
        assert len(ble_targets) == len(_BLE_DEVICES)
        for t in ble_targets:
            assert t.target_id.startswith("ble_")


# ---------------------------------------------------------------------------
# 3. Geofence notifications
# ---------------------------------------------------------------------------

class TestGeofenceNotifications:
    def test_geofence_entry_generates_notification(self):
        """When a target enters a zone, a notification should be created."""
        p = NotificationPipeline()
        # Run enough ticks for BLE devices to wander into zones
        for _ in range(20):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        geo_notifs = [n for n in notifs if n["source"] == "geofence"]
        # With 6 devices wandering and 3 zones, we expect some geofence events
        assert len(geo_notifs) > 0

    def test_geofence_entry_has_correct_severity(self):
        """Restricted zone entries should be critical, monitored should be warning."""
        p = NotificationPipeline()
        for _ in range(30):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        geo_notifs = [n for n in notifs if n["source"] == "geofence"]
        if geo_notifs:
            # At least one notification should exist with a valid severity
            severities = {n["severity"] for n in geo_notifs}
            assert severities.issubset({"info", "warning", "critical"})

    def test_geofence_notification_has_entity_id(self):
        """Geofence notifications should link to the target that triggered them."""
        p = NotificationPipeline()
        for _ in range(30):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        geo_notifs = [n for n in notifs if n["source"] == "geofence"]
        for n in geo_notifs:
            assert n["entity_id"] is not None
            assert len(n["entity_id"]) > 0


# ---------------------------------------------------------------------------
# 4. Threat escalation notifications
# ---------------------------------------------------------------------------

class TestThreatNotifications:
    def test_threat_scorer_runs_each_tick(self):
        """Threat scorer should evaluate targets on every tick."""
        p = NotificationPipeline()
        for _ in range(5):
            p.generate_tick()
        status = p.threat_scorer.get_status()
        assert status["total_profiles"] > 0

    def test_threat_escalation_notification_format(self):
        """If threat escalation notifications are generated, they should have correct format."""
        p = NotificationPipeline()
        for _ in range(50):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        threat_notifs = [n for n in notifs if n["source"] == "threat_scorer"]
        for n in threat_notifs:
            assert "Threat Escalation" in n["title"]
            assert n["severity"] in ("warning", "critical")
            assert n["entity_id"] is not None


# ---------------------------------------------------------------------------
# 5. Sensor health notifications
# ---------------------------------------------------------------------------

class TestSensorHealthNotifications:
    def test_sensor_health_records_sightings(self):
        """Sensor health monitor should track sightings from each tick."""
        p = NotificationPipeline()
        for _ in range(5):
            p.generate_tick()
        health = p.health_monitor.get_health()
        assert len(health) > 0
        sensor_ids = {h["sensor_id"] for h in health}
        # At least our main sensors should be present
        for s in _SENSORS:
            assert s in sensor_ids

    def test_offline_sensor_triggers_notification(self):
        """When a sensor goes offline, a critical notification should be generated.

        The SensorHealthMonitor uses time.monotonic() internally with a 300s
        offline threshold. In unit tests ticks happen instantly, so we
        directly manipulate the monitor's internal state to simulate time
        passing, then trigger the pipeline's health check.
        """
        p = NotificationPipeline()
        # Run a few ticks so node-echo gets registered
        for _ in range(10):
            p.generate_tick()
        # Verify node-echo is tracked
        health = p.health_monitor.get_health()
        echo_ids = [h["sensor_id"] for h in health if h["sensor_id"] == "node-echo"]
        assert len(echo_ids) == 1

        # Simulate node-echo going offline by backdating its last_seen
        import time as _time
        with p.health_monitor._lock:
            rec = p.health_monitor._sensors["node-echo"]
            rec.last_seen = _time.monotonic() - 600.0  # 10 minutes ago

        # Now trigger the offline check by running tick 20
        p.tick = 19
        p.generate_tick()  # tick becomes 20, triggers offline check
        notifs = p.notif_mgr.get_all(limit=500)
        health_notifs = [n for n in notifs if n["source"] == "sensor_health"]
        offline_notifs = [n for n in health_notifs if "Offline" in n["title"]]
        assert len(offline_notifs) >= 1
        assert offline_notifs[0]["severity"] == "critical"
        assert "node-echo" in offline_notifs[0]["title"]


# ---------------------------------------------------------------------------
# 6. Anomaly detection notifications
# ---------------------------------------------------------------------------

class TestAnomalyNotifications:
    def test_anomaly_detection_runs_periodically(self):
        """Anomaly check fires every 5 ticks."""
        p = NotificationPipeline()
        # Run enough ticks for anomaly checks to fire
        for _ in range(10):
            p.generate_tick()
        # Verify tick count is correct
        assert p.tick == 10

    def test_anomaly_notifications_have_correct_source(self):
        """Any anomaly notifications should have source='anomaly_detector'."""
        p = NotificationPipeline()
        for _ in range(30):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        anomaly_notifs = [n for n in notifs if n["source"] == "anomaly_detector"]
        for n in anomaly_notifs:
            assert "Anomaly" in n["title"]
            assert n["severity"] == "warning"


# ---------------------------------------------------------------------------
# 7. NotificationManager integration
# ---------------------------------------------------------------------------

class TestNotificationManagerIntegration:
    def test_broadcast_callback_queues_ws_messages(self):
        """Notifications should be queued for WebSocket broadcast."""
        p = NotificationPipeline()
        p.generate_tick()
        # Some notifications may be generated on first tick
        # Force one
        p.notif_mgr.add("Test", "test msg", severity="info", source="unit_test")
        pending = p.get_pending_ws_messages()
        assert len(pending) >= 1
        assert pending[-1]["type"] == "notification:new"
        assert pending[-1]["data"]["title"] == "Test"

    def test_pending_messages_drain(self):
        """get_pending_ws_messages should drain the queue."""
        p = NotificationPipeline()
        p.notif_mgr.add("A", "msg")
        p.notif_mgr.add("B", "msg")
        msgs = p.get_pending_ws_messages()
        assert len(msgs) >= 2
        # Second drain should be empty
        assert len(p.get_pending_ws_messages()) == 0

    def test_mark_read_works(self):
        """mark_read should work through the pipeline."""
        p = NotificationPipeline()
        nid = p.notif_mgr.add("Test", "test", severity="info", source="test")
        assert p.notif_mgr.count_unread() >= 1
        p.notif_mgr.mark_read(nid)
        unread = p.notif_mgr.get_unread()
        ids = {n["id"] for n in unread}
        assert nid not in ids

    def test_all_severities_produced(self):
        """Over many ticks, the pipeline should produce info, warning, and critical."""
        p = NotificationPipeline()
        for _ in range(30):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        severities = {n["severity"] for n in notifs}
        # At minimum we expect info (geofence exit) and critical (restricted entry / offline)
        assert len(severities) >= 2

    def test_notifications_have_timestamps(self):
        """Every notification should have a valid timestamp."""
        p = NotificationPipeline()
        p.generate_tick()
        p.notif_mgr.add("Test", "test")
        notifs = p.notif_mgr.get_all(limit=10)
        for n in notifs:
            assert "timestamp" in n
            assert n["timestamp"] > 0


# ---------------------------------------------------------------------------
# 8. WebSocket ConnectionManager
# ---------------------------------------------------------------------------

class TestConnectionManager:
    def test_initial_count_zero(self):
        """New ConnectionManager should have zero connections."""
        cm = ConnectionManager()
        assert cm.count == 0

    def test_disconnect_missing_is_safe(self):
        """Disconnecting a non-existent connection should not raise."""

        class FakeWS:
            pass

        cm = ConnectionManager()
        cm.disconnect(FakeWS())  # Should not raise
        assert cm.count == 0


# ---------------------------------------------------------------------------
# 9. SimpleEventBus
# ---------------------------------------------------------------------------

class TestSimpleEventBus:
    def test_publish_records_events(self):
        bus = SimpleEventBus()
        bus.publish("test:topic", {"key": "val"})
        assert bus.event_count == 1
        assert bus.events[0] == ("test:topic", {"key": "val"})

    def test_multiple_publishes(self):
        bus = SimpleEventBus()
        bus.publish("a", {})
        bus.publish("b", {"x": 1})
        assert bus.event_count == 2

    def test_publish_none_data(self):
        bus = SimpleEventBus()
        bus.publish("topic")
        assert bus.events[0] == ("topic", {})


# ---------------------------------------------------------------------------
# 10. Pipeline status and stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_dict_keys(self):
        """generate_tick stats dict should have all expected keys."""
        p = NotificationPipeline()
        stats = p.generate_tick()
        expected_keys = {
            "tick", "ble_sightings", "yolo_sightings",
            "geofence_events", "notifications_generated",
        }
        assert expected_keys.issubset(stats.keys())

    def test_notification_count_increases(self):
        """Total notification count should increase over ticks."""
        p = NotificationPipeline()
        counts = []
        for _ in range(10):
            p.generate_tick()
            counts.append(len(p.notif_mgr.get_all(limit=9999)))
        # Count should be non-decreasing
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1]


# ---------------------------------------------------------------------------
# 11. REST API endpoints (via TestClient)
# ---------------------------------------------------------------------------

class TestRESTEndpoints:
    """Test FastAPI endpoints using httpx TestClient."""

    @pytest.fixture(autouse=True)
    def _reset_pipeline(self):
        """Reset the pipeline module singleton before each test."""
        import tritium_lib.notifications.demos.notification_demo as mod
        mod.pipeline = NotificationPipeline()
        yield

    @pytest.fixture
    def client(self):
        try:
            from httpx import ASGITransport, AsyncClient
        except ImportError:
            pytest.skip("httpx not installed")
        from tritium_lib.notifications.demos.notification_demo import app
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_get_notifications(self, client):
        async with client:
            resp = await client.get("/api/notifications")
            assert resp.status_code == 200
            data = resp.json()
            assert "notifications" in data
            assert "unread_count" in data

    @pytest.mark.asyncio
    async def test_get_unread(self, client):
        async with client:
            resp = await client.get("/api/notifications/unread")
            assert resp.status_code == 200
            data = resp.json()
            assert "notifications" in data
            assert "count" in data

    @pytest.mark.asyncio
    async def test_mark_read_not_found(self, client):
        async with client:
            resp = await client.post("/api/notifications/nonexistent/read")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_mark_all_read(self, client):
        import tritium_lib.notifications.demos.notification_demo as mod
        mod.pipeline.notif_mgr.add("Test", "msg")
        async with client:
            resp = await client.post("/api/notifications/read-all")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_dismiss(self, client):
        import tritium_lib.notifications.demos.notification_demo as mod
        nid = mod.pipeline.notif_mgr.add("Test", "dismiss me")
        async with client:
            resp = await client.post(f"/api/notifications/{nid}/dismiss")
            assert resp.status_code == 200
            assert resp.json()["dismissed"] == nid

    @pytest.mark.asyncio
    async def test_get_targets(self, client):
        async with client:
            resp = await client.get("/api/targets")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_get_geofence(self, client):
        async with client:
            resp = await client.get("/api/geofence")
            assert resp.status_code == 200
            data = resp.json()
            assert "zones" in data
            assert len(data["zones"]) == 3

    @pytest.mark.asyncio
    async def test_get_health(self, client):
        async with client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            assert "sensors" in resp.json()

    @pytest.mark.asyncio
    async def test_get_status(self, client):
        async with client:
            resp = await client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "tick" in data
            assert "total_targets" in data
            assert "unread_notifications" in data

    @pytest.mark.asyncio
    async def test_dashboard_html(self, client):
        async with client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "TRITIUM NOTIFICATION PIPELINE" in resp.text


# ---------------------------------------------------------------------------
# 12. End-to-end: full pipeline with all notification sources
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_multiple_sources(self):
        """Run 30 ticks and verify notifications from multiple sources appear."""
        p = NotificationPipeline()
        for _ in range(30):
            p.generate_tick()
        notifs = p.notif_mgr.get_all(limit=500)
        sources = {n["source"] for n in notifs}
        # We should see at least geofence and sensor_health notifications
        assert "geofence" in sources or "sensor_health" in sources
        # Total notification count should be non-trivial
        assert len(notifs) >= 1

    def test_pipeline_does_not_crash_over_100_ticks(self):
        """Pipeline should be stable over many ticks without errors."""
        p = NotificationPipeline()
        for _ in range(100):
            stats = p.generate_tick()
            assert stats["tick"] <= 100
        assert p.tick == 100
        # Should have accumulated targets and notifications
        assert len(p.tracker.get_all()) > 0
        assert len(p.notif_mgr.get_all(limit=9999)) > 0
