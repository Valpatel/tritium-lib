# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sitaware — the unified situational awareness engine.

25+ tests covering:
  - Engine creation and subsystem wiring
  - OperatingPicture generation and serialization
  - PictureUpdate creation and delta queries
  - Subscriber push notifications
  - Threat level computation
  - Summary generation
  - Target and zone picture queries
  - Health check integration
  - Analytics feed-through
  - Lifecycle (start, stop, reset, shutdown)
  - Edge cases (empty state, unknown targets, time-based filtering)
"""

import time

import pytest

from tritium_lib.sitaware import (
    OperatingPicture,
    PictureUpdate,
    SitAwareEngine,
    UpdateType,
)
from tritium_lib.events.bus import EventBus
from tritium_lib.fusion import FusionEngine
from tritium_lib.alerting import AlertEngine
from tritium_lib.analytics import AnalyticsEngine
from tritium_lib.intelligence.anomaly_engine import AnomalyEngine
from tritium_lib.incident import IncidentManager
from tritium_lib.mission import MissionPlanner
from tritium_lib.monitoring import HealthMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    """Shared EventBus for all test components."""
    return EventBus()


@pytest.fixture
def engine(event_bus):
    """A fresh SitAwareEngine with all subsystems."""
    eng = SitAwareEngine(event_bus=event_bus)
    yield eng
    eng.shutdown()


@pytest.fixture
def engine_with_targets(engine):
    """Engine pre-loaded with a few fused targets."""
    engine.fusion.ingest_ble({
        "mac": "AA:BB:CC:DD:EE:01",
        "rssi": -55,
        "name": "Phone Alpha",
        "position": {"x": 10.0, "y": 20.0},
    })
    engine.fusion.ingest_ble({
        "mac": "AA:BB:CC:DD:EE:02",
        "rssi": -70,
        "name": "Watch Beta",
        "position": {"x": 30.0, "y": 40.0},
    })
    engine.fusion.ingest_camera({
        "class_name": "person",
        "confidence": 0.95,
        "center_x": 10.0,
        "center_y": 20.0,
    })
    return engine


# ---------------------------------------------------------------------------
# 1. Engine creation and subsystem access
# ---------------------------------------------------------------------------

class TestEngineCreation:

    def test_creates_with_defaults(self):
        """Engine creates all subsystems when none provided."""
        engine = SitAwareEngine()
        assert engine.fusion is not None
        assert engine.alerting is not None
        assert engine.anomaly is not None
        assert engine.analytics is not None
        assert engine.health is not None
        assert engine.incidents is not None
        assert engine.missions is not None
        assert engine.event_bus is not None
        engine.shutdown()

    def test_accepts_external_subsystems(self, event_bus):
        """Engine uses externally provided subsystems."""
        fusion = FusionEngine(event_bus=event_bus)
        alerting = AlertEngine(event_bus=event_bus)
        anomaly = AnomalyEngine(event_bus=event_bus)
        analytics = AnalyticsEngine()
        health_mon = HealthMonitor()
        inc_mgr = IncidentManager(event_bus=event_bus)
        mission_mgr = MissionPlanner(event_bus=event_bus)

        engine = SitAwareEngine(
            event_bus=event_bus,
            fusion=fusion,
            alerting=alerting,
            anomaly=anomaly,
            analytics=analytics,
            health=health_mon,
            incidents=inc_mgr,
            missions=mission_mgr,
        )

        assert engine.fusion is fusion
        assert engine.alerting is alerting
        assert engine.anomaly is anomaly
        assert engine.analytics is analytics
        assert engine.health is health_mon
        assert engine.incidents is inc_mgr
        assert engine.missions is mission_mgr
        engine.shutdown()

    def test_shared_event_bus(self, event_bus):
        """All subsystems share the same event bus."""
        engine = SitAwareEngine(event_bus=event_bus)
        assert engine.event_bus is event_bus
        engine.shutdown()


# ---------------------------------------------------------------------------
# 2. OperatingPicture generation
# ---------------------------------------------------------------------------

class TestOperatingPicture:

    def test_empty_picture(self, engine):
        """Empty engine returns a valid empty picture."""
        pic = engine.get_picture()

        assert isinstance(pic, OperatingPicture)
        assert pic.target_count == 0
        assert pic.multi_source_targets == 0
        assert pic.active_alert_count == 0
        assert pic.active_anomaly_count == 0
        assert pic.incident_count == 0
        assert pic.mission_count == 0
        assert pic.threat_level == "green"
        assert "ALL CLEAR" in pic.summary
        assert pic.timestamp > 0

    def test_picture_with_targets(self, engine_with_targets):
        """Picture includes fused targets."""
        pic = engine_with_targets.get_picture()

        assert pic.target_count >= 2
        assert len(pic.targets) == pic.target_count
        assert all("target_id" in t for t in pic.targets)

    def test_picture_to_dict(self, engine):
        """OperatingPicture serializes to dict."""
        pic = engine.get_picture()
        d = pic.to_dict()

        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "targets" in d
        assert "target_count" in d
        assert "alerts" in d
        assert "anomalies" in d
        assert "incidents" in d
        assert "missions" in d
        assert "health" in d
        assert "analytics" in d
        assert "zones" in d
        assert "summary" in d
        assert "threat_level" in d

    def test_picture_includes_health(self, engine):
        """Picture includes health status from all registered checks."""
        pic = engine.get_picture()

        assert "health" in pic.to_dict()
        health = pic.health
        assert "components" in health
        # Should have health checks for all subsystems
        assert "fusion" in health["components"]
        assert "alerting" in health["components"]

    def test_picture_includes_analytics(self, engine):
        """Picture includes analytics snapshot."""
        pic = engine.get_picture()

        assert isinstance(pic.analytics, dict)
        assert "detection_rate" in pic.analytics
        assert "timestamp" in pic.analytics

    def test_picture_zones(self, engine):
        """Picture includes geofence zones."""
        from tritium_lib.tracking.geofence import GeoZone

        engine.fusion.add_zone(GeoZone(
            zone_id="zone-alpha",
            name="Zone Alpha",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        ))

        pic = engine.get_picture()
        assert pic.zone_count == 1
        assert len(pic.zones) == 1
        assert pic.zones[0]["zone_id"] == "zone-alpha"


# ---------------------------------------------------------------------------
# 3. PictureUpdate and delta queries
# ---------------------------------------------------------------------------

class TestPictureUpdates:

    def test_update_creation(self):
        """PictureUpdate creates with sensible defaults."""
        update = PictureUpdate(
            update_type=UpdateType.TARGET_NEW,
            data={"target_id": "ble_test"},
            source="fusion",
            target_id="ble_test",
        )

        assert update.update_type == UpdateType.TARGET_NEW
        assert update.target_id == "ble_test"
        assert update.timestamp > 0
        assert len(update.update_id) == 12

    def test_update_to_dict(self):
        """PictureUpdate serializes to dict."""
        update = PictureUpdate(
            update_type=UpdateType.ALERT_FIRED,
            data={"rule": "test"},
            source="alerting",
            severity="warning",
        )
        d = update.to_dict()

        assert d["update_type"] == "alert_fired"
        assert d["source"] == "alerting"
        assert d["severity"] == "warning"
        assert "update_id" in d

    def test_updates_generated_on_ingest(self, engine):
        """Ingesting sensor data generates picture updates."""
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        updates = engine.get_updates_since(0.0)
        assert len(updates) >= 1
        # First ingestion should be TARGET_NEW
        target_updates = [u for u in updates if u.update_type == UpdateType.TARGET_NEW]
        assert len(target_updates) >= 1

    def test_updates_since_timestamp(self, engine):
        """get_updates_since filters by timestamp."""
        t1 = time.time()

        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        # All updates since t1
        updates = engine.get_updates_since(t1 - 1)
        assert len(updates) >= 1

        # No updates since far future
        future_updates = engine.get_updates_since(time.time() + 1000)
        assert len(future_updates) == 0

    def test_updates_by_type(self, engine):
        """get_updates_by_type filters by UpdateType."""
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        target_updates = engine.get_updates_by_type(UpdateType.TARGET_NEW)
        assert len(target_updates) >= 1

        # No alert updates yet
        alert_updates = engine.get_updates_by_type(UpdateType.ALERT_FIRED)
        assert len(alert_updates) == 0

    def test_update_stream_bounded(self, event_bus):
        """Update stream respects max_updates limit."""
        engine = SitAwareEngine(event_bus=event_bus, max_updates=5)

        for i in range(10):
            engine.fusion.ingest_ble({
                "mac": f"AA:BB:CC:DD:EE:{i:02X}",
                "rssi": -55,
            })

        updates = engine.get_updates_since(0.0)
        assert len(updates) <= 5
        engine.shutdown()

    def test_repeat_ingest_is_target_updated(self, engine):
        """Second ingest of same target should be TARGET_UPDATED not TARGET_NEW."""
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        # Second ingest of same MAC
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -50,
        })

        updates = engine.get_updates_since(0.0)
        new_updates = [u for u in updates if u.update_type == UpdateType.TARGET_NEW]
        updated_updates = [u for u in updates if u.update_type == UpdateType.TARGET_UPDATED]

        assert len(new_updates) == 1  # Only the first should be NEW
        assert len(updated_updates) >= 1  # Subsequent should be UPDATED


# ---------------------------------------------------------------------------
# 4. Subscriber push notifications
# ---------------------------------------------------------------------------

class TestSubscriptions:

    def test_subscribe_receives_updates(self, engine):
        """Subscriber callback receives updates."""
        received = []
        engine.subscribe(lambda u: received.append(u))

        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        assert len(received) >= 1
        assert all(isinstance(u, PictureUpdate) for u in received)

    def test_unsubscribe(self, engine):
        """Unsubscribing stops further callbacks."""
        received = []
        callback = lambda u: received.append(u)
        engine.subscribe(callback)

        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })
        first_count = len(received)

        engine.unsubscribe(callback)

        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:02",
            "rssi": -55,
        })

        assert len(received) == first_count

    def test_unsubscribe_returns_false_for_unknown(self, engine):
        """Unsubscribing a non-subscriber returns False."""
        result = engine.unsubscribe(lambda u: None)
        assert result is False

    def test_subscriber_count(self, engine):
        """subscriber_count tracks active subscribers."""
        assert engine.subscriber_count == 0

        cb1 = lambda u: None
        cb2 = lambda u: None
        engine.subscribe(cb1)
        assert engine.subscriber_count == 1

        engine.subscribe(cb2)
        assert engine.subscriber_count == 2

        engine.unsubscribe(cb1)
        assert engine.subscriber_count == 1

    def test_failing_subscriber_does_not_crash(self, engine):
        """A subscriber that raises an exception doesn't crash the engine."""
        def bad_callback(u):
            raise RuntimeError("subscriber error")

        engine.subscribe(bad_callback)

        # Should not raise
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        # Engine still works
        pic = engine.get_picture()
        assert pic.target_count >= 1

    def test_multiple_subscribers(self, engine):
        """Multiple subscribers all receive the same update."""
        received_a = []
        received_b = []

        engine.subscribe(lambda u: received_a.append(u))
        engine.subscribe(lambda u: received_b.append(u))

        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        assert len(received_a) >= 1
        assert len(received_b) >= 1
        assert len(received_a) == len(received_b)


# ---------------------------------------------------------------------------
# 5. Threat level computation
# ---------------------------------------------------------------------------

class TestThreatLevel:

    def test_green_when_empty(self, engine):
        """Empty engine has GREEN threat level."""
        pic = engine.get_picture()
        assert pic.threat_level == "green"

    def test_yellow_with_alerts(self, engine):
        """Alerts raise threat level to YELLOW."""
        # Evaluate a manual event that will trigger a built-in rule
        engine.alerting.evaluate_event("geofence:enter", {
            "target_id": "ble_test",
            "zone_id": "restricted",
            "zone_name": "Restricted Zone",
            "zone_type": "restricted",
        })

        pic = engine.get_picture()
        # Should be at least yellow with an alert
        assert pic.threat_level in ("yellow", "orange", "red")

    def test_green_with_targets_only(self, engine_with_targets):
        """Targets alone without alerts stay GREEN."""
        pic = engine_with_targets.get_picture()
        assert pic.threat_level == "green"


# ---------------------------------------------------------------------------
# 6. Summary generation
# ---------------------------------------------------------------------------

class TestSummary:

    def test_empty_summary(self, engine):
        """Empty engine produces ALL CLEAR summary."""
        pic = engine.get_picture()
        assert "ALL CLEAR" in pic.summary
        assert "No targets" in pic.summary

    def test_summary_with_targets(self, engine_with_targets):
        """Summary includes target count."""
        pic = engine_with_targets.get_picture()
        assert "target" in pic.summary.lower()
        # Should NOT say "No targets"
        assert "No targets" not in pic.summary

    def test_summary_includes_alerts_when_present(self, engine):
        """Summary mentions alerts when they exist."""
        engine.alerting.evaluate_event("geofence:enter", {
            "target_id": "ble_test",
            "zone_id": "restricted",
            "zone_name": "Restricted Zone",
            "zone_type": "restricted",
        })

        pic = engine.get_picture()
        assert "alert" in pic.summary.lower()


# ---------------------------------------------------------------------------
# 7. Target and zone picture queries
# ---------------------------------------------------------------------------

class TestSpecificQueries:

    def test_get_target_picture(self, engine_with_targets):
        """get_target_picture returns comprehensive target info."""
        targets = engine_with_targets.fusion.get_fused_targets()
        assert len(targets) >= 1

        tid = targets[0].target_id
        result = engine_with_targets.get_target_picture(tid)

        assert result is not None
        assert "target" in result
        assert "alerts" in result
        assert "anomalies" in result
        assert "incidents" in result
        assert "timestamp" in result

    def test_get_target_picture_returns_none_for_unknown(self, engine):
        """get_target_picture returns None for unknown target."""
        result = engine.get_target_picture("nonexistent_target")
        assert result is None

    def test_get_zone_picture(self, engine):
        """get_zone_picture returns zone information."""
        from tritium_lib.tracking.geofence import GeoZone

        engine.fusion.add_zone(GeoZone(
            zone_id="zone-alpha",
            name="Zone Alpha",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        ))

        result = engine.get_zone_picture("zone-alpha")

        assert "zone_activity" in result
        assert "targets" in result
        assert "alerts" in result
        assert "anomalies" in result
        assert "baseline" in result
        assert "timestamp" in result

    def test_get_stats(self, engine):
        """get_stats returns aggregated statistics."""
        stats = engine.get_stats()

        assert "fusion" in stats
        assert "alerting" in stats
        assert "anomaly" in stats
        assert "analytics" in stats
        assert "incidents" in stats
        assert "missions" in stats
        assert "health" in stats
        assert "sitaware" in stats
        assert "timestamp" in stats

        # Sitaware-specific stats
        sa = stats["sitaware"]
        assert "update_count" in sa
        assert "subscriber_count" in sa
        assert "max_updates" in sa
        assert "known_target_ids" in sa


# ---------------------------------------------------------------------------
# 8. Health monitoring integration
# ---------------------------------------------------------------------------

class TestHealthIntegration:

    def test_health_checks_registered(self, engine):
        """Engine registers health checks for all subsystems."""
        components = engine.health.registered_components
        assert "fusion" in components
        assert "alerting" in components
        assert "anomaly" in components
        assert "analytics" in components
        assert "incidents" in components
        assert "missions" in components

    def test_health_all_up(self, engine):
        """All subsystems report UP in a fresh engine."""
        status = engine.health.check_all()
        assert status.overall.value in ("up", "degraded", "unknown")
        # All individual checks should pass
        for name, component in status.components.items():
            assert component.status.value in ("up", "degraded", "unknown"), (
                f"Component {name} is {component.status.value}: {component.error}"
            )


# ---------------------------------------------------------------------------
# 9. Analytics feed-through
# ---------------------------------------------------------------------------

class TestAnalyticsFeedThrough:

    def test_ingest_feeds_analytics(self, engine):
        """Sensor ingestion feeds analytics detection counter."""
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })

        # Analytics should record the detection
        snap = engine.analytics.snapshot()
        detection = snap.get("detection_rate", {})
        assert detection.get("lifetime_count", 0) >= 1


# ---------------------------------------------------------------------------
# 10. Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_start_and_stop(self, engine):
        """Engine starts and stops without error."""
        engine.start()
        engine.stop()

    def test_reset_clears_everything(self, engine_with_targets):
        """Reset clears all state."""
        pic_before = engine_with_targets.get_picture()
        assert pic_before.target_count >= 2

        engine_with_targets.reset()

        pic_after = engine_with_targets.get_picture()
        assert pic_after.target_count == 0

    def test_shutdown_clears_subscribers(self, engine):
        """Shutdown clears all subscribers."""
        engine.subscribe(lambda u: None)
        assert engine.subscriber_count == 1

        engine.shutdown()
        assert engine.subscriber_count == 0

    def test_reset_clears_updates(self, engine):
        """Reset clears the update stream."""
        engine.fusion.ingest_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -55,
        })
        assert len(engine.get_updates_since(0.0)) >= 1

        engine.reset()
        assert len(engine.get_updates_since(0.0)) == 0


# ---------------------------------------------------------------------------
# 11. UpdateType enum
# ---------------------------------------------------------------------------

class TestUpdateType:

    def test_all_update_types_exist(self):
        """All expected UpdateType values exist."""
        expected = [
            "target_new", "target_updated", "target_lost",
            "target_correlated", "alert_fired", "anomaly_detected",
            "incident_created", "incident_updated", "incident_resolved",
            "mission_created", "mission_updated", "mission_completed",
            "health_changed", "zone_breach", "full_refresh",
        ]
        actual = [ut.value for ut in UpdateType]
        for exp in expected:
            assert exp in actual, f"Missing UpdateType: {exp}"

    def test_update_type_is_string(self):
        """UpdateType values are strings."""
        for ut in UpdateType:
            assert isinstance(ut.value, str)


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_duplicate_subscribe_ignored(self, engine):
        """Subscribing the same callback twice doesn't duplicate."""
        received = []
        callback = lambda u: received.append(u)

        engine.subscribe(callback)
        engine.subscribe(callback)
        assert engine.subscriber_count == 1

    def test_get_picture_is_idempotent(self, engine_with_targets):
        """Calling get_picture twice returns consistent results."""
        pic1 = engine_with_targets.get_picture()
        pic2 = engine_with_targets.get_picture()

        assert pic1.target_count == pic2.target_count
        assert pic1.threat_level == pic2.threat_level

    def test_operating_picture_default_values(self):
        """OperatingPicture can be created with all defaults."""
        pic = OperatingPicture()
        d = pic.to_dict()

        assert d["target_count"] == 0
        assert d["threat_level"] == "green"
        assert d["summary"] == ""
        assert d["targets"] == []
        assert d["alerts"] == []
        assert d["anomalies"] == []
        assert d["incidents"] == []
        assert d["missions"] == []

    def test_picture_update_default_values(self):
        """PictureUpdate can be created with minimal args."""
        update = PictureUpdate()
        d = update.to_dict()

        assert d["update_type"] == "full_refresh"
        assert d["source"] == ""
        assert d["severity"] == "info"
        assert len(d["update_id"]) == 12
