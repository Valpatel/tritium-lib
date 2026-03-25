# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.alerting — rules engine for alerts and dispatch."""

import time

import pytest

from tritium_lib.alerting import (
    AlertCondition,
    AlertEngine,
    AlertRecord,
    AlertRule,
    AlertTrigger,
    ConditionOperator,
    DispatchAction,
    NotificationChannel,
    NotificationSeverity,
)
from tritium_lib.events.bus import EventBus
from tritium_lib.notifications import NotificationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs):
    """Create an AlertEngine with sensible test defaults."""
    defaults = {
        "event_bus": EventBus(),
        "notification_manager": NotificationManager(),
        "load_defaults": False,
    }
    defaults.update(kwargs)
    return AlertEngine(**defaults)


def _make_rule(**kwargs):
    """Create a minimal AlertRule for testing."""
    defaults = {
        "rule_id": "test-rule",
        "name": "Test Rule",
        "trigger": AlertTrigger.TARGET_ENTER_ZONE,
        "severity": NotificationSeverity.WARNING,
        "cooldown_seconds": 0,
    }
    defaults.update(kwargs)
    return AlertRule(**defaults)


# ---------------------------------------------------------------------------
# Test AlertEngine initialization
# ---------------------------------------------------------------------------

class TestAlertEngineInit:
    def test_default_init(self):
        engine = AlertEngine()
        assert engine is not None
        stats = engine.get_stats()
        assert stats["total_rules"] > 0  # default rules loaded

    def test_no_defaults(self):
        engine = _make_engine(load_defaults=False)
        stats = engine.get_stats()
        assert stats["total_rules"] == 0

    def test_with_defaults(self):
        engine = _make_engine(load_defaults=True)
        rules = engine.get_rules()
        assert len(rules) >= 4  # at least the 4 built-in rules
        rule_ids = {r.rule_id for r in rules}
        assert "builtin-geofence-entry" in rule_ids
        assert "builtin-threat-level-change" in rule_ids
        assert "builtin-sensor-offline" in rule_ids
        assert "builtin-target-loitering" in rule_ids

    def test_init_without_event_bus(self):
        engine = AlertEngine(event_bus=None, load_defaults=False)
        assert engine.get_stats()["started"] is False

    def test_init_without_notification_manager(self):
        engine = AlertEngine(notification_manager=None, load_defaults=False)
        assert engine is not None


# ---------------------------------------------------------------------------
# Test rule management
# ---------------------------------------------------------------------------

class TestRuleManagement:
    def test_add_rule(self):
        engine = _make_engine()
        rule = _make_rule()
        result = engine.add_rule(rule)
        assert result.rule_id == "test-rule"
        assert engine.get_rule("test-rule") is not None

    def test_remove_rule(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())
        assert engine.remove_rule("test-rule") is True
        assert engine.get_rule("test-rule") is None

    def test_remove_nonexistent_rule(self):
        engine = _make_engine()
        assert engine.remove_rule("nonexistent") is False

    def test_get_rules_list(self):
        engine = _make_engine()
        engine.add_rule(_make_rule(rule_id="r1", name="Rule 1"))
        engine.add_rule(_make_rule(rule_id="r2", name="Rule 2"))
        rules = engine.get_rules()
        assert len(rules) == 2

    def test_enable_disable_rule(self):
        engine = _make_engine()
        rule = _make_rule()
        engine.add_rule(rule)

        assert engine.disable_rule("test-rule") is True
        assert engine.get_rule("test-rule").enabled is False

        assert engine.enable_rule("test-rule") is True
        assert engine.get_rule("test-rule").enabled is True

    def test_enable_nonexistent(self):
        engine = _make_engine()
        assert engine.enable_rule("nope") is False
        assert engine.disable_rule("nope") is False

    def test_set_rule_action(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())
        assert engine.set_rule_action("test-rule", DispatchAction.ESCALATE) is True
        assert engine.set_rule_action("missing", DispatchAction.LOG) is False


# ---------------------------------------------------------------------------
# Test event evaluation — core logic
# ---------------------------------------------------------------------------

class TestEvaluateEvent:
    def test_geofence_enter_fires(self):
        engine = _make_engine()
        rule = _make_rule(
            trigger=AlertTrigger.TARGET_ENTER_ZONE,
            cooldown_seconds=0,
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("geofence:enter", {
            "target_id": "ble_aabb",
            "zone_id": "perimeter",
            "zone_name": "Perimeter",
            "zone_type": "restricted",
        })

        assert len(alerts) == 1
        assert alerts[0].rule_id == "test-rule"
        assert alerts[0].target_id == "ble_aabb"
        assert alerts[0].zone_id == "perimeter"
        assert alerts[0].trigger == "target_enter_zone"
        assert alerts[0].severity == "warning"

    def test_unknown_event_no_fire(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())
        alerts = engine.evaluate_event("unknown.topic", {"foo": "bar"})
        assert len(alerts) == 0

    def test_disabled_rule_no_fire(self):
        engine = _make_engine()
        rule = _make_rule(enabled=False)
        engine.add_rule(rule)
        alerts = engine.evaluate_event("geofence:enter", {"target_id": "x"})
        assert len(alerts) == 0

    def test_condition_filters(self):
        engine = _make_engine()
        rule = _make_rule(
            trigger=AlertTrigger.TARGET_ENTER_ZONE,
            conditions=[
                AlertCondition(
                    field="zone_type",
                    operator=ConditionOperator.EQUALS,
                    value="restricted",
                ),
            ],
        )
        engine.add_rule(rule)

        # Matches
        alerts = engine.evaluate_event("geofence:enter", {
            "target_id": "ble_xx",
            "zone_type": "restricted",
        })
        assert len(alerts) == 1

        # Doesn't match
        alerts = engine.evaluate_event("geofence:enter", {
            "target_id": "ble_yy",
            "zone_type": "monitored",
        })
        assert len(alerts) == 0

    def test_cooldown_suppression(self):
        engine = _make_engine()
        rule = _make_rule(cooldown_seconds=3600)  # 1 hour cooldown
        engine.add_rule(rule)

        alerts1 = engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert len(alerts1) == 1

        alerts2 = engine.evaluate_event("geofence:enter", {"target_id": "t2"})
        assert len(alerts2) == 0

        stats = engine.get_stats()
        assert stats["total_suppressed"] == 1

    def test_multiple_rules_fire(self):
        engine = _make_engine()
        engine.add_rule(_make_rule(rule_id="r1", name="Rule 1"))
        engine.add_rule(_make_rule(rule_id="r2", name="Rule 2"))

        alerts = engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert len(alerts) == 2

    def test_anomaly_alert_maps_to_threat(self):
        engine = _make_engine()
        rule = _make_rule(
            rule_id="threat-rule",
            trigger=AlertTrigger.THREAT_DETECTED,
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("anomaly.alert", {
            "target_id": "ble_ff",
            "severity": "high",
            "detail": "Speed anomaly detected",
        })
        assert len(alerts) == 1
        assert alerts[0].trigger == "threat_detected"

    def test_sensor_offline_fires(self):
        engine = _make_engine()
        rule = _make_rule(
            rule_id="offline-rule",
            trigger=AlertTrigger.DEVICE_OFFLINE,
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("sensor.offline", {
            "device_id": "node-01",
            "last_seen": "2026-03-24T12:00:00",
        })
        assert len(alerts) == 1
        assert alerts[0].device_id == "node-01"

    def test_dwell_event_fires_loitering(self):
        engine = _make_engine()
        rule = _make_rule(
            rule_id="dwell-rule",
            trigger=AlertTrigger.TARGET_LOITER,
            conditions=[
                AlertCondition(
                    field="duration_seconds",
                    operator=ConditionOperator.GREATER_THAN,
                    value=100,
                ),
            ],
        )
        engine.add_rule(rule)

        # Above threshold
        alerts = engine.evaluate_event("dwell.event", {
            "target_id": "ble_abc",
            "zone_id": "lobby",
            "duration_seconds": 600,
        })
        assert len(alerts) == 1

        # Below threshold
        alerts = engine.evaluate_event("dwell.event", {
            "target_id": "ble_def",
            "zone_id": "lobby",
            "duration_seconds": 50,
        })
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Test dispatch actions
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_notify_action_creates_notification(self):
        nm = NotificationManager()
        engine = _make_engine(notification_manager=nm)
        rule = _make_rule()
        engine.add_rule(rule)

        engine.evaluate_event("geofence:enter", {
            "target_id": "ble_aa",
            "zone_id": "z1",
        })

        notifs = nm.get_all()
        assert len(notifs) == 1
        assert notifs[0]["source"] == "alerting"
        assert notifs[0]["title"] == "Test Rule"

    def test_log_action_no_notification(self):
        nm = NotificationManager()
        engine = _make_engine(notification_manager=nm)
        rule = _make_rule()
        engine.add_rule(rule, action=DispatchAction.LOG)

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        notifs = nm.get_all()
        assert len(notifs) == 0

    def test_escalate_publishes_event(self):
        bus = EventBus()
        received = []
        bus.subscribe("alert.escalation", lambda e: received.append(e))

        engine = _make_engine(event_bus=bus)
        rule = _make_rule()
        engine.add_rule(rule, action=DispatchAction.ESCALATE)

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        assert len(received) == 1
        assert received[0].data["rule_id"] == "test-rule"

    def test_dispatch_publishes_command(self):
        bus = EventBus()
        received = []
        bus.subscribe("alert.dispatch", lambda e: received.append(e))

        engine = _make_engine(event_bus=bus)
        rule = _make_rule()
        engine.add_rule(rule, action=DispatchAction.DISPATCH)

        engine.evaluate_event("geofence:enter", {
            "target_id": "ble_hostile",
            "zone_id": "sector-7",
        })

        assert len(received) == 1
        assert received[0].data["target_id"] == "ble_hostile"
        assert received[0].data["zone_id"] == "sector-7"

    def test_custom_action_handler(self):
        handled = []

        def my_handler(record):
            handled.append(record)

        engine = _make_engine()
        engine.register_action_handler(DispatchAction.DISPATCH, my_handler)

        rule = _make_rule()
        engine.add_rule(rule, action=DispatchAction.DISPATCH)

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        assert len(handled) == 1
        assert handled[0].rule_id == "test-rule"

    def test_suppress_action(self):
        nm = NotificationManager()
        engine = _make_engine(notification_manager=nm)
        rule = _make_rule()
        engine.add_rule(rule, action=DispatchAction.SUPPRESS)

        alerts = engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        assert len(alerts) == 1  # Alert still generated
        assert alerts[0].action == "suppress"
        assert len(nm.get_all()) == 0  # But no notification


# ---------------------------------------------------------------------------
# Test EventBus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    def test_start_subscribes(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        engine.add_rule(_make_rule())

        engine.start()
        stats = engine.get_stats()
        assert stats["started"] is True
        assert stats["subscribed_topics"] > 0

    def test_start_without_bus(self):
        engine = _make_engine(event_bus=None)
        engine.start()  # Should not raise
        assert engine.get_stats()["started"] is False

    def test_stop_unsubscribes(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        engine.start()
        engine.stop()
        assert engine.get_stats()["started"] is False

    def test_double_start(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        engine.start()
        engine.start()  # Should be idempotent
        assert engine.get_stats()["started"] is True

    def test_live_event_triggers_alert(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        rule = _make_rule()
        engine.add_rule(rule)
        engine.start()

        # Publish a geofence:enter event on the bus
        bus.publish("geofence:enter", data={
            "target_id": "ble_live",
            "zone_id": "zone-1",
            "zone_name": "Zone One",
            "zone_type": "monitored",
        })

        history = engine.get_history()
        assert len(history) == 1
        assert history[0].target_id == "ble_live"

    def test_live_anomaly_event(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        rule = _make_rule(
            rule_id="anomaly-rule",
            trigger=AlertTrigger.THREAT_DETECTED,
        )
        engine.add_rule(rule)
        engine.start()

        bus.publish("anomaly.alert", data={
            "target_id": "ble_suspicious",
            "severity": "high",
            "detail": "Speed anomaly in zone-lobby",
        })

        history = engine.get_history()
        assert len(history) == 1
        assert history[0].target_id == "ble_suspicious"


# ---------------------------------------------------------------------------
# Test history and query
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_records(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        engine.evaluate_event("geofence:enter", {"target_id": "t2"})

        history = engine.get_history()
        assert len(history) == 2
        # Newest first
        assert history[0].target_id == "t2"
        assert history[1].target_id == "t1"

    def test_history_limit(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        for i in range(10):
            engine.evaluate_event("geofence:enter", {"target_id": f"t{i}"})

        history = engine.get_history(limit=3)
        assert len(history) == 3

    def test_history_filter_by_rule_id(self):
        engine = _make_engine()
        engine.add_rule(_make_rule(rule_id="r1"))
        engine.add_rule(_make_rule(rule_id="r2", name="Rule 2"))

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        history_r1 = engine.get_history(rule_id="r1")
        history_r2 = engine.get_history(rule_id="r2")
        assert len(history_r1) == 1
        assert len(history_r2) == 1

    def test_history_filter_by_severity(self):
        engine = _make_engine()
        engine.add_rule(_make_rule(
            rule_id="warn-rule",
            severity=NotificationSeverity.WARNING,
        ))
        engine.add_rule(_make_rule(
            rule_id="crit-rule",
            severity=NotificationSeverity.CRITICAL,
        ))

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        warnings = engine.get_history(severity="warning")
        criticals = engine.get_history(severity="critical")
        assert len(warnings) == 1
        assert len(criticals) == 1

    def test_history_filter_by_target_id(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "alpha"})
        engine.evaluate_event("geofence:enter", {"target_id": "beta"})

        history = engine.get_history(target_id="alpha")
        assert len(history) == 1
        assert history[0].target_id == "alpha"

    def test_history_filter_by_zone_id(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {
            "target_id": "t1",
            "zone_id": "zone-a",
        })
        engine.evaluate_event("geofence:enter", {
            "target_id": "t2",
            "zone_id": "zone-b",
        })

        history = engine.get_history(zone_id="zone-a")
        assert len(history) == 1

    def test_history_since_filter(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        marker = time.time()
        time.sleep(0.02)
        engine.evaluate_event("geofence:enter", {"target_id": "t2"})

        history = engine.get_history(since=marker)
        assert len(history) == 1
        assert history[0].target_id == "t2"

    def test_clear_history(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert engine.clear_history() == 1
        assert len(engine.get_history()) == 0

    def test_max_history_cap(self):
        engine = _make_engine(max_history=5)
        engine.add_rule(_make_rule())

        for i in range(10):
            engine.evaluate_event("geofence:enter", {"target_id": f"t{i}"})

        history = engine.get_history(limit=100)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Test stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_counters(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        engine.evaluate_event("unknown.event", {})

        stats = engine.get_stats()
        assert stats["total_events_processed"] == 2
        assert stats["total_alerts_fired"] == 1
        assert stats["total_suppressed"] == 0

    def test_rule_stats(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())

        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        rule_stats = engine.get_rule_stats()
        assert len(rule_stats) == 1
        assert rule_stats[0]["fire_count"] == 1
        assert rule_stats[0]["rule_id"] == "test-rule"

    def test_reset_counters(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())
        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        engine.reset_counters()
        stats = engine.get_stats()
        assert stats["total_events_processed"] == 0
        assert stats["total_alerts_fired"] == 0


# ---------------------------------------------------------------------------
# Test AlertRecord
# ---------------------------------------------------------------------------

class TestAlertRecord:
    def test_to_dict(self):
        record = AlertRecord(
            record_id="abc123",
            rule_id="test-rule",
            rule_name="Test Rule",
            trigger="target_enter_zone",
            severity="warning",
            action="notify",
            message="Target entered zone",
            event_data={"target_id": "ble_aa"},
            target_id="ble_aa",
            zone_id="z1",
            timestamp=1000.0,
        )
        d = record.to_dict()
        assert d["record_id"] == "abc123"
        assert d["rule_id"] == "test-rule"
        assert d["severity"] == "warning"
        assert d["target_id"] == "ble_aa"
        assert d["zone_id"] == "z1"

    def test_frozen(self):
        record = AlertRecord(
            record_id="abc",
            rule_id="r",
            rule_name="R",
            trigger="t",
            severity="s",
            action="a",
            message="m",
            event_data={},
        )
        with pytest.raises(AttributeError):
            record.severity = "critical"


# ---------------------------------------------------------------------------
# Test DispatchAction enum
# ---------------------------------------------------------------------------

class TestDispatchAction:
    def test_values(self):
        assert DispatchAction.NOTIFY.value == "notify"
        assert DispatchAction.LOG.value == "log"
        assert DispatchAction.ESCALATE.value == "escalate"
        assert DispatchAction.DISPATCH.value == "dispatch"
        assert DispatchAction.SUPPRESS.value == "suppress"

    def test_from_string(self):
        assert DispatchAction("notify") == DispatchAction.NOTIFY
        assert DispatchAction("escalate") == DispatchAction.ESCALATE


# ---------------------------------------------------------------------------
# Test lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_reset(self):
        engine = _make_engine(load_defaults=True)
        engine.evaluate_event("geofence:enter", {"target_id": "t1"})

        engine.reset()
        assert len(engine.get_rules()) == 0
        assert len(engine.get_history()) == 0
        stats = engine.get_stats()
        assert stats["total_events_processed"] == 0

    def test_start_stop_cycle(self):
        bus = EventBus()
        engine = _make_engine(event_bus=bus)
        engine.start()
        assert engine.get_stats()["started"] is True

        engine.stop()
        assert engine.get_stats()["started"] is False

        # Can restart
        engine.start()
        assert engine.get_stats()["started"] is True
        engine.stop()


# ---------------------------------------------------------------------------
# Test built-in rules with live events
# ---------------------------------------------------------------------------

class TestBuiltinRules:
    def test_builtin_geofence_entry(self):
        engine = _make_engine(load_defaults=True)
        alerts = engine.evaluate_event("geofence:enter", {
            "target_id": "ble_intruder",
            "zone_id": "perimeter",
            "zone_name": "Perimeter",
            "zone_type": "restricted",
        })
        # The builtin-geofence-entry rule should fire
        rule_ids = {a.rule_id for a in alerts}
        assert "builtin-geofence-entry" in rule_ids

    def test_builtin_sensor_offline(self):
        engine = _make_engine(load_defaults=True)
        alerts = engine.evaluate_event("sensor.offline", {
            "device_id": "node-42",
            "last_seen": "10min ago",
        })
        rule_ids = {a.rule_id for a in alerts}
        assert "builtin-sensor-offline" in rule_ids

    def test_builtin_threat_level_change(self):
        engine = _make_engine(load_defaults=True)
        alerts = engine.evaluate_event("anomaly.alert", {
            "target_id": "ble_suspect",
            "severity": "critical",
            "detail": "Extreme speed anomaly",
        })
        rule_ids = {a.rule_id for a in alerts}
        assert "builtin-threat-level-change" in rule_ids

    def test_builtin_loitering(self):
        engine = _make_engine(load_defaults=True)
        alerts = engine.evaluate_event("dwell.event", {
            "target_id": "ble_loiterer",
            "zone_id": "lobby",
            "duration_seconds": 600,
        })
        rule_ids = {a.rule_id for a in alerts}
        assert "builtin-target-loitering" in rule_ids


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_notification_manager(self):
        engine = AlertEngine(
            event_bus=EventBus(),
            notification_manager=None,
            load_defaults=False,
        )
        engine.add_rule(_make_rule())
        # Should not raise even without notification manager
        alerts = engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert len(alerts) == 1

    def test_no_event_bus_for_escalate(self):
        engine = AlertEngine(
            event_bus=None,
            notification_manager=None,
            load_defaults=False,
        )
        engine.add_rule(_make_rule())
        engine.set_rule_action("test-rule", DispatchAction.ESCALATE)
        # Should not raise
        alerts = engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert len(alerts) == 1

    def test_event_with_missing_data(self):
        engine = _make_engine()
        engine.add_rule(_make_rule())
        alerts = engine.evaluate_event("geofence:enter", {})
        assert len(alerts) == 1
        assert alerts[0].target_id == ""

    def test_concurrent_evaluate(self):
        """Verify thread safety by evaluating from multiple threads."""
        import threading

        engine = _make_engine()
        engine.add_rule(_make_rule())
        errors = []

        def evaluate():
            try:
                for i in range(20):
                    engine.evaluate_event("geofence:enter", {"target_id": f"t{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=evaluate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_stats()
        assert stats["total_events_processed"] == 80

    def test_action_handler_exception(self):
        """Custom handler that raises should not crash the engine."""
        def bad_handler(record):
            raise RuntimeError("handler failed")

        engine = _make_engine()
        engine.register_action_handler(DispatchAction.DISPATCH, bad_handler)
        engine.add_rule(_make_rule())
        engine.set_rule_action("test-rule", DispatchAction.DISPATCH)

        # Should not raise
        alerts = engine.evaluate_event("geofence:enter", {"target_id": "t1"})
        assert len(alerts) == 1

    def test_fusion_event_mapping(self):
        engine = _make_engine()
        rule = _make_rule(
            rule_id="correlation-rule",
            trigger=AlertTrigger.CORRELATION_EVENT,
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("fusion.target.correlated", {
            "primary_id": "ble_aa",
            "secondary_id": "wifi_aa",
            "confidence": 0.95,
        })
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Test message rendering
# ---------------------------------------------------------------------------

class TestMessageRendering:
    def test_geofence_message(self):
        engine = _make_engine()
        rule = _make_rule(
            message_template="ALERT: {target_id} entered {zone_name} ({zone_type})",
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("geofence:enter", {
            "target_id": "ble_abc",
            "zone_name": "Perimeter",
            "zone_type": "restricted",
        })
        assert "ble_abc" in alerts[0].message
        assert "Perimeter" in alerts[0].message
        assert "restricted" in alerts[0].message

    def test_sensor_offline_message(self):
        engine = _make_engine()
        rule = _make_rule(
            trigger=AlertTrigger.DEVICE_OFFLINE,
            message_template="OFFLINE: {device_id} last seen {last_seen}",
        )
        engine.add_rule(rule)

        alerts = engine.evaluate_event("sensor.offline", {
            "device_id": "node-01",
            "last_seen": "5min ago",
        })
        assert "node-01" in alerts[0].message
        assert "5min ago" in alerts[0].message
