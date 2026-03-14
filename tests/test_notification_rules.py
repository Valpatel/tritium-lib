# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for notification rule models."""

from datetime import datetime, timedelta, timezone

import pytest

from tritium_lib.models.notification_rules import (
    DEFAULT_RULES,
    NotificationChannel,
    NotificationRule,
    NotificationSeverity,
)


# ------------------------------------------------------------------
# NotificationSeverity
# ------------------------------------------------------------------


class TestNotificationSeverity:
    def test_rank_ordering(self):
        assert NotificationSeverity.DEBUG.rank == 0
        assert NotificationSeverity.CRITICAL.rank == 4

    def test_comparison_operators(self):
        assert NotificationSeverity.WARNING >= NotificationSeverity.INFO
        assert NotificationSeverity.INFO < NotificationSeverity.ERROR
        assert NotificationSeverity.CRITICAL > NotificationSeverity.WARNING
        assert NotificationSeverity.DEBUG <= NotificationSeverity.DEBUG

    def test_equality(self):
        assert NotificationSeverity.INFO == NotificationSeverity.INFO


# ------------------------------------------------------------------
# NotificationChannel
# ------------------------------------------------------------------


class TestNotificationChannel:
    def test_all_channels_exist(self):
        assert NotificationChannel.WEBSOCKET.value == "websocket"
        assert NotificationChannel.MQTT.value == "mqtt"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.LOG.value == "log"


# ------------------------------------------------------------------
# NotificationRule
# ------------------------------------------------------------------


class TestNotificationRule:
    def _make_rule(self, **kwargs):
        defaults = {
            "rule_id": "test-rule",
            "name": "Test Rule",
            "trigger_event": "node_offline",
            "severity_filter": NotificationSeverity.WARNING,
            "channels": [NotificationChannel.WEBSOCKET],
            "cooldown_seconds": 60,
        }
        defaults.update(kwargs)
        return NotificationRule(**defaults)

    def test_basic_creation(self):
        rule = self._make_rule()
        assert rule.rule_id == "test-rule"
        assert rule.name == "Test Rule"
        assert rule.enabled is True
        assert rule.fire_count == 0

    def test_matches_event_exact(self):
        rule = self._make_rule(trigger_event="node_offline")
        assert rule.matches_event("node_offline", NotificationSeverity.WARNING)
        assert not rule.matches_event("battery_low", NotificationSeverity.WARNING)

    def test_matches_event_wildcard(self):
        rule = self._make_rule(trigger_event="*")
        assert rule.matches_event("node_offline", NotificationSeverity.WARNING)
        assert rule.matches_event("battery_low", NotificationSeverity.CRITICAL)
        assert rule.matches_event("anything", NotificationSeverity.ERROR)

    def test_matches_event_severity_filter(self):
        rule = self._make_rule(severity_filter=NotificationSeverity.ERROR)
        assert not rule.matches_event("node_offline", NotificationSeverity.WARNING)
        assert rule.matches_event("node_offline", NotificationSeverity.ERROR)
        assert rule.matches_event("node_offline", NotificationSeverity.CRITICAL)

    def test_matches_event_disabled(self):
        rule = self._make_rule(enabled=False)
        assert not rule.matches_event("node_offline", NotificationSeverity.CRITICAL)

    def test_matches_device_no_filter(self):
        rule = self._make_rule()
        assert rule.matches_device("any-device")

    def test_matches_device_with_filter(self):
        rule = self._make_rule(device_filter=["dev-001", "dev-002"])
        assert rule.matches_device("dev-001")
        assert rule.matches_device("dev-002")
        assert not rule.matches_device("dev-003")

    def test_cooldown_never_fired(self):
        rule = self._make_rule(cooldown_seconds=60)
        assert rule.is_cooled_down()

    def test_cooldown_not_elapsed(self):
        now = datetime.now(timezone.utc)
        rule = self._make_rule(cooldown_seconds=60)
        rule.last_fired_at = now - timedelta(seconds=30)
        assert not rule.is_cooled_down(now=now)

    def test_cooldown_elapsed(self):
        now = datetime.now(timezone.utc)
        rule = self._make_rule(cooldown_seconds=60)
        rule.last_fired_at = now - timedelta(seconds=90)
        assert rule.is_cooled_down(now=now)

    def test_cooldown_zero_always_ready(self):
        now = datetime.now(timezone.utc)
        rule = self._make_rule(cooldown_seconds=0)
        rule.last_fired_at = now
        assert rule.is_cooled_down(now=now)

    def test_render_message(self):
        rule = self._make_rule(template="Alert: {event} on {device_id}")
        msg = rule.render_message(event="node_offline", device_id="esp-001")
        assert msg == "Alert: node_offline on esp-001"

    def test_render_message_missing_placeholder(self):
        rule = self._make_rule(template="Alert: {event} — {unknown}")
        msg = rule.render_message(event="test")
        assert msg == "Alert: test — {unknown}"

    def test_record_firing(self):
        rule = self._make_rule()
        assert rule.fire_count == 0
        assert rule.last_fired_at is None

        now = datetime.now(timezone.utc)
        rule.record_firing(now=now)
        assert rule.fire_count == 1
        assert rule.last_fired_at == now
        assert rule.updated_at == now

        rule.record_firing()
        assert rule.fire_count == 2

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        rule = self._make_rule(created_at=now)
        d = rule.to_dict()
        assert d["rule_id"] == "test-rule"
        assert d["severity_filter"] == "warning"
        assert d["channels"] == ["websocket"]
        assert d["created_at"] == now.isoformat()
        assert d["enabled"] is True

    def test_from_dict_roundtrip(self):
        now = datetime.now(timezone.utc)
        rule = self._make_rule(
            created_at=now,
            channels=[NotificationChannel.MQTT, NotificationChannel.LOG],
            device_filter=["dev-001"],
        )
        rule.record_firing(now=now)

        d = rule.to_dict()
        restored = NotificationRule.from_dict(d)

        assert restored.rule_id == rule.rule_id
        assert restored.name == rule.name
        assert restored.trigger_event == rule.trigger_event
        assert restored.severity_filter == rule.severity_filter
        assert restored.channels == rule.channels
        assert restored.cooldown_seconds == rule.cooldown_seconds
        assert restored.template == rule.template
        assert restored.enabled == rule.enabled
        assert restored.device_filter == rule.device_filter
        assert restored.fire_count == rule.fire_count
        assert restored.created_at == rule.created_at
        assert restored.last_fired_at == rule.last_fired_at

    def test_from_dict_minimal(self):
        rule = NotificationRule.from_dict({"rule_id": "min"})
        assert rule.rule_id == "min"
        assert rule.trigger_event == "*"
        assert rule.severity_filter == NotificationSeverity.INFO
        assert rule.channels == [NotificationChannel.WEBSOCKET]


# ------------------------------------------------------------------
# Default rules
# ------------------------------------------------------------------


class TestDefaultRules:
    def test_default_rules_exist(self):
        assert len(DEFAULT_RULES) >= 3

    def test_default_rules_are_valid(self):
        for rule in DEFAULT_RULES:
            assert rule.rule_id
            assert isinstance(rule.severity_filter, NotificationSeverity)
            assert len(rule.channels) > 0
            assert rule.enabled is True

    def test_critical_rule_catches_all_critical(self):
        critical_rule = next(r for r in DEFAULT_RULES if r.rule_id == "default-critical")
        assert critical_rule.trigger_event == "*"
        assert critical_rule.severity_filter == NotificationSeverity.CRITICAL
        assert critical_rule.matches_event("anything", NotificationSeverity.CRITICAL)
        assert not critical_rule.matches_event("anything", NotificationSeverity.WARNING)
