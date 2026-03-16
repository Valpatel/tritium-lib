# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AlertRule models."""

from datetime import datetime, timezone, timedelta

import pytest

from tritium_lib.models.alert_rules import (
    AlertCondition,
    AlertRule,
    AlertTrigger,
    ConditionOperator,
    DEFAULT_ALERT_RULES,
)
from tritium_lib.models.notification_rules import (
    NotificationChannel,
    NotificationSeverity,
)


class TestAlertCondition:
    def test_equals(self):
        c = AlertCondition(field="alliance", operator=ConditionOperator.EQUALS, value="hostile")
        assert c.evaluate({"alliance": "hostile"})
        assert not c.evaluate({"alliance": "friendly"})

    def test_not_equals(self):
        c = AlertCondition(field="status", operator=ConditionOperator.NOT_EQUALS, value="online")
        assert c.evaluate({"status": "offline"})
        assert not c.evaluate({"status": "online"})

    def test_greater_than(self):
        c = AlertCondition(field="rssi", operator=ConditionOperator.GREATER_THAN, value=-70)
        assert c.evaluate({"rssi": -50})
        assert not c.evaluate({"rssi": -80})

    def test_less_than(self):
        c = AlertCondition(field="battery_level", operator=ConditionOperator.LESS_THAN, value=10)
        assert c.evaluate({"battery_level": 5})
        assert not c.evaluate({"battery_level": 50})

    def test_gte_lte(self):
        c = AlertCondition(field="speed", operator=ConditionOperator.GREATER_EQUAL, value=5.0)
        assert c.evaluate({"speed": 5.0})
        assert c.evaluate({"speed": 10.0})
        assert not c.evaluate({"speed": 4.9})

    def test_contains(self):
        c = AlertCondition(field="name", operator=ConditionOperator.CONTAINS, value="Phone")
        assert c.evaluate({"name": "iPhone 15"})
        assert not c.evaluate({"name": "Galaxy Watch"})

    def test_in_list(self):
        c = AlertCondition(field="type", operator=ConditionOperator.IN_LIST, value=["phone", "watch"])
        assert c.evaluate({"type": "phone"})
        assert not c.evaluate({"type": "laptop"})

    def test_missing_field(self):
        c = AlertCondition(field="missing", operator=ConditionOperator.EQUALS, value="x")
        assert not c.evaluate({"other": "y"})

    def test_regex(self):
        c = AlertCondition(field="mac", operator=ConditionOperator.REGEX, value=r"^AA:BB")
        assert c.evaluate({"mac": "AA:BB:CC:DD:EE:FF"})
        assert not c.evaluate({"mac": "11:22:33:44:55:66"})

    def test_serialization(self):
        c = AlertCondition(field="rssi", operator=ConditionOperator.LESS_THAN, value=-70)
        d = c.to_dict()
        assert d["field"] == "rssi"
        assert d["operator"] == "lt"
        c2 = AlertCondition.from_dict(d)
        assert c2.field == "rssi"
        assert c2.operator == ConditionOperator.LESS_THAN


class TestAlertRule:
    def _make_rule(self, **kwargs):
        defaults = {
            "rule_id": "test-rule",
            "name": "Test Rule",
            "trigger": AlertTrigger.TARGET_NEW,
            "severity": NotificationSeverity.WARNING,
        }
        defaults.update(kwargs)
        return AlertRule(**defaults)

    def test_matches_trigger(self):
        r = self._make_rule()
        assert r.matches("target_new", {})
        assert not r.matches("device_offline", {})

    def test_matches_disabled(self):
        r = self._make_rule(enabled=False)
        assert not r.matches("target_new", {})

    def test_matches_conditions(self):
        r = self._make_rule(conditions=[
            AlertCondition(field="alliance", operator=ConditionOperator.EQUALS, value="hostile"),
            AlertCondition(field="rssi", operator=ConditionOperator.GREATER_THAN, value=-70),
        ])
        assert r.matches("target_new", {"alliance": "hostile", "rssi": -50})
        assert not r.matches("target_new", {"alliance": "hostile", "rssi": -80})
        assert not r.matches("target_new", {"alliance": "friendly", "rssi": -50})

    def test_zone_filter(self):
        r = self._make_rule(zone_filter=["zone-1", "zone-2"])
        assert r.matches("target_new", {"zone_id": "zone-1"})
        assert not r.matches("target_new", {"zone_id": "zone-3"})

    def test_target_filter(self):
        r = self._make_rule(target_filter=["hostile"])
        assert r.matches("target_new", {"alliance": "hostile"})
        assert not r.matches("target_new", {"alliance": "friendly"})

    def test_cooldown(self):
        r = self._make_rule(cooldown_seconds=60)
        assert r.is_cooled_down()

        now = datetime.now(timezone.utc)
        r.record_firing(now)
        assert not r.is_cooled_down(now + timedelta(seconds=30))
        assert r.is_cooled_down(now + timedelta(seconds=61))

    def test_cooldown_zero(self):
        r = self._make_rule(cooldown_seconds=0)
        r.record_firing()
        assert r.is_cooled_down()

    def test_render_message(self):
        r = self._make_rule(message_template="Alert: {target_id} in {zone_id}")
        msg = r.render_message(target_id="ble_aa:bb", zone_id="perimeter")
        assert msg == "Alert: ble_aa:bb in perimeter"

    def test_record_firing(self):
        r = self._make_rule()
        assert r.fire_count == 0
        assert r.last_fired_at is None

        r.record_firing()
        assert r.fire_count == 1
        assert r.last_fired_at is not None
        assert r.updated_at is not None

    def test_serialization_roundtrip(self):
        r = self._make_rule(
            conditions=[AlertCondition(field="rssi", operator=ConditionOperator.LESS_THAN, value=-70)],
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.MQTT],
            zone_filter=["z1"],
            target_filter=["hostile"],
            tags=["perimeter"],
        )
        r.record_firing()

        d = r.to_dict()
        assert d["rule_id"] == "test-rule"
        assert d["trigger"] == "target_new"
        assert len(d["conditions"]) == 1

        r2 = AlertRule.from_dict(d)
        assert r2.rule_id == "test-rule"
        assert r2.trigger == AlertTrigger.TARGET_NEW
        assert len(r2.conditions) == 1
        assert r2.conditions[0].operator == ConditionOperator.LESS_THAN
        assert r2.fire_count == 1
        assert r2.last_fired_at is not None


class TestDefaultAlertRules:
    def test_defaults_exist(self):
        assert len(DEFAULT_ALERT_RULES) >= 3

    def test_defaults_valid(self):
        for rule in DEFAULT_ALERT_RULES:
            assert rule.rule_id
            assert rule.trigger is not None
            assert rule.severity is not None
            assert len(rule.channels) > 0
            d = rule.to_dict()
            r2 = AlertRule.from_dict(d)
            assert r2.rule_id == rule.rule_id
