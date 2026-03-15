# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for proximity alert models."""

import pytest

from tritium_lib.models.proximity import (
    AlliancePair,
    ProximityAlert,
    ProximityAlertType,
    ProximityRule,
    ProximitySeverity,
    classify_proximity_severity,
    DEFAULT_PROXIMITY_RULES,
)


class TestProximityAlert:
    """Tests for ProximityAlert dataclass."""

    def test_create_default(self):
        alert = ProximityAlert()
        assert alert.alert_id  # should have UUID
        assert alert.distance_m == 0.0
        assert alert.threshold_m == 10.0
        assert alert.alert_type == "breach"
        assert alert.severity == "medium"
        assert alert.acknowledged is False

    def test_to_dict_roundtrip(self):
        alert = ProximityAlert(
            target_a_id="ble_aa:bb:cc",
            target_b_id="mesh_node1",
            target_a_alliance="hostile",
            target_b_alliance="friendly",
            distance_m=5.5,
            threshold_m=10.0,
            alert_type="breach",
            severity="high",
            position_a=(10.0, 20.0),
            position_b=(15.0, 25.0),
            rule_id="rule_1",
        )
        d = alert.to_dict()
        assert d["target_a_id"] == "ble_aa:bb:cc"
        assert d["distance_m"] == 5.5
        assert d["position_a"] == [10.0, 20.0]
        assert d["position_b"] == [15.0, 25.0]

        # Roundtrip
        restored = ProximityAlert.from_dict(d)
        assert restored.target_a_id == alert.target_a_id
        assert restored.distance_m == alert.distance_m
        assert restored.rule_id == alert.rule_id

    def test_from_dict_defaults(self):
        alert = ProximityAlert.from_dict({})
        assert alert.distance_m == 0.0
        assert alert.threshold_m == 10.0


class TestProximityRule:
    """Tests for ProximityRule dataclass."""

    def test_create_default(self):
        rule = ProximityRule()
        assert rule.threshold_m == 10.0
        assert rule.cooldown_s == 60.0
        assert rule.enabled is True

    def test_matches_alliance_hostile_friendly(self):
        rule = ProximityRule(alliance_pair="hostile_friendly")
        assert rule.matches_alliance("hostile", "friendly") is True
        assert rule.matches_alliance("friendly", "hostile") is True
        assert rule.matches_alliance("friendly", "friendly") is False
        assert rule.matches_alliance("hostile", "hostile") is False

    def test_matches_alliance_any_different(self):
        rule = ProximityRule(alliance_pair="any_different")
        assert rule.matches_alliance("hostile", "friendly") is True
        assert rule.matches_alliance("unknown", "friendly") is True
        assert rule.matches_alliance("friendly", "friendly") is False

    def test_to_dict_roundtrip(self):
        rule = ProximityRule(
            name="Test Rule",
            alliance_pair="hostile_friendly",
            threshold_m=25.0,
            cooldown_s=120.0,
        )
        d = rule.to_dict()
        restored = ProximityRule.from_dict(d)
        assert restored.name == "Test Rule"
        assert restored.threshold_m == 25.0
        assert restored.cooldown_s == 120.0


class TestClassifyProximitySeverity:
    """Tests for severity classification."""

    def test_critical(self):
        assert classify_proximity_severity(2.0, 10.0) == "critical"

    def test_high(self):
        assert classify_proximity_severity(3.0, 10.0) == "high"

    def test_medium(self):
        assert classify_proximity_severity(6.0, 10.0) == "medium"

    def test_low(self):
        assert classify_proximity_severity(8.0, 10.0) == "low"

    def test_zero_threshold(self):
        assert classify_proximity_severity(5.0, 0.0) == "critical"


class TestDefaultRules:
    """Tests for default proximity rules."""

    def test_default_rules_exist(self):
        assert len(DEFAULT_PROXIMITY_RULES) >= 1

    def test_hostile_friendly_rule(self):
        rule = DEFAULT_PROXIMITY_RULES[0]
        assert rule.alliance_pair == "hostile_friendly"
        assert rule.threshold_m == 10.0
        assert rule.enabled is True


class TestEnums:
    """Tests for proximity enums."""

    def test_alert_types(self):
        assert ProximityAlertType.APPROACH.value == "approach"
        assert ProximityAlertType.BREACH.value == "breach"
        assert ProximityAlertType.DEPARTURE.value == "departure"

    def test_severity_levels(self):
        assert ProximitySeverity.LOW.value == "low"
        assert ProximitySeverity.CRITICAL.value == "critical"

    def test_alliance_pairs(self):
        assert AlliancePair.HOSTILE_FRIENDLY.value == "hostile_friendly"
        assert AlliancePair.ANY_DIFFERENT.value == "any_different"
