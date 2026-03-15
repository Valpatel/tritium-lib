# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for autonomous decision models."""
import pytest
from tritium_lib.models.autonomous import (
    AutonomousDecision,
    AutonomousDecisionLog,
    AutonomousDecisionType,
    AutonomousTrigger,
    EdgeAlertRule,
    OverrideState,
)


class TestAutonomousDecision:
    def test_create_default(self):
        d = AutonomousDecision()
        assert d.decision_type == AutonomousDecisionType.ALERT
        assert d.trigger == AutonomousTrigger.UNKNOWN_BLE
        assert d.sc_override == OverrideState.PENDING
        assert d.confidence == 0.5

    def test_create_with_values(self):
        d = AutonomousDecision(
            decision_id="test_001",
            device_id="tritium-43c-001",
            decision_type=AutonomousDecisionType.ESCALATE,
            trigger=AutonomousTrigger.MOTION_DETECTED,
            confidence=0.85,
            action_taken="publish_alert",
            target_id="ble_AA:BB:CC:DD:EE:FF",
            threshold_value=-60.0,
            measured_value=-45.0,
        )
        assert d.decision_id == "test_001"
        assert d.device_id == "tritium-43c-001"
        assert d.decision_type == AutonomousDecisionType.ESCALATE
        assert d.trigger == AutonomousTrigger.MOTION_DETECTED
        assert d.confidence == 0.85

    def test_override_states(self):
        assert OverrideState.PENDING.value == "pending"
        assert OverrideState.CONFIRMED.value == "confirmed"
        assert OverrideState.OVERRIDDEN.value == "overridden"
        assert OverrideState.EXPIRED.value == "expired"

    def test_trigger_types(self):
        triggers = [
            AutonomousTrigger.UNKNOWN_BLE,
            AutonomousTrigger.THREAT_FEED_MATCH,
            AutonomousTrigger.MOTION_DETECTED,
            AutonomousTrigger.ACOUSTIC_EVENT,
            AutonomousTrigger.RSSI_ANOMALY,
            AutonomousTrigger.GEOFENCE_BREACH,
            AutonomousTrigger.SIGNAL_STRENGTH,
            AutonomousTrigger.PATTERN_MATCH,
            AutonomousTrigger.MANUAL,
        ]
        assert len(triggers) == 9

    def test_decision_types(self):
        types = [
            AutonomousDecisionType.ALERT,
            AutonomousDecisionType.CLASSIFY,
            AutonomousDecisionType.ESCALATE,
            AutonomousDecisionType.LOCKDOWN,
            AutonomousDecisionType.EVADE,
            AutonomousDecisionType.REPORT,
        ]
        assert len(types) == 6

    def test_serialization(self):
        d = AutonomousDecision(
            decision_id="ser_001",
            device_id="dev_001",
            confidence=0.9,
        )
        data = d.model_dump()
        assert data["decision_id"] == "ser_001"
        assert data["device_id"] == "dev_001"
        assert data["confidence"] == 0.9
        assert data["sc_override"] == "pending"

    def test_trigger_data(self):
        d = AutonomousDecision(
            trigger_data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -45},
        )
        assert d.trigger_data["mac"] == "AA:BB:CC:DD:EE:FF"
        assert d.trigger_data["rssi"] == -45


class TestEdgeAlertRule:
    def test_create_default(self):
        r = EdgeAlertRule()
        assert r.enabled is True
        assert r.trigger == AutonomousTrigger.UNKNOWN_BLE
        assert r.decision_type == AutonomousDecisionType.ALERT
        assert r.cooldown_seconds == 30.0

    def test_create_motion_rule(self):
        r = EdgeAlertRule(
            rule_id="motion_01",
            name="RF Motion Alert",
            trigger=AutonomousTrigger.MOTION_DETECTED,
            threshold=8.0,
            min_confidence=0.6,
            cooldown_seconds=60.0,
        )
        assert r.rule_id == "motion_01"
        assert r.threshold == 8.0
        assert r.min_confidence == 0.6

    def test_serialization(self):
        r = EdgeAlertRule(rule_id="test", name="Test Rule")
        data = r.model_dump()
        assert data["rule_id"] == "test"
        assert data["enabled"] is True


class TestAutonomousDecisionLog:
    def test_create_empty(self):
        log = AutonomousDecisionLog(device_id="dev_001")
        assert log.total_decisions == 0
        assert log.accuracy_rate == 0.0

    def test_compute_accuracy(self):
        log = AutonomousDecisionLog(
            device_id="dev_001",
            confirmed=8,
            overridden=2,
        )
        rate = log.compute_accuracy()
        assert rate == 0.8
        assert log.accuracy_rate == 0.8

    def test_compute_accuracy_no_reviews(self):
        log = AutonomousDecisionLog(device_id="dev_001")
        rate = log.compute_accuracy()
        assert rate == 0.0

    def test_with_decisions(self):
        decisions = [
            AutonomousDecision(decision_id=f"d_{i}", device_id="dev_001")
            for i in range(5)
        ]
        log = AutonomousDecisionLog(
            device_id="dev_001",
            total_decisions=5,
            decisions=decisions,
        )
        assert len(log.decisions) == 5


class TestImports:
    """Verify autonomous models are accessible from the main models package."""

    def test_import_from_models(self):
        from tritium_lib.models import (
            AutonomousDecision,
            AutonomousDecisionLog,
            AutonomousDecisionType,
            AutonomousTrigger,
            EdgeAlertRule,
            OverrideState,
        )
        assert AutonomousDecision is not None
        assert AutonomousDecisionLog is not None
        assert AutonomousDecisionType is not None
        assert AutonomousTrigger is not None
        assert EdgeAlertRule is not None
        assert OverrideState is not None
