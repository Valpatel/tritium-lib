# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Autonomous decision models for edge-initiated actions.

When edge devices detect high-threat conditions locally (unknown BLE in
restricted zone, motion detected, acoustic event), they can make autonomous
decisions without waiting for SC classification. These models track those
decisions so SC can audit, validate, and optionally override them.

Decision types:
  - ALERT: edge published an immediate alert
  - CLASSIFY: edge classified a target locally
  - ESCALATE: edge escalated threat level autonomously
  - LOCKDOWN: edge triggered local lockdown (LED, siren, etc.)
  - EVADE: edge initiated evasive maneuver (mobile units)
  - REPORT: edge sent a priority report to SC

Override states:
  - PENDING: SC has not yet reviewed
  - CONFIRMED: SC agrees with edge decision
  - OVERRIDDEN: SC disagrees and has issued correction
  - EXPIRED: decision aged out without SC review
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AutonomousDecisionType(str, Enum):
    """Types of autonomous decisions an edge device can make."""
    ALERT = "alert"
    CLASSIFY = "classify"
    ESCALATE = "escalate"
    LOCKDOWN = "lockdown"
    EVADE = "evade"
    REPORT = "report"


class AutonomousTrigger(str, Enum):
    """What triggered the autonomous decision."""
    UNKNOWN_BLE = "unknown_ble"
    THREAT_FEED_MATCH = "threat_feed_match"
    MOTION_DETECTED = "motion_detected"
    ACOUSTIC_EVENT = "acoustic_event"
    RSSI_ANOMALY = "rssi_anomaly"
    GEOFENCE_BREACH = "geofence_breach"
    SIGNAL_STRENGTH = "signal_strength"
    PATTERN_MATCH = "pattern_match"
    MANUAL = "manual"


class OverrideState(str, Enum):
    """SC override status for an autonomous decision."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    OVERRIDDEN = "overridden"
    EXPIRED = "expired"


class AutonomousDecision(BaseModel):
    """A decision made autonomously by an edge device.

    Edge devices publish these when they take local action without
    waiting for SC. SC receives them, logs them, and can confirm
    or override the decision.
    """
    decision_id: str = ""
    device_id: str = ""
    decision_type: AutonomousDecisionType = AutonomousDecisionType.ALERT
    trigger: AutonomousTrigger = AutonomousTrigger.UNKNOWN_BLE
    confidence: float = 0.5
    action_taken: str = ""
    description: str = ""

    # What triggered it
    target_id: str = ""
    trigger_data: dict = Field(default_factory=dict)

    # SC oversight
    sc_override: OverrideState = OverrideState.PENDING
    override_reason: str = ""
    override_by: str = ""
    override_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Thresholds that triggered this decision
    threshold_value: float = 0.0
    measured_value: float = 0.0


class EdgeAlertRule(BaseModel):
    """A local threshold rule stored on edge devices (NVS).

    Defines conditions under which the edge device should autonomously
    publish an alert without waiting for SC classification.
    """
    rule_id: str = ""
    name: str = ""
    enabled: bool = True

    # Trigger conditions
    trigger: AutonomousTrigger = AutonomousTrigger.UNKNOWN_BLE
    decision_type: AutonomousDecisionType = AutonomousDecisionType.ALERT
    threshold: float = 0.0
    min_confidence: float = 0.5

    # Action configuration
    action: str = "publish_alert"
    mqtt_topic_suffix: str = "alert/autonomous"
    cooldown_seconds: float = 30.0

    # Metadata
    description: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class AutonomousDecisionLog(BaseModel):
    """Aggregated log of autonomous decisions from a device."""
    device_id: str = ""
    total_decisions: int = 0
    confirmed: int = 0
    overridden: int = 0
    pending: int = 0
    expired: int = 0
    accuracy_rate: float = 0.0
    decisions: list[AutonomousDecision] = Field(default_factory=list)
    last_decision_at: Optional[datetime] = None

    def compute_accuracy(self) -> float:
        """Compute accuracy as confirmed / (confirmed + overridden)."""
        reviewed = self.confirmed + self.overridden
        if reviewed == 0:
            return 0.0
        self.accuracy_rate = self.confirmed / reviewed
        return self.accuracy_rate
