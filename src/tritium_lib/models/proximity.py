# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Proximity alert models for entity-to-entity distance monitoring.

Defines the shared contract for dynamic proximity alerting between
tracked targets. When two targets of differing alliances approach
within a configurable threshold, a ProximityAlert is emitted.

Used by tritium-sc's proximity monitor and the automation engine.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProximityAlertType(str, Enum):
    """Type of proximity alert."""
    APPROACH = "approach"          # Targets moving closer
    BREACH = "breach"              # Threshold crossed
    DWELL = "dwell"                # Staying within threshold
    DEPARTURE = "departure"       # Targets moving apart after breach


class ProximitySeverity(str, Enum):
    """Severity of a proximity alert."""
    LOW = "low"              # >75% of threshold
    MEDIUM = "medium"        # 50-75% of threshold
    HIGH = "high"            # 25-50% of threshold
    CRITICAL = "critical"    # <25% of threshold


class AlliancePair(str, Enum):
    """Alliance pair combinations for proximity rules."""
    HOSTILE_FRIENDLY = "hostile_friendly"
    HOSTILE_UNKNOWN = "hostile_unknown"
    UNKNOWN_FRIENDLY = "unknown_friendly"
    ANY_DIFFERENT = "any_different"        # Any two targets with different alliances


@dataclass
class ProximityAlert:
    """An alert fired when two targets come within a configured distance.

    Attributes
    ----------
    alert_id:
        Unique identifier for this alert instance.
    target_a_id:
        First target's ID.
    target_b_id:
        Second target's ID.
    target_a_alliance:
        Alliance of target A (friendly, hostile, unknown).
    target_b_alliance:
        Alliance of target B.
    distance_m:
        Current distance in meters between the two targets.
    threshold_m:
        The configured threshold that triggered this alert.
    alert_type:
        Type of proximity event (approach, breach, dwell, departure).
    severity:
        Severity classification based on distance vs threshold ratio.
    timestamp:
        When the alert was generated (epoch seconds).
    position_a:
        Position of target A as (x, y) in local coordinates.
    position_b:
        Position of target B as (x, y) in local coordinates.
    rule_id:
        ID of the proximity rule that triggered this alert.
    acknowledged:
        Whether an operator has acknowledged this alert.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_a_id: str = ""
    target_b_id: str = ""
    target_a_alliance: str = ""
    target_b_alliance: str = ""
    distance_m: float = 0.0
    threshold_m: float = 10.0
    alert_type: str = "breach"
    severity: str = "medium"
    timestamp: float = field(default_factory=time.time)
    position_a: tuple[float, float] = (0.0, 0.0)
    position_b: tuple[float, float] = (0.0, 0.0)
    rule_id: str = ""
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "target_a_id": self.target_a_id,
            "target_b_id": self.target_b_id,
            "target_a_alliance": self.target_a_alliance,
            "target_b_alliance": self.target_b_alliance,
            "distance_m": round(self.distance_m, 2),
            "threshold_m": self.threshold_m,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "position_a": list(self.position_a),
            "position_b": list(self.position_b),
            "rule_id": self.rule_id,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProximityAlert:
        pos_a = d.get("position_a", [0.0, 0.0])
        pos_b = d.get("position_b", [0.0, 0.0])
        return cls(
            alert_id=d.get("alert_id", str(uuid.uuid4())),
            target_a_id=d.get("target_a_id", ""),
            target_b_id=d.get("target_b_id", ""),
            target_a_alliance=d.get("target_a_alliance", ""),
            target_b_alliance=d.get("target_b_alliance", ""),
            distance_m=d.get("distance_m", 0.0),
            threshold_m=d.get("threshold_m", 10.0),
            alert_type=d.get("alert_type", "breach"),
            severity=d.get("severity", "medium"),
            timestamp=d.get("timestamp", time.time()),
            position_a=tuple(pos_a) if isinstance(pos_a, list) else pos_a,
            position_b=tuple(pos_b) if isinstance(pos_b, list) else pos_b,
            rule_id=d.get("rule_id", ""),
            acknowledged=d.get("acknowledged", False),
        )


@dataclass
class ProximityRule:
    """A rule defining when to generate proximity alerts.

    Attributes
    ----------
    rule_id:
        Unique identifier for this rule.
    name:
        Human-readable name for the rule.
    alliance_pair:
        Which alliance combinations this rule applies to.
    threshold_m:
        Distance threshold in meters — alert when targets are closer.
    cooldown_s:
        Minimum seconds between alerts for the same target pair.
    enabled:
        Whether this rule is active.
    notify_on_approach:
        If True, also fire alerts when targets are approaching threshold
        (not just when breached).
    approach_factor:
        Multiplier for approach detection (e.g., 1.5 = alert at 1.5x threshold
        if targets are closing).
    """

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Proximity Alert"
    alliance_pair: str = "hostile_friendly"
    threshold_m: float = 10.0
    cooldown_s: float = 60.0
    enabled: bool = True
    notify_on_approach: bool = False
    approach_factor: float = 1.5

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "alliance_pair": self.alliance_pair,
            "threshold_m": self.threshold_m,
            "cooldown_s": self.cooldown_s,
            "enabled": self.enabled,
            "notify_on_approach": self.notify_on_approach,
            "approach_factor": self.approach_factor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProximityRule:
        return cls(
            rule_id=d.get("rule_id", str(uuid.uuid4())),
            name=d.get("name", "Proximity Alert"),
            alliance_pair=d.get("alliance_pair", "hostile_friendly"),
            threshold_m=d.get("threshold_m", 10.0),
            cooldown_s=d.get("cooldown_s", 60.0),
            enabled=d.get("enabled", True),
            notify_on_approach=d.get("notify_on_approach", False),
            approach_factor=d.get("approach_factor", 1.5),
        )

    def matches_alliance(self, alliance_a: str, alliance_b: str) -> bool:
        """Check if a pair of alliances matches this rule."""
        if self.alliance_pair == AlliancePair.ANY_DIFFERENT.value:
            return alliance_a != alliance_b

        pair = self.alliance_pair
        combo = f"{alliance_a}_{alliance_b}"
        reverse = f"{alliance_b}_{alliance_a}"
        return pair == combo or pair == reverse


def classify_proximity_severity(distance_m: float, threshold_m: float) -> str:
    """Classify severity based on how close distance is to threshold.

    Returns
    -------
    str:
        One of: "low", "medium", "high", "critical"
    """
    if threshold_m <= 0:
        return ProximitySeverity.CRITICAL.value

    ratio = distance_m / threshold_m
    if ratio < 0.25:
        return ProximitySeverity.CRITICAL.value
    elif ratio < 0.50:
        return ProximitySeverity.HIGH.value
    elif ratio < 0.75:
        return ProximitySeverity.MEDIUM.value
    return ProximitySeverity.LOW.value


# Default proximity rules
DEFAULT_PROXIMITY_RULES: list[ProximityRule] = [
    ProximityRule(
        rule_id="default_hostile_friendly",
        name="Hostile approaching friendly asset",
        alliance_pair=AlliancePair.HOSTILE_FRIENDLY.value,
        threshold_m=10.0,
        cooldown_s=60.0,
        enabled=True,
    ),
    ProximityRule(
        rule_id="default_unknown_friendly",
        name="Unknown target near friendly asset",
        alliance_pair=AlliancePair.UNKNOWN_FRIENDLY.value,
        threshold_m=15.0,
        cooldown_s=120.0,
        enabled=True,
    ),
]
