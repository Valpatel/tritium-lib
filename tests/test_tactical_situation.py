# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TacticalSituation model."""

from tritium_lib.models.tactical_situation import (
    AmyStatus,
    FleetHealth,
    TargetCountsSummary,
    TacticalSituation,
    ThreatLevel,
)


def test_threat_level_enum():
    assert ThreatLevel.GREEN.value == "green"
    assert ThreatLevel.BLACK.value == "black"


def test_amy_status_enum():
    assert AmyStatus.THINKING.value == "thinking"


def test_fleet_health_defaults():
    fh = FleetHealth()
    assert fh.health_pct == 100.0


def test_fleet_health_pct():
    fh = FleetHealth(total_devices=10, online=7, offline=2, degraded=1)
    assert fh.health_pct == 80.0


def test_fleet_health_roundtrip():
    fh = FleetHealth(total_devices=5, online=3, offline=1, degraded=1, avg_battery_pct=72.5)
    d = fh.to_dict()
    fh2 = FleetHealth.from_dict(d)
    assert fh2.total_devices == 5
    assert fh2.avg_battery_pct == 72.5


def test_target_counts_summary_roundtrip():
    tc = TargetCountsSummary(total=20, friendly=5, hostile=3, unknown=12, new_last_hour=4)
    d = tc.to_dict()
    tc2 = TargetCountsSummary.from_dict(d)
    assert tc2.total == 20
    assert tc2.new_last_hour == 4


def test_tactical_situation_defaults():
    sit = TacticalSituation()
    assert sit.threat_level == ThreatLevel.GREEN
    assert sit.amy_status == AmyStatus.ONLINE
    assert sit.timestamp is not None
    assert not sit.is_critical


def test_escalate_deescalate():
    sit = TacticalSituation(threat_level=ThreatLevel.GREEN)
    sit.escalate()
    assert sit.threat_level == ThreatLevel.YELLOW
    sit.escalate()
    assert sit.threat_level == ThreatLevel.ORANGE
    sit.escalate()
    assert sit.threat_level == ThreatLevel.RED
    assert sit.is_critical
    sit.escalate()
    assert sit.threat_level == ThreatLevel.BLACK
    # Can't escalate past BLACK
    sit.escalate()
    assert sit.threat_level == ThreatLevel.BLACK
    # Deescalate
    sit.deescalate()
    assert sit.threat_level == ThreatLevel.RED
    # Back to green
    sit.deescalate()
    sit.deescalate()
    sit.deescalate()
    assert sit.threat_level == ThreatLevel.GREEN
    # Can't deescalate past GREEN
    sit.deescalate()
    assert sit.threat_level == ThreatLevel.GREEN


def test_generate_sitrep():
    sit = TacticalSituation(
        threat_level=ThreatLevel.ORANGE,
        target_counts=TargetCountsSummary(total=15, friendly=5, hostile=3, unknown=7, new_last_hour=2),
        active_alerts=4,
        active_investigations=1,
        fleet_health=FleetHealth(total_devices=8, online=6, offline=2),
        amy_status=AmyStatus.ALERTING,
        summary_text="Multiple hostile contacts detected",
    )
    sitrep = sit.generate_sitrep()
    assert "ORANGE" in sitrep
    assert "15 total" in sitrep
    assert "5F / 3H / 7U" in sitrep
    assert "Active alerts: 4" in sitrep
    assert "6/8 online" in sitrep
    assert "alerting" in sitrep
    assert "Multiple hostile contacts" in sitrep


def test_roundtrip():
    sit = TacticalSituation(
        threat_level=ThreatLevel.RED,
        target_counts=TargetCountsSummary(total=50, hostile=10),
        active_alerts=7,
        fleet_health=FleetHealth(total_devices=20, online=18, degraded=1, offline=1),
        amy_status=AmyStatus.THINKING,
        site_id="alpha",
        notes="Commander notes",
    )
    d = sit.to_dict()
    sit2 = TacticalSituation.from_dict(d)
    assert sit2.threat_level == ThreatLevel.RED
    assert sit2.target_counts.hostile == 10
    assert sit2.active_alerts == 7
    assert sit2.fleet_health.online == 18
    assert sit2.amy_status == AmyStatus.THINKING
    assert sit2.site_id == "alpha"
    assert sit2.notes == "Commander notes"
