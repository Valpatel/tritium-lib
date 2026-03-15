# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for dwell event models."""

from datetime import datetime, timezone

from tritium_lib.models.dwell import (
    DwellEvent,
    DwellSeverity,
    DwellState,
    DWELL_RADIUS_M,
    DWELL_THRESHOLD_S,
    classify_dwell_severity,
)


def test_dwell_event_basic():
    """DwellEvent can be created with minimal fields."""
    d = DwellEvent(target_id="ble_aa:bb:cc", duration_s=600.0)
    assert d.target_id == "ble_aa:bb:cc"
    assert d.duration_s == 600.0
    assert d.state == DwellState.ACTIVE
    assert d.severity == DwellSeverity.NORMAL


def test_dwell_event_full():
    """DwellEvent with all fields."""
    d = DwellEvent(
        target_id="ble_aa:bb:cc",
        event_id="dwell_abc123",
        position_lat=40.7128,
        position_lng=-74.0060,
        position_x=100.0,
        position_y=200.0,
        start_time=datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc),
        duration_s=1800.0,
        zone_id="zone_1",
        zone_name="Front Yard",
        state=DwellState.ENDED,
        severity=DwellSeverity.EXTENDED,
        radius_m=15.0,
        target_name="Unknown Phone",
        target_alliance="unknown",
        target_type="phone",
    )
    assert d.event_id == "dwell_abc123"
    assert d.position_lat == 40.7128
    assert d.zone_name == "Front Yard"
    assert d.state == DwellState.ENDED


def test_dwell_duration_display():
    """Duration display formatting."""
    d = DwellEvent(target_id="t1", duration_s=300.0)
    assert d.duration_display == "5m 0s"

    d2 = DwellEvent(target_id="t2", duration_s=3661.0)
    assert d2.duration_display == "1h 1m"

    d3 = DwellEvent(target_id="t3", duration_s=45.0)
    assert d3.duration_display == "0m 45s"


def test_dwell_duration_minutes():
    """Duration in minutes property."""
    d = DwellEvent(target_id="t1", duration_s=600.0)
    assert d.duration_minutes == 10.0


def test_classify_dwell_severity():
    """Severity classification by duration."""
    assert classify_dwell_severity(200) == DwellSeverity.NORMAL
    assert classify_dwell_severity(600) == DwellSeverity.NORMAL
    assert classify_dwell_severity(1000) == DwellSeverity.EXTENDED
    assert classify_dwell_severity(3700) == DwellSeverity.PROLONGED
    assert classify_dwell_severity(15000) == DwellSeverity.CRITICAL


def test_dwell_constants():
    """Default constants are sensible."""
    assert DWELL_THRESHOLD_S == 300
    assert DWELL_RADIUS_M == 15.0


def test_dwell_model_dump():
    """DwellEvent can be serialized to dict."""
    d = DwellEvent(
        target_id="ble_aa:bb:cc",
        event_id="dwell_test",
        duration_s=600.0,
        state=DwellState.ACTIVE,
        severity=DwellSeverity.NORMAL,
    )
    data = d.model_dump()
    assert data["target_id"] == "ble_aa:bb:cc"
    assert data["state"] == "active"
    assert data["severity"] == "normal"


def test_dwell_model_roundtrip():
    """DwellEvent can roundtrip through dict serialization."""
    d = DwellEvent(
        target_id="ble_aa:bb:cc",
        event_id="dwell_test",
        duration_s=600.0,
        start_time=datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc),
    )
    data = d.model_dump(mode="json")
    d2 = DwellEvent(**data)
    assert d2.target_id == d.target_id
    assert d2.duration_s == d.duration_s


def test_dwell_state_enum():
    """DwellState enum values."""
    assert DwellState.ACTIVE == "active"
    assert DwellState.ENDED == "ended"
    assert DwellState.EXPIRED == "expired"


def test_dwell_severity_enum():
    """DwellSeverity enum values."""
    assert DwellSeverity.NORMAL == "normal"
    assert DwellSeverity.EXTENDED == "extended"
    assert DwellSeverity.PROLONGED == "prolonged"
    assert DwellSeverity.CRITICAL == "critical"


def test_dwell_import_from_init():
    """DwellEvent can be imported from tritium_lib.models."""
    from tritium_lib.models import DwellEvent as DE, classify_dwell_severity as cds
    assert DE is DwellEvent
    assert cds is classify_dwell_severity
