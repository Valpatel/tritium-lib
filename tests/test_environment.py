# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the EnvironmentReading model."""

import pytest
from tritium_lib.models.environment import (
    EnvironmentReading,
    EnvironmentSnapshot,
    EnvironmentSource,
)


def test_environment_reading_basic():
    """Test basic environment reading creation."""
    r = EnvironmentReading(
        source_id="mesh_abc123",
        source_type=EnvironmentSource.MESHTASTIC,
        temperature_c=22.5,
        humidity_pct=45.0,
        pressure_hpa=1013.25,
    )
    assert r.source_id == "mesh_abc123"
    assert r.temperature_c == 22.5
    assert r.humidity_pct == 45.0
    assert r.pressure_hpa == 1013.25
    assert r.has_data is True


def test_temperature_f_conversion():
    """Test Celsius to Fahrenheit conversion."""
    r = EnvironmentReading(source_id="test", temperature_c=0.0)
    assert r.temperature_f == pytest.approx(32.0)

    r2 = EnvironmentReading(source_id="test", temperature_c=100.0)
    assert r2.temperature_f == pytest.approx(212.0)

    r3 = EnvironmentReading(source_id="test")
    assert r3.temperature_f is None


def test_has_data_empty():
    """Test has_data returns False when no sensor values."""
    r = EnvironmentReading(source_id="empty")
    assert r.has_data is False


def test_summary_line():
    """Test human-readable summary."""
    r = EnvironmentReading(
        source_id="test",
        temperature_c=22.22,
        humidity_pct=45.0,
        pressure_hpa=1013.25,
    )
    s = r.summary_line()
    assert "72F" in s
    assert "45%" in s
    assert "1013" in s


def test_summary_line_empty():
    r = EnvironmentReading(source_id="test")
    assert r.summary_line() == "No data"


def test_environment_snapshot_aggregation():
    """Test snapshot averaging across multiple sources."""
    snap = EnvironmentSnapshot(readings=[
        EnvironmentReading(source_id="a", temperature_c=20.0, humidity_pct=40.0),
        EnvironmentReading(source_id="b", temperature_c=24.0, humidity_pct=50.0),
    ])
    assert snap.avg_temperature_c == pytest.approx(22.0)
    assert snap.avg_humidity_pct == pytest.approx(45.0)


def test_environment_snapshot_empty():
    snap = EnvironmentSnapshot()
    assert snap.avg_temperature_c is None
    assert snap.avg_humidity_pct is None
    assert "No environment data" in snap.summary()


def test_environment_source_enum():
    """Test all source types."""
    assert EnvironmentSource.MESHTASTIC == "meshtastic"
    assert EnvironmentSource.EDGE_DEVICE == "edge_device"
    assert EnvironmentSource.WEATHER_API == "weather_api"


def test_serialization_roundtrip():
    """Test JSON serialization roundtrip."""
    r = EnvironmentReading(
        source_id="mesh_abc",
        source_type=EnvironmentSource.MESHTASTIC,
        temperature_c=22.5,
        humidity_pct=45.0,
        pressure_hpa=1013.25,
    )
    data = r.model_dump()
    r2 = EnvironmentReading(**data)
    assert r2.source_id == r.source_id
    assert r2.temperature_c == r.temperature_c


def test_quality_bounds():
    """Test quality field validates bounds."""
    r = EnvironmentReading(source_id="test", quality=0.5)
    assert r.quality == 0.5

    with pytest.raises(Exception):
        EnvironmentReading(source_id="test", quality=1.5)
