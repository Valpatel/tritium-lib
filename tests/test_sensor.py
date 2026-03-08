# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sensor reading models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tritium_lib.models.sensor import SensorReading


def _utc(hour=12, minute=0, second=0):
    return datetime(2026, 3, 7, hour, minute, second, tzinfo=timezone.utc)


class TestSensorReading:
    def test_create_scalar(self):
        r = SensorReading(
            device_id="dev-1",
            sensor_type="temperature",
            value=23.5,
            unit="celsius",
        )
        assert r.device_id == "dev-1"
        assert r.sensor_type == "temperature"
        assert r.value == 23.5
        assert r.unit == "celsius"
        assert r.quality == 1.0
        assert r.timestamp is None

    def test_create_with_dict_value(self):
        r = SensorReading(
            device_id="dev-2",
            sensor_type="imu",
            value={"accel_x": 0.01, "accel_y": -0.02, "accel_z": 9.81},
            unit="g",
        )
        assert isinstance(r.value, dict)
        assert r.value["accel_z"] == 9.81

    def test_create_with_list_value(self):
        r = SensorReading(
            device_id="dev-3",
            sensor_type="gps",
            value=[37.7749, -122.4194, 15.0],
            unit="deg",
        )
        assert isinstance(r.value, list)
        assert len(r.value) == 3

    def test_quality_bounds(self):
        r_low = SensorReading(
            device_id="dev-4", sensor_type="temp", value=20.0, quality=0.0
        )
        assert r_low.quality == 0.0

        r_high = SensorReading(
            device_id="dev-5", sensor_type="temp", value=20.0, quality=1.0
        )
        assert r_high.quality == 1.0

    def test_default_quality(self):
        r = SensorReading(device_id="dev-6", sensor_type="humidity", value=55.0)
        assert r.quality == 1.0

    def test_default_unit(self):
        r = SensorReading(device_id="dev-7", sensor_type="custom", value=42.0)
        assert r.unit == ""

    def test_with_timestamp(self):
        ts = _utc()
        r = SensorReading(
            device_id="dev-8",
            sensor_type="temperature",
            value=21.0,
            timestamp=ts,
        )
        assert r.timestamp == ts

    def test_serialization(self):
        r = SensorReading(
            device_id="dev-9",
            sensor_type="humidity",
            value=65.2,
            unit="percent",
            quality=0.95,
            timestamp=_utc(),
        )
        d = r.model_dump()
        assert d["device_id"] == "dev-9"
        assert d["sensor_type"] == "humidity"
        assert d["value"] == 65.2
        assert d["quality"] == 0.95

    def test_json_roundtrip_scalar(self):
        r = SensorReading(
            device_id="dev-10",
            sensor_type="temperature",
            value=25.0,
            unit="celsius",
            quality=0.99,
            timestamp=_utc(),
        )
        json_str = r.model_dump_json()
        r2 = SensorReading.model_validate_json(json_str)
        assert r2.device_id == r.device_id
        assert r2.value == r.value
        assert r2.unit == r.unit
        assert r2.quality == r.quality

    def test_json_roundtrip_dict(self):
        r = SensorReading(
            device_id="dev-11",
            sensor_type="imu",
            value={"gyro_x": 1.0, "gyro_y": 2.0, "gyro_z": 3.0},
            unit="deg/s",
        )
        json_str = r.model_dump_json()
        r2 = SensorReading.model_validate_json(json_str)
        assert r2.value == r.value

    def test_json_roundtrip_list(self):
        r = SensorReading(
            device_id="dev-12",
            sensor_type="spectrum",
            value=[100.0, 200.0, 300.0, 400.0],
        )
        json_str = r.model_dump_json()
        r2 = SensorReading.model_validate_json(json_str)
        assert r2.value == r.value

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            SensorReading(device_id="dev-13", sensor_type="temp")  # missing value
        with pytest.raises(ValidationError):
            SensorReading(device_id="dev-14", value=20.0)  # missing sensor_type
        with pytest.raises(ValidationError):
            SensorReading(sensor_type="temp", value=20.0)  # missing device_id

    def test_from_dict(self):
        data = {
            "device_id": "dev-15",
            "sensor_type": "pressure",
            "value": 1013.25,
            "unit": "hPa",
        }
        r = SensorReading.model_validate(data)
        assert r.sensor_type == "pressure"
        assert r.value == 1013.25

    def test_negative_value(self):
        r = SensorReading(
            device_id="dev-16",
            sensor_type="temperature",
            value=-40.0,
            unit="celsius",
        )
        assert r.value == -40.0

    def test_zero_value(self):
        r = SensorReading(
            device_id="dev-17",
            sensor_type="temperature",
            value=0.0,
            unit="celsius",
        )
        assert r.value == 0.0

    def test_empty_dict_value(self):
        r = SensorReading(
            device_id="dev-18",
            sensor_type="custom",
            value={},
        )
        assert r.value == {}

    def test_empty_list_value(self):
        r = SensorReading(
            device_id="dev-19",
            sensor_type="custom",
            value=[],
        )
        assert r.value == []
