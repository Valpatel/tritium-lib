# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sensor placement and configuration models."""

import math

import pytest

from tritium_lib.models.sensor_config import (
    MountingType,
    SensorArray,
    SensorPlacement,
    SensorPosition,
    SensorStatus,
    SensorType,
)


class TestSensorPlacement:
    """Tests for the SensorPlacement model."""

    def test_create_minimal(self):
        sp = SensorPlacement(sensor_id="cam_01")
        assert sp.sensor_id == "cam_01"
        assert sp.sensor_type == SensorType.UNKNOWN
        assert sp.height_m == 2.0
        assert sp.fov_degrees == 360.0
        assert sp.rotation_degrees == 0.0
        assert sp.coverage_radius_m == 50.0
        assert sp.mounting_type == MountingType.UNKNOWN
        assert sp.status == SensorStatus.UNKNOWN

    def test_create_full(self):
        sp = SensorPlacement(
            sensor_id="ble_scanner_01",
            device_id="node_alpha",
            sensor_type=SensorType.BLE_RADIO,
            position=SensorPosition(latitude=33.123, longitude=-96.456, z=3.0),
            height_m=3.0,
            fov_degrees=360.0,
            rotation_degrees=0.0,
            coverage_radius_m=30.0,
            mounting_type=MountingType.WALL,
            status=SensorStatus.ONLINE,
            frequency_mhz=2400.0,
            tx_power_dbm=4.0,
            sensitivity_dbm=-95.0,
            label="Front Door BLE",
            model="ESP32-S3",
        )
        assert sp.sensor_type == SensorType.BLE_RADIO
        assert sp.mounting_type == MountingType.WALL
        assert sp.status == SensorStatus.ONLINE
        assert sp.position.latitude == 33.123
        assert sp.label == "Front Door BLE"

    def test_omnidirectional(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=360.0)
        assert sp.is_omnidirectional
        assert not sp.is_directional

    def test_directional(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=120.0)
        assert not sp.is_omnidirectional
        assert sp.is_directional

    def test_passive(self):
        sp = SensorPlacement(sensor_id="x", tx_power_dbm=0.0)
        assert sp.is_passive

    def test_active(self):
        sp = SensorPlacement(sensor_id="x", tx_power_dbm=4.0)
        assert not sp.is_passive

    def test_coverage_area_omni(self):
        sp = SensorPlacement(sensor_id="x", coverage_radius_m=10.0, fov_degrees=360.0)
        expected = math.pi * 10.0 * 10.0
        assert abs(sp.coverage_area_m2() - expected) < 0.01

    def test_coverage_area_sector(self):
        sp = SensorPlacement(sensor_id="x", coverage_radius_m=10.0, fov_degrees=90.0)
        expected = math.pi * 10.0 * 10.0 * 0.25
        assert abs(sp.coverage_area_m2() - expected) < 0.01

    def test_contains_bearing_omni(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=360.0)
        assert sp.contains_bearing(0)
        assert sp.contains_bearing(180)
        assert sp.contains_bearing(359)

    def test_contains_bearing_directional(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=90.0, rotation_degrees=0.0)
        assert sp.contains_bearing(0)
        assert sp.contains_bearing(45)
        assert not sp.contains_bearing(90)
        assert not sp.contains_bearing(180)

    def test_contains_bearing_rotated(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=90.0, rotation_degrees=180.0)
        assert sp.contains_bearing(180)
        assert sp.contains_bearing(200)
        assert not sp.contains_bearing(0)
        assert not sp.contains_bearing(90)

    def test_contains_bearing_wraparound(self):
        sp = SensorPlacement(sensor_id="x", fov_degrees=90.0, rotation_degrees=350.0)
        assert sp.contains_bearing(350)
        assert sp.contains_bearing(10)
        assert not sp.contains_bearing(100)

    def test_serialization_roundtrip(self):
        sp = SensorPlacement(
            sensor_id="test_sensor",
            sensor_type=SensorType.CAMERA,
            fov_degrees=120.0,
            height_m=5.0,
            mounting_type=MountingType.POLE,
        )
        data = sp.model_dump()
        restored = SensorPlacement(**data)
        assert restored.sensor_id == "test_sensor"
        assert restored.sensor_type == SensorType.CAMERA
        assert restored.fov_degrees == 120.0

    def test_json_roundtrip(self):
        sp = SensorPlacement(
            sensor_id="json_test",
            sensor_type=SensorType.WIFI_RADIO,
        )
        json_str = sp.model_dump_json()
        restored = SensorPlacement.model_validate_json(json_str)
        assert restored.sensor_id == "json_test"


class TestSensorArray:
    """Tests for the SensorArray model."""

    def test_empty_array(self):
        arr = SensorArray(array_id="arr_01")
        assert arr.sensor_count == 0
        assert arr.sensor_ids() == []
        assert arr.total_coverage_area_m2() == 0.0

    def test_array_with_sensors(self):
        sensors = [
            SensorPlacement(sensor_id="s1", sensor_type=SensorType.BLE_RADIO, status=SensorStatus.ONLINE),
            SensorPlacement(sensor_id="s2", sensor_type=SensorType.CAMERA, status=SensorStatus.ONLINE),
            SensorPlacement(sensor_id="s3", sensor_type=SensorType.BLE_RADIO, status=SensorStatus.OFFLINE),
        ]
        arr = SensorArray(array_id="arr_02", sensors=sensors, purpose="trilateration")
        assert arr.sensor_count == 3
        assert len(arr.sensor_ids()) == 3
        assert len(arr.by_type(SensorType.BLE_RADIO)) == 2
        assert len(arr.by_type(SensorType.CAMERA)) == 1
        assert len(arr.online_sensors()) == 2


class TestSensorEnums:
    """Tests for sensor-related enumerations."""

    def test_sensor_type_values(self):
        assert SensorType.BLE_RADIO.value == "ble_radio"
        assert SensorType.CAMERA.value == "camera"
        assert SensorType.MICROPHONE.value == "microphone"

    def test_mounting_type_values(self):
        assert MountingType.WALL.value == "wall"
        assert MountingType.DRONE.value == "drone"
        assert MountingType.EMBEDDED.value == "embedded"

    def test_sensor_status_values(self):
        assert SensorStatus.ONLINE.value == "online"
        assert SensorStatus.DEGRADED.value == "degraded"
