# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for MeshNodeExtended model."""

import time
import pytest
from tritium_lib.models.mesh_node_extended import (
    MeshNodeExtended,
    MeshNodePosition,
    MeshNodeDeviceMetrics,
    MeshNodeEnvironment,
    MeshNodeRadioMetrics,
)


class TestMeshNodePosition:
    def test_has_fix_with_coords(self):
        pos = MeshNodePosition(latitude=30.0, longitude=-97.0)
        assert pos.has_fix is True

    def test_no_fix_without_coords(self):
        pos = MeshNodePosition()
        assert pos.has_fix is False

    def test_no_fix_with_none(self):
        pos = MeshNodePosition(latitude=None, longitude=None)
        assert pos.has_fix is False


class TestMeshNodeDeviceMetrics:
    def test_basic(self):
        dm = MeshNodeDeviceMetrics(
            battery_level=85,
            voltage=3.95,
            channel_utilization=12.5,
            air_util_tx=3.2,
        )
        assert dm.battery_level == 85
        assert dm.voltage == 3.95


class TestMeshNodeEnvironment:
    def test_basic(self):
        env = MeshNodeEnvironment(
            temperature=22.5,
            relative_humidity=45.0,
            barometric_pressure=1013.25,
        )
        assert env.temperature == 22.5


class TestMeshNodeRadioMetrics:
    def test_hops_away(self):
        rm = MeshNodeRadioMetrics(hop_start=3, hop_limit=1)
        assert rm.hops_away == 2

    def test_hops_away_none(self):
        rm = MeshNodeRadioMetrics(snr=5.0)
        assert rm.hops_away is None


class TestMeshNodeExtended:
    def test_basic_creation(self):
        node = MeshNodeExtended(
            node_id="!ba33ff38",
            long_name="T-LoRa Pager",
            short_name="TLP",
            hw_model="TLORA_V2_1_1P6",
            firmware_version="2.7.19",
        )
        assert node.node_id == "!ba33ff38"
        assert node.display_name == "T-LoRa Pager"
        assert node.has_position is False

    def test_display_name_fallbacks(self):
        n1 = MeshNodeExtended(node_id="!abc", long_name="Long")
        assert n1.display_name == "Long"

        n2 = MeshNodeExtended(node_id="!abc", short_name="Srt")
        assert n2.display_name == "Srt"

        n3 = MeshNodeExtended(node_id="!abc")
        assert n3.display_name == "!abc"

    def test_has_position(self):
        node = MeshNodeExtended(
            node_id="!abc",
            position=MeshNodePosition(latitude=30.0, longitude=-97.0),
        )
        assert node.has_position is True

    def test_battery_percent(self):
        node = MeshNodeExtended(
            node_id="!abc",
            device_metrics=MeshNodeDeviceMetrics(battery_level=72),
        )
        assert node.battery_percent == 72

    def test_battery_percent_none(self):
        node = MeshNodeExtended(node_id="!abc")
        assert node.battery_percent is None

    def test_age_seconds(self):
        now = int(time.time())
        node = MeshNodeExtended(node_id="!abc", last_heard=now - 60)
        age = node.age_seconds
        assert age is not None
        assert 59 <= age <= 62

    def test_age_seconds_none(self):
        node = MeshNodeExtended(node_id="!abc")
        assert node.age_seconds is None

    def test_from_meshtastic_node(self):
        raw = {
            "user": {
                "id": "!ba33ff38",
                "longName": "T-LoRa Pager",
                "shortName": "TLP",
                "hwModel": "TLORA_V2_1_1P6",
                "role": "CLIENT",
                "isLicensed": False,
                "macaddr": "aa:bb:cc:dd:ee:ff",
                "num": 3124666168,
            },
            "position": {
                "latitude": 30.2672,
                "longitude": -97.7431,
                "altitude": 150,
                "satsInView": 12,
            },
            "deviceMetrics": {
                "batteryLevel": 85,
                "voltage": 3.95,
                "channelUtilization": 12.5,
                "airUtilTx": 3.2,
                "uptimeSeconds": 86400,
            },
            "environmentMetrics": {
                "temperature": 22.5,
                "relativeHumidity": 45.0,
                "barometricPressure": 1013.25,
            },
            "snr": 8.5,
            "rssi": -75,
            "lastHeard": int(time.time()) - 30,
            "isFavorite": True,
            "viaMqtt": False,
            "hopsAway": 1,
            "firmwareVersion": "2.7.19",
        }

        node = MeshNodeExtended.from_meshtastic_node("!ba33ff38", raw)

        assert node.node_id == "!ba33ff38"
        assert node.long_name == "T-LoRa Pager"
        assert node.short_name == "TLP"
        assert node.hw_model == "TLORA_V2_1_1P6"
        assert node.firmware_version == "2.7.19"
        assert node.role == "CLIENT"
        assert node.has_position is True
        assert node.position.latitude == 30.2672
        assert node.position.longitude == -97.7431
        assert node.position.altitude == 150
        assert node.device_metrics.battery_level == 85
        assert node.device_metrics.voltage == 3.95
        assert node.device_metrics.channel_utilization == 12.5
        assert node.environment.temperature == 22.5
        assert node.environment.barometric_pressure == 1013.25
        assert node.radio.snr == 8.5
        assert node.radio.rssi == -75
        assert node.is_favorite is True
        assert node.hops_away == 1
        assert node.target_id == "mesh_ba33ff38"

    def test_from_meshtastic_node_minimal(self):
        raw = {"user": {"id": "!dead"}}
        node = MeshNodeExtended.from_meshtastic_node("!dead", raw)
        assert node.node_id == "!dead"
        assert node.has_position is False
        assert node.battery_percent is None
        assert node.target_id == "mesh_dead"

    def test_serialization(self):
        node = MeshNodeExtended(
            node_id="!abc",
            long_name="Test",
            position=MeshNodePosition(latitude=30.0, longitude=-97.0),
            device_metrics=MeshNodeDeviceMetrics(battery_level=50),
        )
        d = node.model_dump()
        assert d["node_id"] == "!abc"
        assert d["position"]["latitude"] == 30.0
        assert d["device_metrics"]["battery_level"] == 50

        # Round-trip
        node2 = MeshNodeExtended.model_validate(d)
        assert node2.node_id == "!abc"
        assert node2.has_position is True


class TestImportFromInit:
    def test_import_from_models(self):
        from tritium_lib.models import (
            MeshNodeExtended,
            MeshNodePosition,
            MeshNodeDeviceMetrics,
            MeshNodeEnvironment,
            MeshNodeRadioMetrics,
        )
        assert MeshNodeExtended is not None
        assert MeshNodePosition is not None
