# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for command models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tritium_lib.models.command import Command, CommandStatus, CommandType


def _utc(hour=12, minute=0, second=0):
    return datetime(2026, 3, 7, hour, minute, second, tzinfo=timezone.utc)


class TestCommandType:
    def test_all_values(self):
        expected = {
            "reboot", "gpio_set", "gpio_read", "config_update", "ota_url",
            "ota_rollback", "identify", "sleep", "wifi_add", "mesh_send", "custom",
        }
        assert {t.value for t in CommandType} == expected

    def test_string_enum(self):
        assert CommandType.REBOOT == "reboot"
        assert isinstance(CommandType.OTA_URL, str)


class TestCommandStatus:
    def test_all_values(self):
        expected = {"pending", "delivered", "acked", "failed", "expired"}
        assert {s.value for s in CommandStatus} == expected

    def test_string_enum(self):
        assert CommandStatus.PENDING == "pending"


class TestCommand:
    def test_create_minimal(self):
        cmd = Command(id="cmd-1", device_id="dev-a", type=CommandType.REBOOT)
        assert cmd.id == "cmd-1"
        assert cmd.device_id == "dev-a"
        assert cmd.type == CommandType.REBOOT
        assert cmd.status == CommandStatus.PENDING
        assert cmd.payload == {}
        assert cmd.created_at is None
        assert cmd.error is None

    def test_create_full(self):
        cmd = Command(
            id="cmd-2",
            device_id="dev-b",
            type=CommandType.CONFIG_UPDATE,
            payload={"key": "heartbeat_interval", "value": 30},
            status=CommandStatus.DELIVERED,
            created_at=_utc(10),
            delivered_at=_utc(10, 5),
            acked_at=None,
            expires_at=_utc(11),
            error=None,
        )
        assert cmd.payload["key"] == "heartbeat_interval"
        assert cmd.status == CommandStatus.DELIVERED
        assert cmd.delivered_at == _utc(10, 5)

    def test_all_command_types(self):
        for ct in CommandType:
            cmd = Command(id="t", device_id="d", type=ct)
            assert cmd.type == ct

    def test_all_statuses(self):
        for st in CommandStatus:
            cmd = Command(id="t", device_id="d", type=CommandType.REBOOT, status=st)
            assert cmd.status == st

    def test_serialization(self):
        cmd = Command(
            id="cmd-3",
            device_id="dev-c",
            type=CommandType.GPIO_SET,
            payload={"pin": 5, "value": 1},
            status=CommandStatus.ACKED,
            created_at=_utc(),
        )
        d = cmd.model_dump()
        assert d["type"] == "gpio_set"
        assert d["status"] == "acked"
        assert d["payload"] == {"pin": 5, "value": 1}

    def test_json_roundtrip(self):
        cmd = Command(
            id="cmd-4",
            device_id="dev-d",
            type=CommandType.OTA_URL,
            payload={"url": "https://example.com/fw.bin"},
            status=CommandStatus.FAILED,
            created_at=_utc(),
            error="Download timeout",
        )
        json_str = cmd.model_dump_json()
        cmd2 = Command.model_validate_json(json_str)
        assert cmd2.id == cmd.id
        assert cmd2.type == cmd.type
        assert cmd2.error == "Download timeout"
        assert cmd2.payload == cmd.payload

    def test_empty_payload(self):
        cmd = Command(id="cmd-5", device_id="dev-e", type=CommandType.IDENTIFY)
        assert cmd.payload == {}

    def test_complex_payload(self):
        payload = {
            "networks": [
                {"ssid": "net1", "password": "pass1"},
                {"ssid": "net2", "password": "pass2"},
            ]
        }
        cmd = Command(
            id="cmd-6", device_id="dev-f", type=CommandType.WIFI_ADD, payload=payload
        )
        assert len(cmd.payload["networks"]) == 2

    def test_expired_command(self):
        cmd = Command(
            id="cmd-7",
            device_id="dev-g",
            type=CommandType.SLEEP,
            status=CommandStatus.EXPIRED,
            created_at=_utc(8),
            expires_at=_utc(9),
        )
        assert cmd.status == CommandStatus.EXPIRED
        assert cmd.expires_at is not None

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Command(id="cmd-8", device_id="dev-h")  # missing type

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            Command(id="cmd-9", device_id="dev-i", type="nonexistent")

    def test_from_dict(self):
        data = {
            "id": "cmd-10",
            "device_id": "dev-j",
            "type": "mesh_send",
            "payload": {"data": "hello"},
        }
        cmd = Command.model_validate(data)
        assert cmd.type == CommandType.MESH_SEND
        assert cmd.status == CommandStatus.PENDING
