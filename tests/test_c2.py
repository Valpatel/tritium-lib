# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.c2 — Command and Control protocol."""

import time

import pytest

from tritium_lib.c2 import (
    C2Channel,
    C2CommandType,
    C2Controller,
    C2Envelope,
    C2Priority,
    C2Status,
    CommandHistory,
    CommandResult,
    ConfigCommand,
    DiagnosticCommand,
    ObserveCommand,
    PatrolCommand,
    ScanCommand,
)


# ===========================================================================
# Command payload tests
# ===========================================================================

class TestScanCommand:
    def test_defaults(self):
        cmd = ScanCommand()
        assert cmd.type == C2CommandType.SCAN
        assert cmd.ble_channels == [37, 38, 39]
        assert cmd.wifi_bands == ["2.4GHz"]
        assert cmd.scan_duration_s == 10.0
        assert cmd.scan_interval_s == 0.0
        assert cmd.active_scan is False
        assert cmd.filter_rssi_min == -100

    def test_custom_params(self):
        cmd = ScanCommand(
            ble_channels=[37],
            wifi_bands=["2.4GHz", "5GHz"],
            scan_duration_s=30.0,
            active_scan=True,
            filter_rssi_min=-70,
        )
        assert cmd.ble_channels == [37]
        assert cmd.wifi_bands == ["2.4GHz", "5GHz"]
        assert cmd.scan_duration_s == 30.0
        assert cmd.active_scan is True
        assert cmd.filter_rssi_min == -70

    def test_to_dict(self):
        cmd = ScanCommand()
        d = cmd.to_dict()
        assert d["type"] == "scan"
        assert d["ble_channels"] == [37, 38, 39]
        assert d["scan_duration_s"] == 10.0


class TestPatrolCommand:
    def test_defaults(self):
        cmd = PatrolCommand()
        assert cmd.type == C2CommandType.PATROL
        assert cmd.waypoints == []
        assert cmd.loop is False
        assert cmd.speed_mps == 1.0

    def test_with_waypoints(self):
        wp = [
            {"lat": 40.0, "lng": -74.0, "dwell_s": 10},
            {"lat": 40.1, "lng": -74.1, "dwell_s": 20},
        ]
        cmd = PatrolCommand(waypoints=wp, loop=True, speed_mps=2.5)
        assert len(cmd.waypoints) == 2
        assert cmd.loop is True
        assert cmd.speed_mps == 2.5

    def test_to_dict(self):
        cmd = PatrolCommand(
            waypoints=[{"lat": 40.0, "lng": -74.0}],
            loop=True,
        )
        d = cmd.to_dict()
        assert d["type"] == "patrol"
        assert d["loop"] is True
        assert len(d["waypoints"]) == 1


class TestObserveCommand:
    def test_defaults(self):
        cmd = ObserveCommand()
        assert cmd.type == C2CommandType.OBSERVE
        assert cmd.target_id == ""
        assert cmd.sensor_modes == ["ble", "wifi"]
        assert cmd.duration_s == 60.0

    def test_observe_target(self):
        cmd = ObserveCommand(
            target_id="ble_aabbccdd",
            sensor_modes=["ble"],
            duration_s=120.0,
            report_interval_s=2.0,
        )
        assert cmd.target_id == "ble_aabbccdd"
        assert cmd.duration_s == 120.0
        assert cmd.report_interval_s == 2.0

    def test_to_dict(self):
        cmd = ObserveCommand(target_id="ble_001")
        d = cmd.to_dict()
        assert d["type"] == "observe"
        assert d["target_id"] == "ble_001"


class TestConfigCommand:
    def test_defaults(self):
        cmd = ConfigCommand()
        assert cmd.type == C2CommandType.CONFIGURE
        assert cmd.updates == {}
        assert cmd.restart_required is False
        assert cmd.validate_before_apply is True

    def test_with_updates(self):
        cmd = ConfigCommand(
            updates={"heartbeat_interval_s": 5, "wifi_ssid": "tritium"},
            restart_required=True,
        )
        assert cmd.updates["heartbeat_interval_s"] == 5
        assert cmd.restart_required is True

    def test_to_dict(self):
        cmd = ConfigCommand(updates={"key": "val"})
        d = cmd.to_dict()
        assert d["type"] == "configure"
        assert d["updates"] == {"key": "val"}


class TestDiagnosticCommand:
    def test_defaults(self):
        cmd = DiagnosticCommand()
        assert cmd.type == C2CommandType.DIAGNOSTIC
        assert "system" in cmd.sections
        assert "memory" in cmd.sections
        assert cmd.include_logs is False
        assert cmd.log_lines == 100
        assert cmd.include_config is True

    def test_custom_sections(self):
        cmd = DiagnosticCommand(
            sections=["system", "network"],
            include_logs=True,
            log_lines=50,
        )
        assert cmd.sections == ["system", "network"]
        assert cmd.include_logs is True
        assert cmd.log_lines == 50

    def test_to_dict(self):
        cmd = DiagnosticCommand()
        d = cmd.to_dict()
        assert d["type"] == "diagnostic"
        assert isinstance(d["sections"], list)


# ===========================================================================
# CommandResult tests
# ===========================================================================

class TestCommandResult:
    def test_success(self):
        r = CommandResult(success=True, detail="scan started")
        assert r.success is True
        assert r.detail == "scan started"
        assert r.received_at > 0

    def test_failure(self):
        r = CommandResult(success=False, error_code="TIMEOUT", detail="no response")
        assert r.success is False
        assert r.error_code == "TIMEOUT"

    def test_to_dict(self):
        r = CommandResult(success=True, data={"targets_found": 5})
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"]["targets_found"] == 5
        assert d["received_at"] > 0


# ===========================================================================
# C2Channel tests
# ===========================================================================

class TestC2Channel:
    def test_default_channel_noop(self):
        ch = C2Channel()
        assert ch.channel_name == "noop"
        assert ch.is_connected is True
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        assert ch.send(env) is True

    def test_custom_channel(self):
        sent = []

        class MockChannel(C2Channel):
            def send(self, envelope):
                sent.append(envelope)
                return True

            @property
            def channel_name(self):
                return "mock"

        ch = MockChannel()
        assert ch.channel_name == "mock"
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        ch.send(env)
        assert len(sent) == 1

    def test_failing_channel(self):
        class FailChannel(C2Channel):
            def send(self, envelope):
                return False

            @property
            def is_connected(self):
                return False

        ch = FailChannel()
        assert ch.is_connected is False
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        assert ch.send(env) is False


# ===========================================================================
# C2Envelope tests
# ===========================================================================

class TestC2Envelope:
    def test_auto_id_and_timestamp(self):
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        assert len(env.command_id) == 16
        assert env.created_at > 0

    def test_command_type(self):
        env = C2Envelope(command=PatrolCommand(), device_id="dev1")
        assert env.command_type == C2CommandType.PATROL

    def test_command_type_none(self):
        env = C2Envelope(device_id="dev1")
        assert env.command_type is None

    def test_target_description_device(self):
        env = C2Envelope(command=ScanCommand(), device_id="esp32-001")
        assert env.target_description == "esp32-001"

    def test_target_description_broadcast_group(self):
        env = C2Envelope(
            command=ScanCommand(),
            is_broadcast=True,
            group="perimeter",
        )
        assert env.target_description == "group:perimeter"

    def test_target_description_broadcast_all(self):
        env = C2Envelope(command=ScanCommand(), is_broadcast=True)
        assert env.target_description == "all"

    def test_is_expired(self):
        env = C2Envelope(
            command=ScanCommand(),
            device_id="dev1",
            ttl_s=0.01,
            created_at=time.time() - 1.0,
        )
        assert env.is_expired is True

    def test_not_expired(self):
        env = C2Envelope(command=ScanCommand(), device_id="dev1", ttl_s=300.0)
        assert env.is_expired is False

    def test_zero_ttl_never_expires(self):
        env = C2Envelope(
            command=ScanCommand(),
            device_id="dev1",
            ttl_s=0,
            created_at=time.time() - 99999,
        )
        assert env.is_expired is False

    def test_is_terminal(self):
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        env.status = C2Status.DRAFT
        assert env.is_terminal is False
        env.status = C2Status.COMPLETED
        assert env.is_terminal is True
        env.status = C2Status.FAILED
        assert env.is_terminal is True
        env.status = C2Status.EXPIRED
        assert env.is_terminal is True
        env.status = C2Status.CANCELLED
        assert env.is_terminal is True

    def test_to_dict(self):
        env = C2Envelope(
            command=ScanCommand(),
            device_id="dev1",
            operator="amy",
            priority=C2Priority.HIGH,
            tags=["urgent"],
        )
        d = env.to_dict()
        assert d["device_id"] == "dev1"
        assert d["operator"] == "amy"
        assert d["priority"] == "high"
        assert d["command_type"] == "scan"
        assert d["tags"] == ["urgent"]
        assert d["command"]["type"] == "scan"


# ===========================================================================
# CommandHistory tests
# ===========================================================================

class TestCommandHistory:
    def test_record_and_get(self):
        h = CommandHistory()
        env = C2Envelope(command=ScanCommand(), device_id="dev1")
        h.record(env)
        assert h.get(env.command_id) is env
        assert h.total == 1

    def test_query_by_device(self):
        h = CommandHistory()
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        h.record(C2Envelope(command=ScanCommand(), device_id="dev2"))
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        results = h.query(device_id="dev1")
        assert len(results) == 2

    def test_query_by_operator(self):
        h = CommandHistory()
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1", operator="amy"))
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1", operator="matt"))
        results = h.query(operator="amy")
        assert len(results) == 1

    def test_query_by_command_type(self):
        h = CommandHistory()
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        h.record(C2Envelope(command=PatrolCommand(), device_id="dev1"))
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        results = h.query(command_type=C2CommandType.SCAN)
        assert len(results) == 2

    def test_query_by_status(self):
        h = CommandHistory()
        env1 = C2Envelope(command=ScanCommand(), device_id="dev1")
        env1.status = C2Status.SENT
        env2 = C2Envelope(command=ScanCommand(), device_id="dev1")
        env2.status = C2Status.COMPLETED
        h.record(env1)
        h.record(env2)
        results = h.query(status=C2Status.SENT)
        assert len(results) == 1

    def test_query_limit(self):
        h = CommandHistory()
        for _ in range(10):
            h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        results = h.query(limit=3)
        assert len(results) == 3

    def test_query_time_range(self):
        h = CommandHistory()
        now = time.time()
        old = C2Envelope(command=ScanCommand(), device_id="dev1")
        old.created_at = now - 1000
        new = C2Envelope(command=ScanCommand(), device_id="dev1")
        new.created_at = now
        h.record(old)
        h.record(new)
        results = h.query(since=now - 10)
        assert len(results) == 1

    def test_summary(self):
        h = CommandHistory()
        env1 = C2Envelope(command=ScanCommand(), device_id="dev1")
        env1.status = C2Status.SENT
        env2 = C2Envelope(command=PatrolCommand(), device_id="dev1")
        env2.status = C2Status.COMPLETED
        h.record(env1)
        h.record(env2)
        s = h.summary()
        assert s["total"] == 2
        assert s["by_status"]["sent"] == 1
        assert s["by_status"]["completed"] == 1
        assert s["by_type"]["scan"] == 1
        assert s["by_type"]["patrol"] == 1

    def test_clear(self):
        h = CommandHistory()
        h.record(C2Envelope(command=ScanCommand(), device_id="dev1"))
        assert h.total == 1
        h.clear()
        assert h.total == 0

    def test_max_size_eviction(self):
        h = CommandHistory(max_size=5)
        envs = []
        for i in range(10):
            e = C2Envelope(command=ScanCommand(), device_id=f"dev{i}")
            h.record(e)
            envs.append(e)
        assert h.total == 5
        # Oldest should be evicted
        assert h.get(envs[0].command_id) is None
        # Newest should be present
        assert h.get(envs[9].command_id) is not None


# ===========================================================================
# C2Controller tests
# ===========================================================================

class TestC2ControllerSendDevice:
    def test_send_device(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand(), operator="amy")
        assert env.device_id == "esp32-001"
        assert env.status == C2Status.SENT
        assert env.operator == "amy"
        assert env.is_broadcast is False
        assert c2.history.total == 1

    def test_send_device_with_priority(self):
        c2 = C2Controller()
        env = c2.send_device(
            "esp32-001",
            ScanCommand(),
            priority=C2Priority.CRITICAL,
        )
        assert env.priority == C2Priority.CRITICAL

    def test_send_device_with_tags(self):
        c2 = C2Controller()
        env = c2.send_device(
            "esp32-001",
            ScanCommand(),
            tags=["urgent", "perimeter"],
        )
        assert env.tags == ["urgent", "perimeter"]


class TestC2ControllerSendBroadcast:
    def test_broadcast_to_group(self):
        c2 = C2Controller()
        env = c2.send_broadcast(
            ConfigCommand(updates={"interval": 5}),
            group="perimeter",
            operator="amy",
        )
        assert env.is_broadcast is True
        assert env.group == "perimeter"
        assert env.status == C2Status.SENT

    def test_broadcast_to_all(self):
        c2 = C2Controller()
        env = c2.send_broadcast(DiagnosticCommand())
        assert env.is_broadcast is True
        assert env.group == ""
        assert env.target_description == "all"


class TestC2ControllerQueue:
    def test_queue_command(self):
        c2 = C2Controller()
        env = c2.queue_command("esp32-001", ScanCommand(), operator="amy")
        assert env.status == C2Status.QUEUED
        assert c2.queue_size == 1
        assert c2.history.total == 1

    def test_flush_queue(self):
        c2 = C2Controller()
        c2.queue_command("esp32-001", ScanCommand())
        c2.queue_command("esp32-002", PatrolCommand())
        dispatched = c2.flush_queue()
        assert len(dispatched) == 2
        assert c2.queue_size == 0
        for env in dispatched:
            assert env.status == C2Status.SENT

    def test_flush_queue_by_device(self):
        c2 = C2Controller()
        c2.queue_command("esp32-001", ScanCommand())
        c2.queue_command("esp32-002", PatrolCommand())
        dispatched = c2.flush_queue(device_id="esp32-001")
        assert len(dispatched) == 1
        assert c2.queue_size == 1

    def test_flush_expired(self):
        c2 = C2Controller()
        env = c2.queue_command("esp32-001", ScanCommand(), ttl_s=0.01)
        # Force it to be expired
        env.created_at = time.time() - 10.0
        dispatched = c2.flush_queue()
        assert len(dispatched) == 0
        assert env.status == C2Status.EXPIRED

    def test_pending_for(self):
        c2 = C2Controller()
        c2.queue_command("esp32-001", ScanCommand())
        c2.queue_command("esp32-001", PatrolCommand())
        c2.queue_command("esp32-002", ScanCommand())
        pending = c2.pending_for("esp32-001")
        assert len(pending) == 2

    def test_queue_priority_order(self):
        c2 = C2Controller()
        c2.queue_command("esp32-001", ScanCommand(), priority=C2Priority.LOW)
        c2.queue_command("esp32-001", PatrolCommand(), priority=C2Priority.CRITICAL)
        c2.queue_command("esp32-001", ObserveCommand(), priority=C2Priority.NORMAL)
        dispatched = c2.flush_queue()
        assert dispatched[0].command_type == C2CommandType.PATROL
        assert dispatched[1].command_type == C2CommandType.OBSERVE
        assert dispatched[2].command_type == C2CommandType.SCAN


class TestC2ControllerResultHandling:
    def test_record_success(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        result = CommandResult(success=True, detail="found 3 targets")
        assert c2.record_result(env.command_id, result) is True
        assert env.status == C2Status.COMPLETED
        assert env.result is result
        assert env.completed_at > 0

    def test_record_failure(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        result = CommandResult(success=False, error_code="HW_ERR")
        assert c2.record_result(env.command_id, result) is True
        assert env.status == C2Status.FAILED

    def test_record_unknown_id(self):
        c2 = C2Controller()
        result = CommandResult(success=True)
        assert c2.record_result("nonexistent", result) is False

    def test_record_terminal_rejected(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        c2.record_result(env.command_id, CommandResult(success=True))
        # Second attempt on completed command should be rejected
        assert c2.record_result(env.command_id, CommandResult(success=False)) is False
        assert env.status == C2Status.COMPLETED  # unchanged


class TestC2ControllerLifecycle:
    def test_mark_delivered(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert c2.mark_delivered(env.command_id) is True
        assert env.status == C2Status.DELIVERED
        assert env.delivered_at > 0

    def test_mark_acknowledged(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert c2.mark_acknowledged(env.command_id) is True
        assert env.status == C2Status.ACKNOWLEDGED

    def test_mark_executing(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert c2.mark_executing(env.command_id) is True
        assert env.status == C2Status.EXECUTING

    def test_full_lifecycle(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert env.status == C2Status.SENT

        c2.mark_delivered(env.command_id)
        assert env.status == C2Status.DELIVERED

        c2.mark_acknowledged(env.command_id)
        assert env.status == C2Status.ACKNOWLEDGED

        c2.mark_executing(env.command_id)
        assert env.status == C2Status.EXECUTING

        c2.record_result(env.command_id, CommandResult(success=True, detail="done"))
        assert env.status == C2Status.COMPLETED

    def test_cancel(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert c2.cancel(env.command_id) is True
        assert env.status == C2Status.CANCELLED
        # Can't cancel again
        assert c2.cancel(env.command_id) is False

    def test_cancel_queued_removes_from_queue(self):
        c2 = C2Controller()
        env = c2.queue_command("esp32-001", ScanCommand())
        assert c2.queue_size == 1
        c2.cancel(env.command_id)
        assert c2.queue_size == 0

    def test_retry_failed(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        c2.record_result(env.command_id, CommandResult(success=False))
        assert env.status == C2Status.FAILED
        assert c2.retry(env.command_id) is True
        assert env.status == C2Status.SENT
        assert env.retries == 1

    def test_retry_exhausted(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        env.max_retries = 1
        c2.record_result(env.command_id, CommandResult(success=False))
        c2.retry(env.command_id)
        # Fail again
        env.status = C2Status.FAILED
        # Out of retries
        assert c2.retry(env.command_id) is False

    def test_retry_non_failed_rejected(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        assert c2.retry(env.command_id) is False  # SENT, not FAILED

    def test_lifecycle_marks_on_terminal_rejected(self):
        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand())
        c2.record_result(env.command_id, CommandResult(success=True))
        assert c2.mark_delivered(env.command_id) is False
        assert c2.mark_acknowledged(env.command_id) is False
        assert c2.mark_executing(env.command_id) is False


class TestC2ControllerWithChannel:
    def test_failing_channel_marks_failed(self):
        class FailChannel(C2Channel):
            def send(self, envelope):
                return False

        c2 = C2Controller(channel=FailChannel())
        env = c2.send_device("esp32-001", ScanCommand())
        assert env.status == C2Status.FAILED

    def test_custom_channel_receives_envelopes(self):
        sent = []

        class RecordChannel(C2Channel):
            def send(self, envelope):
                sent.append(envelope)
                return True

            @property
            def channel_name(self):
                return "record"

        c2 = C2Controller(channel=RecordChannel())
        c2.send_device("esp32-001", ScanCommand())
        c2.send_broadcast(ConfigCommand(updates={"k": "v"}), group="all")
        assert len(sent) == 2
        assert c2.summary()["channel"] == "record"


class TestC2ControllerSummary:
    def test_summary(self):
        c2 = C2Controller()
        c2.send_device("esp32-001", ScanCommand())
        c2.queue_command("esp32-002", PatrolCommand())
        s = c2.summary()
        assert s["queue_size"] == 1
        assert s["channel"] == "noop"
        assert s["channel_connected"] is True
        assert s["history"]["total"] == 2

    def test_active_commands(self):
        c2 = C2Controller()
        c2.send_device("esp32-001", ScanCommand())
        c2.send_device("esp32-002", PatrolCommand())
        env3 = c2.send_device("esp32-003", ObserveCommand())
        c2.record_result(env3.command_id, CommandResult(success=True))
        active = c2.active_commands()
        # env3 is COMPLETED (terminal), so only 2 active
        assert len(active) == 2
