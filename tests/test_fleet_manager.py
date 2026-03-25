# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.fleet — FleetManager, groups, commands, monitoring."""

import time

import pytest

from tritium_lib.fleet import (
    CommandPriority,
    CommandQueue,
    CommandStatus,
    DeviceGroup,
    DeviceStatus,
    FleetDevice,
    FleetManager,
    HeartbeatMonitor,
    QueuedCommand,
    StaleDevice,
)


# ---------------------------------------------------------------------------
# FleetDevice
# ---------------------------------------------------------------------------

class TestFleetDevice:
    def test_create_defaults(self):
        dev = FleetDevice(device_id="esp32-001")
        assert dev.device_id == "esp32-001"
        assert dev.device_type == "esp32"
        assert dev.status == DeviceStatus.OFFLINE
        assert dev.capabilities == []
        assert dev.firmware_version == "unknown"
        assert dev.battery_pct is None

    def test_create_full(self):
        dev = FleetDevice(
            device_id="cam-002",
            device_type="rpi4",
            device_name="Front Gate Camera",
            status=DeviceStatus.ONLINE,
            capabilities=["camera", "wifi"],
            group="perimeter",
            tags=["outdoor", "critical"],
            firmware_version="2.1.0",
            ip_address="10.0.0.42",
            wifi_rssi=-55,
            free_heap=180_000,
            uptime_s=7200,
            battery_pct=85.0,
        )
        assert dev.device_name == "Front Gate Camera"
        assert dev.is_online
        assert dev.has_cap("camera")
        assert not dev.has_cap("ble")
        assert dev.has_capability  # at least one cap

    def test_to_dict(self):
        dev = FleetDevice(device_id="n1", capabilities=["ble"])
        d = dev.to_dict()
        assert d["device_id"] == "n1"
        assert d["capabilities"] == ["ble"]
        assert d["status"] == "offline"

    def test_is_online_flag(self):
        dev = FleetDevice(device_id="n1", status=DeviceStatus.ONLINE)
        assert dev.is_online
        dev.status = DeviceStatus.STALE
        assert not dev.is_online


# ---------------------------------------------------------------------------
# DeviceGroup
# ---------------------------------------------------------------------------

class TestDeviceGroup:
    def test_create(self):
        grp = DeviceGroup(group_id="perimeter", name="Perimeter Nodes")
        assert grp.size == 0
        assert grp.group_id == "perimeter"

    def test_add_remove(self):
        grp = DeviceGroup(group_id="g1")
        assert grp.add("dev-1")
        assert grp.add("dev-2")
        assert not grp.add("dev-1")  # duplicate
        assert grp.size == 2
        assert "dev-1" in grp
        assert grp.remove("dev-1")
        assert not grp.remove("dev-1")  # already gone
        assert grp.size == 1

    def test_to_dict(self):
        grp = DeviceGroup(group_id="g1", name="Group One", device_ids=["a", "b"])
        d = grp.to_dict()
        assert d["group_id"] == "g1"
        assert d["device_ids"] == ["a", "b"]


# ---------------------------------------------------------------------------
# CommandQueue
# ---------------------------------------------------------------------------

class TestCommandQueue:
    def test_enqueue_dequeue(self):
        q = CommandQueue()
        cmd = q.enqueue("esp32-001", "reboot")
        assert cmd.command_type == "reboot"
        assert cmd.device_id == "esp32-001"
        assert cmd.status == CommandStatus.QUEUED
        assert q.pending_count == 1

        out = q.dequeue()
        assert out is not None
        assert out.command_id == cmd.command_id
        assert out.status == CommandStatus.DISPATCHED
        assert q.pending_count == 0

    def test_dequeue_empty(self):
        q = CommandQueue()
        assert q.dequeue() is None

    def test_priority_ordering(self):
        q = CommandQueue()
        q.enqueue("d1", "low_task", priority=CommandPriority.LOW)
        q.enqueue("d2", "critical_task", priority=CommandPriority.CRITICAL)
        q.enqueue("d3", "normal_task", priority=CommandPriority.NORMAL)

        first = q.dequeue()
        assert first is not None
        assert first.command_type == "critical_task"

        second = q.dequeue()
        assert second is not None
        assert second.command_type == "normal_task"

        third = q.dequeue()
        assert third is not None
        assert third.command_type == "low_task"

    def test_dequeue_by_device(self):
        q = CommandQueue()
        q.enqueue("dev-A", "reboot")
        q.enqueue("dev-B", "scan")
        q.enqueue("dev-A", "identify")

        cmd = q.dequeue(device_id="dev-B")
        assert cmd is not None
        assert cmd.command_type == "scan"
        assert q.pending_count == 2

    def test_ack_and_fail(self):
        q = CommandQueue()
        cmd = q.enqueue("d1", "reboot")
        q.dequeue()  # dispatch it
        assert q.ack(cmd.command_id)
        assert cmd.status == CommandStatus.ACKED
        assert cmd.acked_at > 0

        cmd2 = q.enqueue("d2", "ota")
        q.dequeue()
        assert q.fail(cmd2.command_id, error="timeout")
        assert cmd2.status == CommandStatus.FAILED
        assert cmd2.error == "timeout"

    def test_ack_unknown_returns_false(self):
        q = CommandQueue()
        assert not q.ack("nonexistent")
        assert not q.fail("nonexistent")

    def test_pending_for_device(self):
        q = CommandQueue()
        q.enqueue("d1", "reboot")
        q.enqueue("d1", "scan")
        q.enqueue("d2", "reboot")
        assert len(q.pending_for("d1")) == 2
        assert len(q.pending_for("d2")) == 1
        assert len(q.pending_for("d3")) == 0

    def test_peek_does_not_remove(self):
        q = CommandQueue()
        q.enqueue("d1", "reboot")
        cmd = q.peek()
        assert cmd is not None
        assert q.pending_count == 1  # still there
        cmd2 = q.peek(device_id="d1")
        assert cmd2 is not None
        assert cmd2.command_id == cmd.command_id

    def test_get_by_id(self):
        q = CommandQueue()
        cmd = q.enqueue("d1", "reboot")
        assert q.get(cmd.command_id) is cmd
        assert q.get("nonexistent") is None

    def test_history_after_dequeue(self):
        q = CommandQueue()
        q.enqueue("d1", "reboot")
        q.dequeue()
        assert len(q.history) == 1

    def test_payload(self):
        q = CommandQueue()
        cmd = q.enqueue("d1", "ota", payload={"url": "http://fw.bin"})
        assert cmd.payload["url"] == "http://fw.bin"


# ---------------------------------------------------------------------------
# HeartbeatMonitor
# ---------------------------------------------------------------------------

class TestHeartbeatMonitor:
    def _make_devices(self) -> dict[str, FleetDevice]:
        return {
            "d1": FleetDevice(
                device_id="d1",
                status=DeviceStatus.ONLINE,
                last_heartbeat=100.0,
            ),
            "d2": FleetDevice(
                device_id="d2",
                status=DeviceStatus.ONLINE,
                last_heartbeat=50.0,
            ),
            "d3": FleetDevice(
                device_id="d3",
                status=DeviceStatus.OFFLINE,
                last_heartbeat=10.0,
            ),
        }

    def test_check_stale(self):
        devices = self._make_devices()
        mon = HeartbeatMonitor(devices)
        stale = mon.check_stale(timeout_s=30, now=120.0)
        # d1 is 20s old -> not stale
        # d2 is 70s old -> stale
        # d3 is OFFLINE -> skipped
        assert len(stale) == 1
        assert stale[0].device_id == "d2"
        assert stale[0].seconds_since == 70.0

    def test_check_stale_never_heartbeat(self):
        devices = {
            "d1": FleetDevice(
                device_id="d1",
                status=DeviceStatus.ONLINE,
                last_heartbeat=0,
            ),
        }
        mon = HeartbeatMonitor(devices)
        stale = mon.check_stale(timeout_s=60, now=1000.0)
        assert len(stale) == 1
        assert stale[0].seconds_since == float("inf")

    def test_mark_stale(self):
        devices = self._make_devices()
        mon = HeartbeatMonitor(devices)
        affected = mon.mark_stale(timeout_s=30, now=120.0)
        assert "d2" in affected
        assert devices["d2"].status == DeviceStatus.STALE
        assert devices["d1"].status == DeviceStatus.ONLINE  # not affected

    def test_mark_offline(self):
        devices = self._make_devices()
        mon = HeartbeatMonitor(devices)
        affected = mon.mark_offline(timeout_s=30, now=120.0)
        assert "d2" in affected
        assert devices["d2"].status == DeviceStatus.OFFLINE

    def test_already_stale_not_double_marked(self):
        devices = {
            "d1": FleetDevice(
                device_id="d1",
                status=DeviceStatus.STALE,
                last_heartbeat=10.0,
            ),
        }
        mon = HeartbeatMonitor(devices)
        affected = mon.mark_stale(timeout_s=30, now=120.0)
        assert affected == []  # already stale


# ---------------------------------------------------------------------------
# FleetManager — registration
# ---------------------------------------------------------------------------

class TestFleetManagerRegistration:
    def test_register_and_get(self):
        mgr = FleetManager()
        dev = mgr.register("esp32-001", device_type="esp32-s3", capabilities=["ble", "wifi"])
        assert dev.device_id == "esp32-001"
        assert dev.device_type == "esp32-s3"
        assert "ble" in dev.capabilities
        assert mgr.get("esp32-001") is dev

    def test_register_duplicate_raises(self):
        mgr = FleetManager()
        mgr.register("n1")
        with pytest.raises(ValueError, match="already registered"):
            mgr.register("n1")

    def test_unregister(self):
        mgr = FleetManager()
        mgr.register("n1")
        assert mgr.unregister("n1")
        assert mgr.get("n1") is None
        assert not mgr.unregister("n1")  # already gone

    def test_contains_and_len(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.register("n2")
        assert "n1" in mgr
        assert "n3" not in mgr
        assert len(mgr) == 2

    def test_iteration(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.register("n2")
        ids = [d.device_id for d in mgr]
        assert set(ids) == {"n1", "n2"}


# ---------------------------------------------------------------------------
# FleetManager — heartbeat
# ---------------------------------------------------------------------------

class TestFleetManagerHeartbeat:
    def test_heartbeat_updates_telemetry(self):
        mgr = FleetManager()
        mgr.register("n1")
        assert mgr.heartbeat(
            "n1",
            firmware_version="1.2.0",
            wifi_rssi=-48,
            free_heap=200_000,
            uptime_s=3600,
            now=100.0,
        )
        dev = mgr.get("n1")
        assert dev is not None
        assert dev.status == DeviceStatus.ONLINE
        assert dev.firmware_version == "1.2.0"
        assert dev.wifi_rssi == -48
        assert dev.free_heap == 200_000
        assert dev.uptime_s == 3600
        assert dev.last_heartbeat == 100.0

    def test_heartbeat_unknown_device(self):
        mgr = FleetManager()
        assert not mgr.heartbeat("nonexistent", now=1.0)

    def test_heartbeat_partial_update(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.heartbeat("n1", firmware_version="1.0", wifi_rssi=-50, now=10.0)
        # Second heartbeat only updates RSSI
        mgr.heartbeat("n1", wifi_rssi=-40, now=20.0)
        dev = mgr.get("n1")
        assert dev is not None
        assert dev.firmware_version == "1.0"  # unchanged
        assert dev.wifi_rssi == -40  # updated
        assert dev.last_heartbeat == 20.0


# ---------------------------------------------------------------------------
# FleetManager — groups
# ---------------------------------------------------------------------------

class TestFleetManagerGroups:
    def test_create_group(self):
        mgr = FleetManager()
        grp = mgr.create_group("perimeter", name="Perimeter Sensors")
        assert grp.group_id == "perimeter"
        assert grp.name == "Perimeter Sensors"

    def test_create_duplicate_group_raises(self):
        mgr = FleetManager()
        mgr.create_group("g1")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_group("g1")

    def test_assign_to_group(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.register("n2")
        mgr.create_group("perimeter")
        assert mgr.assign_to_group("n1", "perimeter")
        assert mgr.assign_to_group("n2", "perimeter")
        grp = mgr.get_group("perimeter")
        assert grp is not None
        assert grp.size == 2
        dev = mgr.get("n1")
        assert dev is not None
        assert dev.group == "perimeter"

    def test_reassign_removes_from_old_group(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.create_group("g1")
        mgr.create_group("g2")
        mgr.assign_to_group("n1", "g1")
        mgr.assign_to_group("n1", "g2")
        assert mgr.get_group("g1").size == 0
        assert mgr.get_group("g2").size == 1

    def test_register_with_group_auto_creates(self):
        mgr = FleetManager()
        mgr.register("n1", group="auto-group")
        grp = mgr.get_group("auto-group")
        assert grp is not None
        assert "n1" in grp

    def test_delete_group_clears_membership(self):
        mgr = FleetManager()
        mgr.register("n1", group="g1")
        assert mgr.delete_group("g1")
        dev = mgr.get("n1")
        assert dev is not None
        assert dev.group == ""
        assert not mgr.delete_group("g1")  # already gone

    def test_remove_from_group(self):
        mgr = FleetManager()
        mgr.register("n1", group="g1")
        assert mgr.remove_from_group("n1")
        assert mgr.get("n1").group == ""
        assert not mgr.remove_from_group("n1")  # no group now

    def test_list_groups(self):
        mgr = FleetManager()
        mgr.create_group("g1")
        mgr.create_group("g2")
        groups = mgr.list_groups()
        assert len(groups) == 2

    def test_assign_unknown_device_returns_false(self):
        mgr = FleetManager()
        assert not mgr.assign_to_group("ghost", "g1")

    def test_assign_auto_creates_group(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.assign_to_group("n1", "new-group")
        assert mgr.get_group("new-group") is not None


# ---------------------------------------------------------------------------
# FleetManager — listing and filtering
# ---------------------------------------------------------------------------

class TestFleetManagerListing:
    def _setup(self) -> FleetManager:
        mgr = FleetManager()
        mgr.register("n1", device_type="esp32", capabilities=["ble", "wifi"], group="perimeter")
        mgr.register("n2", device_type="rpi4", capabilities=["camera"], group="interior")
        mgr.register("n3", device_type="esp32", capabilities=["ble"], group="perimeter")
        mgr.heartbeat("n1", wifi_rssi=-40, free_heap=250_000, now=100.0)
        mgr.heartbeat("n2", wifi_rssi=-60, free_heap=500_000, now=100.0)
        # n3 never sends heartbeat — stays OFFLINE
        return mgr

    def test_list_all(self):
        mgr = self._setup()
        assert len(mgr.list_devices()) == 3

    def test_filter_by_status(self):
        mgr = self._setup()
        online = mgr.list_devices(status=DeviceStatus.ONLINE)
        assert len(online) == 2

    def test_filter_by_group(self):
        mgr = self._setup()
        perimeter = mgr.list_devices(group="perimeter")
        assert len(perimeter) == 2

    def test_filter_by_capability(self):
        mgr = self._setup()
        ble_devices = mgr.list_devices(capability="ble")
        assert len(ble_devices) == 2
        camera_devices = mgr.list_devices(capability="camera")
        assert len(camera_devices) == 1

    def test_filter_by_device_type(self):
        mgr = self._setup()
        esp_devices = mgr.list_devices(device_type="esp32")
        assert len(esp_devices) == 2

    def test_combined_filters(self):
        mgr = self._setup()
        result = mgr.list_devices(status=DeviceStatus.ONLINE, capability="ble")
        assert len(result) == 1
        assert result[0].device_id == "n1"

    def test_counts(self):
        mgr = self._setup()
        assert mgr.device_count == 3
        assert mgr.online_count == 2
        assert mgr.offline_count == 1


# ---------------------------------------------------------------------------
# FleetManager — health & summary
# ---------------------------------------------------------------------------

class TestFleetManagerHealth:
    def test_empty_fleet(self):
        mgr = FleetManager()
        assert mgr.health_score() == 0.0

    def test_healthy_fleet(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.register("n2")
        mgr.heartbeat("n1", wifi_rssi=-35, free_heap=280_000, now=1.0)
        mgr.heartbeat("n2", wifi_rssi=-40, free_heap=260_000, now=1.0)
        score = mgr.health_score()
        assert score > 0.8

    def test_all_offline(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.register("n2")
        # No heartbeats — both OFFLINE
        assert mgr.health_score() == 0.0

    def test_summary(self):
        mgr = FleetManager()
        mgr.register("n1", group="g1")
        mgr.heartbeat("n1", wifi_rssi=-50, free_heap=200_000, now=1.0)
        mgr.commands.enqueue("n1", "reboot")
        s = mgr.summary()
        assert s["total_devices"] == 1
        assert s["online"] == 1
        assert s["pending_commands"] == 1
        assert "g1" in s["groups"]


# ---------------------------------------------------------------------------
# FleetManager — monitor integration
# ---------------------------------------------------------------------------

class TestFleetManagerMonitor:
    def test_monitor_detects_stale(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.heartbeat("n1", now=10.0)
        stale = mgr.monitor.check_stale(timeout_s=30, now=50.0)
        assert len(stale) == 1
        assert stale[0].device_id == "n1"

    def test_monitor_marks_stale(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.heartbeat("n1", now=10.0)
        affected = mgr.monitor.mark_stale(timeout_s=30, now=50.0)
        assert "n1" in affected
        assert mgr.get("n1").status == DeviceStatus.STALE

    def test_monitor_marks_offline(self):
        mgr = FleetManager()
        mgr.register("n1")
        mgr.heartbeat("n1", now=10.0)
        affected = mgr.monitor.mark_offline(timeout_s=30, now=350.0)
        assert "n1" in affected
        assert mgr.get("n1").status == DeviceStatus.OFFLINE


# ---------------------------------------------------------------------------
# FleetManager — unregister cleans up groups
# ---------------------------------------------------------------------------

class TestFleetManagerUnregisterCleanup:
    def test_unregister_removes_from_group(self):
        mgr = FleetManager()
        mgr.register("n1", group="g1")
        grp = mgr.get_group("g1")
        assert grp is not None and grp.size == 1
        mgr.unregister("n1")
        assert grp.size == 0


# ---------------------------------------------------------------------------
# QueuedCommand
# ---------------------------------------------------------------------------

class TestQueuedCommand:
    def test_to_dict(self):
        cmd = QueuedCommand(
            command_id="abc",
            device_id="d1",
            command_type="reboot",
            priority=CommandPriority.HIGH,
        )
        d = cmd.to_dict()
        assert d["command_id"] == "abc"
        assert d["priority"] == "high"
        assert d["status"] == "queued"

    def test_is_expired(self):
        now = time.monotonic()
        cmd = QueuedCommand(
            command_id="x",
            device_id="d1",
            command_type="reboot",
            expires_at=now - 10,  # already expired
        )
        assert cmd.is_expired

    def test_not_expired_when_zero(self):
        cmd = QueuedCommand(
            command_id="x",
            device_id="d1",
            command_type="reboot",
            expires_at=0,
        )
        assert not cmd.is_expired
