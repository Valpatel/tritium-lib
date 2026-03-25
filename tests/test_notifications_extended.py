# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Extended notification system tests — manager lifecycle, filtering, broadcast."""

import time
import threading

import pytest

from tritium_lib.notifications import Notification, NotificationManager


# ── Notification model ──────────────────────────────────────────────

class TestNotification:
    """Tests for the Notification dataclass."""

    def test_basic_fields(self):
        n = Notification(
            id="abc123", title="Alert", message="Something happened",
            severity="warning", source="camera", timestamp=1000.0,
        )
        assert n.id == "abc123"
        assert n.title == "Alert"
        assert n.message == "Something happened"
        assert n.severity == "warning"
        assert n.source == "camera"
        assert n.timestamp == 1000.0
        assert n.read is False

    def test_optional_entity_id(self):
        n = Notification(
            id="x", title="T", message="M",
            severity="info", source="s", timestamp=0,
            entity_id="target_42",
        )
        assert n.entity_id == "target_42"

    def test_to_dict(self):
        n = Notification(
            id="n1", title="Test", message="msg",
            severity="critical", source="ble", timestamp=12345.0,
            entity_id="t1",
        )
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["title"] == "Test"
        assert d["severity"] == "critical"
        assert d["entity_id"] == "t1"
        assert d["read"] is False

    def test_read_default_false(self):
        n = Notification(
            id="x", title="T", message="M",
            severity="info", source="s", timestamp=0,
        )
        assert n.read is False

    def test_read_can_be_set_true(self):
        n = Notification(
            id="x", title="T", message="M",
            severity="info", source="s", timestamp=0, read=True,
        )
        assert n.read is True


# ── NotificationManager ────────────────────────────────────────────

class TestNotificationManager:
    """Tests for NotificationManager core operations."""

    def test_add_returns_id(self):
        mgr = NotificationManager()
        nid = mgr.add("Title", "Message")
        assert isinstance(nid, str)
        assert len(nid) > 0

    def test_add_with_severity(self):
        mgr = NotificationManager()
        mgr.add("T", "M", severity="critical", source="sensor")
        items = mgr.get_all()
        assert items[0]["severity"] == "critical"
        assert items[0]["source"] == "sensor"

    def test_invalid_severity_defaults_to_info(self):
        mgr = NotificationManager()
        mgr.add("T", "M", severity="invalid_level")
        items = mgr.get_all()
        assert items[0]["severity"] == "info"

    def test_add_with_entity_id(self):
        mgr = NotificationManager()
        mgr.add("T", "M", entity_id="ble_AA:BB:CC")
        items = mgr.get_all()
        assert items[0]["entity_id"] == "ble_AA:BB:CC"

    def test_get_all_returns_newest_first(self):
        mgr = NotificationManager()
        mgr.add("First", "1")
        mgr.add("Second", "2")
        mgr.add("Third", "3")
        items = mgr.get_all()
        assert items[0]["title"] == "Third"
        assert items[-1]["title"] == "First"

    def test_get_all_limit(self):
        mgr = NotificationManager()
        for i in range(10):
            mgr.add(f"N{i}", "msg")
        items = mgr.get_all(limit=3)
        assert len(items) == 3

    def test_get_all_since_filter(self):
        mgr = NotificationManager()
        mgr.add("Old", "old")
        cutoff = time.time()
        time.sleep(0.01)
        mgr.add("New", "new")
        items = mgr.get_all(since=cutoff)
        assert len(items) == 1
        assert items[0]["title"] == "New"

    def test_get_unread(self):
        mgr = NotificationManager()
        id1 = mgr.add("A", "a")
        mgr.add("B", "b")
        mgr.mark_read(id1)
        unread = mgr.get_unread()
        assert len(unread) == 1
        assert unread[0]["title"] == "B"

    def test_mark_read(self):
        mgr = NotificationManager()
        nid = mgr.add("T", "M")
        assert mgr.mark_read(nid) is True
        assert mgr.count_unread() == 0

    def test_mark_read_nonexistent(self):
        mgr = NotificationManager()
        assert mgr.mark_read("nonexistent") is False

    def test_mark_all_read(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        mgr.add("C", "c")
        count = mgr.mark_all_read()
        assert count == 3
        assert mgr.count_unread() == 0

    def test_mark_all_read_returns_zero_when_all_read(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.mark_all_read()
        assert mgr.mark_all_read() == 0

    def test_count_unread(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        assert mgr.count_unread() == 2
        mgr.mark_all_read()
        assert mgr.count_unread() == 0

    def test_clear(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        count = mgr.clear()
        assert count == 2
        assert mgr.get_all() == []
        assert mgr.count_unread() == 0

    def test_clear_empty(self):
        mgr = NotificationManager()
        assert mgr.clear() == 0

    def test_max_notifications_evicts_oldest(self):
        mgr = NotificationManager(max_notifications=5)
        for i in range(10):
            mgr.add(f"N{i}", f"msg{i}")
        items = mgr.get_all(limit=100)
        assert len(items) == 5
        # Most recent should be N9
        assert items[0]["title"] == "N9"


# ── Broadcast callback ─────────────────────────────────────────────

class TestNotificationBroadcast:
    """Tests for the broadcast callback mechanism."""

    def test_broadcast_called_on_add(self):
        received = []
        mgr = NotificationManager(broadcast=lambda msg: received.append(msg))
        mgr.add("Alert", "Test")
        assert len(received) == 1
        assert received[0]["type"] == "notification:new"
        assert received[0]["data"]["title"] == "Alert"

    def test_broadcast_exception_does_not_crash(self):
        def bad_broadcast(msg):
            raise RuntimeError("broadcast failed")

        mgr = NotificationManager(broadcast=bad_broadcast)
        # Should not raise
        nid = mgr.add("T", "M")
        assert isinstance(nid, str)

    def test_no_broadcast_when_none(self):
        mgr = NotificationManager(broadcast=None)
        nid = mgr.add("T", "M")
        assert isinstance(nid, str)


# ── Thread safety ───────────────────────────────────────────────────

class TestNotificationThreadSafety:
    """Verify NotificationManager is thread-safe."""

    def test_concurrent_adds(self):
        mgr = NotificationManager()
        errors = []

        def add_batch(start):
            try:
                for i in range(50):
                    mgr.add(f"Thread-{start}-{i}", "msg")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mgr.count_unread() == 200
