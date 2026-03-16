# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.notifications — Notification model and NotificationManager."""

import time
import pytest

from tritium_lib.notifications import Notification, NotificationManager


# ---------------------------------------------------------------------------
# Notification dataclass
# ---------------------------------------------------------------------------

class TestNotification:
    def test_to_dict(self):
        n = Notification(
            id="abc123",
            title="Test",
            message="Hello",
            severity="info",
            source="unit_test",
            timestamp=1000.0,
        )
        d = n.to_dict()
        assert d["id"] == "abc123"
        assert d["title"] == "Test"
        assert d["severity"] == "info"
        assert d["read"] is False
        assert d["entity_id"] is None

    def test_with_entity_id(self):
        n = Notification(
            id="x", title="T", message="M",
            severity="critical", source="s",
            timestamp=0.0, entity_id="target-42",
        )
        assert n.entity_id == "target-42"
        assert n.to_dict()["entity_id"] == "target-42"


# ---------------------------------------------------------------------------
# NotificationManager basics
# ---------------------------------------------------------------------------

class TestNotificationManager:
    def test_add_returns_id(self):
        mgr = NotificationManager()
        nid = mgr.add("Title", "Message")
        assert isinstance(nid, str)
        assert len(nid) == 12

    def test_get_all(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        all_notifs = mgr.get_all()
        assert len(all_notifs) == 2
        # Newest first
        assert all_notifs[0]["title"] == "B"
        assert all_notifs[1]["title"] == "A"

    def test_get_unread(self):
        mgr = NotificationManager()
        nid = mgr.add("A", "a")
        mgr.add("B", "b")
        assert len(mgr.get_unread()) == 2
        mgr.mark_read(nid)
        unread = mgr.get_unread()
        assert len(unread) == 1
        assert unread[0]["title"] == "B"

    def test_mark_read_returns_false_for_missing(self):
        mgr = NotificationManager()
        assert mgr.mark_read("nonexistent") is False

    def test_mark_all_read(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        count = mgr.mark_all_read()
        assert count == 2
        assert mgr.count_unread() == 0

    def test_count_unread(self):
        mgr = NotificationManager()
        assert mgr.count_unread() == 0
        mgr.add("A", "a")
        assert mgr.count_unread() == 1

    def test_severity_validation(self):
        mgr = NotificationManager()
        mgr.add("A", "a", severity="invalid")
        notifs = mgr.get_all()
        assert notifs[0]["severity"] == "info"

    def test_valid_severities(self):
        mgr = NotificationManager()
        for sev in ("info", "warning", "critical"):
            mgr.add("T", "m", severity=sev)
        notifs = mgr.get_all()
        severities = {n["severity"] for n in notifs}
        assert severities == {"info", "warning", "critical"}

    def test_max_notifications(self):
        mgr = NotificationManager(max_notifications=5)
        for i in range(10):
            mgr.add(f"N{i}", f"msg{i}")
        all_notifs = mgr.get_all(limit=100)
        assert len(all_notifs) == 5

    def test_get_all_with_limit(self):
        mgr = NotificationManager()
        for i in range(10):
            mgr.add(f"N{i}", f"msg{i}")
        assert len(mgr.get_all(limit=3)) == 3

    def test_get_all_since(self):
        mgr = NotificationManager()
        mgr.add("Old", "old")
        cutoff = time.time()
        time.sleep(0.01)
        mgr.add("New", "new")
        result = mgr.get_all(since=cutoff)
        assert len(result) == 1
        assert result[0]["title"] == "New"

    def test_clear(self):
        mgr = NotificationManager()
        mgr.add("A", "a")
        mgr.add("B", "b")
        removed = mgr.clear()
        assert removed == 2
        assert len(mgr.get_all()) == 0

    def test_entity_id_stored(self):
        mgr = NotificationManager()
        mgr.add("Alert", "target spotted", entity_id="target-7")
        notifs = mgr.get_all()
        assert notifs[0]["entity_id"] == "target-7"


# ---------------------------------------------------------------------------
# Broadcast callback
# ---------------------------------------------------------------------------

class TestBroadcast:
    def test_broadcast_called(self):
        messages = []
        mgr = NotificationManager(broadcast=messages.append)
        mgr.add("Test", "hello")
        assert len(messages) == 1
        assert messages[0]["type"] == "notification:new"
        assert messages[0]["data"]["title"] == "Test"

    def test_broadcast_exception_does_not_break(self):
        def bad_broadcast(msg):
            raise RuntimeError("boom")

        mgr = NotificationManager(broadcast=bad_broadcast)
        # Should not raise
        nid = mgr.add("Test", "hello")
        assert isinstance(nid, str)
        assert mgr.count_unread() == 1
