# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AuditStore — security compliance audit logging."""

import time

import pytest

from tritium_lib.store.audit_log import (
    AuditStore,
    AuditEntry,
    AuditSeverity,
)


@pytest.fixture
def store():
    """Create an in-memory AuditStore."""
    return AuditStore(db_path=":memory:")


# -- Basic logging tests ---------------------------------------------------

class TestBasicLogging:
    def test_log_entry(self, store):
        row_id = store.log("user:admin", "login", "Authenticated via JWT")
        assert row_id >= 1

    def test_log_with_all_fields(self, store):
        row_id = store.log(
            actor="user:operator",
            action="delete_target",
            detail="Removed target ble_AA:BB",
            severity="warning",
            resource="target",
            resource_id="ble_AA:BB",
            ip_address="192.168.1.50",
            metadata={"reason": "duplicate"},
        )
        entry = store.get_entry(row_id)
        assert entry is not None
        assert entry.actor == "user:operator"
        assert entry.action == "delete_target"
        assert entry.severity == "warning"
        assert entry.resource == "target"
        assert entry.resource_id == "ble_AA:BB"
        assert entry.ip_address == "192.168.1.50"
        assert entry.metadata["reason"] == "duplicate"

    def test_log_timestamp(self, store):
        before = time.time()
        row_id = store.log("system", "startup", "System started")
        after = time.time()
        entry = store.get_entry(row_id)
        assert before <= entry.timestamp <= after

    def test_log_custom_timestamp(self, store):
        row_id = store.log("system", "test", timestamp=1234567890.0)
        entry = store.get_entry(row_id)
        assert entry.timestamp == 1234567890.0

    def test_multiple_entries(self, store):
        for i in range(10):
            store.log(f"user:{i}", "action", f"Entry {i}")
        assert store.count() == 10


# -- Query tests -----------------------------------------------------------

class TestQuerying:
    def test_query_all(self, store):
        store.log("user:admin", "login")
        store.log("user:admin", "view")
        store.log("plugin:acoustic", "classify")
        entries = store.query()
        assert len(entries) == 3

    def test_query_by_actor(self, store):
        store.log("user:admin", "login")
        store.log("plugin:acoustic", "classify")
        entries = store.query(actor="user:admin")
        assert len(entries) == 1
        assert entries[0].actor == "user:admin"

    def test_query_by_action(self, store):
        store.log("user:admin", "login")
        store.log("user:admin", "logout")
        entries = store.query(action="login")
        assert len(entries) == 1
        assert entries[0].action == "login"

    def test_query_by_severity(self, store):
        store.log("system", "error", severity="error")
        store.log("system", "info", severity="info")
        entries = store.query(severity="error")
        assert len(entries) == 1

    def test_query_by_resource(self, store):
        store.log("user:admin", "create", resource="target")
        store.log("user:admin", "create", resource="user")
        entries = store.query(resource="target")
        assert len(entries) == 1

    def test_query_by_time_range(self, store):
        store.log("system", "a", timestamp=1000.0)
        store.log("system", "b", timestamp=2000.0)
        store.log("system", "c", timestamp=3000.0)
        entries = store.query(start_time=1500.0, end_time=2500.0)
        assert len(entries) == 1
        assert entries[0].action == "b"

    def test_query_limit(self, store):
        for i in range(20):
            store.log("user", "action")
        entries = store.query(limit=5)
        assert len(entries) == 5

    def test_query_offset(self, store):
        for i in range(10):
            store.log("user", "action", detail=str(i), timestamp=1000.0 + i)
        entries = store.query(limit=3, offset=5)
        assert len(entries) == 3

    def test_query_order(self, store):
        """Results should be most-recent first."""
        store.log("system", "first", timestamp=1000.0)
        store.log("system", "second", timestamp=2000.0)
        entries = store.query()
        assert entries[0].action == "second"
        assert entries[1].action == "first"


# -- Count tests -----------------------------------------------------------

class TestCounting:
    def test_count_all(self, store):
        for _ in range(5):
            store.log("user", "action")
        assert store.count() == 5

    def test_count_filtered(self, store):
        store.log("user:admin", "login")
        store.log("user:admin", "login")
        store.log("user:op", "login")
        assert store.count(actor="user:admin") == 2
        assert store.count(action="login") == 3

    def test_count_by_severity(self, store):
        store.log("sys", "err", severity="error")
        store.log("sys", "err", severity="error")
        store.log("sys", "info", severity="info")
        assert store.count(severity="error") == 2


# -- Statistics tests ------------------------------------------------------

class TestStats:
    def test_get_stats(self, store):
        store.log("user:admin", "login", severity="info")
        store.log("plugin:acoustic", "classify", severity="warning")
        store.log("system", "error", severity="error")
        stats = store.get_stats()
        assert stats["total_entries"] == 3
        assert stats["unique_actors"] == 3
        assert "info" in stats["by_severity"]
        assert "login" in stats["top_actions"]
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None

    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total_entries"] == 0
        assert stats["unique_actors"] == 0


# -- Cleanup tests ---------------------------------------------------------

class TestCleanup:
    def test_cleanup(self, store):
        for i in range(20):
            store.log("user", "action", timestamp=1000.0 + i)
        deleted = store.cleanup(keep=10)
        assert deleted == 10
        assert store.count() == 10

    def test_cleanup_no_op(self, store):
        for _ in range(5):
            store.log("user", "action")
        deleted = store.cleanup(keep=10)
        assert deleted == 0

    def test_clear(self, store):
        for _ in range(5):
            store.log("user", "action")
        deleted = store.clear()
        assert deleted == 5
        assert store.count() == 0


# -- AuditEntry dataclass tests -------------------------------------------

class TestAuditEntry:
    def test_to_dict(self):
        entry = AuditEntry(
            id=1,
            timestamp=1000.0,
            actor="user:admin",
            action="login",
            detail="Logged in",
            severity="info",
        )
        d = entry.to_dict()
        assert d["id"] == 1
        assert d["actor"] == "user:admin"
        assert d["action"] == "login"
        assert d["severity"] == "info"

    def test_default_values(self):
        entry = AuditEntry()
        assert entry.actor == ""
        assert entry.metadata == {}


# -- Severity enum tests ---------------------------------------------------

class TestSeverity:
    def test_severity_values(self):
        assert AuditSeverity.DEBUG == "debug"
        assert AuditSeverity.INFO == "info"
        assert AuditSeverity.WARNING == "warning"
        assert AuditSeverity.ERROR == "error"
        assert AuditSeverity.CRITICAL == "critical"


# -- Import tests ----------------------------------------------------------

class TestImports:
    def test_import_from_store(self):
        from tritium_lib.store import AuditStore, AuditEntry, AuditSeverity
        assert AuditStore is not None
