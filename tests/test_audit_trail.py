# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.audit — persistent audit trail for compliance."""

import os
import tempfile
import time

import pytest

from tritium_lib.audit import (
    AuditTrail,
    AuditEntry,
    AuditSeverity,
    AuditQuery,
    AuditAction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trail():
    """In-memory AuditTrail for fast tests."""
    t = AuditTrail(db_path=":memory:")
    yield t
    t.close()


@pytest.fixture
def trail_on_disk(tmp_path):
    """AuditTrail backed by an on-disk SQLite file."""
    db = tmp_path / "audit.db"
    t = AuditTrail(db_path=str(db))
    yield t
    t.close()


# ---------------------------------------------------------------------------
# 1. Basic recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_returns_entry(self, trail):
        entry = trail.record("user:admin", "login", details="JWT auth")
        assert isinstance(entry, AuditEntry)
        assert entry.actor == "user:admin"
        assert entry.action == "login"
        assert entry.details == "JWT auth"
        assert entry.entry_id  # non-empty UUID
        assert entry.db_id >= 1

    def test_record_auto_timestamp(self, trail):
        before = time.time()
        entry = trail.record("user:admin", "login")
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_record_custom_timestamp(self, trail):
        entry = trail.record("system", "test", timestamp=1234567890.0)
        assert entry.timestamp == 1234567890.0

    def test_record_all_fields(self, trail):
        entry = trail.record(
            actor="user:operator",
            action="delete_target",
            resource="target",
            resource_id="ble_AA:BB",
            details="Removed stale target",
            severity="warning",
            source_ip="192.168.1.50",
            metadata={"reason": "duplicate"},
        )
        assert entry.resource == "target"
        assert entry.resource_id == "ble_AA:BB"
        assert entry.severity == "warning"
        assert entry.source_ip == "192.168.1.50"
        assert entry.metadata["reason"] == "duplicate"

    def test_record_unique_entry_ids(self, trail):
        e1 = trail.record("user", "a")
        e2 = trail.record("user", "b")
        assert e1.entry_id != e2.entry_id


# ---------------------------------------------------------------------------
# 2. Typed compliance helpers
# ---------------------------------------------------------------------------

class TestComplianceHelpers:
    def test_target_accessed(self, trail):
        entry = trail.record_target_accessed("user:analyst", "ble_AA:BB:CC",
                                              details="Viewed dossier")
        assert entry.action == AuditAction.TARGET_ACCESSED
        assert entry.resource == "target"
        assert entry.resource_id == "ble_AA:BB:CC"
        assert entry.severity == "info"

    def test_target_created(self, trail):
        entry = trail.record_target_created("system", "det_person_1")
        assert entry.action == AuditAction.TARGET_CREATED
        assert entry.resource == "target"

    def test_target_deleted(self, trail):
        entry = trail.record_target_deleted("user:admin", "ble_00:11",
                                             source_ip="10.0.0.5")
        assert entry.action == AuditAction.TARGET_DELETED
        assert entry.severity == "warning"
        assert entry.source_ip == "10.0.0.5"

    def test_zone_modified(self, trail):
        entry = trail.record_zone_modified("user:admin", "zone_alpha",
                                            details="Expanded perimeter")
        assert entry.action == AuditAction.ZONE_MODIFIED
        assert entry.resource == "zone"
        assert entry.resource_id == "zone_alpha"

    def test_alert_acknowledged(self, trail):
        entry = trail.record_alert_acknowledged("user:op", "alert_42")
        assert entry.action == AuditAction.ALERT_ACKNOWLEDGED
        assert entry.resource == "alert"
        assert entry.resource_id == "alert_42"

    def test_report_generated(self, trail):
        entry = trail.record_report_generated("user:analyst", "rpt_2026_q1",
                                               details="Quarterly compliance")
        assert entry.action == AuditAction.REPORT_GENERATED
        assert entry.resource == "report"

    def test_config_changed(self, trail):
        entry = trail.record_config_changed(
            "user:admin", "mqtt.broker_url",
            old_value="localhost", new_value="10.0.0.1",
        )
        assert entry.action == AuditAction.CONFIG_CHANGED
        assert entry.resource == "config"
        assert entry.resource_id == "mqtt.broker_url"
        assert entry.metadata["old_value"] == "localhost"
        assert entry.metadata["new_value"] == "10.0.0.1"
        assert "localhost" in entry.details
        assert "10.0.0.1" in entry.details
        assert entry.severity == "warning"

    def test_auth_login(self, trail):
        entry = trail.record_auth_login("user:admin", source_ip="192.168.1.1")
        assert entry.action == AuditAction.AUTH_LOGIN
        assert entry.resource == "auth"
        assert "Login successful" in entry.details

    def test_auth_failed(self, trail):
        entry = trail.record_auth_failed("user:unknown", source_ip="10.0.0.99")
        assert entry.action == AuditAction.AUTH_FAILED
        assert entry.severity == "warning"

    def test_data_exported(self, trail):
        entry = trail.record_data_exported("user:admin", resource_id="export_001")
        assert entry.action == AuditAction.DATA_EXPORTED
        assert entry.resource == "data"

    def test_data_purged(self, trail):
        entry = trail.record_data_purged("user:admin", details="Purged 90-day data")
        assert entry.action == AuditAction.DATA_PURGED
        assert entry.severity == "critical"


# ---------------------------------------------------------------------------
# 3. AuditQuery — search and filter
# ---------------------------------------------------------------------------

class TestAuditQuery:
    def test_query_all(self, trail):
        trail.record("user:a", "login")
        trail.record("user:b", "logout")
        trail.record("system", "error")
        results = trail.search(AuditQuery())
        assert len(results) == 3

    def test_query_by_actor(self, trail):
        trail.record("user:admin", "login")
        trail.record("plugin:acoustic", "classify")
        results = trail.search(AuditQuery().by_actor("user:admin"))
        assert len(results) == 1
        assert results[0].actor == "user:admin"

    def test_query_by_action(self, trail):
        trail.record("user:admin", "login")
        trail.record("user:admin", "logout")
        results = trail.search(AuditQuery().by_action("login"))
        assert len(results) == 1

    def test_query_by_severity(self, trail):
        trail.record("system", "startup", severity="info")
        trail.record("system", "crash", severity="critical")
        results = trail.search(AuditQuery().with_severity("critical"))
        assert len(results) == 1
        assert results[0].action == "crash"

    def test_query_by_resource(self, trail):
        trail.record("user", "create", resource="target")
        trail.record("user", "modify", resource="zone")
        results = trail.search(AuditQuery().with_resource("target"))
        assert len(results) == 1

    def test_query_by_resource_id(self, trail):
        trail.record("user", "access", resource="target", resource_id="ble_01")
        trail.record("user", "access", resource="target", resource_id="ble_02")
        results = trail.search(AuditQuery().with_resource_id("ble_01"))
        assert len(results) == 1

    def test_query_by_time_range(self, trail):
        trail.record("sys", "a", timestamp=1000.0)
        trail.record("sys", "b", timestamp=2000.0)
        trail.record("sys", "c", timestamp=3000.0)
        results = trail.search(AuditQuery().since(1500.0).until(2500.0))
        assert len(results) == 1
        assert results[0].action == "b"

    def test_query_by_ip(self, trail):
        trail.record("user", "login", source_ip="10.0.0.1")
        trail.record("user", "login", source_ip="10.0.0.2")
        results = trail.search(AuditQuery().from_ip("10.0.0.1"))
        assert len(results) == 1

    def test_query_containing_keyword(self, trail):
        trail.record("user", "action", details="Viewed target dossier")
        trail.record("user", "action", details="Changed config value")
        results = trail.search(AuditQuery().containing("dossier"))
        assert len(results) == 1

    def test_query_limit_and_offset(self, trail):
        for i in range(20):
            trail.record("user", "action", details=str(i), timestamp=1000.0 + i)
        results = trail.search(AuditQuery().limit(5).offset(10))
        assert len(results) == 5

    def test_query_chained_filters(self, trail):
        trail.record("user:admin", "login", severity="info", source_ip="10.0.0.1")
        trail.record("user:admin", "delete_target", severity="warning", source_ip="10.0.0.1")
        trail.record("user:op", "login", severity="info", source_ip="10.0.0.2")
        results = trail.search(
            AuditQuery()
            .by_actor("user:admin")
            .with_severity("warning")
            .from_ip("10.0.0.1")
        )
        assert len(results) == 1
        assert results[0].action == "delete_target"

    def test_query_order_most_recent_first(self, trail):
        trail.record("sys", "first", timestamp=1000.0)
        trail.record("sys", "second", timestamp=2000.0)
        results = trail.search(AuditQuery())
        assert results[0].action == "second"
        assert results[1].action == "first"


# ---------------------------------------------------------------------------
# 4. Retrieval by ID
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_get_by_entry_id(self, trail):
        entry = trail.record("user", "test")
        fetched = trail.get_by_id(entry.entry_id)
        assert fetched is not None
        assert fetched.entry_id == entry.entry_id
        assert fetched.actor == "user"

    def test_get_by_db_id(self, trail):
        entry = trail.record("user", "test")
        fetched = trail.get_by_db_id(entry.db_id)
        assert fetched is not None
        assert fetched.db_id == entry.db_id

    def test_get_missing_entry(self, trail):
        assert trail.get_by_id("nonexistent") is None
        assert trail.get_by_db_id(99999) is None


# ---------------------------------------------------------------------------
# 5. Counting
# ---------------------------------------------------------------------------

class TestCounting:
    def test_count_all(self, trail):
        for _ in range(7):
            trail.record("user", "action")
        assert trail.count() == 7

    def test_count_with_query(self, trail):
        trail.record("user:admin", "login")
        trail.record("user:admin", "login")
        trail.record("user:op", "login")
        assert trail.count(AuditQuery().by_actor("user:admin")) == 2


# ---------------------------------------------------------------------------
# 6. Export for compliance
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_all(self, trail):
        trail.record("user", "a", timestamp=1000.0)
        trail.record("user", "b", timestamp=2000.0)
        exported = trail.export()
        assert len(exported) == 2
        # Export is chronological (ascending)
        assert exported[0]["action"] == "a"
        assert exported[1]["action"] == "b"

    def test_export_time_range(self, trail):
        trail.record("sys", "a", timestamp=1000.0)
        trail.record("sys", "b", timestamp=2000.0)
        trail.record("sys", "c", timestamp=3000.0)
        exported = trail.export(start_time=1500.0, end_time=2500.0)
        assert len(exported) == 1
        assert exported[0]["action"] == "b"

    def test_export_by_actions(self, trail):
        trail.record("sys", "login")
        trail.record("sys", "logout")
        trail.record("sys", "error")
        exported = trail.export(actions=["login", "logout"])
        assert len(exported) == 2
        actions = {e["action"] for e in exported}
        assert actions == {"login", "logout"}

    def test_export_single_action(self, trail):
        trail.record("sys", "login")
        trail.record("sys", "logout")
        exported = trail.export(actions=["login"])
        assert len(exported) == 1

    def test_export_returns_dicts(self, trail):
        trail.record("user", "test", metadata={"key": "val"})
        exported = trail.export()
        assert isinstance(exported[0], dict)
        assert "entry_id" in exported[0]
        assert "timestamp" in exported[0]
        assert exported[0]["metadata"]["key"] == "val"


# ---------------------------------------------------------------------------
# 7. Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_basic(self, trail):
        trail.record("user:admin", "login", severity="info")
        trail.record("plugin:acoustic", "classify", severity="warning")
        trail.record("system", "error", severity="error")
        stats = trail.get_stats()
        assert stats["total_entries"] == 3
        assert stats["unique_actors"] == 3
        assert "info" in stats["by_severity"]
        assert "login" in stats["top_actions"]
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None

    def test_stats_empty(self, trail):
        stats = trail.get_stats()
        assert stats["total_entries"] == 0
        assert stats["unique_actors"] == 0


# ---------------------------------------------------------------------------
# 8. Rotation / cleanup
# ---------------------------------------------------------------------------

class TestRotation:
    def test_rotate_prunes_oldest(self, trail):
        for i in range(25):
            trail.record("user", "action", timestamp=1000.0 + i)
        deleted = trail.rotate(keep=10)
        assert deleted == 15
        assert trail.count() == 10

    def test_rotate_no_op_when_under_limit(self, trail):
        for _ in range(5):
            trail.record("user", "action")
        deleted = trail.rotate(keep=10)
        assert deleted == 0

    def test_auto_rotate_on_threshold(self):
        # max_entries=10, threshold=1.1 -> auto-rotate at 11 entries
        trail = AuditTrail(db_path=":memory:", max_entries=10,
                           auto_rotate_threshold=1.1)
        for i in range(12):
            trail.record("user", "action", timestamp=1000.0 + i)
        # After auto-rotation, should be <= max_entries
        assert trail.count() <= 10
        trail.close()

    def test_clear_all(self, trail):
        for _ in range(5):
            trail.record("user", "action")
        deleted = trail.clear()
        assert deleted == 5
        assert trail.count() == 0


# ---------------------------------------------------------------------------
# 9. Persistence (on-disk)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        db = str(tmp_path / "audit.db")

        t1 = AuditTrail(db_path=db)
        entry = t1.record("user:admin", "login", details="First session")
        entry_id = entry.entry_id
        t1.close()

        t2 = AuditTrail(db_path=db)
        fetched = t2.get_by_id(entry_id)
        assert fetched is not None
        assert fetched.actor == "user:admin"
        assert fetched.details == "First session"
        t2.close()

    def test_creates_parent_directories(self, tmp_path):
        db = str(tmp_path / "deep" / "nested" / "audit.db")
        t = AuditTrail(db_path=db)
        t.record("test", "test")
        assert t.count() == 1
        t.close()
        assert os.path.exists(db)


# ---------------------------------------------------------------------------
# 10. AuditEntry model
# ---------------------------------------------------------------------------

class TestAuditEntryModel:
    def test_to_dict(self):
        entry = AuditEntry.create(
            actor="user:admin",
            action="login",
            details="JWT auth",
            severity="info",
        )
        d = entry.to_dict()
        assert d["actor"] == "user:admin"
        assert d["action"] == "login"
        assert d["details"] == "JWT auth"
        assert "entry_id" in d

    def test_create_factory(self):
        entry = AuditEntry.create(
            actor="system",
            action="startup",
        )
        assert entry.entry_id  # non-empty
        assert entry.timestamp > 0
        assert entry.actor == "system"

    def test_immutable(self):
        entry = AuditEntry.create(actor="user", action="test")
        with pytest.raises(AttributeError):
            entry.actor = "modified"

    def test_default_values(self):
        entry = AuditEntry()
        assert entry.actor == ""
        assert entry.metadata == {}
        assert entry.severity == "info"


# ---------------------------------------------------------------------------
# 11. AuditAction enum
# ---------------------------------------------------------------------------

class TestAuditActionEnum:
    def test_all_required_actions_present(self):
        assert AuditAction.TARGET_ACCESSED == "target_accessed"
        assert AuditAction.ZONE_MODIFIED == "zone_modified"
        assert AuditAction.ALERT_ACKNOWLEDGED == "alert_acknowledged"
        assert AuditAction.REPORT_GENERATED == "report_generated"
        assert AuditAction.CONFIG_CHANGED == "config_changed"

    def test_action_is_string(self):
        assert isinstance(AuditAction.TARGET_ACCESSED, str)
        assert AuditAction.AUTH_LOGIN == "auth_login"

    def test_action_count(self):
        # Ensure we have a substantial set of actions
        assert len(AuditAction) >= 30


# ---------------------------------------------------------------------------
# 12. AuditSeverity enum
# ---------------------------------------------------------------------------

class TestAuditSeverityEnum:
    def test_severity_values(self):
        assert AuditSeverity.DEBUG == "debug"
        assert AuditSeverity.INFO == "info"
        assert AuditSeverity.WARNING == "warning"
        assert AuditSeverity.ERROR == "error"
        assert AuditSeverity.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# 13. AuditQuery builder
# ---------------------------------------------------------------------------

class TestAuditQueryBuilder:
    def test_empty_query_builds_empty_where(self):
        q = AuditQuery()
        where, params = q.build_sql()
        assert where == ""
        assert params == []

    def test_single_filter(self):
        q = AuditQuery().by_actor("user:admin")
        where, params = q.build_sql()
        assert "actor = ?" in where
        assert "user:admin" in params

    def test_chained_filters(self):
        q = (AuditQuery()
             .by_actor("user:admin")
             .by_action("login")
             .with_severity("info"))
        where, params = q.build_sql()
        assert "actor = ?" in where
        assert "action = ?" in where
        assert "severity = ?" in where
        assert len(params) == 3

    def test_time_range_filter(self):
        q = AuditQuery().since(1000.0).until(2000.0)
        where, params = q.build_sql()
        assert "timestamp >= ?" in where
        assert "timestamp <= ?" in where
        assert 1000.0 in params
        assert 2000.0 in params

    def test_keyword_filter(self):
        q = AuditQuery().containing("dossier")
        where, params = q.build_sql()
        assert "LIKE" in where
        assert "%dossier%" in params


# ---------------------------------------------------------------------------
# 14. Import paths
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_from_audit_package(self):
        from tritium_lib.audit import AuditTrail, AuditEntry, AuditQuery, AuditAction
        assert AuditTrail is not None
        assert AuditEntry is not None
        assert AuditQuery is not None
        assert AuditAction is not None

    def test_import_individual_modules(self):
        from tritium_lib.audit.trail import AuditTrail
        from tritium_lib.audit.entry import AuditEntry
        from tritium_lib.audit.query import AuditQuery
        from tritium_lib.audit.actions import AuditAction
        assert all([AuditTrail, AuditEntry, AuditQuery, AuditAction])
