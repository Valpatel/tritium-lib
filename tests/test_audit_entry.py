# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.audit.entry — AuditEntry model."""

import time

from tritium_lib.audit.entry import AuditEntry, AuditSeverity


class TestAuditSeverity:
    def test_enum_values(self):
        assert AuditSeverity.DEBUG == "debug"
        assert AuditSeverity.INFO == "info"
        assert AuditSeverity.WARNING == "warning"
        assert AuditSeverity.ERROR == "error"
        assert AuditSeverity.CRITICAL == "critical"

    def test_enum_count(self):
        assert len(AuditSeverity) == 5


class TestAuditEntry:
    def test_default_construction(self):
        e = AuditEntry()
        assert e.entry_id == ""
        assert e.timestamp == 0.0
        assert e.actor == ""
        assert e.action == ""
        assert e.resource == ""
        assert e.resource_id == ""
        assert e.details == ""
        assert e.severity == "info"
        assert e.source_ip == ""
        assert e.metadata == {}
        assert e.db_id == 0

    def test_frozen_dataclass(self):
        e = AuditEntry(actor="admin")
        try:
            e.actor = "other"
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass

    def test_to_dict(self):
        e = AuditEntry(
            entry_id="abc123",
            timestamp=1000.0,
            actor="user:admin",
            action="config_changed",
            resource="zone",
            resource_id="zone_1",
            details="Changed threshold",
            severity="warning",
            source_ip="10.0.0.1",
            metadata={"old": 5, "new": 10},
            db_id=42,
        )
        d = e.to_dict()
        assert d["entry_id"] == "abc123"
        assert d["timestamp"] == 1000.0
        assert d["actor"] == "user:admin"
        assert d["action"] == "config_changed"
        assert d["resource"] == "zone"
        assert d["resource_id"] == "zone_1"
        assert d["details"] == "Changed threshold"
        assert d["severity"] == "warning"
        assert d["source_ip"] == "10.0.0.1"
        assert d["metadata"] == {"old": 5, "new": 10}
        assert d["db_id"] == 42

    def test_to_dict_metadata_is_copy(self):
        meta = {"key": "value"}
        e = AuditEntry(metadata=meta)
        d = e.to_dict()
        d["metadata"]["key"] = "changed"
        assert e.metadata["key"] == "value"

    def test_create_factory(self):
        e = AuditEntry.create(
            actor="plugin:acoustic",
            action="target_detected",
            resource="target",
            resource_id="ble_AA:BB:CC",
            details="New BLE device found",
            severity="info",
            source_ip="192.168.1.1",
            metadata={"rssi": -55},
        )
        assert e.actor == "plugin:acoustic"
        assert e.action == "target_detected"
        assert e.resource == "target"
        assert e.resource_id == "ble_AA:BB:CC"
        assert e.details == "New BLE device found"
        assert e.severity == "info"
        assert e.source_ip == "192.168.1.1"
        assert e.metadata == {"rssi": -55}
        assert len(e.entry_id) > 0  # UUID generated
        assert e.timestamp > 0  # auto-generated

    def test_create_auto_timestamp(self):
        before = time.time()
        e = AuditEntry.create(actor="test", action="test_action")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_create_custom_timestamp(self):
        e = AuditEntry.create(
            actor="test", action="test_action", timestamp=12345.0
        )
        assert e.timestamp == 12345.0

    def test_create_no_metadata(self):
        e = AuditEntry.create(actor="test", action="test")
        assert e.metadata == {}

    def test_create_unique_ids(self):
        e1 = AuditEntry.create(actor="test", action="a1")
        e2 = AuditEntry.create(actor="test", action="a2")
        assert e1.entry_id != e2.entry_id
