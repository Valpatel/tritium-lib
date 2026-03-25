# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.audit.query — fluent AuditQuery builder."""

from tritium_lib.audit.query import AuditQuery


class TestAuditQueryDefaults:
    def test_default_values(self):
        q = AuditQuery()
        assert q.actor is None
        assert q.action is None
        assert q.severity is None
        assert q.resource is None
        assert q.resource_id is None
        assert q.start_time is None
        assert q.end_time is None
        assert q.source_ip is None
        assert q.keyword is None
        assert q.max_results == 100
        assert q.skip == 0


class TestAuditQueryFluent:
    def test_by_actor(self):
        q = AuditQuery().by_actor("user:admin")
        assert q.actor == "user:admin"

    def test_by_action(self):
        q = AuditQuery().by_action("config_changed")
        assert q.action == "config_changed"

    def test_with_severity(self):
        q = AuditQuery().with_severity("warning")
        assert q.severity == "warning"

    def test_with_resource(self):
        q = AuditQuery().with_resource("zone")
        assert q.resource == "zone"

    def test_with_resource_id(self):
        q = AuditQuery().with_resource_id("zone_alpha")
        assert q.resource_id == "zone_alpha"

    def test_since(self):
        q = AuditQuery().since(1000.0)
        assert q.start_time == 1000.0

    def test_until(self):
        q = AuditQuery().until(2000.0)
        assert q.end_time == 2000.0

    def test_from_ip(self):
        q = AuditQuery().from_ip("10.0.0.1")
        assert q.source_ip == "10.0.0.1"

    def test_containing(self):
        q = AuditQuery().containing("threshold")
        assert q.keyword == "threshold"

    def test_limit(self):
        q = AuditQuery().limit(25)
        assert q.max_results == 25

    def test_offset(self):
        q = AuditQuery().offset(10)
        assert q.skip == 10

    def test_chaining(self):
        q = (
            AuditQuery()
            .by_actor("user:admin")
            .by_action("target_accessed")
            .since(1000.0)
            .until(2000.0)
            .with_severity("info")
            .with_resource("target")
            .with_resource_id("ble_ABC")
            .limit(50)
            .offset(5)
        )
        assert q.actor == "user:admin"
        assert q.action == "target_accessed"
        assert q.start_time == 1000.0
        assert q.end_time == 2000.0
        assert q.severity == "info"
        assert q.resource == "target"
        assert q.resource_id == "ble_ABC"
        assert q.max_results == 50
        assert q.skip == 5

    def test_returns_self(self):
        q = AuditQuery()
        assert q.by_actor("x") is q
        assert q.by_action("y") is q
        assert q.with_severity("z") is q
        assert q.with_resource("r") is q
        assert q.with_resource_id("ri") is q
        assert q.since(1.0) is q
        assert q.until(2.0) is q
        assert q.from_ip("ip") is q
        assert q.containing("kw") is q
        assert q.limit(10) is q
        assert q.offset(5) is q


class TestAuditQueryBuildSQL:
    def test_empty_query(self):
        where, params = AuditQuery().build_sql()
        assert where == ""
        assert params == []

    def test_single_actor(self):
        where, params = AuditQuery().by_actor("admin").build_sql()
        assert "WHERE" in where
        assert "actor = ?" in where
        assert params == ["admin"]

    def test_multiple_filters(self):
        where, params = (
            AuditQuery()
            .by_actor("admin")
            .by_action("delete")
            .with_severity("error")
            .build_sql()
        )
        assert "WHERE" in where
        assert "actor = ?" in where
        assert "action = ?" in where
        assert "severity = ?" in where
        assert len(params) == 3
        assert "admin" in params
        assert "delete" in params
        assert "error" in params

    def test_time_range(self):
        where, params = (
            AuditQuery()
            .since(100.0)
            .until(200.0)
            .build_sql()
        )
        assert "timestamp >= ?" in where
        assert "timestamp <= ?" in where
        assert 100.0 in params
        assert 200.0 in params

    def test_resource_filter(self):
        where, params = (
            AuditQuery()
            .with_resource("zone")
            .with_resource_id("zone_1")
            .build_sql()
        )
        assert "resource = ?" in where
        assert "resource_id = ?" in where
        assert "zone" in params
        assert "zone_1" in params

    def test_ip_filter(self):
        where, params = AuditQuery().from_ip("10.0.0.1").build_sql()
        assert "ip_address = ?" in where
        assert params == ["10.0.0.1"]

    def test_keyword_filter(self):
        where, params = AuditQuery().containing("threshold").build_sql()
        assert "detail LIKE ?" in where
        assert params == ["%threshold%"]

    def test_all_filters(self):
        where, params = (
            AuditQuery()
            .by_actor("admin")
            .by_action("config_changed")
            .with_severity("warning")
            .with_resource("zone")
            .with_resource_id("z1")
            .since(100.0)
            .until(200.0)
            .from_ip("10.0.0.1")
            .containing("test")
            .build_sql()
        )
        assert where.startswith("WHERE ")
        assert " AND " in where
        assert len(params) == 9
