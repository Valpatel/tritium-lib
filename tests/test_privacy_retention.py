# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.privacy.retention — data retention policies."""

import time

from tritium_lib.privacy.retention import (
    DEFAULT_RETENTION,
    DataCategory,
    PurgeResult,
    RetentionManager,
    RetentionPolicy,
)


class TestDataCategory:
    def test_values(self):
        assert DataCategory.REALTIME_SIGHTINGS == "realtime_sightings"
        assert DataCategory.AUDIT_TRAIL == "audit_trail"
        assert len(DataCategory) == 9


class TestRetentionPolicy:
    def test_retention_days(self):
        p = RetentionPolicy(category="test", retention_seconds=86400 * 7)
        assert p.retention_days == 7.0

    def test_is_expired_true(self):
        p = RetentionPolicy(category="test", retention_seconds=3600)
        old_ts = time.time() - 7200  # 2 hours ago
        assert p.is_expired(old_ts) is True

    def test_is_expired_false(self):
        p = RetentionPolicy(category="test", retention_seconds=3600)
        recent_ts = time.time() - 60  # 1 minute ago
        assert p.is_expired(recent_ts) is False

    def test_is_expired_disabled(self):
        p = RetentionPolicy(category="test", retention_seconds=1, enabled=False)
        old_ts = time.time() - 10000
        assert p.is_expired(old_ts) is False

    def test_is_expired_with_now(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        assert p.is_expired(0.0, now=200.0) is True
        assert p.is_expired(150.0, now=200.0) is False

    def test_to_dict(self):
        p = RetentionPolicy(
            category="test",
            retention_seconds=86400,
            enabled=True,
            description="Test policy",
            legal_basis="consent",
        )
        d = p.to_dict()
        assert d["category"] == "test"
        assert d["retention_seconds"] == 86400
        assert d["retention_days"] == 1.0
        assert d["enabled"] is True
        assert d["description"] == "Test policy"


class TestPurgeResult:
    def test_success(self):
        r = PurgeResult(
            category="test",
            purged_count=10,
            cutoff_timestamp=1000.0,
        )
        assert r.success is True

    def test_failure(self):
        r = PurgeResult(
            category="test",
            purged_count=0,
            cutoff_timestamp=1000.0,
            errors=["Database error"],
        )
        assert r.success is False

    def test_to_dict(self):
        r = PurgeResult(
            category="test",
            purged_count=5,
            cutoff_timestamp=1000.0,
            execution_time=0.5,
        )
        d = r.to_dict()
        assert d["category"] == "test"
        assert d["purged_count"] == 5
        assert d["success"] is True


class TestRetentionManager:
    def test_default_policies(self):
        mgr = RetentionManager()
        policies = mgr.list_policies()
        assert len(policies) > 0
        categories = {p.category for p in policies}
        assert DataCategory.REALTIME_SIGHTINGS in categories
        assert DataCategory.AUDIT_TRAIL in categories

    def test_custom_policies(self):
        p = RetentionPolicy(category="custom", retention_seconds=100)
        mgr = RetentionManager(policies={"custom": p})
        assert mgr.get_policy("custom") is not None
        assert mgr.get_policy("realtime_sightings") is None

    def test_get_policy(self):
        mgr = RetentionManager()
        p = mgr.get_policy(DataCategory.REALTIME_SIGHTINGS)
        assert p is not None
        assert p.retention_seconds == DEFAULT_RETENTION[DataCategory.REALTIME_SIGHTINGS]

    def test_set_policy(self):
        mgr = RetentionManager()
        new_p = RetentionPolicy(category="new_cat", retention_seconds=500)
        mgr.set_policy(new_p)
        assert mgr.get_policy("new_cat") is not None

    def test_remove_policy(self):
        mgr = RetentionManager()
        assert mgr.remove_policy(DataCategory.REALTIME_SIGHTINGS) is True
        assert mgr.get_policy(DataCategory.REALTIME_SIGHTINGS) is None
        assert mgr.remove_policy("nonexistent") is False

    def test_register_handler(self):
        mgr = RetentionManager()
        handler = lambda cat, cutoff: 0
        mgr.register_handler("test", handler)
        # No assertion failure = success

    def test_unregister_handler(self):
        mgr = RetentionManager()
        handler = lambda cat, cutoff: 0
        mgr.register_handler("test", handler)
        assert mgr.unregister_handler("test") is True
        assert mgr.unregister_handler("test") is False

    def test_enforce_calls_handlers(self):
        calls = []
        def handler(cat, cutoff):
            calls.append((cat, cutoff))
            return 5

        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", handler)
        now = 1000.0
        results = mgr.enforce(now=now)
        assert len(results) == 1
        assert results[0].purged_count == 5
        assert results[0].success is True
        assert len(calls) == 1
        assert calls[0][1] == 900.0  # now - retention_seconds

    def test_enforce_skips_disabled(self):
        calls = []
        p = RetentionPolicy(category="test", retention_seconds=100, enabled=False)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", lambda c, t: calls.append(1) or 0)
        results = mgr.enforce()
        assert len(results) == 0
        assert len(calls) == 0

    def test_enforce_skips_no_handler(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        results = mgr.enforce()
        assert len(results) == 0

    def test_enforce_handler_error(self):
        def bad_handler(cat, cutoff):
            raise RuntimeError("DB unavailable")

        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", bad_handler)
        results = mgr.enforce(now=1000.0)
        assert len(results) == 1
        assert results[0].success is False
        assert "DB unavailable" in results[0].errors[0]

    def test_enforce_category(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", lambda c, t: 3)
        result = mgr.enforce_category("test", now=1000.0)
        assert result is not None
        assert result.purged_count == 3

    def test_enforce_category_no_policy(self):
        mgr = RetentionManager(policies={})
        result = mgr.enforce_category("nonexistent")
        assert result is None

    def test_enforce_category_no_handler(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        result = mgr.enforce_category("test")
        assert result is None

    def test_history(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", lambda c, t: 1)
        mgr.enforce(now=1000.0)
        mgr.enforce(now=2000.0)
        assert len(mgr.history) == 2

    def test_clear_history(self):
        p = RetentionPolicy(category="test", retention_seconds=100)
        mgr = RetentionManager(policies={"test": p})
        mgr.register_handler("test", lambda c, t: 1)
        mgr.enforce(now=1000.0)
        count = mgr.clear_history()
        assert count == 1
        assert len(mgr.history) == 0

    def test_export(self):
        mgr = RetentionManager()
        exported = mgr.export()
        assert "policies" in exported
        assert "handlers_registered" in exported
        assert "history_count" in exported
