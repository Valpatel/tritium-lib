# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.proximity_monitor."""

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tritium_lib.tracking.proximity_monitor import ProximityMonitor


# ---- Minimal target stub ----

@dataclass
class _StubTarget:
    target_id: str
    alliance: str
    position: tuple


class _StubTracker:
    """Minimal tracker with get_all() -> dict."""

    def __init__(self, targets: list[_StubTarget]):
        self._targets = {t.target_id: t for t in targets}

    def get_all(self):
        return dict(self._targets)


# ---- Helpers ----

def _make_monitor(targets=None, event_bus=None, **kwargs):
    """Create a ProximityMonitor with a temp data dir (no disk side-effects)."""
    tmp = tempfile.mkdtemp()
    tracker = _StubTracker(targets or []) if targets is not None else None
    return ProximityMonitor(
        target_tracker=tracker,
        event_bus=event_bus,
        data_dir=tmp,
        **kwargs,
    )


class TestProximityMonitorInit:
    def test_default_rules_seeded(self):
        mon = _make_monitor()
        rules = mon.list_rules()
        assert len(rules) >= 1
        assert rules[0].alliance_pair == "hostile_friendly"

    def test_stats_initial(self):
        mon = _make_monitor()
        stats = mon.get_stats()
        assert stats["running"] is False
        assert stats["scans_completed"] == 0
        assert stats["alerts_fired"] == 0


class TestRuleManagement:
    def test_add_rule(self):
        mon = _make_monitor()
        from tritium_lib.tracking.proximity_monitor import ProximityMonitor
        # Use the fallback ProximityRule (always available via the module's imports)
        try:
            from tritium_lib.models.proximity import ProximityRule
        except ImportError:
            from tritium_lib.tracking.proximity_monitor import ProximityRule
        rule = ProximityRule(rule_id="test_rule", name="Test", threshold_m=5.0)
        mon.add_rule(rule)
        assert any(r.rule_id == "test_rule" for r in mon.list_rules())

    def test_remove_rule(self):
        mon = _make_monitor()
        try:
            from tritium_lib.models.proximity import ProximityRule
        except ImportError:
            from tritium_lib.tracking.proximity_monitor import ProximityRule
        rule = ProximityRule(rule_id="del_me", name="Delete Me")
        mon.add_rule(rule)
        assert mon.remove_rule("del_me") is True
        assert not any(r.rule_id == "del_me" for r in mon.list_rules())

    def test_remove_nonexistent_rule(self):
        mon = _make_monitor()
        assert mon.remove_rule("no_such_rule") is False

    def test_update_rule(self):
        mon = _make_monitor()
        try:
            from tritium_lib.models.proximity import ProximityRule
        except ImportError:
            from tritium_lib.tracking.proximity_monitor import ProximityRule
        rule = ProximityRule(rule_id="upd", name="Before", threshold_m=10.0)
        mon.add_rule(rule)
        assert mon.update_rule("upd", {"name": "After", "threshold_m": 20.0}) is True
        updated = [r for r in mon.list_rules() if r.rule_id == "upd"][0]
        assert updated.name == "After"
        assert updated.threshold_m == 20.0

    def test_update_nonexistent_rule(self):
        mon = _make_monitor()
        assert mon.update_rule("nope", {"name": "X"}) is False


class TestScanning:
    def test_scan_no_tracker(self):
        mon = _make_monitor(targets=None)
        # Should not raise
        mon._scan()
        assert mon.get_stats()["scans_completed"] == 0

    def test_scan_single_target_no_alert(self):
        targets = [_StubTarget("t1", "friendly", (0.0, 0.0))]
        mon = _make_monitor(targets=targets)
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 0

    def test_scan_same_alliance_no_alert(self):
        targets = [
            _StubTarget("t1", "friendly", (0.0, 0.0)),
            _StubTarget("t2", "friendly", (1.0, 1.0)),
        ]
        mon = _make_monitor(targets=targets)
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 0

    def test_scan_detects_breach(self):
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),  # distance = 5m
        ]
        mon = _make_monitor(targets=targets)
        # Default rule: hostile_friendly, threshold 10m
        mon._scan()
        stats = mon.get_stats()
        assert stats["alerts_fired"] == 1
        assert stats["active_breaches"] == 1

    def test_scan_no_breach_outside_threshold(self):
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (100.0, 0.0)),  # distance = 100m
        ]
        mon = _make_monitor(targets=targets)
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 0

    def test_cooldown_prevents_spam(self):
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),
        ]
        mon = _make_monitor(targets=targets)
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 1
        # Second scan within cooldown should NOT fire again
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 1

    def test_departure_alert(self):
        targets_close = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),
        ]
        mon = _make_monitor(targets=targets_close)
        mon._scan()
        assert mon.get_stats()["alerts_fired"] == 1
        assert mon.get_stats()["active_breaches"] == 1

        # Move targets apart — replace tracker
        targets_far = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (100.0, 0.0)),
        ]
        mon._tracker = _StubTracker(targets_far)
        mon._scan()
        # Should have departure alert + original = 2
        assert mon.get_stats()["alerts_fired"] == 2
        assert mon.get_stats()["active_breaches"] == 0


class TestQueryAPI:
    def test_get_recent_alerts(self):
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),
        ]
        mon = _make_monitor(targets=targets)
        mon._scan()
        alerts = mon.get_recent_alerts(limit=10)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "breach"

    def test_get_active_breaches(self):
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),
        ]
        mon = _make_monitor(targets=targets)
        mon._scan()
        breaches = mon.get_active_breaches()
        assert len(breaches) == 1

    def test_acknowledge_alert_not_found(self):
        mon = _make_monitor()
        assert mon.acknowledge_alert("no_such_id") is False


class TestEventBus:
    def test_alert_published_to_event_bus(self):
        bus = MagicMock()
        targets = [
            _StubTarget("hostile_1", "hostile", (0.0, 0.0)),
            _StubTarget("friendly_1", "friendly", (3.0, 4.0)),
        ]
        mon = _make_monitor(targets=targets, event_bus=bus)
        mon._scan()
        bus.publish.assert_called()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "proximity:alert"


class TestPairKey:
    def test_pair_key_canonical(self):
        key1 = ProximityMonitor._pair_key("a", "b", "r1")
        key2 = ProximityMonitor._pair_key("b", "a", "r1")
        assert key1 == key2

    def test_pair_key_different_rules(self):
        key1 = ProximityMonitor._pair_key("a", "b", "r1")
        key2 = ProximityMonitor._pair_key("a", "b", "r2")
        assert key1 != key2


class TestLifecycle:
    def test_start_stop(self):
        mon = _make_monitor()
        mon.start()
        assert mon.get_stats()["running"] is True
        mon.stop()
        assert mon.get_stats()["running"] is False

    def test_start_idempotent(self):
        mon = _make_monitor()
        mon.start()
        mon.start()  # should not raise or double-start
        assert mon.get_stats()["running"] is True
        mon.stop()
