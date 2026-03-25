# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for access pattern analysis module."""

from __future__ import annotations

import time

import pytest

from tritium_lib.intelligence.access_patterns import (
    AccessAnomaly,
    AccessEvent,
    AccessPattern,
    AccessPatternAnalyzer,
    FrequencyReport,
    PiggybackAlert,
    TailgateAlert,
    detect_piggybacking,
    detect_tailgating,
    frequency_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enter(target_id: str, zone_id: str, ts: float, authorized: bool | None = None) -> AccessEvent:
    return AccessEvent(
        target_id=target_id,
        zone_id=zone_id,
        event_type="enter",
        timestamp=ts,
        authorized=authorized,
    )


def _make_exit(target_id: str, zone_id: str, ts: float) -> AccessEvent:
    return AccessEvent(
        target_id=target_id,
        zone_id=zone_id,
        event_type="exit",
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# AccessEvent tests
# ---------------------------------------------------------------------------

class TestAccessEvent:
    def test_create_basic(self):
        ev = AccessEvent(
            target_id="ble_aa", zone_id="zone_lobby", event_type="enter", timestamp=1000.0,
        )
        assert ev.target_id == "ble_aa"
        assert ev.zone_id == "zone_lobby"
        assert ev.event_type == "enter"
        assert ev.timestamp == 1000.0

    def test_default_timestamp(self):
        before = time.time()
        ev = AccessEvent(target_id="t1", zone_id="z1", event_type="enter")
        assert ev.timestamp >= before

    def test_to_dict(self):
        ev = AccessEvent(
            target_id="t1", zone_id="z1", event_type="exit",
            timestamp=5000.0, authorized=True, source="camera",
            position=(10.0, 20.0),
        )
        d = ev.to_dict()
        assert d["target_id"] == "t1"
        assert d["position"] == [10.0, 20.0]
        assert d["authorized"] is True
        assert d["source"] == "camera"

    def test_from_dict(self):
        d = {
            "target_id": "t2",
            "zone_id": "z2",
            "event_type": "enter",
            "timestamp": 3000.0,
            "authorized": False,
            "source": "ble",
            "position": [1.0, 2.0],
        }
        ev = AccessEvent.from_dict(d)
        assert ev.target_id == "t2"
        assert ev.authorized is False
        assert ev.position == (1.0, 2.0)

    def test_from_dict_no_position(self):
        d = {"target_id": "t3", "zone_id": "z3", "event_type": "exit"}
        ev = AccessEvent.from_dict(d)
        assert ev.position is None


# ---------------------------------------------------------------------------
# detect_tailgating tests
# ---------------------------------------------------------------------------

class TestDetectTailgating:
    def test_no_events(self):
        assert detect_tailgating([]) == []

    def test_single_entry(self):
        events = [_make_enter("t1", "z1", 1000.0)]
        assert detect_tailgating(events) == []

    def test_same_target_not_flagged(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t1", "z1", 1001.0),
        ]
        assert detect_tailgating(events) == []

    def test_two_targets_within_threshold(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1002.0),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 1
        assert alerts[0].leader_target_id == "t1"
        assert alerts[0].follower_target_id == "t2"
        assert alerts[0].gap_seconds == pytest.approx(2.0)

    def test_two_targets_outside_threshold(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1010.0),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 0

    def test_severity_high_for_very_close(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1000.5),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "high"

    def test_severity_low_for_edge(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1002.5),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 1
        assert alerts[0].severity == "low"

    def test_exit_events_ignored(self):
        events = [
            _make_exit("t1", "z1", 1000.0),
            _make_exit("t2", "z1", 1001.0),
        ]
        assert detect_tailgating(events) == []

    def test_multiple_zones(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1001.0),
            _make_enter("t3", "z2", 2000.0),
            _make_enter("t4", "z2", 2001.0),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 2
        zone_ids = {a.zone_id for a in alerts}
        assert zone_ids == {"z1", "z2"}

    def test_zero_threshold(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1000.0),
        ]
        assert detect_tailgating(events, threshold_seconds=0.0) == []

    def test_chain_tailgating(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1001.0),
            _make_enter("t3", "z1", 1002.0),
        ]
        alerts = detect_tailgating(events, threshold_seconds=3.0)
        assert len(alerts) == 2


# ---------------------------------------------------------------------------
# detect_piggybacking tests
# ---------------------------------------------------------------------------

class TestDetectPiggybacking:
    def test_no_events(self):
        assert detect_piggybacking([]) == []

    def test_all_authorized_no_alerts(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1005.0),
        ]
        alerts = detect_piggybacking(events, authorized_targets={"t1", "t2"})
        assert len(alerts) == 0

    def test_unauthorized_target_flagged(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1005.0),
        ]
        alerts = detect_piggybacking(events, authorized_targets={"t1"})
        assert len(alerts) == 1
        assert alerts[0].target_id == "t2"
        assert alerts[0].preceding_authorized_id == "t1"

    def test_unauthorized_with_no_preceding_auth(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
        ]
        alerts = detect_piggybacking(events, authorized_targets=set())
        assert len(alerts) == 1
        assert alerts[0].preceding_authorized_id == ""
        assert alerts[0].severity == "low"

    def test_event_level_authorization(self):
        events = [
            _make_enter("t1", "z1", 1000.0, authorized=True),
            _make_enter("t2", "z1", 1005.0, authorized=False),
        ]
        alerts = detect_piggybacking(events, authorized_targets=None)
        assert len(alerts) == 1
        assert alerts[0].target_id == "t2"

    def test_high_severity_close_follow(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1001.0),
        ]
        alerts = detect_piggybacking(
            events,
            authorized_targets={"t1"},
            follow_window_seconds=10.0,
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "high"

    def test_outside_follow_window(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1020.0),
        ]
        alerts = detect_piggybacking(
            events,
            authorized_targets={"t1"},
            follow_window_seconds=10.0,
        )
        assert len(alerts) == 1
        # No preceding authorized within window
        assert alerts[0].preceding_authorized_id == ""


# ---------------------------------------------------------------------------
# frequency_analysis tests
# ---------------------------------------------------------------------------

class TestFrequencyAnalysis:
    def test_no_events(self):
        report = frequency_analysis("t1", "z1", [])
        assert report.total_visits == 0
        assert report.visits_per_day == 0.0

    def test_basic_frequency(self):
        base = 1000000.0
        day = 86400.0
        events = [
            _make_enter("t1", "z1", base),
            _make_exit("t1", "z1", base + 300),
            _make_enter("t1", "z1", base + day),
            _make_exit("t1", "z1", base + day + 600),
            _make_enter("t1", "z1", base + 2 * day),
            _make_exit("t1", "z1", base + 2 * day + 120),
        ]
        report = frequency_analysis("t1", "z1", events)
        assert report.total_visits == 3
        assert report.visits_per_day == pytest.approx(3.0 / 2.0, rel=0.1)

    def test_dwell_time_computed(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_exit("t1", "z1", 1060.0),
            _make_enter("t1", "z1", 2000.0),
            _make_exit("t1", "z1", 2120.0),
        ]
        report = frequency_analysis("t1", "z1", events)
        # (60 + 120) / 2 = 90
        assert report.avg_dwell_seconds == pytest.approx(90.0)

    def test_filters_target_and_zone(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t2", "z1", 1001.0),
            _make_enter("t1", "z2", 1002.0),
        ]
        report = frequency_analysis("t1", "z1", events)
        assert report.total_visits == 1

    def test_time_range_filter(self):
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_enter("t1", "z1", 2000.0),
            _make_enter("t1", "z1", 3000.0),
        ]
        report = frequency_analysis("t1", "z1", events, time_range=(1500.0, 2500.0))
        assert report.total_visits == 1

    def test_to_dict(self):
        report = FrequencyReport(
            target_id="t1", zone_id="z1", total_visits=5,
            visits_per_day=2.5, visits_per_week=17.5,
            time_range=(1000.0, 2000.0), peak_hours=[9, 17],
            avg_dwell_seconds=120.0, last_visit=2000.0,
        )
        d = report.to_dict()
        assert d["total_visits"] == 5
        assert d["visits_per_week"] == 17.5


# ---------------------------------------------------------------------------
# AccessPattern tests
# ---------------------------------------------------------------------------

class TestAccessPattern:
    def test_to_dict(self):
        pattern = AccessPattern(
            target_id="t1", zone_id="z1", total_entries=10,
            avg_dwell_seconds=120.5, std_dwell_seconds=30.2,
        )
        d = pattern.to_dict()
        assert d["target_id"] == "t1"
        assert d["avg_dwell_seconds"] == 120.5
        assert d["std_dwell_seconds"] == 30.2


# ---------------------------------------------------------------------------
# AccessPatternAnalyzer tests
# ---------------------------------------------------------------------------

class TestAccessPatternAnalyzer:
    def test_record_and_retrieve(self):
        analyzer = AccessPatternAnalyzer()
        ev = _make_enter("t1", "z1", 1000.0)
        analyzer.record_access(ev)
        events = analyzer.get_events(target_id="t1", zone_id="z1")
        assert len(events) == 1
        assert events[0].target_id == "t1"

    def test_record_batch(self):
        analyzer = AccessPatternAnalyzer()
        events = [
            _make_enter("t1", "z1", 1000.0),
            _make_exit("t1", "z1", 1060.0),
            _make_enter("t2", "z1", 1100.0),
        ]
        count = analyzer.record_access_batch(events)
        assert count == 3

    def test_learn_pattern_insufficient_data(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        pattern = analyzer.learn_pattern("t1", "z1")
        assert pattern is None

    def test_learn_pattern_sufficient_data(self):
        analyzer = AccessPatternAnalyzer()
        base = 1000000.0
        for i in range(10):
            ts = base + i * 3600
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))

        pattern = analyzer.learn_pattern("t1", "z1")
        assert pattern is not None
        assert pattern.total_entries == 10
        assert pattern.total_exits == 10
        assert pattern.avg_dwell_seconds == pytest.approx(300.0)
        assert pattern.avg_interval_seconds == pytest.approx(3600.0, rel=0.01)

    def test_get_pattern(self):
        analyzer = AccessPatternAnalyzer()
        base = 1000000.0
        for i in range(10):
            ts = base + i * 3600
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")

        pattern = analyzer.get_pattern("t1", "z1")
        assert pattern is not None
        assert pattern.target_id == "t1"

    def test_check_access_new_zone(self):
        analyzer = AccessPatternAnalyzer()
        # t1 has been in z1 but not z2
        for i in range(6):
            analyzer.record_access(_make_enter("t1", "z1", 1000.0 + i * 100))

        ev = _make_enter("t1", "z2", 2000.0)
        anomalies = analyzer.check_access(ev)
        assert any(a.anomaly_type == "new_zone" for a in anomalies)

    def test_check_access_unusual_time(self):
        analyzer = AccessPatternAnalyzer()
        # Build pattern: all entries at a specific hour
        # Use a timestamp that maps to hour 10 in localtime
        import calendar
        import datetime
        # Create a reference time at 10:00 local
        ref = datetime.datetime(2026, 3, 25, 10, 0, 0)
        base_ts = ref.timestamp()

        for i in range(10):
            ts = base_ts + i * 86400  # same hour each day
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")

        # Now access at a very different hour (3 AM)
        odd_ref = datetime.datetime(2026, 3, 25, 3, 0, 0)
        odd_ts = odd_ref.timestamp()
        ev = _make_enter("t1", "z1", odd_ts)
        anomalies = analyzer.check_access(ev)
        assert any(a.anomaly_type == "unusual_time" for a in anomalies)

    def test_check_access_unusual_frequency(self):
        analyzer = AccessPatternAnalyzer()
        # Build pattern: entries every ~3600s
        base = 1000000.0
        for i in range(10):
            ts = base + i * 3600
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")

        # Now access way too soon (10 seconds after last entry vs ~3600s avg)
        ev = _make_enter("t1", "z1", base + 9 * 3600 + 10)
        anomalies = analyzer.check_access(ev)
        assert any(a.anomaly_type == "unusual_frequency" for a in anomalies)

    def test_check_access_unusual_dwell(self):
        analyzer = AccessPatternAnalyzer()
        # Build pattern: dwell ~300s each time
        base = 1000000.0
        for i in range(10):
            ts = base + i * 7200
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")

        # Now create an enter followed by a very late exit
        analyzer.record_access(_make_enter("t1", "z1", base + 20 * 7200))
        ev = _make_exit("t1", "z1", base + 20 * 7200 + 50000)
        anomalies = analyzer.check_access(ev)
        assert any(a.anomaly_type == "unusual_dwell" for a in anomalies)

    def test_check_access_no_anomaly_normal(self):
        analyzer = AccessPatternAnalyzer()
        base = 1000000.0
        for i in range(10):
            ts = base + i * 3600
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")

        # Normal access: right on schedule
        ev = _make_enter("t1", "z1", base + 10 * 3600)
        anomalies = analyzer.check_access(ev)
        freq_anomalies = [a for a in anomalies if a.anomaly_type == "unusual_frequency"]
        assert len(freq_anomalies) == 0

    def test_detect_tailgating_in_zone(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t2", "z1", 1001.0))
        alerts = analyzer.detect_tailgating_in_zone("z1", threshold_seconds=3.0)
        assert len(alerts) == 1

    def test_detect_piggybacking_in_zone(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0, authorized=True))
        analyzer.record_access(_make_enter("t2", "z1", 1005.0, authorized=False))
        alerts = analyzer.detect_piggybacking_in_zone("z1")
        assert len(alerts) == 1
        assert alerts[0].target_id == "t2"

    def test_frequency_analysis_for_target(self):
        analyzer = AccessPatternAnalyzer()
        base = 1000000.0
        for i in range(5):
            ts = base + i * 86400
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 600))
        report = analyzer.frequency_analysis_for_target("t1", "z1")
        assert report.total_visits == 5

    def test_get_zone_ids(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t1", "z2", 2000.0))
        zones = analyzer.get_zone_ids()
        assert set(zones) == {"z1", "z2"}

    def test_get_target_ids(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t2", "z1", 2000.0))
        targets = analyzer.get_target_ids()
        assert set(targets) == {"t1", "t2"}

    def test_get_target_ids_filtered_by_zone(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t2", "z2", 2000.0))
        targets = analyzer.get_target_ids(zone_id="z1")
        assert targets == ["t1"]

    def test_get_stats(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        stats = analyzer.get_stats()
        assert stats["total_events"] == 1
        assert stats["zone_count"] == 1
        assert stats["target_zone_pairs"] == 1

    def test_clear_all(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.clear()
        assert analyzer.get_stats()["total_events"] == 0

    def test_clear_by_zone(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t1", "z2", 2000.0))
        analyzer.clear(zone_id="z1")
        assert "z1" not in analyzer.get_zone_ids()
        assert "z2" in analyzer.get_zone_ids()

    def test_clear_by_target(self):
        analyzer = AccessPatternAnalyzer()
        analyzer.record_access(_make_enter("t1", "z1", 1000.0))
        analyzer.record_access(_make_enter("t2", "z1", 2000.0))
        analyzer.clear(target_id="t1")
        targets = analyzer.get_target_ids()
        assert "t1" not in targets

    def test_get_all_patterns(self):
        analyzer = AccessPatternAnalyzer()
        base = 1000000.0
        for i in range(10):
            ts = base + i * 3600
            analyzer.record_access(_make_enter("t1", "z1", ts))
            analyzer.record_access(_make_exit("t1", "z1", ts + 300))
        analyzer.learn_pattern("t1", "z1")
        patterns = analyzer.get_all_patterns()
        assert len(patterns) == 1

    def test_event_bus_publish_on_anomaly(self):
        published = []

        class MockBus:
            def publish(self, topic, data):
                published.append((topic, data))

        analyzer = AccessPatternAnalyzer(event_bus=MockBus())
        # Record enough to learn, then trigger new_zone anomaly
        for i in range(6):
            analyzer.record_access(_make_enter("t1", "z1", 1000.0 + i * 100))
        ev = _make_enter("t1", "z2", 5000.0)
        anomalies = analyzer.check_access(ev)
        assert len(anomalies) > 0
        assert len(published) > 0
        assert published[0][0] == "access:anomaly"


# ---------------------------------------------------------------------------
# AccessAnomaly tests
# ---------------------------------------------------------------------------

class TestAccessAnomaly:
    def test_to_dict(self):
        anomaly = AccessAnomaly(
            target_id="t1", zone_id="z1", anomaly_type="unusual_time",
            description="odd hour", score=0.7, severity="high",
            timestamp=5000.0,
        )
        d = anomaly.to_dict()
        assert d["anomaly_type"] == "unusual_time"
        assert d["score"] == 0.7
        assert d["severity"] == "high"

    def test_default_timestamp(self):
        before = time.time()
        anomaly = AccessAnomaly(target_id="t1", zone_id="z1")
        assert anomaly.timestamp >= before


# ---------------------------------------------------------------------------
# TailgateAlert / PiggybackAlert serialization tests
# ---------------------------------------------------------------------------

class TestAlertSerialization:
    def test_tailgate_to_dict(self):
        alert = TailgateAlert(
            zone_id="z1", leader_target_id="t1",
            follower_target_id="t2", leader_timestamp=1000.0,
            follower_timestamp=1001.5, gap_seconds=1.5,
            severity="medium",
        )
        d = alert.to_dict()
        assert d["gap_seconds"] == 1.5
        assert d["severity"] == "medium"

    def test_piggyback_to_dict(self):
        alert = PiggybackAlert(
            zone_id="z1", target_id="t2", timestamp=2000.0,
            preceding_authorized_id="t1", gap_seconds=3.0,
            severity="high",
        )
        d = alert.to_dict()
        assert d["gap_seconds"] == 3.0
        assert d["preceding_authorized_id"] == "t1"

    def test_piggyback_to_dict_no_gap(self):
        alert = PiggybackAlert(
            zone_id="z1", target_id="t2", timestamp=2000.0,
        )
        d = alert.to_dict()
        assert "gap_seconds" not in d
