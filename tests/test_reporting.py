# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.reporting — situation reports, daily summaries, incident reports."""

import json
import time

import pytest

from tritium_lib.reporting import (
    SitRepGenerator,
    SitRep,
    DailySummary,
    IncidentReport,
    TargetBreakdown,
    ThreatSummary,
    ZoneActivity,
    AnomalySummary,
    EventTimeline,
)
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.store.event_store import EventStore, TacticalEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker() -> TargetTracker:
    """Empty TargetTracker."""
    return TargetTracker()


@pytest.fixture
def event_store() -> EventStore:
    """In-memory EventStore."""
    s = EventStore(":memory:")
    yield s
    s.close()


def _add_sim_target(tracker: TargetTracker, tid: str, alliance: str = "friendly",
                    asset_type: str = "rover", pos: tuple[float, float] = (0.0, 0.0),
                    threat_score: float = 0.0) -> None:
    """Helper to insert a simulation target into the tracker."""
    tracker.update_from_simulation({
        "target_id": tid,
        "name": tid[:8],
        "alliance": alliance,
        "asset_type": asset_type,
        "position": {"x": pos[0], "y": pos[1]},
        "heading": 0.0,
        "speed": 1.0,
        "battery": 0.9,
        "status": "active",
    })
    # Manually set threat_score since sim doesn't set it
    t = tracker.get_target(tid)
    if t is not None:
        t.threat_score = threat_score


def _populate_tracker(tracker: TargetTracker) -> None:
    """Add a mix of targets to the tracker."""
    _add_sim_target(tracker, "sim_rover_1", "friendly", "rover", (10, 20))
    _add_sim_target(tracker, "sim_drone_1", "friendly", "drone", (30, 40))
    _add_sim_target(tracker, "sim_hostile_1", "hostile", "person", (50, 60), threat_score=0.8)
    _add_sim_target(tracker, "sim_hostile_2", "hostile", "vehicle", (70, 80), threat_score=0.5)
    _add_sim_target(tracker, "sim_unknown_1", "unknown", "person", (0, 0), threat_score=0.1)

    # Add a BLE target
    tracker.update_from_ble({
        "mac": "AA:BB:CC:DD:EE:FF",
        "name": "iPhone",
        "rssi": -45,
        "device_type": "phone",
        "position": {"x": 5.0, "y": 5.0},
    })


def _populate_events(store: EventStore) -> None:
    """Add a variety of events to the store."""
    now = time.time()
    store.record("target_detected", severity="info", source="ble_scanner",
                 target_id="ble_aabbccddee", summary="New BLE device detected",
                 timestamp=now - 100)
    store.record("geofence_enter", severity="warning", source="geofence",
                 target_id="sim_hostile_1", summary="Hostile entered zone Alpha",
                 data={"zone_name": "Alpha", "zone_id": "zone_1"},
                 timestamp=now - 80)
    store.record("anomaly_detected", severity="warning", source="anomaly_engine",
                 target_id="ble_aabbccddee", summary="Unusual BLE activity spike",
                 timestamp=now - 60)
    store.record("geofence_exit", severity="info", source="geofence",
                 target_id="sim_hostile_1", summary="Hostile exited zone Alpha",
                 data={"zone_name": "Alpha", "zone_id": "zone_1"},
                 timestamp=now - 40)
    store.record("target_classified", severity="info", source="classifier",
                 target_id="sim_unknown_1", summary="Target classified as person",
                 timestamp=now - 20)
    store.record("threat_escalated", severity="error", source="escalation",
                 target_id="sim_hostile_1", summary="Threat escalated to hostile",
                 timestamp=now - 10)
    store.record("system_alert", severity="critical", source="system",
                 summary="Sensor node offline",
                 timestamp=now - 5)


# ---------------------------------------------------------------------------
# TargetBreakdown tests
# ---------------------------------------------------------------------------

class TestTargetBreakdown:
    def test_from_empty_targets(self):
        bd = TargetBreakdown.from_targets([])
        assert bd.total == 0
        assert bd.by_source == {}
        assert bd.fused_count == 0

    def test_from_targets_counts_sources(self, tracker):
        _populate_tracker(tracker)
        targets = tracker.get_all()
        bd = TargetBreakdown.from_targets(targets)
        assert bd.total == 6
        assert "simulation" in bd.by_source
        assert "ble" in bd.by_source
        assert bd.by_source["simulation"] == 5
        assert bd.by_source["ble"] == 1

    def test_counts_alliances(self, tracker):
        _populate_tracker(tracker)
        targets = tracker.get_all()
        bd = TargetBreakdown.from_targets(targets)
        assert bd.by_alliance["friendly"] == 2
        assert bd.by_alliance["hostile"] == 2
        assert bd.by_alliance["unknown"] == 2  # 1 sim_unknown + 1 BLE

    def test_counts_fused_targets(self):
        t = TrackedTarget(
            target_id="fused_1", name="Fused", alliance="unknown",
            asset_type="device", confirming_sources={"ble", "wifi"},
        )
        bd = TargetBreakdown.from_targets([t])
        assert bd.fused_count == 1

    def test_to_dict(self, tracker):
        _populate_tracker(tracker)
        bd = TargetBreakdown.from_targets(tracker.get_all())
        d = bd.to_dict()
        assert isinstance(d, dict)
        assert d["total"] == 6
        assert isinstance(d["by_source"], dict)


# ---------------------------------------------------------------------------
# ThreatSummary tests
# ---------------------------------------------------------------------------

class TestThreatSummary:
    def test_from_targets_categorizes_threats(self, tracker):
        _populate_tracker(tracker)
        targets = tracker.get_all()
        ts = ThreatSummary.from_targets(targets)
        assert ts.total_assessed == 6
        assert ts.high_threat == 1   # threat_score 0.8
        assert ts.medium_threat == 1  # threat_score 0.5
        assert ts.low_threat == 1     # threat_score 0.1
        assert ts.no_threat == 3      # the rest
        assert ts.hostile_count == 2

    def test_suspicious_targets_listed(self, tracker):
        _populate_tracker(tracker)
        targets = tracker.get_all()
        ts = ThreatSummary.from_targets(targets)
        assert "sim_hostile_1" in ts.suspicious_targets
        assert "sim_hostile_2" in ts.suspicious_targets

    def test_empty_targets(self):
        ts = ThreatSummary.from_targets([])
        assert ts.total_assessed == 0
        assert ts.high_threat == 0

    # --- Situational assessor (assess=True) — the SitRep threat-level fix -----

    def test_assess_false_default_ignores_alliance(self):
        # Default (assess=False) buckets strictly on the stored threat_score.
        # A hostile the sim never scored reads no_threat — this is exactly the
        # all-zero distribution the bug produced.
        tk = TargetTracker()
        _add_sim_target(tk, "h1", "hostile", "person", (10, 10))   # score 0.0
        _add_sim_target(tk, "f1", "friendly", "rover", (12, 12))
        ts = ThreatSummary.from_targets(tk.get_all())
        assert ts.no_threat == 2
        assert ts.high_threat == ts.medium_threat == ts.low_threat == 0
        assert ts.hostile_count == 1

    def test_assess_populates_levels_for_unscored_hostiles(self):
        # The reported gap: hostiles get an alliance but the sim never writes a
        # threat_score, so High/Med/Low stayed 0 despite live hostiles.  With
        # assess=True the situational assessor grades each hostile by its range
        # to the friendly asset.
        tk = TargetTracker()
        _add_sim_target(tk, "f1", "friendly", "rover", (0, 0))
        _add_sim_target(tk, "h_far", "hostile", "person", (500, 0))     # distant -> Low
        _add_sim_target(tk, "h_close", "hostile", "person", (100, 0))   # closing -> Medium
        _add_sim_target(tk, "h_contact", "hostile", "person", (20, 0))  # contact -> High
        ts = ThreatSummary.from_targets(tk.get_all(), assess=True)
        assert ts.high_threat == 1
        assert ts.medium_threat == 1
        assert ts.low_threat == 1
        assert ts.hostile_count == 3
        assert ts.no_threat == 1  # the friendly is never a threat

    def test_assess_hostile_floor_is_low_without_assets(self):
        # A hostile with nothing to threaten still reads at least Low, never
        # no_threat.
        tk = TargetTracker()
        _add_sim_target(tk, "h1", "hostile", "person", (10, 10))
        ts = ThreatSummary.from_targets(tk.get_all(), assess=True)
        assert ts.low_threat == 1
        assert ts.no_threat == 0

    def test_assess_behavioral_score_still_wins(self):
        # A high behavioral threat_score is preserved: effective = max(stored,
        # situational).  A distant hostile with a stored 0.9 stays High.
        tk = TargetTracker()
        _add_sim_target(tk, "h1", "hostile", "person", (500, 0), threat_score=0.9)
        ts = ThreatSummary.from_targets(tk.get_all(), assess=True)
        assert ts.high_threat == 1

    def test_assess_unknown_only_threatens_on_breach(self):
        # An unknown far from assets is not a threat (no noise flood); an unknown
        # that has closed on a friendly asset escalates.
        tk = TargetTracker()
        _add_sim_target(tk, "f1", "friendly", "rover", (0, 0))
        _add_sim_target(tk, "u_far", "unknown", "phone", (500, 0))   # -> no_threat
        _add_sim_target(tk, "u_near", "unknown", "person", (20, 0))  # -> High
        ts = ThreatSummary.from_targets(tk.get_all(), assess=True)
        assert ts.no_threat == 2  # friendly + distant unknown
        assert ts.high_threat == 1


# ---------------------------------------------------------------------------
# ZoneActivity tests
# ---------------------------------------------------------------------------

class TestZoneActivity:
    def test_from_geofence_events(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        za = ZoneActivity.from_events(events)
        assert za.entries == 1
        assert za.exits == 1
        assert za.total_events == 2
        assert "Alpha" in za.zones_active
        assert za.most_active_zone == "Alpha"

    def test_no_zone_events(self):
        za = ZoneActivity.from_events([])
        assert za.total_events == 0
        assert za.entries == 0
        assert za.zones_active == []


# ---------------------------------------------------------------------------
# AnomalySummary tests
# ---------------------------------------------------------------------------

class TestAnomalySummary:
    def test_from_events_counts_anomalies(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        ans = AnomalySummary.from_events(events)
        # anomaly_detected(warning) + geofence_enter(warning) + threat_escalated(error) + system_alert(critical)
        assert ans.total_anomalies >= 4
        assert "warning" in ans.by_severity or "error" in ans.by_severity

    def test_top_anomalies_sorted_by_severity(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        ans = AnomalySummary.from_events(events)
        if len(ans.top_anomalies) >= 2:
            # Critical should appear before warning
            severities = [a["severity"] for a in ans.top_anomalies]
            if "critical" in severities and "warning" in severities:
                assert severities.index("critical") < severities.index("warning")

    def test_empty_events(self):
        ans = AnomalySummary.from_events([])
        assert ans.total_anomalies == 0
        assert ans.top_anomalies == []

    def test_proximity_breach_high_severity_counts(self, event_store):
        # Severity-vocab reconcile (tick-70 note): proximity:breach events carry
        # the proximity vocab (low/medium/high/critical), not the EventStore vocab
        # (warning/error/critical).  A "high" breach must count as an anomaly, not
        # be silently dropped because "high" is not in the EventStore vocab.
        now = time.time()
        event_store.record("proximity:breach", severity="high", source="proximity",
                           target_id="h1", summary="Hostile breached friendly space",
                           timestamp=now - 5)
        event_store.record("proximity:breach", severity="medium", source="proximity",
                           target_id="h2", summary="Hostile closing", timestamp=now - 4)
        event_store.record("proximity:breach", severity="low", source="proximity",
                           target_id="h3", summary="Outer-ring contact", timestamp=now - 3)
        events = event_store.query_time_range(limit=100)
        ans = AnomalySummary.from_events(events)
        # high + medium count; low stays out as routine (mirrors info/debug).
        assert ans.total_anomalies == 2
        assert ans.by_severity.get("high") == 1
        assert ans.by_severity.get("medium") == 1
        assert "low" not in ans.by_severity

    def test_high_breach_outranks_warning_in_top(self, event_store):
        # A "high" proximity breach should rank above a "warning" event in the
        # top-anomalies list (combined severity rank: high~error > warning).
        now = time.time()
        event_store.record("system_alert", severity="warning", source="system",
                           summary="minor warning", timestamp=now - 10)
        event_store.record("proximity:breach", severity="high", source="proximity",
                           target_id="h1", summary="breach", timestamp=now - 9)
        events = event_store.query_time_range(limit=100)
        ans = AnomalySummary.from_events(events)
        sev = [a["severity"] for a in ans.top_anomalies]
        assert sev.index("high") < sev.index("warning")


# ---------------------------------------------------------------------------
# EventTimeline tests
# ---------------------------------------------------------------------------

class TestEventTimeline:
    def test_from_events_builds_timeline(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        tl = EventTimeline.from_events(events)
        assert tl.total_events > 0
        assert len(tl.events) > 0
        assert tl.period_start > 0
        assert tl.period_end >= tl.period_start

    def test_chronological_order(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        tl = EventTimeline.from_events(events)
        timestamps = [e["timestamp"] for e in tl.events]
        assert timestamps == sorted(timestamps)

    def test_severity_filtering(self, event_store):
        _populate_events(event_store)
        events = event_store.query_time_range(limit=100)
        tl_all = EventTimeline.from_events(events, min_severity="info")
        tl_warn = EventTimeline.from_events(events, min_severity="warning")
        assert tl_warn.total_events <= tl_all.total_events

    def test_empty_events(self):
        tl = EventTimeline.from_events([])
        assert tl.total_events == 0
        assert tl.events == []


# ---------------------------------------------------------------------------
# SitRep output format tests
# ---------------------------------------------------------------------------

class TestSitRepFormats:
    def _make_sitrep(self, tracker, event_store):
        _populate_tracker(tracker)
        _populate_events(event_store)
        gen = SitRepGenerator(tracker=tracker, event_store=event_store)
        return gen.generate()

    def test_to_json_valid(self, tracker, event_store):
        report = self._make_sitrep(tracker, event_store)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Situation Report"
        assert parsed["targets"]["total"] == 6
        assert "threats" in parsed
        assert "zones" in parsed
        assert "anomalies" in parsed
        assert "timeline" in parsed

    def test_to_text_contains_sections(self, tracker, event_store):
        report = self._make_sitrep(tracker, event_store)
        text = report.to_text()
        assert "TARGET SUMMARY" in text
        assert "THREAT ASSESSMENT" in text
        assert "ZONE ACTIVITY" in text
        assert "ANOMALIES" in text
        assert "END OF REPORT" in text

    def test_to_html_valid_structure(self, tracker, event_store):
        report = self._make_sitrep(tracker, event_store)
        h = report.to_html()
        assert "<!DOCTYPE html>" in h
        assert "Target Summary" in h
        assert "Threat Assessment" in h
        assert "#00f0ff" in h  # cyberpunk cyan

    def test_to_dict_serializable(self, tracker, event_store):
        report = self._make_sitrep(tracker, event_store)
        d = report.to_dict()
        # Should be fully JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 100


# ---------------------------------------------------------------------------
# SitRepGenerator tests
# ---------------------------------------------------------------------------

class TestSitRepGenerator:
    def test_generate_without_events(self, tracker):
        _populate_tracker(tracker)
        gen = SitRepGenerator(tracker=tracker)
        report = gen.generate()
        assert report.targets.total == 6
        assert report.zones.total_events == 0

    def test_generate_with_events(self, tracker, event_store):
        _populate_tracker(tracker)
        _populate_events(event_store)
        gen = SitRepGenerator(tracker=tracker, event_store=event_store)
        report = gen.generate()
        assert report.targets.total == 6
        assert report.anomalies.total_anomalies > 0

    def test_custom_title(self, tracker):
        gen = SitRepGenerator(tracker=tracker, title="Custom Title")
        report = gen.generate()
        assert report.title == "Custom Title"

    def test_notes_included(self, tracker):
        gen = SitRepGenerator(tracker=tracker)
        report = gen.generate(notes="Testing notes")
        assert report.notes == "Testing notes"
        assert "Testing notes" in report.to_text()

    def test_custom_time_range(self, tracker, event_store):
        _populate_events(event_store)
        gen = SitRepGenerator(tracker=tracker, event_store=event_store)
        # Very old range should yield no events
        report = gen.generate(event_time_range=(0.0, 1.0))
        assert report.timeline.total_events == 0

    def test_empty_tracker(self):
        tracker = TargetTracker()
        gen = SitRepGenerator(tracker=tracker)
        report = gen.generate()
        assert report.targets.total == 0
        assert report.threats.total_assessed == 0


# ---------------------------------------------------------------------------
# DailySummary tests
# ---------------------------------------------------------------------------

class TestDailySummary:
    def test_from_tracker_and_events(self, tracker, event_store):
        _populate_tracker(tracker)
        _populate_events(event_store)
        summary = DailySummary.from_tracker_and_events(tracker, event_store)
        assert summary.total_targets_seen == 6
        assert summary.total_events >= 7
        assert summary.zone_entries == 1
        assert summary.zone_exits == 1
        assert summary.anomaly_count > 0
        assert summary.date  # should have today's date

    def test_without_event_store(self, tracker):
        _populate_tracker(tracker)
        summary = DailySummary.from_tracker_and_events(tracker)
        assert summary.total_targets_seen == 6
        assert summary.total_events == 0

    def test_text_output(self, tracker):
        _populate_tracker(tracker)
        summary = DailySummary.from_tracker_and_events(tracker)
        text = summary.to_text()
        assert "DAILY SUMMARY" in text
        assert "Targets seen:" in text

    def test_json_output(self, tracker):
        _populate_tracker(tracker)
        summary = DailySummary.from_tracker_and_events(tracker)
        j = summary.to_json()
        parsed = json.loads(j)
        assert parsed["total_targets_seen"] == 6

    def test_html_output(self, tracker):
        _populate_tracker(tracker)
        summary = DailySummary.from_tracker_and_events(tracker)
        h = summary.to_html()
        assert "<!DOCTYPE html>" in h
        assert "Daily Summary" in h

    def test_custom_date(self, tracker):
        summary = DailySummary.from_tracker_and_events(tracker, date="2026-03-24")
        assert summary.date == "2026-03-24"

    def test_threats_detected_count(self, tracker):
        _populate_tracker(tracker)
        summary = DailySummary.from_tracker_and_events(tracker)
        # DailySummary applies the situational assessor (assess=True).  In this
        # fixture every hostile/unknown sits within ~60 m of a friendly asset —
        # well inside the 150 m tactical envelope — so all four escalate to High.
        # threats_detected = high + medium.
        assert summary.threats_detected == 4

    def test_top_event_types(self, tracker, event_store):
        _populate_events(event_store)
        summary = DailySummary.from_tracker_and_events(tracker, event_store)
        assert isinstance(summary.top_event_types, list)
        if summary.top_event_types:
            name, count = summary.top_event_types[0]
            assert isinstance(name, str)
            assert count > 0


# ---------------------------------------------------------------------------
# IncidentReport tests
# ---------------------------------------------------------------------------

class TestIncidentReport:
    def test_basic_construction(self):
        ir = IncidentReport(
            title="Test Incident",
            incident_id="inc-001",
            severity="warning",
            description="Something happened",
            target_ids=["ble_aabb"],
        )
        assert ir.title == "Test Incident"
        assert ir.severity == "warning"
        assert "ble_aabb" in ir.target_ids

    def test_text_output(self):
        ir = IncidentReport(
            title="Hostile in Zone",
            severity="critical",
            description="A hostile target entered restricted zone.",
            target_ids=["sim_hostile_1"],
            location=(40.7, -74.0),
            recommendations=["Dispatch drone", "Alert operator"],
        )
        text = ir.to_text()
        assert "INCIDENT REPORT" in text
        assert "Hostile in Zone" in text
        assert "CRITICAL" in text
        assert "sim_hostile_1" in text
        assert "Dispatch drone" in text
        assert "40.7" in text

    def test_json_output(self):
        ir = IncidentReport(
            title="Test", severity="info",
            target_ids=["t1", "t2"],
        )
        j = ir.to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Test"
        assert len(parsed["target_ids"]) == 2

    def test_html_output(self):
        ir = IncidentReport(
            title="HTML Test", severity="error",
            description="Error level incident",
            target_ids=["ble_ff"],
            related_events=[{
                "event_id": "ev1", "timestamp": time.time(),
                "event_type": "alert", "severity": "error",
                "summary": "Alert triggered", "target_id": "ble_ff",
            }],
            recommendations=["Check sensor"],
        )
        h = ir.to_html()
        assert "<!DOCTYPE html>" in h
        assert "Incident Report" in h
        assert "Check sensor" in h

    def test_from_event(self, tracker, event_store):
        _populate_tracker(tracker)
        _populate_events(event_store)
        events = event_store.query_by_type("threat_escalated")
        assert len(events) > 0
        ev = events[0]
        ir = IncidentReport.from_event(ev, tracker=tracker, event_store=event_store)
        assert ir.title
        assert ir.incident_id == ev.event_id
        assert ir.severity == "error"
        # Should have found related events
        assert len(ir.related_events) >= 1

    def test_from_event_without_tracker(self, event_store):
        _populate_events(event_store)
        events = event_store.query_by_type("system_alert")
        ev = events[0]
        ir = IncidentReport.from_event(ev)
        assert ir.severity == "critical"
        assert ir.target_ids == []  # system alert has no target_id

    def test_location_from_event(self, event_store):
        eid = event_store.record(
            "test_located", severity="info",
            position_lat=33.0, position_lng=-117.0,
            summary="Located event",
        )
        ev = event_store.get_event(eid)
        ir = IncidentReport.from_event(ev)
        assert ir.location is not None
        assert ir.location[0] == pytest.approx(33.0)
        assert ir.location[1] == pytest.approx(-117.0)
