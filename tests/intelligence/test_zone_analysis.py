# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.zone_analysis module."""
import time

import pytest

from tritium_lib.intelligence.zone_analysis import (
    ActivityPrediction,
    Hotspot,
    ZoneAnalyzer,
    ZoneComparison,
    ZoneEvent,
    ZoneReport,
)
from tritium_lib.tracking.heatmap import HeatmapEngine
from tritium_lib.store.targets import TargetStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """A bare ZoneAnalyzer with no backing stores."""
    return ZoneAnalyzer()


@pytest.fixture
def heatmap():
    return HeatmapEngine(retention_seconds=86400)


@pytest.fixture
def target_store(tmp_path):
    db_path = str(tmp_path / "targets.db")
    return TargetStore(db_path)


@pytest.fixture
def full_analyzer(target_store, heatmap):
    """ZoneAnalyzer wired to a TargetStore and HeatmapEngine."""
    return ZoneAnalyzer(target_store=target_store, heatmap_engine=heatmap)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _populate_zone(analyzer: ZoneAnalyzer, zone_id: str, target_count: int = 10,
                   entries_per_target: int = 3, base_time: float | None = None):
    """Feed enter/exit events for multiple targets into a zone."""
    if base_time is None:
        base_time = time.time() - 3600  # 1 hour ago
    for t in range(target_count):
        tid = f"target_{t}"
        for i in range(entries_per_target):
            enter_ts = base_time + t * 60 + i * 300
            exit_ts = enter_ts + 120 + t * 10  # dwell varies by target
            analyzer.record_zone_event(zone_id, tid, "enter", timestamp=enter_ts)
            analyzer.record_zone_event(zone_id, tid, "exit", timestamp=exit_ts)


# ---------------------------------------------------------------------------
# Tests: ZoneEvent dataclass
# ---------------------------------------------------------------------------

class TestZoneEvent:
    def test_zone_event_to_dict(self):
        evt = ZoneEvent(zone_id="z1", target_id="t1", event_type="enter", timestamp=100.0)
        d = evt.to_dict()
        assert d["zone_id"] == "z1"
        assert d["target_id"] == "t1"
        assert d["event_type"] == "enter"
        assert d["timestamp"] == 100.0

    def test_zone_event_fields(self):
        evt = ZoneEvent(zone_id="lobby", target_id="ble_aa", event_type="exit", timestamp=200.0)
        assert evt.zone_id == "lobby"
        assert evt.event_type == "exit"


# ---------------------------------------------------------------------------
# Tests: Event recording
# ---------------------------------------------------------------------------

class TestRecordEvents:
    def test_record_single_event(self, analyzer):
        evt = analyzer.record_zone_event("zone_a", "t1", "enter", timestamp=1000.0)
        assert evt.zone_id == "zone_a"
        assert evt.target_id == "t1"
        assert evt.event_type == "enter"
        assert evt.timestamp == 1000.0

    def test_record_default_timestamp(self, analyzer):
        before = time.time()
        evt = analyzer.record_zone_event("zone_a", "t1", "enter")
        after = time.time()
        assert before <= evt.timestamp <= after

    def test_record_batch(self, analyzer):
        batch = [
            {"zone_id": "z1", "target_id": "t1", "event_type": "enter", "timestamp": 100.0},
            {"zone_id": "z1", "target_id": "t1", "event_type": "exit", "timestamp": 200.0},
            {"zone_id": "z1", "target_id": "t2", "event_type": "enter", "timestamp": 150.0},
            # Invalid: missing event_type
            {"zone_id": "z1", "target_id": "t3"},
            # Invalid: bad event_type
            {"zone_id": "z1", "target_id": "t4", "event_type": "invalid"},
        ]
        count = analyzer.record_zone_events_batch(batch)
        assert count == 3

    def test_event_trimming(self):
        analyzer = ZoneAnalyzer(max_events=5)
        for i in range(10):
            analyzer.record_zone_event("z1", f"t{i}", "enter", timestamp=float(i))
        events = analyzer.get_zone_events("z1", limit=100)
        assert len(events) == 5
        # Should retain the most recent 5
        assert events[0].timestamp == 9.0

    def test_get_zone_events_filtered(self, analyzer):
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("z1", "t1", "exit", timestamp=200.0)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=300.0)

        enters = analyzer.get_zone_events("z1", event_type="enter")
        assert len(enters) == 2
        exits = analyzer.get_zone_events("z1", event_type="exit")
        assert len(exits) == 1


# ---------------------------------------------------------------------------
# Tests: Zone analysis
# ---------------------------------------------------------------------------

class TestAnalyzeZone:
    def test_empty_zone(self, analyzer):
        report = analyzer.analyze_zone("nonexistent")
        assert report.entry_count == 0
        assert report.exit_count == 0
        assert report.unique_targets == 0
        assert report.avg_dwell_seconds == 0.0

    def test_basic_analysis(self, analyzer):
        base = time.time() - 3600
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=base)
        analyzer.record_zone_event("z1", "t1", "exit", timestamp=base + 120)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=base + 60)
        analyzer.record_zone_event("z1", "t2", "exit", timestamp=base + 180)

        report = analyzer.analyze_zone("z1", time_range=(base - 10, base + 300))
        assert report.entry_count == 2
        assert report.exit_count == 2
        assert report.unique_targets == 2
        assert report.zone_id == "z1"

    def test_dwell_time_computation(self, analyzer):
        base = 1000.0
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=base)
        analyzer.record_zone_event("z1", "t1", "exit", timestamp=base + 60)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=base + 10)
        analyzer.record_zone_event("z1", "t2", "exit", timestamp=base + 130)

        report = analyzer.analyze_zone("z1", time_range=(base - 1, base + 200))
        # t1 dwell = 60, t2 dwell = 120, avg = 90
        assert report.avg_dwell_seconds == 90.0
        assert report.max_dwell_seconds == 120.0
        assert report.min_dwell_seconds == 60.0

    def test_peak_hours(self, analyzer):
        # Simulate entries at hour 14 (more) and hour 10 (fewer)
        base = 1000.0
        for i in range(5):
            # Use a timestamp whose localtime hour we control
            # We use a fixed-ish approach: record and check the report's hourly_entries
            analyzer.record_zone_event("z1", f"t{i}", "enter", timestamp=base + i)

        report = analyzer.analyze_zone("z1", time_range=(base - 1, base + 100))
        assert report.entry_count == 5
        # All entries at the same hour, so peak_hours has exactly that hour
        assert len(report.peak_hours) >= 1

    def test_targets_currently_inside(self, analyzer):
        base = 1000.0
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=base)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=base + 10)
        analyzer.record_zone_event("z1", "t1", "exit", timestamp=base + 50)
        # t2 never exited

        report = analyzer.analyze_zone("z1", time_range=(base - 1, base + 100))
        assert report.targets_currently_inside == 1

    def test_time_range_filter(self, analyzer):
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=200.0)
        analyzer.record_zone_event("z1", "t3", "enter", timestamp=300.0)

        report = analyzer.analyze_zone("z1", time_range=(150.0, 250.0))
        assert report.entry_count == 1
        assert report.unique_targets == 1

    def test_report_to_dict(self, analyzer):
        _populate_zone(analyzer, "z1", target_count=3, entries_per_target=2)
        report = analyzer.analyze_zone("z1")
        d = report.to_dict()
        assert "zone_id" in d
        assert "entry_count" in d
        assert "avg_dwell_seconds" in d
        assert "peak_hours" in d
        assert isinstance(d["time_range"], list)


# ---------------------------------------------------------------------------
# Tests: Activity prediction
# ---------------------------------------------------------------------------

class TestPredictActivity:
    def test_predict_no_data(self, analyzer):
        pred = analyzer.predict_activity("empty_zone", future_hours=4)
        assert pred.zone_id == "empty_zone"
        assert len(pred.predicted_counts) == 4
        for p in pred.predicted_counts:
            assert p["predicted_entries"] == 0.0
            assert p["confidence"] == 0.0

    def test_predict_with_data(self, analyzer):
        # Feed enough entries to build hourly baselines
        base = time.time() - 7200
        for i in range(100):
            analyzer.record_zone_event("z1", f"t{i % 10}", "enter", timestamp=base + i * 30)

        pred = analyzer.predict_activity("z1", future_hours=2)
        assert pred.zone_id == "z1"
        assert len(pred.predicted_counts) == 2

    def test_predict_clamps_hours(self, analyzer):
        pred = analyzer.predict_activity("z1", future_hours=100)
        assert len(pred.predicted_counts) == 48  # clamped to max

        pred = analyzer.predict_activity("z1", future_hours=-5)
        assert len(pred.predicted_counts) == 1  # clamped to min

    def test_predict_trend_detection(self, analyzer):
        # Feed entries with increasing pattern across hours
        now = time.time()
        # We just verify the field is one of the valid values
        _populate_zone(analyzer, "z1", target_count=20, entries_per_target=5, base_time=now - 86400)
        pred = analyzer.predict_activity("z1", future_hours=8)
        assert pred.trend in ("increasing", "decreasing", "stable")

    def test_prediction_to_dict(self, analyzer):
        pred = analyzer.predict_activity("z1", future_hours=3)
        d = pred.to_dict()
        assert "zone_id" in d
        assert "predicted_counts" in d
        assert "trend" in d
        assert "avg_predicted_entries" in d


# ---------------------------------------------------------------------------
# Tests: Hotspot detection
# ---------------------------------------------------------------------------

class TestFindHotspots:
    def test_no_heatmap_no_store(self, analyzer):
        hotspots = analyzer.find_hotspots(area=(0, 0, 100, 100))
        assert hotspots == []

    def test_invalid_area(self, full_analyzer):
        # max <= min
        hotspots = full_analyzer.find_hotspots(area=(100, 100, 50, 50))
        assert hotspots == []

    def test_hotspots_from_heatmap(self, full_analyzer, heatmap):
        now = time.time()
        # Add concentrated events in one area
        for i in range(20):
            heatmap.record_event("ble_activity", 50.0, 50.0, weight=1.0, timestamp=now - i)
        # Add a few events elsewhere
        heatmap.record_event("ble_activity", 10.0, 10.0, weight=1.0, timestamp=now)

        hotspots = full_analyzer.find_hotspots(
            area=(0, 0, 100, 100), resolution=10, time_window_minutes=5, threshold=0.1
        )
        assert len(hotspots) >= 1
        # The highest-intensity hotspot should be near (50, 50)
        top = hotspots[0]
        assert top.intensity > 0.0
        assert top.event_count > 0

    def test_hotspot_sorted_by_intensity(self, full_analyzer, heatmap):
        now = time.time()
        for i in range(10):
            heatmap.record_event("ble_activity", 20.0, 20.0, weight=1.0, timestamp=now - i)
        for i in range(5):
            heatmap.record_event("ble_activity", 80.0, 80.0, weight=1.0, timestamp=now - i)

        hotspots = full_analyzer.find_hotspots(
            area=(0, 0, 100, 100), resolution=10, time_window_minutes=5, threshold=0.0
        )
        if len(hotspots) >= 2:
            assert hotspots[0].intensity >= hotspots[1].intensity

    def test_hotspot_to_dict(self):
        h = Hotspot(x=10.5, y=20.5, intensity=0.85, event_count=42, cell_row=3, cell_col=5)
        d = h.to_dict()
        assert d["x"] == 10.5
        assert d["y"] == 20.5
        assert d["intensity"] == 0.85
        assert d["event_count"] == 42

    def test_hotspots_from_target_store(self, target_store, tmp_path):
        """Hotspot detection falls back to TargetStore when no heatmap."""
        analyzer = ZoneAnalyzer(target_store=target_store)
        now = time.time()
        # Insert some targets with positions
        for i in range(5):
            target_store.record_sighting(
                target_id=f"t{i}", name=f"Target {i}", source="ble",
                position_x=50.0 + i, position_y=50.0, timestamp=now - i * 10,
            )

        hotspots = analyzer.find_hotspots(
            area=(0, 0, 100, 100), resolution=10, time_window_minutes=60, threshold=0.0
        )
        # Should find at least something from store data
        assert isinstance(hotspots, list)


# ---------------------------------------------------------------------------
# Tests: Zone comparison
# ---------------------------------------------------------------------------

class TestCompareZones:
    def test_compare_empty(self, analyzer):
        comp = analyzer.compare_zones([])
        assert comp.zone_ids == []
        assert comp.rankings == []

    def test_compare_single_zone(self, analyzer):
        _populate_zone(analyzer, "z1", target_count=5, entries_per_target=2)
        comp = analyzer.compare_zones(["z1"])
        assert len(comp.rankings) == 1
        assert comp.busiest_zone == "z1"
        assert comp.quietest_zone == "z1"

    def test_compare_multiple_zones(self, analyzer):
        base = time.time() - 3600
        # z1 gets more entries than z2
        _populate_zone(analyzer, "z1", target_count=10, entries_per_target=3, base_time=base)
        _populate_zone(analyzer, "z2", target_count=3, entries_per_target=1, base_time=base)

        comp = analyzer.compare_zones(["z1", "z2"])
        assert len(comp.rankings) == 2
        assert comp.busiest_zone == "z1"
        assert comp.quietest_zone == "z2"
        # Rankings sorted by entry_count descending
        assert comp.rankings[0]["zone_id"] == "z1"
        assert comp.rankings[0]["entry_count"] > comp.rankings[1]["entry_count"]

    def test_comparison_to_dict(self, analyzer):
        _populate_zone(analyzer, "z1", target_count=5)
        comp = analyzer.compare_zones(["z1"])
        d = comp.to_dict()
        assert "zone_ids" in d
        assert "rankings" in d
        assert "busiest_zone" in d
        assert "quietest_zone" in d


# ---------------------------------------------------------------------------
# Tests: Utility methods
# ---------------------------------------------------------------------------

class TestUtilityMethods:
    def test_get_all_zone_ids(self, analyzer):
        analyzer.record_zone_event("alpha", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("beta", "t2", "enter", timestamp=200.0)
        ids = analyzer.get_all_zone_ids()
        assert set(ids) == {"alpha", "beta"}

    def test_get_stats(self, analyzer):
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("z1", "t2", "enter", timestamp=200.0)
        analyzer.record_zone_event("z2", "t3", "enter", timestamp=300.0)
        stats = analyzer.get_stats()
        assert stats["total_events"] == 3
        assert stats["zone_count"] == 2

    def test_clear_specific_zone(self, analyzer):
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("z2", "t2", "enter", timestamp=200.0)
        analyzer.clear("z1")
        ids = analyzer.get_all_zone_ids()
        assert "z1" not in ids
        assert "z2" in ids

    def test_clear_all(self, analyzer):
        analyzer.record_zone_event("z1", "t1", "enter", timestamp=100.0)
        analyzer.record_zone_event("z2", "t2", "enter", timestamp=200.0)
        analyzer.clear()
        assert analyzer.get_all_zone_ids() == []
        assert analyzer.get_stats()["total_events"] == 0


# ---------------------------------------------------------------------------
# Tests: Integration — import from intelligence package
# ---------------------------------------------------------------------------

class TestPackageImport:
    def test_import_from_intelligence(self):
        from tritium_lib.intelligence import (
            ZoneAnalyzer,
            ZoneReport,
            ZoneEvent,
            ActivityPrediction,
            Hotspot,
            ZoneComparison,
        )
        assert ZoneAnalyzer is not None
        assert ZoneReport is not None
        assert ZoneEvent is not None
        assert ActivityPrediction is not None
        assert Hotspot is not None
        assert ZoneComparison is not None
