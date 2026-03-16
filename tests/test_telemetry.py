# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.telemetry — session telemetry and performance monitoring."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time

import pytest

from tritium_lib.sim_engine.telemetry import (
    GameEvent,
    MetricType,
    PerformanceBudget,
    TelemetryDashboard,
    TelemetryPoint,
    TelemetrySession,
    _esc,
)


# ===========================================================================
# MetricType enum
# ===========================================================================


class TestMetricType:
    def test_all_members_present(self):
        expected = {
            "FPS", "FRAME_TIME", "SIM_TIME", "DRAW_CALLS", "TRIANGLES",
            "ENTITIES", "PARTICLES", "PROJECTILES", "MEMORY", "CUSTOM",
        }
        assert set(m.name for m in MetricType) == expected

    def test_values_are_lowercase(self):
        for m in MetricType:
            assert m.value == m.name.lower()

    def test_metric_type_count(self):
        assert len(MetricType) == 10


# ===========================================================================
# TelemetryPoint
# ===========================================================================


class TestTelemetryPoint:
    def test_basic_construction(self):
        p = TelemetryPoint(
            timestamp=1.0, sim_time=0.5, tick=10, metric="fps", value=60.0
        )
        assert p.timestamp == 1.0
        assert p.sim_time == 0.5
        assert p.tick == 10
        assert p.metric == "fps"
        assert p.value == 60.0
        assert p.tags == {}

    def test_with_tags(self):
        p = TelemetryPoint(
            timestamp=1.0, sim_time=0.5, tick=10,
            metric="fps", value=60.0,
            tags={"phase": "riot", "wave": 3},
        )
        assert p.tags["phase"] == "riot"
        assert p.tags["wave"] == 3

    def test_to_dict(self):
        p = TelemetryPoint(
            timestamp=1.5, sim_time=0.75, tick=15,
            metric="entities", value=120.0,
            tags={"zone": "alpha"},
        )
        d = p.to_dict()
        assert d["timestamp"] == 1.5
        assert d["sim_time"] == 0.75
        assert d["tick"] == 15
        assert d["metric"] == "entities"
        assert d["value"] == 120.0
        assert d["tags"] == {"zone": "alpha"}

    def test_to_dict_default_tags(self):
        p = TelemetryPoint(timestamp=0, sim_time=0, tick=0, metric="x", value=1.0)
        assert p.to_dict()["tags"] == {}

    def test_tags_are_independent_copies(self):
        tags = {"a": 1}
        p = TelemetryPoint(timestamp=0, sim_time=0, tick=0, metric="x", value=1.0, tags=tags)
        d = p.to_dict()
        d["tags"]["b"] = 2
        assert "b" not in p.tags


# ===========================================================================
# GameEvent
# ===========================================================================


class TestGameEvent:
    def test_basic_construction(self):
        e = GameEvent(
            timestamp=2.0, sim_time=1.0, tick=20,
            event_type="kill",
            data={"attacker": "u1", "victim": "u2"},
        )
        assert e.event_type == "kill"
        assert e.data["attacker"] == "u1"

    def test_default_data(self):
        e = GameEvent(timestamp=0, sim_time=0, tick=0, event_type="spawn")
        assert e.data == {}

    def test_to_dict(self):
        e = GameEvent(
            timestamp=3.0, sim_time=1.5, tick=30,
            event_type="explosion",
            data={"x": 10, "y": 20, "radius": 5},
        )
        d = e.to_dict()
        assert d["event_type"] == "explosion"
        assert d["data"]["radius"] == 5
        assert d["tick"] == 30


# ===========================================================================
# TelemetrySession — construction & set_tick
# ===========================================================================


class TestTelemetrySessionInit:
    def test_default_session_id(self):
        s = TelemetrySession()
        assert len(s.session_id) == 12

    def test_custom_session_id(self):
        s = TelemetrySession(session_id="test123")
        assert s.session_id == "test123"

    def test_metadata(self):
        s = TelemetrySession(metadata={"preset": "urban"})
        assert s.metadata["preset"] == "urban"

    def test_default_metadata(self):
        s = TelemetrySession()
        assert s.metadata == {}

    def test_start_time_is_recent(self):
        before = time.time()
        s = TelemetrySession()
        after = time.time()
        assert before <= s.start_time <= after

    def test_empty_metrics_and_events(self):
        s = TelemetrySession()
        assert s.metrics == {}
        assert s.events == []

    def test_set_tick(self):
        s = TelemetrySession()
        s.set_tick(42, 2.1)
        assert s._tick == 42
        assert s._sim_time == 2.1


# ===========================================================================
# TelemetrySession — record_metric
# ===========================================================================


class TestRecordMetric:
    def test_record_single_metric(self):
        s = TelemetrySession()
        s.set_tick(1, 0.016)
        p = s.record_metric("fps", 60.0)
        assert isinstance(p, TelemetryPoint)
        assert p.metric == "fps"
        assert p.value == 60.0
        assert p.tick == 1

    def test_record_with_tags(self):
        s = TelemetrySession()
        p = s.record_metric("fps", 55.0, tags={"phase": "riot"})
        assert p.tags["phase"] == "riot"

    def test_metric_stored_in_series(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0)
        s.record_metric("fps", 58.0)
        assert len(s.metrics["fps"]) == 2

    def test_multiple_metrics(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0)
        s.record_metric("entities", 100.0)
        assert "fps" in s.metrics
        assert "entities" in s.metrics

    def test_timestamp_is_positive(self):
        s = TelemetrySession()
        p = s.record_metric("fps", 60.0)
        assert p.timestamp >= 0


# ===========================================================================
# TelemetrySession — record_event
# ===========================================================================


class TestRecordEvent:
    def test_record_event(self):
        s = TelemetrySession()
        s.set_tick(5, 0.5)
        e = s.record_event("kill", {"attacker": "u1"})
        assert isinstance(e, GameEvent)
        assert e.event_type == "kill"
        assert e.tick == 5

    def test_event_default_data(self):
        s = TelemetrySession()
        e = s.record_event("phase_change")
        assert e.data == {}

    def test_events_appended(self):
        s = TelemetrySession()
        s.record_event("spawn")
        s.record_event("kill")
        s.record_event("explosion")
        assert len(s.events) == 3


# ===========================================================================
# TelemetrySession — record_frame
# ===========================================================================


class TestRecordFrame:
    def test_records_all_standard_metrics(self):
        s = TelemetrySession()
        s.set_tick(1, 0.016)
        s.record_frame(
            fps=60, frame_time=16.6, draw_calls=80,
            triangles=50000, entity_count=120,
            particle_count=30, projectile_count=5,
        )
        assert len(s.metrics) == 7
        assert MetricType.FPS.value in s.metrics
        assert MetricType.FRAME_TIME.value in s.metrics
        assert MetricType.DRAW_CALLS.value in s.metrics
        assert MetricType.TRIANGLES.value in s.metrics
        assert MetricType.ENTITIES.value in s.metrics
        assert MetricType.PARTICLES.value in s.metrics
        assert MetricType.PROJECTILES.value in s.metrics

    def test_frame_values_correct(self):
        s = TelemetrySession()
        s.record_frame(fps=45, frame_time=22.2, draw_calls=50,
                       triangles=30000, entity_count=80,
                       particle_count=10, projectile_count=2)
        assert s.metrics["fps"][0].value == 45.0
        assert s.metrics["frame_time"][0].value == 22.2
        assert s.metrics["draw_calls"][0].value == 50.0

    def test_multiple_frames(self):
        s = TelemetrySession()
        for i in range(10):
            s.set_tick(i, i * 0.016)
            s.record_frame(fps=60 - i, frame_time=16.6 + i, draw_calls=80,
                           triangles=50000, entity_count=100 + i,
                           particle_count=20, projectile_count=3)
        assert len(s.metrics["fps"]) == 10


# ===========================================================================
# TelemetrySession — get_series
# ===========================================================================


class TestGetSeries:
    def _populated_session(self) -> TelemetrySession:
        s = TelemetrySession()
        # Manually set start_time so timestamps are deterministic
        s.start_time = time.time()
        for i in range(5):
            s.set_tick(i, i * 0.1)
            p = s.record_metric("fps", 60.0 - i)
            # Override timestamp for deterministic tests
            p.timestamp = float(i)
        return s

    def test_get_all(self):
        s = self._populated_session()
        series = s.get_series("fps")
        assert len(series) == 5

    def test_get_nonexistent_metric(self):
        s = self._populated_session()
        assert s.get_series("nonexistent") == []

    def test_time_range_filter_start(self):
        s = self._populated_session()
        series = s.get_series("fps", start=2.0)
        assert len(series) == 3  # timestamps 2, 3, 4

    def test_time_range_filter_end(self):
        s = self._populated_session()
        series = s.get_series("fps", end=2.0)
        assert len(series) == 3  # timestamps 0, 1, 2

    def test_time_range_filter_both(self):
        s = self._populated_session()
        series = s.get_series("fps", start=1.0, end=3.0)
        assert len(series) == 3  # timestamps 1, 2, 3


# ===========================================================================
# TelemetrySession — get_events
# ===========================================================================


class TestGetEvents:
    def _session_with_events(self) -> TelemetrySession:
        s = TelemetrySession()
        for i, etype in enumerate(["spawn", "kill", "explosion", "spawn", "kill"]):
            s.set_tick(i, i * 0.1)
            e = s.record_event(etype, {"idx": i})
            e.timestamp = float(i)
        return s

    def test_get_all_events(self):
        s = self._session_with_events()
        assert len(s.get_events()) == 5

    def test_filter_by_type(self):
        s = self._session_with_events()
        kills = s.get_events(event_type="kill")
        assert len(kills) == 2

    def test_filter_by_type_no_match(self):
        s = self._session_with_events()
        assert s.get_events(event_type="dance") == []

    def test_time_range(self):
        s = self._session_with_events()
        events = s.get_events(start=1.0, end=3.0)
        assert len(events) == 3

    def test_combined_type_and_time(self):
        s = self._session_with_events()
        kills = s.get_events(event_type="kill", start=0.0, end=2.0)
        assert len(kills) == 1  # only the kill at timestamp 1


# ===========================================================================
# TelemetrySession — get_statistics
# ===========================================================================


class TestGetStatistics:
    def _session_with_values(self, values: list[float]) -> TelemetrySession:
        s = TelemetrySession()
        for i, v in enumerate(values):
            s.set_tick(i, i * 0.1)
            p = s.record_metric("test", v)
            p.timestamp = float(i)
        return s

    def test_empty_metric(self):
        s = TelemetrySession()
        assert s.get_statistics("nonexistent") == {}

    def test_basic_stats(self):
        s = self._session_with_values([10, 20, 30, 40, 50])
        stats = s.get_statistics("test")
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["avg"] == 30.0
        assert stats["count"] == 5.0

    def test_p50(self):
        s = self._session_with_values([10, 20, 30, 40, 50])
        stats = s.get_statistics("test")
        assert stats["p50"] == 30.0

    def test_single_value(self):
        s = self._session_with_values([42.0])
        stats = s.get_statistics("test")
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["avg"] == 42.0
        assert stats["p50"] == 42.0
        assert stats["std_dev"] == 0.0

    def test_std_dev(self):
        vals = [60, 60, 60, 60, 60]
        s = self._session_with_values(vals)
        stats = s.get_statistics("test")
        assert stats["std_dev"] == 0.0

    def test_std_dev_nonzero(self):
        vals = [10, 20, 30]
        s = self._session_with_values(vals)
        stats = s.get_statistics("test")
        assert stats["std_dev"] > 0

    def test_p95_p99(self):
        vals = list(range(1, 101))  # 1 to 100
        s = self._session_with_values(vals)
        stats = s.get_statistics("test")
        assert stats["p95"] >= 95
        assert stats["p99"] >= 99


# ===========================================================================
# TelemetrySession — get_timeline
# ===========================================================================


class TestGetTimeline:
    def test_empty_timeline(self):
        s = TelemetrySession()
        assert s.get_timeline() == []

    def test_mixed_timeline(self):
        s = TelemetrySession()
        p = s.record_metric("fps", 60.0)
        p.timestamp = 1.0
        e = s.record_event("kill")
        e.timestamp = 0.5
        timeline = s.get_timeline()
        assert len(timeline) == 2
        assert timeline[0]["type"] == "event"  # 0.5 before 1.0
        assert timeline[1]["type"] == "metric"

    def test_timeline_sorted_by_timestamp(self):
        s = TelemetrySession()
        for i in range(5):
            p = s.record_metric("fps", 60.0 - i)
            p.timestamp = float(4 - i)  # reverse order
        timeline = s.get_timeline()
        timestamps = [e["timestamp"] for e in timeline]
        assert timestamps == sorted(timestamps)


# ===========================================================================
# TelemetrySession — find_drops
# ===========================================================================


class TestFindDrops:
    def test_no_data(self):
        s = TelemetrySession()
        assert s.find_drops("fps", 30) == []

    def test_no_drops(self):
        s = TelemetrySession()
        for i in range(10):
            p = s.record_metric("fps", 60.0)
            p.timestamp = float(i)
        assert s.find_drops("fps", 30) == []

    def test_single_drop(self):
        s = TelemetrySession()
        values = [60, 60, 20, 15, 60, 60]
        for i, v in enumerate(values):
            p = s.record_metric("fps", float(v))
            p.timestamp = float(i)
        drops = s.find_drops("fps", 30)
        assert len(drops) == 1
        assert drops[0]["min_value"] == 15.0
        assert drops[0]["samples"] == 2

    def test_multiple_drops(self):
        s = TelemetrySession()
        values = [60, 10, 60, 20, 60]
        for i, v in enumerate(values):
            p = s.record_metric("fps", float(v))
            p.timestamp = float(i)
        drops = s.find_drops("fps", 30)
        assert len(drops) == 2

    def test_drop_at_end(self):
        s = TelemetrySession()
        values = [60, 60, 10, 5]
        for i, v in enumerate(values):
            p = s.record_metric("fps", float(v))
            p.timestamp = float(i)
        drops = s.find_drops("fps", 30)
        assert len(drops) == 1
        assert drops[0]["min_value"] == 5.0

    def test_drop_avg_value(self):
        s = TelemetrySession()
        values = [60, 20, 10, 60]
        for i, v in enumerate(values):
            p = s.record_metric("fps", float(v))
            p.timestamp = float(i)
        drops = s.find_drops("fps", 30)
        assert drops[0]["avg_value"] == 15.0

    def test_all_below_threshold(self):
        s = TelemetrySession()
        for i in range(5):
            p = s.record_metric("fps", 10.0)
            p.timestamp = float(i)
        drops = s.find_drops("fps", 30)
        assert len(drops) == 1
        assert drops[0]["samples"] == 5


# ===========================================================================
# TelemetrySession — save_json
# ===========================================================================


class TestSaveJson:
    def test_save_and_load(self):
        s = TelemetrySession(session_id="json_test", metadata={"preset": "city"})
        s.set_tick(1, 0.1)
        s.record_metric("fps", 60.0, tags={"phase": "calm"})
        s.record_event("spawn", {"unit": "infantry"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "json_test"
            assert data["metadata"]["preset"] == "city"
            assert "fps" in data["metrics"]
            assert len(data["events"]) == 1
            assert data["events"][0]["event_type"] == "spawn"
        finally:
            os.unlink(path)

    def test_json_valid_structure(self):
        s = TelemetrySession()
        s.record_frame(fps=60, frame_time=16.6, draw_calls=80,
                       triangles=50000, entity_count=100,
                       particle_count=20, projectile_count=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "session_id" in data
            assert "start_time" in data
            assert "metrics" in data
            assert "events" in data
        finally:
            os.unlink(path)


# ===========================================================================
# TelemetrySession — save_csv
# ===========================================================================


class TestSaveCsv:
    def test_csv_header(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            s.save_csv(path)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == ["timestamp", "sim_time", "tick", "metric", "value", "tags"]
        finally:
            os.unlink(path)

    def test_csv_rows(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0)
        s.record_metric("fps", 55.0)
        s.record_metric("entities", 100.0)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            s.save_csv(path)
            with open(path) as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) == 4  # header + 3 data rows
        finally:
            os.unlink(path)

    def test_csv_filter_metrics(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0)
        s.record_metric("entities", 100.0)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            s.save_csv(path, metrics=["fps"])
            with open(path) as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) == 2  # header + 1 fps row
            assert rows[1][3] == "fps"
        finally:
            os.unlink(path)

    def test_csv_tags_serialization(self):
        s = TelemetrySession()
        s.record_metric("fps", 60.0, tags={"phase": "riot", "wave": "3"})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            s.save_csv(path)
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                row = next(reader)
            tag_str = row[5]
            assert "phase=riot" in tag_str
            assert "wave=3" in tag_str
        finally:
            os.unlink(path)


# ===========================================================================
# TelemetrySession — summary
# ===========================================================================


class TestSummary:
    def test_empty_session_summary(self):
        s = TelemetrySession(session_id="empty")
        summary = s.summary()
        assert summary["session_id"] == "empty"
        assert summary["total_events"] == 0
        assert summary["peak_entities"] == 0
        assert summary["total_metric_points"] == 0

    def test_summary_with_data(self):
        s = TelemetrySession(session_id="test", metadata={"preset": "urban"})
        for i in range(10):
            s.set_tick(i, i * 0.1)
            p = s.record_metric("fps", 60.0 - i)
            p.timestamp = float(i)
            p2 = s.record_metric("entities", 100.0 + i * 10)
            p2.timestamp = float(i)
        s.record_event("kill")
        summary = s.summary()
        assert summary["session_id"] == "test"
        assert summary["avg_fps"] == pytest.approx(55.5)
        assert summary["min_fps"] == 51.0
        assert summary["peak_entities"] == 190.0
        assert summary["total_events"] == 1
        assert summary["metadata"]["preset"] == "urban"

    def test_summary_duration(self):
        s = TelemetrySession()
        p = s.record_metric("fps", 60.0)
        p.timestamp = 10.0
        summary = s.summary()
        assert summary["duration"] == 10.0


# ===========================================================================
# TelemetryDashboard
# ===========================================================================


class TestTelemetryDashboard:
    def _sample_session(self) -> TelemetrySession:
        s = TelemetrySession(session_id="dash_test", metadata={"preset": "riot"})
        for i in range(20):
            s.set_tick(i, i * 0.1)
            fps = 60.0 if i < 15 else 20.0
            s.record_frame(
                fps=fps, frame_time=1000.0 / max(fps, 1),
                draw_calls=80 + i, triangles=50000 + i * 100,
                entity_count=100 + i * 5,
                particle_count=20 + i, projectile_count=3,
            )
            # Override timestamps for deterministic ordering
            for metric_series in s.metrics.values():
                if metric_series and metric_series[-1].tick == i:
                    metric_series[-1].timestamp = float(i)
        s.record_event("spawn", {"unit": "infantry"})
        s.events[-1].timestamp = 0.5
        s.record_event("kill", {"attacker": "u1", "victim": "u2"})
        s.events[-1].timestamp = 5.0
        return s

    def test_generates_html(self):
        s = self._sample_session()
        dashboard = TelemetryDashboard()
        html = dashboard.generate_report(s)
        assert "<!DOCTYPE html>" in html
        assert "dash_test" in html

    def test_html_contains_charts(self):
        s = self._sample_session()
        html = TelemetryDashboard().generate_report(s)
        assert "<svg" in html
        assert "FPS Over Time" in html
        assert "Entity Count" in html
        assert "Draw Calls" in html

    def test_html_contains_stats_table(self):
        s = self._sample_session()
        html = TelemetryDashboard().generate_report(s)
        assert "Performance Statistics" in html
        assert "FPS" in html

    def test_html_contains_events(self):
        s = self._sample_session()
        html = TelemetryDashboard().generate_report(s)
        assert "Event Timeline" in html
        assert "spawn" in html
        assert "kill" in html

    def test_html_contains_drops(self):
        s = self._sample_session()
        html = TelemetryDashboard().generate_report(s)
        assert "FPS Drops" in html

    def test_html_contains_summary_cards(self):
        s = self._sample_session()
        html = TelemetryDashboard().generate_report(s)
        assert "summary-card" in html
        assert "Duration" in html
        assert "Avg FPS" in html

    def test_empty_session_report(self):
        s = TelemetrySession(session_id="empty")
        html = TelemetryDashboard().generate_report(s)
        assert "<!DOCTYPE html>" in html
        assert "No data" in html

    def test_html_escapes_special_chars(self):
        s = TelemetrySession(session_id="<script>alert(1)</script>")
        html = TelemetryDashboard().generate_report(s)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_svg_line_chart_with_threshold(self):
        s = self._sample_session()
        dashboard = TelemetryDashboard()
        html = dashboard.generate_report(s)
        # The FPS chart has threshold=30
        assert "stroke-dasharray" in html


# ===========================================================================
# PerformanceBudget — check
# ===========================================================================


class TestPerformanceBudgetCheck:
    def _session_with_frames(
        self, fps_values: list[float], entity_values: list[float] | None = None,
    ) -> TelemetrySession:
        s = TelemetrySession()
        for i, fps in enumerate(fps_values):
            s.set_tick(i, i * 0.016)
            entities = entity_values[i] if entity_values else 100
            s.record_frame(
                fps=fps, frame_time=1000.0 / max(fps, 0.1),
                draw_calls=50, triangles=30000,
                entity_count=int(entities), particle_count=10, projectile_count=2,
            )
        return s

    def test_no_violations(self):
        s = self._session_with_frames([60] * 10)
        budget = PerformanceBudget()
        violations = budget.check(s)
        assert len(violations) == 0

    def test_fps_violation(self):
        s = self._session_with_frames([60, 60, 20, 15, 60])
        budget = PerformanceBudget(fps_target=30)
        violations = budget.check(s)
        fps_v = [v for v in violations if v["metric"] == "fps"]
        assert len(fps_v) == 1
        assert fps_v[0]["violations"] == 2
        assert fps_v[0]["worst_value"] == 15.0

    def test_entity_violation(self):
        s = self._session_with_frames(
            [60] * 5,
            entity_values=[100, 200, 600, 700, 100],
        )
        budget = PerformanceBudget(max_entities=500)
        violations = budget.check(s)
        ent_v = [v for v in violations if v["metric"] == "entities"]
        assert len(ent_v) == 1
        assert ent_v[0]["violations"] == 2

    def test_pct_violated(self):
        s = self._session_with_frames([20] * 10)
        budget = PerformanceBudget(fps_target=30)
        violations = budget.check(s)
        fps_v = [v for v in violations if v["metric"] == "fps"]
        assert fps_v[0]["pct_violated"] == 100.0

    def test_empty_session(self):
        s = TelemetrySession()
        budget = PerformanceBudget()
        assert budget.check(s) == []

    def test_frame_time_violation(self):
        s = self._session_with_frames([30] * 5)  # frame_time ~33ms > 16.67
        budget = PerformanceBudget(max_frame_time=16.67)
        violations = budget.check(s)
        ft_v = [v for v in violations if v["metric"] == "frame_time"]
        assert len(ft_v) == 1

    def test_custom_budgets(self):
        budget = PerformanceBudget(
            fps_target=30, max_entities=1000,
            max_particles=500, max_draw_calls=200,
        )
        assert budget.fps_target == 30
        assert budget.max_entities == 1000


# ===========================================================================
# PerformanceBudget — grade
# ===========================================================================


class TestPerformanceBudgetGrade:
    def _session_with_fps(self, values: list[float]) -> TelemetrySession:
        s = TelemetrySession()
        for i, fps in enumerate(values):
            s.set_tick(i, i * 0.016)
            s.record_frame(
                fps=fps, frame_time=1000.0 / max(fps, 0.1),
                draw_calls=50, triangles=30000,
                entity_count=100, particle_count=10, projectile_count=2,
            )
        return s

    def test_grade_a(self):
        s = self._session_with_fps([60] * 100)
        budget = PerformanceBudget()
        assert budget.grade(s) == "A"

    def test_grade_b(self):
        # ~2-3% violation
        vals = [60] * 97 + [20] * 3
        s = self._session_with_fps(vals)
        budget = PerformanceBudget(fps_target=30)
        grade = budget.grade(s)
        assert grade == "B"

    def test_grade_f(self):
        s = self._session_with_fps([10] * 100)
        budget = PerformanceBudget(fps_target=30)
        assert budget.grade(s) == "F"

    def test_empty_session_grade(self):
        s = TelemetrySession()
        budget = PerformanceBudget()
        assert budget.grade(s) == "A"

    def test_grade_d(self):
        # ~20% violation
        vals = [60] * 80 + [20] * 20
        s = self._session_with_fps(vals)
        budget = PerformanceBudget(fps_target=30)
        grade = budget.grade(s)
        assert grade in ("C", "D")  # depends on weighted calculation


# ===========================================================================
# Utility function
# ===========================================================================


class TestEsc:
    def test_escapes_angle_brackets(self):
        assert _esc("<script>") == "&lt;script&gt;"

    def test_escapes_ampersand(self):
        assert _esc("a & b") == "a &amp; b"

    def test_escapes_quotes(self):
        assert _esc('"hello"') == "&quot;hello&quot;"

    def test_plain_text_unchanged(self):
        assert _esc("hello world") == "hello world"
