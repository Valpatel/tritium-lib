# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine performance test and coverage report tools."""
from __future__ import annotations

import os
import statistics
import tempfile
import time

import pytest

from tritium_lib.sim_engine.demos.perf_test import (
    DEFAULT_PHASES,
    PerfReport,
    PerfRunner,
    Phase,
    TickSample,
    _esc,
)
from tritium_lib.sim_engine.demos.test_report import (
    ModuleInfo,
    CoverageReport,
    discover_sim_engine_modules,
    find_test_files,
    generate_coverage_report,
)
from tritium_lib.sim_engine.telemetry import (
    MetricType,
    PerformanceBudget,
    TelemetryDashboard,
    TelemetrySession,
)


# ---------------------------------------------------------------------------
# Phase and TickSample dataclass tests
# ---------------------------------------------------------------------------


class TestPhase:
    """Test Phase dataclass."""

    def test_default_phase(self) -> None:
        p = Phase("TEST", 10.0)
        assert p.name == "TEST"
        assert p.duration == 10.0
        assert p.description == ""
        assert p.setup_fn == ""

    def test_phase_with_setup(self) -> None:
        p = Phase("RIOT", 60.0, "Combat phase", setup_fn="start_riot")
        assert p.setup_fn == "start_riot"
        assert p.description == "Combat phase"


class TestTickSample:
    """Test TickSample dataclass."""

    def test_creation(self) -> None:
        s = TickSample(
            tick=1,
            phase="PEACEFUL",
            wall_time=0.5,
            tick_duration_ms=2.0,
            entity_count=50,
            civilian_count=25,
            vehicle_count=12,
            police_count=10,
            crowd_count=3,
            particle_count=0,
            fire_count=0,
            event_count=0,
        )
        assert s.tick == 1
        assert s.phase == "PEACEFUL"
        assert s.entity_count == 50
        assert s.memory_bytes == 0  # default


# ---------------------------------------------------------------------------
# PerfReport tests
# ---------------------------------------------------------------------------


class TestPerfReport:
    """Test PerfReport aggregation and worst_frames."""

    def _make_samples(self, n: int = 20) -> list[TickSample]:
        samples = []
        for i in range(n):
            samples.append(
                TickSample(
                    tick=i,
                    phase="TEST",
                    wall_time=i * 0.05,
                    tick_duration_ms=2.0 + (i % 5) * 3.0,
                    entity_count=50 + i,
                    civilian_count=25,
                    vehicle_count=12,
                    police_count=10,
                    crowd_count=3 + i,
                    particle_count=i,
                    fire_count=0,
                    event_count=0,
                )
            )
        return samples

    def test_worst_frames_returns_n(self) -> None:
        samples = self._make_samples(20)
        report = PerfReport(
            duration_s=1.0,
            total_ticks=20,
            phases=["TEST"],
            avg_fps=100.0,
            min_fps=50.0,
            min_fps_tick=0,
            min_fps_phase="TEST",
            min_fps_events=[],
            max_entity_count=69,
            max_entity_tick=19,
            max_particle_count=19,
            max_particle_tick=19,
            grade="A",
            report_path="/tmp/test.html",
            samples=samples,
        )
        worst = report.worst_frames(5)
        assert len(worst) == 5
        # Should be sorted by tick_duration_ms descending
        durations = [w.tick_duration_ms for w in worst]
        assert durations == sorted(durations, reverse=True)

    def test_worst_frames_less_than_n(self) -> None:
        samples = self._make_samples(3)
        report = PerfReport(
            duration_s=0.15,
            total_ticks=3,
            phases=["TEST"],
            avg_fps=100.0,
            min_fps=50.0,
            min_fps_tick=0,
            min_fps_phase="TEST",
            min_fps_events=[],
            max_entity_count=52,
            max_entity_tick=2,
            max_particle_count=2,
            max_particle_tick=2,
            grade="A",
            report_path="/tmp/test.html",
            samples=samples,
        )
        worst = report.worst_frames(10)
        assert len(worst) == 3


# ---------------------------------------------------------------------------
# PerfRunner tests (short phases for speed)
# ---------------------------------------------------------------------------


class TestPerfRunner:
    """Test the PerfRunner with minimal phase durations."""

    def _short_phases(self) -> list[Phase]:
        return [
            Phase("PEACEFUL", 0.2, "Short baseline"),
            Phase("RIOT", 0.3, "Short riot", setup_fn="start_riot"),
            Phase("AFTERMATH", 0.2, "Short aftermath"),
        ]

    def test_runner_creates_sim(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        assert runner.sim is None
        report = runner.run()
        assert runner.sim is not None
        assert report.total_ticks > 0

    def test_runner_produces_samples(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        report = runner.run()
        assert len(report.samples) == report.total_ticks
        assert all(isinstance(s, TickSample) for s in report.samples)

    def test_runner_has_all_phases(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        report = runner.run()
        phases_seen = set(s.phase for s in report.samples)
        assert "PEACEFUL" in phases_seen
        assert "RIOT" in phases_seen
        assert "AFTERMATH" in phases_seen

    def test_runner_fps_positive(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        report = runner.run()
        assert report.avg_fps > 0
        assert report.min_fps > 0

    def test_runner_grade_is_letter(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        report = runner.run()
        assert report.grade in ("A", "B", "C", "D", "F")

    def test_runner_generates_html_report(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            runner = PerfRunner(
                phases=self._short_phases(), dt=0.05, report_path=path
            )
            report = runner.run()
            assert os.path.exists(path)
            with open(path) as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "Performance Test Summary" in html
            assert "Performance Grade" in html
            assert "Phase Timeline" in html
            assert "Worst 5 Frames" in html
        finally:
            os.unlink(path)

    def test_runner_telemetry_recorded(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        runner.run()
        assert runner.telemetry is not None
        fps_series = runner.telemetry.get_series(MetricType.FPS.value)
        assert len(fps_series) > 0
        entity_series = runner.telemetry.get_series(MetricType.ENTITIES.value)
        assert len(entity_series) > 0

    def test_runner_events_recorded(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        runner.run()
        assert runner.telemetry is not None
        events = runner.telemetry.events
        # Should have phase_start events at minimum
        event_types = [e.event_type for e in events]
        assert "phase_start_PEACEFUL" in event_types
        assert "phase_start_RIOT" in event_types
        assert "phase_start_AFTERMATH" in event_types
        # Riot setup should inject riot_start and molotov
        assert "riot_start" in event_types
        assert "molotov" in event_types

    def test_runner_entity_peak_tracked(self) -> None:
        runner = PerfRunner(phases=self._short_phases(), dt=0.05)
        report = runner.run()
        assert report.max_entity_count > 0
        assert report.max_entity_tick >= 0

    def test_runner_custom_fps_target(self) -> None:
        runner = PerfRunner(
            phases=self._short_phases(), dt=0.05, fps_target=10000.0
        )
        report = runner.run()
        # With an absurdly high target, most frames will be "below target"
        # so grade should be poor
        assert report.grade in ("C", "D", "F")


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------


class TestEscape:
    """Test HTML escaping utility."""

    def test_escapes_html_chars(self) -> None:
        assert _esc("<script>") == "&lt;script&gt;"
        assert _esc('a="b"') == 'a=&quot;b&quot;'
        assert _esc("a&b") == "a&amp;b"

    def test_passthrough_safe_strings(self) -> None:
        assert _esc("hello world") == "hello world"
        assert _esc("123") == "123"


# ---------------------------------------------------------------------------
# Test report / module discovery
# ---------------------------------------------------------------------------


class TestModuleDiscovery:
    """Test sim_engine module discovery."""

    def test_discovers_modules(self) -> None:
        modules = discover_sim_engine_modules()
        assert len(modules) > 10
        names = [m.name for m in modules]
        # Should find core modules
        assert any("crowd" in n for n in names)
        assert any("world" in n for n in names)
        assert any("telemetry" in n for n in names)

    def test_modules_importable(self) -> None:
        modules = discover_sim_engine_modules()
        importable = [m for m in modules if m.importable]
        assert len(importable) > 5

    def test_module_info_has_counts(self) -> None:
        modules = discover_sim_engine_modules()
        importable = [m for m in modules if m.importable]
        assert len(importable) > 0
        # At least some modules should have classes
        with_classes = [m for m in importable if m.class_count > 0]
        assert len(with_classes) > 0


class TestTestFileDiscovery:
    """Test finding test files."""

    def test_finds_test_files(self) -> None:
        mapping = find_test_files()
        assert len(mapping) > 0

    def test_maps_to_modules(self) -> None:
        mapping = find_test_files()
        # At least one module should have mapped test files
        has_mapped = any(
            not k.startswith("_test:") for k in mapping.keys()
        )
        assert has_mapped


class TestCoverageReportGeneration:
    """Test HTML coverage report generation."""

    def test_generates_report(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report = generate_coverage_report(output_path=path)
            assert report.total_modules > 0
            assert report.modules_importable > 0
            assert os.path.exists(path)
            with open(path) as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "Test Coverage Report" in html
        finally:
            os.unlink(path)

    def test_report_has_coverage_pct(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report = generate_coverage_report(output_path=path)
            assert 0.0 <= report.coverage_pct <= 100.0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Default phases sanity
# ---------------------------------------------------------------------------


class TestDefaultPhases:
    """Test that DEFAULT_PHASES are well-formed."""

    def test_three_phases(self) -> None:
        assert len(DEFAULT_PHASES) == 3

    def test_phase_names(self) -> None:
        names = [p.name for p in DEFAULT_PHASES]
        assert "PEACEFUL" in names
        assert "RIOT" in names
        assert "AFTERMATH" in names

    def test_riot_has_setup(self) -> None:
        riot = [p for p in DEFAULT_PHASES if p.name == "RIOT"][0]
        assert riot.setup_fn == "start_riot"

    def test_total_duration(self) -> None:
        total = sum(p.duration for p in DEFAULT_PHASES)
        assert total == 120.0
