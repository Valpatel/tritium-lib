# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Automated performance test for city3d demo.

Runs the game server, starts a riot, records telemetry, generates report.

Usage:
    python3 -m tritium_lib.sim_engine.demos.perf_test
    # Generates: /tmp/tritium_perf_report.html
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.sim_engine.demos.city_sim_backend import CitySim
from tritium_lib.sim_engine.crowd import CrowdEvent, CrowdMood
from tritium_lib.sim_engine.telemetry import (
    TelemetrySession,
    TelemetryDashboard,
    PerformanceBudget,
    MetricType,
)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


@dataclass
class Phase:
    """A named phase of the performance test."""

    name: str
    duration: float  # seconds of sim time to run
    description: str = ""
    setup_fn: str = ""  # method name on PerfRunner to call at phase start


@dataclass
class TickSample:
    """Per-tick metrics captured during the test."""

    tick: int
    phase: str
    wall_time: float  # seconds since test start
    tick_duration_ms: float
    entity_count: int
    civilian_count: int
    vehicle_count: int
    police_count: int
    crowd_count: int
    particle_count: int
    fire_count: int
    event_count: int
    memory_bytes: int = 0


@dataclass
class PerfReport:
    """Aggregated results from a performance test run."""

    duration_s: float
    total_ticks: int
    phases: list[str]
    avg_fps: float
    min_fps: float
    min_fps_tick: int
    min_fps_phase: str
    min_fps_events: list[str]
    max_entity_count: int
    max_entity_tick: int
    max_particle_count: int
    max_particle_tick: int
    grade: str
    report_path: str
    samples: list[TickSample] = field(default_factory=list)

    def worst_frames(self, n: int = 5) -> list[TickSample]:
        """Return the N frames with highest tick duration."""
        return sorted(self.samples, key=lambda s: s.tick_duration_ms, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Performance test runner
# ---------------------------------------------------------------------------


DEFAULT_PHASES = [
    Phase("PEACEFUL", 30.0, "Baseline — city at rest"),
    Phase("RIOT", 60.0, "Combat load — riot in progress", setup_fn="start_riot"),
    Phase("AFTERMATH", 30.0, "Cleanup — riot winding down"),
]


class PerfRunner:
    """Runs the city simulation through performance test phases.

    Parameters
    ----------
    phases : list[Phase] or None
        Test phases to run. Defaults to PEACEFUL/RIOT/AFTERMATH.
    dt : float
        Tick delta-time in seconds. Default 0.05 (20 tps target).
    seed : int
        Random seed for the simulation.
    fps_target : float
        Target FPS for grading. Default 60.
    report_path : str
        Output path for the HTML report.
    """

    def __init__(
        self,
        phases: list[Phase] | None = None,
        dt: float = 0.05,
        seed: int = 42,
        fps_target: float = 60.0,
        report_path: str = "/tmp/tritium_perf_report.html",
    ) -> None:
        self.phases = phases or list(DEFAULT_PHASES)
        self.dt = dt
        self.seed = seed
        self.fps_target = fps_target
        self.report_path = report_path

        self.sim: CitySim | None = None
        self.telemetry: TelemetrySession | None = None
        self.samples: list[TickSample] = []
        self._current_phase: str = ""
        self._phase_events: dict[str, list[str]] = {}

    # -- phase setup callbacks -------------------------------------------

    def start_riot(self) -> None:
        """Inject crowd events to start a riot."""
        if self.sim is None:
            return
        cx = self.sim.width / 2
        cy = self.sim.height / 2
        # Agitate the crowd
        self.sim.inject_crowd_event(
            "riot_start", (cx, cy), radius=40.0, intensity=0.9
        )
        self._record_event("riot_start")
        # Inject molotov after a few ticks
        self.sim.inject_crowd_event(
            "molotov", (cx + 15, cy + 10), radius=15.0, intensity=0.8
        )
        self._record_event("molotov")

    # -- internal helpers ------------------------------------------------

    def _record_event(self, event_type: str) -> None:
        """Record a named event in telemetry and phase tracking."""
        if self.telemetry is not None:
            self.telemetry.record_event(
                event_type, {"phase": self._current_phase}
            )
        self._phase_events.setdefault(self._current_phase, []).append(event_type)

    def _sample_tick(
        self, tick: int, wall_elapsed: float, tick_dur_ms: float
    ) -> TickSample:
        """Capture a snapshot of the sim state for this tick."""
        sim = self.sim
        assert sim is not None

        crowd_count = len(sim.crowd.members) if sim.crowd else 0
        fire_count = sum(
            1 for s in sim.buildings if s.health < s.max_health * 0.9
        )
        event_count = len(sim.events)

        sample = TickSample(
            tick=tick,
            phase=self._current_phase,
            wall_time=wall_elapsed,
            tick_duration_ms=tick_dur_ms,
            entity_count=(
                len(sim.civilians)
                + len(sim.city_vehicles)
                + len(sim.police_units)
                + crowd_count
            ),
            civilian_count=len(sim.civilians),
            vehicle_count=len(sim.city_vehicles),
            police_count=len(sim.police_units),
            crowd_count=crowd_count,
            particle_count=0,  # CitySim doesn't track particles directly
            fire_count=fire_count,
            event_count=event_count,
        )
        self.samples.append(sample)
        return sample

    # -- main run --------------------------------------------------------

    def run(self) -> PerfReport:
        """Execute all phases and return a PerfReport.

        Returns
        -------
        PerfReport
            Aggregated test results including the HTML report path.
        """
        # Set up the simulation
        self.sim = CitySim(seed=self.seed)
        self.sim.setup()
        self.telemetry = TelemetrySession(
            metadata={
                "type": "perf_test",
                "seed": self.seed,
                "dt": self.dt,
                "phases": [p.name for p in self.phases],
            }
        )
        self.samples = []
        self._phase_events = {}

        total_tick = 0
        test_start = time.perf_counter()

        for phase in self.phases:
            self._current_phase = phase.name
            self._record_event(f"phase_start_{phase.name}")

            # Call phase setup if specified
            if phase.setup_fn:
                fn = getattr(self, phase.setup_fn, None)
                if fn is not None:
                    fn()

            ticks_for_phase = max(1, int(phase.duration / self.dt))

            for i in range(ticks_for_phase):
                t0 = time.perf_counter()
                self.sim.tick(self.dt)
                t1 = time.perf_counter()

                tick_ms = (t1 - t0) * 1000.0
                wall_elapsed = t1 - test_start
                fps = 1000.0 / tick_ms if tick_ms > 0 else 9999.0

                # Record in telemetry session
                self.telemetry.set_tick(total_tick, self.sim.sim_time)
                self.telemetry.record_metric(
                    MetricType.FPS.value, fps, tags={"phase": phase.name}
                )
                self.telemetry.record_metric(
                    MetricType.FRAME_TIME.value,
                    tick_ms,
                    tags={"phase": phase.name},
                )

                sample = self._sample_tick(total_tick, wall_elapsed, tick_ms)
                self.telemetry.record_metric(
                    MetricType.ENTITIES.value,
                    float(sample.entity_count),
                    tags={"phase": phase.name},
                )
                self.telemetry.record_metric(
                    MetricType.PARTICLES.value,
                    float(sample.particle_count),
                    tags={"phase": phase.name},
                )

                # Detect notable events during riot phase
                if phase.name == "RIOT":
                    for ev in self.sim.events:
                        ev_type = ev.get("type", "")
                        if ev_type and ev_type not in ("crowd_event",):
                            self._record_event(ev_type)

                total_tick += 1

        test_end = time.perf_counter()
        total_duration = test_end - test_start

        # Build the report
        report = self._build_report(total_duration, total_tick)

        # Generate HTML report
        self._generate_html(report)

        return report

    def _build_report(self, duration_s: float, total_ticks: int) -> PerfReport:
        """Aggregate samples into a PerfReport."""
        if not self.samples:
            return PerfReport(
                duration_s=duration_s,
                total_ticks=total_ticks,
                phases=[p.name for p in self.phases],
                avg_fps=0.0,
                min_fps=0.0,
                min_fps_tick=0,
                min_fps_phase="",
                min_fps_events=[],
                max_entity_count=0,
                max_entity_tick=0,
                max_particle_count=0,
                max_particle_tick=0,
                grade="F",
                report_path=self.report_path,
                samples=self.samples,
            )

        tick_durations = [s.tick_duration_ms for s in self.samples]
        fps_values = [
            1000.0 / d if d > 0 else 9999.0 for d in tick_durations
        ]
        avg_fps = statistics.mean(fps_values)
        min_fps = min(fps_values)
        min_idx = fps_values.index(min_fps)
        min_sample = self.samples[min_idx]

        max_entity_sample = max(self.samples, key=lambda s: s.entity_count)
        max_particle_sample = max(self.samples, key=lambda s: s.particle_count)

        # Events near the worst frame
        min_events = self._phase_events.get(min_sample.phase, [])

        # Grade using PerformanceBudget
        grade = "A"
        if self.telemetry is not None:
            budget = PerformanceBudget(fps_target=self.fps_target)
            grade = budget.grade(self.telemetry)

        return PerfReport(
            duration_s=duration_s,
            total_ticks=total_ticks,
            phases=[p.name for p in self.phases],
            avg_fps=avg_fps,
            min_fps=min_fps,
            min_fps_tick=min_sample.tick,
            min_fps_phase=min_sample.phase,
            min_fps_events=min_events[:5],
            max_entity_count=max_entity_sample.entity_count,
            max_entity_tick=max_entity_sample.tick,
            max_particle_count=max_particle_sample.particle_count,
            max_particle_tick=max_particle_sample.tick,
            grade=grade,
            report_path=self.report_path,
            samples=self.samples,
        )

    def _generate_html(self, report: PerfReport) -> str:
        """Generate an HTML report and write to report_path.

        Returns the HTML string.
        """
        if self.telemetry is None:
            return ""

        dashboard = TelemetryDashboard()
        base_html = dashboard.generate_report(self.telemetry)

        # Inject performance-test-specific sections before closing </body>
        phase_markers = self._build_phase_markers_html()
        worst_frames = self._build_worst_frames_html(report)
        grade_section = self._build_grade_html(report)
        summary_section = self._build_summary_html(report)

        injection = f"""
<h2 style="color:#ff2a6d;margin-top:30px;">Performance Test Summary</h2>
{summary_section}

<h2 style="color:#ff2a6d;margin-top:30px;">Performance Grade</h2>
{grade_section}

<h2 style="color:#ff2a6d;margin-top:30px;">Phase Timeline</h2>
{phase_markers}

<h2 style="color:#ff2a6d;margin-top:30px;">Worst 5 Frames</h2>
{worst_frames}
"""
        html = base_html.replace("</body>", injection + "</body>")

        with open(self.report_path, "w") as f:
            f.write(html)

        return html

    def _build_phase_markers_html(self) -> str:
        """Build HTML table of phase boundaries and events."""
        rows = ""
        for phase in self.phases:
            phase_samples = [s for s in self.samples if s.phase == phase.name]
            if not phase_samples:
                continue
            start_tick = phase_samples[0].tick
            end_tick = phase_samples[-1].tick
            fps_vals = [
                1000.0 / s.tick_duration_ms
                if s.tick_duration_ms > 0
                else 9999.0
                for s in phase_samples
            ]
            avg_fps = statistics.mean(fps_vals) if fps_vals else 0.0
            events = self._phase_events.get(phase.name, [])
            unique_events = sorted(set(events))
            rows += (
                f"<tr><td>{_esc(phase.name)}</td>"
                f"<td>{start_tick}-{end_tick}</td>"
                f"<td>{len(phase_samples)}</td>"
                f"<td>{avg_fps:.1f}</td>"
                f"<td>{_esc(', '.join(unique_events[:8]))}</td></tr>\n"
            )

        return f"""<table style="border-collapse:collapse;width:100%;margin:10px 0;">
<tr style="background:#1a1a2e;color:#00f0ff;">
<th style="border:1px solid #333;padding:6px;">Phase</th>
<th style="border:1px solid #333;padding:6px;">Ticks</th>
<th style="border:1px solid #333;padding:6px;">Samples</th>
<th style="border:1px solid #333;padding:6px;">Avg FPS</th>
<th style="border:1px solid #333;padding:6px;">Events</th></tr>
{rows}
</table>"""

    def _build_worst_frames_html(self, report: PerfReport) -> str:
        """Build HTML table of worst N frames."""
        worst = report.worst_frames(5)
        rows = ""
        for s in worst:
            fps = 1000.0 / s.tick_duration_ms if s.tick_duration_ms > 0 else 9999.0
            rows += (
                f"<tr><td>{s.tick}</td>"
                f"<td>{_esc(s.phase)}</td>"
                f"<td>{s.tick_duration_ms:.2f}</td>"
                f"<td>{fps:.1f}</td>"
                f"<td>{s.entity_count}</td>"
                f"<td>{s.crowd_count}</td>"
                f"<td>{s.fire_count}</td></tr>\n"
            )

        return f"""<table style="border-collapse:collapse;width:100%;margin:10px 0;">
<tr style="background:#1a1a2e;color:#00f0ff;">
<th style="border:1px solid #333;padding:6px;">Tick</th>
<th style="border:1px solid #333;padding:6px;">Phase</th>
<th style="border:1px solid #333;padding:6px;">Duration (ms)</th>
<th style="border:1px solid #333;padding:6px;">FPS</th>
<th style="border:1px solid #333;padding:6px;">Entities</th>
<th style="border:1px solid #333;padding:6px;">Crowd</th>
<th style="border:1px solid #333;padding:6px;">Fires</th></tr>
{rows}
</table>"""

    def _build_grade_html(self, report: PerfReport) -> str:
        """Build the grade display section."""
        color_map = {"A": "#05ffa1", "B": "#00f0ff", "C": "#fcee0a", "D": "#ff8800", "F": "#ff2a6d"}
        color = color_map.get(report.grade, "#ccc")
        return (
            f'<div style="text-align:center;margin:20px 0;">'
            f'<span style="font-size:4em;font-weight:bold;color:{color};">'
            f'{_esc(report.grade)}</span>'
            f'<div style="color:#888;margin-top:8px;">Target: {self.fps_target:.0f} FPS</div>'
            f'</div>'
        )

    def _build_summary_html(self, report: PerfReport) -> str:
        """Build key metrics summary cards."""
        return f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:15px 0;">
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Duration</div>
  <div style="color:#05ffa1;font-size:1.4em;font-weight:bold;">{report.duration_s:.1f}s</div>
</div>
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Total Ticks</div>
  <div style="color:#05ffa1;font-size:1.4em;font-weight:bold;">{report.total_ticks}</div>
</div>
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Avg FPS</div>
  <div style="color:#05ffa1;font-size:1.4em;font-weight:bold;">{report.avg_fps:.1f}</div>
</div>
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Min FPS</div>
  <div style="color:#ff2a6d;font-size:1.4em;font-weight:bold;">{report.min_fps:.1f}</div>
</div>
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Peak Entities</div>
  <div style="color:#05ffa1;font-size:1.4em;font-weight:bold;">{report.max_entity_count}</div>
</div>
<div style="background:#12121a;border:1px solid #333;padding:12px;border-radius:4px;">
  <div style="color:#888;font-size:0.85em;">Phases</div>
  <div style="color:#05ffa1;font-size:1.4em;font-weight:bold;">{len(report.phases)}</div>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the performance test and print summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tritium city3d performance test"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Tick delta-time in seconds (default 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)"
    )
    parser.add_argument(
        "--fps-target",
        type=float,
        default=60.0,
        help="Target FPS for grading (default 60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/tritium_perf_report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--peaceful",
        type=float,
        default=30.0,
        help="Peaceful phase duration in seconds (default 30)",
    )
    parser.add_argument(
        "--riot",
        type=float,
        default=60.0,
        help="Riot phase duration in seconds (default 60)",
    )
    parser.add_argument(
        "--aftermath",
        type=float,
        default=30.0,
        help="Aftermath phase duration in seconds (default 30)",
    )
    args = parser.parse_args()

    phases = [
        Phase("PEACEFUL", args.peaceful, "Baseline -- city at rest"),
        Phase("RIOT", args.riot, "Combat load -- riot in progress", setup_fn="start_riot"),
        Phase("AFTERMATH", args.aftermath, "Cleanup -- riot winding down"),
    ]

    runner = PerfRunner(
        phases=phases,
        dt=args.dt,
        seed=args.seed,
        fps_target=args.fps_target,
        report_path=args.output,
    )

    print("\n=== TRITIUM PERF TEST ===")
    print(f"Phases: {', '.join(p.name for p in phases)}")
    print(f"Target FPS: {args.fps_target}")
    print(f"Seed: {args.seed}, dt: {args.dt}")
    print("Running...\n")

    report = runner.run()

    # Print summary
    worst = report.worst_frames(1)
    worst_events = ", ".join(report.min_fps_events[:3]) if report.min_fps_events else "none"

    print(f"=== TRITIUM PERF REPORT ===")
    print(
        f"Duration: {report.duration_s:.1f}s across {len(report.phases)} phases"
    )
    print(
        f"Avg FPS: {report.avg_fps:.1f} (target: {args.fps_target:.0f})"
    )
    print(
        f"Min FPS: {report.min_fps:.1f} at tick {report.min_fps_tick} "
        f"(Phase: {report.min_fps_phase}, Events: {worst_events})"
    )
    print(f"Entity peak: {report.max_entity_count} at tick {report.max_entity_tick}")
    print(
        f"Particle peak: {report.max_particle_count} at tick {report.max_particle_tick}"
    )
    print(f"Grade: {report.grade}")
    print(f"Report: {report.report_path}")


if __name__ == "__main__":
    main()
