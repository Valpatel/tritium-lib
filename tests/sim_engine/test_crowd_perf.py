# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Performance tests for CrowdSimulator spatial-grid optimization.

NOTE: These tests are sensitive to system load (CPU contention, swap
pressure).  They are marked ``@pytest.mark.slow`` and will be skipped
automatically when the 1-minute load average exceeds 20 or available
memory is below 4 GB (i.e., when the machine is too busy for reliable
micro-benchmarks).
"""

from __future__ import annotations

import os
import random
import time

import pytest

from tritium_lib.sim_engine.crowd import CrowdMood, CrowdSimulator


def _system_too_loaded() -> bool:
    """Return True when the machine is under too much pressure for perf tests."""
    try:
        load_1m = os.getloadavg()[0]
        if load_1m > 20:
            return True
    except OSError:
        pass
    # Check available memory (Linux only)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    if avail_kb < 4 * 1024 * 1024:  # <4 GB
                        return True
                    break
    except (OSError, ValueError):
        pass
    return False


_skip_loaded = pytest.mark.skipif(
    _system_too_loaded(),
    reason="System too loaded for reliable perf benchmarks "
           f"(load={os.getloadavg()[0]:.1f})",
)


BOUNDS = (0.0, 0.0, 100.0, 100.0)


@_skip_loaded
class TestCrowdPerformance:
    """Verify that the spatial-grid optimization keeps tick times low."""

    def _make_sim(self, count: int, mood: CrowdMood = CrowdMood.CALM) -> CrowdSimulator:
        """Create a simulator with *count* members spread across the bounds."""
        sim = CrowdSimulator(BOUNDS, max_members=count + 10)
        sim.spawn_crowd((50.0, 50.0), count, radius=45.0, mood=mood)
        return sim

    def test_crowd_500_under_50ms(self):
        """500 crowd members must tick in under 50ms at P95.

        Profiling shows ~13ms on an idle machine.  We use a generous budget
        (50ms) so that the test still passes when the host is under heavy
        load (swap active, parallel test suites, etc.).
        """
        random.seed(42)
        sim = self._make_sim(500)

        # Warm-up ticks
        for _ in range(3):
            sim.tick(0.1)

        times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            sim.tick(0.1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        p95 = sorted(times)[94]
        avg = sum(times) / len(times)
        assert p95 < 50.0, (
            f"P95 tick time {p95:.2f}ms exceeds 50ms target (avg={avg:.2f}ms)"
        )

    def test_crowd_200_under_25ms(self):
        """200 crowd members must tick in under 25ms at P95."""
        random.seed(123)
        sim = self._make_sim(200)

        for _ in range(3):
            sim.tick(0.1)

        times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            sim.tick(0.1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        p95 = sorted(times)[94]
        assert p95 < 25.0, f"P95 tick time {p95:.2f}ms exceeds 25ms target"

    def test_agitated_crowd_500_under_100ms(self):
        """500 agitated members (more movement, more interactions) under 100ms.

        Agitated members do more work per tick (group center targeting,
        separation forces, event scanning), so the budget is higher than calm.
        Profiled at ~20-30ms idle; budget set to 100ms for loaded systems.
        """
        random.seed(99)
        sim = self._make_sim(500, mood=CrowdMood.AGITATED)

        for _ in range(3):
            sim.tick(0.1)

        times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            sim.tick(0.1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        p95 = sorted(times)[94]
        assert p95 < 100.0, f"P95 tick time {p95:.2f}ms exceeds 100ms target"

    def test_scaling_subquadratic(self):
        """Doubling crowd size should NOT quadruple tick time (confirms O(n*k) not O(n^2))."""
        random.seed(7)
        sim_small = self._make_sim(100)
        sim_large = self._make_sim(400)

        # Warm up
        for _ in range(3):
            sim_small.tick(0.1)
            sim_large.tick(0.1)

        def measure(sim: CrowdSimulator, runs: int = 50) -> float:
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                sim.tick(0.1)
                times.append(time.perf_counter() - start)
            return sum(times) / len(times)

        avg_small = measure(sim_small)
        avg_large = measure(sim_large)

        # With O(n^2), 4x members -> 16x time.
        # With spatial grid, 4x members -> ~4-6x time (linear in n, constant k).
        ratio = avg_large / max(avg_small, 1e-9)
        assert ratio < 10.0, (
            f"4x members caused {ratio:.1f}x slowdown — "
            f"expected <10x for sub-quadratic scaling "
            f"(small={avg_small*1000:.2f}ms, large={avg_large*1000:.2f}ms)"
        )
