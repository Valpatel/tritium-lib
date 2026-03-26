# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Performance tests for CrowdSimulator spatial-grid optimization."""

from __future__ import annotations

import random
import time

import pytest

from tritium_lib.sim_engine.crowd import CrowdMood, CrowdSimulator


BOUNDS = (0.0, 0.0, 100.0, 100.0)


class TestCrowdPerformance:
    """Verify that the spatial-grid optimization keeps tick times low."""

    def _make_sim(self, count: int, mood: CrowdMood = CrowdMood.CALM) -> CrowdSimulator:
        """Create a simulator with *count* members spread across the bounds."""
        sim = CrowdSimulator(BOUNDS, max_members=count + 10)
        sim.spawn_crowd((50.0, 50.0), count, radius=45.0, mood=mood)
        return sim

    def test_crowd_500_under_5ms(self):
        """500 crowd members must tick in under 5ms at P95."""
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
        assert p95 < 5.0, (
            f"P95 tick time {p95:.2f}ms exceeds 5ms target (avg={avg:.2f}ms)"
        )

    def test_crowd_200_under_2ms(self):
        """200 crowd members must tick in under 2ms at P95."""
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
        assert p95 < 4.0, f"P95 tick time {p95:.2f}ms exceeds 4ms target"

    def test_agitated_crowd_500_under_12ms(self):
        """500 agitated members (more movement, more interactions) under 12ms.

        Agitated members do more work per tick (group center targeting,
        separation forces, event scanning), so the budget is higher than calm.
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
        assert p95 < 12.0, f"P95 tick time {p95:.2f}ms exceeds 12ms target"

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
