# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Performance benchmark for the game_ai simulation engine.

Run: python3 -m tritium_lib.game_ai.demos.demo_perf

Tests scaling from 50 to 1000 agents. Measures tick time, FPS, and memory usage.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc

import numpy as np

from tritium_lib.game_ai.steering_np import SteeringSystem
from tritium_lib.game_ai.city_sim import NeighborhoodSim
from tritium_lib.game_ai.ambient_np import AmbientSimulatorNP, PEDESTRIAN, VEHICLE


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_steering(agent_count: int, duration: float) -> dict:
    """Benchmark the SteeringSystem with flocking agents."""
    gc.collect()
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    ss = SteeringSystem(max_agents=agent_count + 16)
    for _ in range(agent_count):
        x = np.random.uniform(0, 500)
        y = np.random.uniform(0, 500)
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        ss.add_agent(
            pos=(x, y), vel=(vx, vy),
            max_speed=np.random.uniform(1.5, 3.0),
            max_force=2.5,
            behavior=(
                SteeringSystem.WANDER
                | SteeringSystem.SEPARATE
                | SteeringSystem.ALIGN
                | SteeringSystem.COHERE
            ),
        )

    dt = 0.05
    frame = 0
    tick_times = []
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        t0 = time.perf_counter()
        ss.tick(dt)
        t1 = time.perf_counter()
        tick_times.append(t1 - t0)

        # Wrap positions
        pos = ss.positions[:ss.count]
        pos[:, 0] = pos[:, 0] % 500.0
        pos[:, 1] = pos[:, 1] % 500.0

        frame += 1

    elapsed = time.time() - t_start
    mem_after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    tick_ms = np.array(tick_times) * 1000
    return {
        "agents": agent_count,
        "system": "steering",
        "frames": frame,
        "elapsed_s": elapsed,
        "fps": frame / max(elapsed, 0.001),
        "tick_avg_ms": float(tick_ms.mean()),
        "tick_p50_ms": float(np.percentile(tick_ms, 50)),
        "tick_p95_ms": float(np.percentile(tick_ms, 95)),
        "tick_max_ms": float(tick_ms.max()),
        "memory_mb": (mem_after - mem_before) / (1024 * 1024),
    }


def bench_city(resident_count: int, duration: float) -> dict:
    """Benchmark the NeighborhoodSim."""
    gc.collect()
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    sim = NeighborhoodSim(
        num_residents=resident_count,
        bounds=((0, 0), (500, 500)),
        seed=42,
    )
    sim.populate()

    dt = 1.0
    sim_hour = 8.0  # Start at 8 AM (busy time)
    frame = 0
    tick_times = []
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        t0 = time.perf_counter()
        sim.tick(dt, sim_hour)
        t1 = time.perf_counter()
        tick_times.append(t1 - t0)

        sim_hour += 0.01
        if sim_hour >= 24.0:
            sim_hour -= 24.0
        frame += 1

    elapsed = time.time() - t_start
    mem_after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    tick_ms = np.array(tick_times) * 1000
    return {
        "agents": resident_count,
        "system": "city_sim",
        "frames": frame,
        "elapsed_s": elapsed,
        "fps": frame / max(elapsed, 0.001),
        "tick_avg_ms": float(tick_ms.mean()),
        "tick_p50_ms": float(np.percentile(tick_ms, 50)),
        "tick_p95_ms": float(np.percentile(tick_ms, 95)),
        "tick_max_ms": float(tick_ms.max()),
        "memory_mb": (mem_after - mem_before) / (1024 * 1024),
    }


def bench_ambient(entity_count: int, duration: float) -> dict:
    """Benchmark the AmbientSimulatorNP."""
    gc.collect()
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    sim = AmbientSimulatorNP(
        bounds=((0, 0), (500, 500)),
        max_entities=entity_count + 64,
        seed=42,
    )
    # Spawn 80% pedestrians, 20% vehicles
    ped_count = int(entity_count * 0.8)
    veh_count = entity_count - ped_count
    sim.spawn_batch(PEDESTRIAN, ped_count)
    sim.spawn_batch(VEHICLE, veh_count)

    dt = 0.1
    frame = 0
    tick_times = []
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        t0 = time.perf_counter()
        sim.tick(dt, current_hour=12.0)
        t1 = time.perf_counter()
        tick_times.append(t1 - t0)
        frame += 1

    elapsed = time.time() - t_start
    mem_after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    tick_ms = np.array(tick_times) * 1000
    return {
        "agents": entity_count,
        "system": "ambient_np",
        "frames": frame,
        "elapsed_s": elapsed,
        "fps": frame / max(elapsed, 0.001),
        "tick_avg_ms": float(tick_ms.mean()),
        "tick_p50_ms": float(np.percentile(tick_ms, 50)),
        "tick_p95_ms": float(np.percentile(tick_ms, 95)),
        "tick_max_ms": float(tick_ms.max()),
        "memory_mb": (mem_after - mem_before) / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(results: list[dict], title: str) -> None:
    """Print a formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Agents':>6} | {'Tick avg':>10} | {'Tick p95':>10} | "
          f"{'FPS':>8} | {'Memory':>10} | {'Frames':>7}")
    print(f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-"
          f"{'-' * 8}-+-{'-' * 10}-+-{'-' * 7}")

    for r in results:
        print(f"  {r['agents']:>6} | "
              f"{r['tick_avg_ms']:>8.2f}ms | "
              f"{r['tick_p95_ms']:>8.2f}ms | "
              f"{r['fps']:>8.1f} | "
              f"{r['memory_mb']:>8.2f}MB | "
              f"{r['frames']:>7}")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tritium game_ai performance benchmark")
    parser.add_argument("--headless", action="store_true",
                        help="No effect (benchmark is always text output)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration per test in seconds (default 5)")
    parser.add_argument("--agents", type=str, default="50,100,200,500,1000",
                        help="Comma-separated agent counts to test")
    args = parser.parse_args()

    agent_counts = [int(x.strip()) for x in args.agents.split(",")]
    duration = args.duration

    np.random.seed(42)

    print("\n" + "=" * 80)
    print("  TRITIUM GAME AI PERFORMANCE BENCHMARK")
    print(f"  Duration per test: {duration}s | Agent counts: {agent_counts}")
    print("=" * 80)

    # --- Steering system benchmark ---
    print("\n  Running SteeringSystem (flocking) benchmarks...")
    steering_results = []
    for n in agent_counts:
        sys.stdout.write(f"    {n:>5} agents... ")
        sys.stdout.flush()
        r = bench_steering(n, duration)
        steering_results.append(r)
        sys.stdout.write(f"done ({r['fps']:.0f} FPS, {r['tick_avg_ms']:.2f}ms/tick)\n")

    print_table(steering_results, "SteeringSystem (Flocking: separate + align + cohere + wander)")

    # --- City sim benchmark ---
    print("\n  Running NeighborhoodSim benchmarks...")
    city_results = []
    city_counts = [c for c in agent_counts if c <= 500]  # City sim is O(n) per resident
    for n in city_counts:
        sys.stdout.write(f"    {n:>5} residents... ")
        sys.stdout.flush()
        r = bench_city(n, duration)
        city_results.append(r)
        sys.stdout.write(f"done ({r['fps']:.0f} FPS, {r['tick_avg_ms']:.2f}ms/tick)\n")

    print_table(city_results, "NeighborhoodSim (Daily schedules + vehicle pathfinding)")

    # --- Ambient NP benchmark ---
    print("\n  Running AmbientSimulatorNP benchmarks...")
    ambient_results = []
    for n in agent_counts:
        sys.stdout.write(f"    {n:>5} entities... ")
        sys.stdout.flush()
        r = bench_ambient(n, duration)
        ambient_results.append(r)
        sys.stdout.write(f"done ({r['fps']:.0f} FPS, {r['tick_avg_ms']:.2f}ms/tick)\n")

    print_table(ambient_results, "AmbientSimulatorNP (Vectorized ambient movement)")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    if steering_results:
        best = steering_results[0]
        worst = steering_results[-1]
        print(f"  Steering:  {best['agents']} agents @ {best['fps']:.0f} FPS"
              f" -> {worst['agents']} agents @ {worst['fps']:.0f} FPS")
    if city_results:
        best = city_results[0]
        worst = city_results[-1]
        print(f"  City sim:  {best['agents']} residents @ {best['fps']:.0f} FPS"
              f" -> {worst['agents']} residents @ {worst['fps']:.0f} FPS")
    if ambient_results:
        best = ambient_results[0]
        worst = ambient_results[-1]
        print(f"  Ambient:   {best['agents']} entities @ {best['fps']:.0f} FPS"
              f" -> {worst['agents']} entities @ {worst['fps']:.0f} FPS")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
