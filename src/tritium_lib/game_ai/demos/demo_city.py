# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""City life simulation visual demo.

Run: python3 -m tritium_lib.game_ai.demos.demo_city

Shows 50 residents with daily schedules in a neighborhood.
Cars drive on roads, people walk between buildings.
Simulated clock: 1 real second = 10 sim minutes.

Falls back to ASCII terminal output if matplotlib unavailable.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

try:
    import matplotlib
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        pass
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from tritium_lib.game_ai.city_sim import (
    Building,
    BuildingType,
    NeighborhoodSim,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE = 500.0
NUM_RESIDENTS = 50
TIME_SCALE = 10.0 / 60.0  # 10 sim minutes per real second (in hours/sec)
START_HOUR = 6.0  # Start at 6 AM
DT = 1.0  # 1 second physics step

# Activity colors for matplotlib
ACTIVITY_COLORS = {
    "sleeping": "#555555",
    "napping": "#666666",
    "waking_up": "#888888",
    "commuting": "#fcee0a",
    "working": "#05ffa1",
    "at_school": "#00b8ff",
    "lunch": "#ff8c00",
    "walking": "#00f0ff",
    "walking_dog": "#00ddcc",
    "jogging": "#ff2a6d",
    "playing": "#ff66ff",
    "relaxing": "#7755ff",
    "gardening": "#33cc33",
    "shopping": "#ffaa00",
    "delivering": "#ff4444",
}

# ASCII activity symbols
ACTIVITY_CHARS = {
    "sleeping": "z",
    "napping": "z",
    "waking_up": ".",
    "commuting": "C",
    "working": "W",
    "at_school": "S",
    "lunch": "L",
    "walking": "w",
    "walking_dog": "d",
    "jogging": "j",
    "playing": "p",
    "relaxing": "r",
    "gardening": "g",
    "shopping": "$",
    "delivering": "D",
}

# Building type colors
BUILDING_COLORS = {
    BuildingType.HOME: "#1a1a2e",
    BuildingType.OFFICE: "#2a2a4e",
    BuildingType.SCHOOL: "#1a3a3e",
    BuildingType.GROCERY: "#3a2a1e",
    BuildingType.PARK: "#0a2a0a",
    BuildingType.RESTAURANT: "#3a1a2e",
    BuildingType.GAS_STATION: "#2a2a2a",
}


def hour_to_string(hour: float) -> str:
    """Convert fractional hour to HH:MM string."""
    h = int(hour) % 24
    m = int((hour % 1.0) * 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12:2d}:{m:02d} {ampm}"


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------

def run_matplotlib(duration: float) -> None:
    """Run city demo with matplotlib animation."""
    sim = NeighborhoodSim(num_residents=NUM_RESIDENTS,
                          bounds=((0, 0), (WORLD_SIZE, WORLD_SIZE)),
                          seed=42)
    sim.populate()

    fig, (ax_map, ax_stats) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("Tritium City Life Simulation", color="#00f0ff", fontsize=16)

    # Map axes
    ax_map.set_facecolor("#0e0e14")
    ax_map.set_xlim(0, WORLD_SIZE)
    ax_map.set_ylim(0, WORLD_SIZE)
    ax_map.set_aspect("equal")
    ax_map.set_title("Neighborhood", color="#aaaaaa", fontsize=12)
    ax_map.tick_params(colors="#555555")
    for spine in ax_map.spines.values():
        spine.set_color("#333333")

    # Stats axes
    ax_stats.set_facecolor("#0e0e14")
    ax_stats.set_title("Activity Stats", color="#aaaaaa", fontsize=12)
    ax_stats.tick_params(colors="#555555")
    for spine in ax_stats.spines.values():
        spine.set_color("#333333")

    # Draw buildings (static)
    building_size = {
        BuildingType.HOME: 8,
        BuildingType.OFFICE: 15,
        BuildingType.SCHOOL: 18,
        BuildingType.GROCERY: 12,
        BuildingType.PARK: 20,
        BuildingType.RESTAURANT: 10,
        BuildingType.GAS_STATION: 10,
    }
    for b in sim.buildings:
        sz = building_size.get(b.building_type, 8)
        color = BUILDING_COLORS.get(b.building_type, "#1a1a2e")
        rect = Rectangle(
            (b.position[0] - sz / 2, b.position[1] - sz / 2),
            sz, sz,
            facecolor=color, edgecolor="#333333", linewidth=0.5,
            alpha=0.7, zorder=1,
        )
        ax_map.add_patch(rect)

    # People scatter
    people_scatter = ax_map.scatter([], [], s=25, zorder=3, label="People")
    # Vehicle scatter
    vehicle_scatter = ax_map.scatter([], [], s=60, marker="s", zorder=2,
                                     edgecolors="#ffaa00", linewidth=0.5,
                                     label="Vehicles")

    clock_text = ax_map.text(
        10, WORLD_SIZE - 15, "", color="#fcee0a", fontsize=14,
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0f", edgecolor="#333333"),
        zorder=10,
    )
    fps_text = ax_map.text(
        10, WORLD_SIZE - 35, "", color="#888888", fontsize=9,
        fontfamily="monospace", zorder=10,
    )

    sim_hour = [START_HOUR]
    frame_count = [0]
    t_start = [time.time()]
    t_end = time.time() + duration

    def update(frame_num):
        now = time.time()
        if now >= t_end:
            plt.close(fig)
            return ()

        # Advance simulation time
        real_dt = 0.05  # animation interval
        sim_hours_elapsed = real_dt * TIME_SCALE * 60  # TIME_SCALE is hours/sec already
        sim_hour[0] += real_dt * TIME_SCALE
        if sim_hour[0] >= 24.0:
            sim_hour[0] -= 24.0

        # Run several physics ticks per frame
        for _ in range(3):
            sim.tick(DT, sim_hour[0])

        frame_count[0] += 1
        elapsed = now - t_start[0]
        fps = frame_count[0] / max(elapsed, 0.001)

        # Gather people positions and colors
        px, py, pcolors = [], [], []
        for r in sim.residents:
            px.append(r.position[0])
            py.append(r.position[1])
            pcolors.append(ACTIVITY_COLORS.get(r.current_activity, "#ffffff"))

        if px:
            people_scatter.set_offsets(list(zip(px, py)))
            people_scatter.set_color(pcolors)

        # Gather vehicle positions
        vx, vy, vcolors = [], [], []
        for v in sim.vehicles:
            vx.append(v.position[0])
            vy.append(v.position[1])
            vcolors.append("#fcee0a" if v.driving else "#666666")

        if vx:
            vehicle_scatter.set_offsets(list(zip(vx, vy)))
            vehicle_scatter.set_color(vcolors)

        # Update clock
        clock_text.set_text(f"  {hour_to_string(sim_hour[0])}  ")
        fps_text.set_text(
            f"FPS: {fps:.0f}  Elapsed: {elapsed:.0f}s/{duration:.0f}s"
        )

        # Update stats bar chart
        stats = sim.get_statistics()
        activities = stats.get("activities", {})
        ax_stats.clear()
        ax_stats.set_facecolor("#0e0e14")
        ax_stats.set_title("Activity Breakdown", color="#aaaaaa", fontsize=11)

        if activities:
            sorted_acts = sorted(activities.items(), key=lambda x: -x[1])
            labels = [a[0].replace("_", "\n") for a in sorted_acts]
            values = [a[1] for a in sorted_acts]
            colors = [ACTIVITY_COLORS.get(a[0], "#ffffff") for a in sorted_acts]
            bars = ax_stats.barh(labels, values, color=colors, edgecolor="#333333")
            ax_stats.set_xlim(0, max(values) * 1.3 if values else 10)
            ax_stats.tick_params(colors="#888888", labelsize=8)
            for spine in ax_stats.spines.values():
                spine.set_color("#333333")
            # Add count labels
            for bar, val in zip(bars, values):
                ax_stats.text(
                    bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", color="#cccccc", fontsize=8,
                )

        # Vehicle stats text at bottom
        veh_driving = stats.get("vehicles_driving", 0)
        veh_parked = stats.get("vehicles_parked", 0)
        ax_stats.text(
            0.05, -0.05,
            f"Vehicles: {veh_driving} driving, {veh_parked} parked",
            transform=ax_stats.transAxes, color="#ffaa00", fontsize=9,
        )

        return ()

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    elapsed = time.time() - t_start[0]
    print(f"\nCity sim ran for {elapsed:.1f}s, simulated {(sim_hour[0] - START_HOUR) % 24:.1f} hours")
    stats = sim.get_statistics()
    print(f"Final stats: {stats}")


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------

def run_terminal(duration: float) -> None:
    """Run city demo with ASCII terminal output."""
    sim = NeighborhoodSim(num_residents=NUM_RESIDENTS,
                          bounds=((0, 0), (WORLD_SIZE, WORLD_SIZE)),
                          seed=42)
    sim.populate()

    sim_hour = START_HOUR
    frame = 0
    t_start = time.time()
    t_end = t_start + duration
    cols, rows = 80, 25

    try:
        while time.time() < t_end:
            real_dt = 0.1
            sim_hour += real_dt * TIME_SCALE
            if sim_hour >= 24.0:
                sim_hour -= 24.0

            for _ in range(3):
                sim.tick(DT, sim_hour)

            frame += 1
            elapsed = time.time() - t_start
            fps = frame / max(elapsed, 0.001)

            if frame % 5 == 0:
                # Build ASCII grid
                grid = [["." for _ in range(cols)] for _ in range(rows)]

                # Draw buildings
                for b in sim.buildings:
                    gc = int(b.position[0] / WORLD_SIZE * cols) % cols
                    gr = int(b.position[1] / WORLD_SIZE * rows) % rows
                    bchar = {
                        BuildingType.HOME: "#",
                        BuildingType.OFFICE: "B",
                        BuildingType.SCHOOL: "S",
                        BuildingType.GROCERY: "G",
                        BuildingType.PARK: "P",
                        BuildingType.RESTAURANT: "R",
                    }
                    grid[gr][gc] = bchar.get(b.building_type, "#")

                # Draw vehicles
                for v in sim.vehicles:
                    gc = int(v.position[0] / WORLD_SIZE * cols) % cols
                    gr = int(v.position[1] / WORLD_SIZE * rows) % rows
                    grid[gr][gc] = "V" if v.driving else "v"

                # Draw people
                for r in sim.residents:
                    gc = int(r.position[0] / WORLD_SIZE * cols) % cols
                    gr = int(r.position[1] / WORLD_SIZE * rows) % rows
                    grid[gr][gc] = ACTIVITY_CHARS.get(r.current_activity, "?")

                stats = sim.get_statistics()
                activities = stats.get("activities", {})

                lines = []
                lines.append(f"  {hour_to_string(sim_hour)}  |  FPS: {fps:.0f}  |  "
                             f"Frame: {frame}  |  Elapsed: {elapsed:.0f}s")
                lines.append("-" * cols)
                for row in grid:
                    lines.append("".join(row))
                lines.append("-" * cols)

                # Stats line
                act_parts = [f"{k}:{v}" for k, v in sorted(activities.items(), key=lambda x: -x[1])]
                lines.append("  ".join(act_parts[:6]))
                vd = stats.get("vehicles_driving", 0)
                vp = stats.get("vehicles_parked", 0)
                lines.append(f"  Vehicles: {vd} driving, {vp} parked")

                sys.stdout.write("\033[H\033[J")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            time.sleep(max(0, 0.1 - (time.time() - t_start - (frame - 1) * 0.1)))
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    print(f"\nCity sim ran for {elapsed:.1f}s")
    stats = sim.get_statistics()
    print(f"Final stats: {stats}")


# ---------------------------------------------------------------------------
# Headless
# ---------------------------------------------------------------------------

def run_headless(duration: float) -> None:
    """Run city sim headless, print summary."""
    sim = NeighborhoodSim(num_residents=NUM_RESIDENTS,
                          bounds=((0, 0), (WORLD_SIZE, WORLD_SIZE)),
                          seed=42)
    sim.populate()

    sim_hour = START_HOUR
    frame = 0
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        sim_hour += 0.05 * TIME_SCALE
        if sim_hour >= 24.0:
            sim_hour -= 24.0
        sim.tick(DT, sim_hour)
        frame += 1

    elapsed = time.time() - t_start
    fps = frame / max(elapsed, 0.001)
    stats = sim.get_statistics()

    print("=" * 55)
    print("  Tritium City Life Demo -- Headless Results")
    print("=" * 55)
    print(f"  Residents:     {stats['total_residents']}")
    print(f"  Vehicles:      {stats['total_vehicles']}")
    print(f"  Buildings:     {stats['total_buildings']}")
    print(f"  Duration:      {elapsed:.1f}s")
    print(f"  Frames:        {frame}")
    print(f"  FPS:           {fps:.1f}")
    print(f"  Tick time:     {elapsed / max(frame, 1) * 1000:.2f} ms")
    print(f"  Final hour:    {hour_to_string(sim_hour)}")
    print(f"  Veh driving:   {stats['vehicles_driving']}")
    print(f"  Veh parked:    {stats['vehicles_parked']}")
    print(f"  Activities:")
    for activity, count in sorted(stats.get("activities", {}).items(), key=lambda x: -x[1]):
        print(f"    {activity:20s} {count:3d}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tritium city life simulation demo")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    args = parser.parse_args()

    if args.headless:
        run_headless(args.duration)
    elif HAS_MPL:
        run_matplotlib(args.duration)
    else:
        print("matplotlib not available, falling back to terminal renderer")
        run_terminal(args.duration)


if __name__ == "__main__":
    main()
