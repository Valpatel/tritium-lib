# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GTA-style full city simulation — ALL game_ai modules combined.

Run: python3 -m tritium_lib.game_ai.demos.demo_full

One window, everything running together:
- NeighborhoodSim with 50 residents on daily schedules
- PersonRFProfile + VehicleRFProfile for every entity
- Steering behaviors for smooth movement
- Buildings, roads, people, cars, RF emissions
- Stats panel with clock, activity breakdown, RF counts, FPS

10x accelerated time: 1 real second = 10 sim minutes.
Cyberpunk color scheme matching Tritium.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, FancyBboxPatch
    HAS_MPL = True
except (ImportError, Exception):
    HAS_MPL = False

from tritium_lib.game_ai.city_sim import (
    ActivityState,
    Building,
    BuildingType,
    NeighborhoodSim,
    Resident,
    SimVehicle,
    state_movement_type,
    state_rf_emission,
    state_visible_on_map,
)
from tritium_lib.game_ai.rf_signatures import (
    PersonRFProfile,
    RFSignatureGenerator,
    VehicleRFProfile,
)
from tritium_lib.game_ai.steering import Vec2


# ---------------------------------------------------------------------------
# Constants — cyberpunk palette
# ---------------------------------------------------------------------------

WORLD_SIZE = 500.0
NUM_RESIDENTS = 50
TIME_SCALE = 10.0 / 60.0  # 10 sim minutes per real second (hours/sec)
START_HOUR = 6.0
SIM_DT = 1.0  # simulation dt in seconds per tick
TICKS_PER_FRAME = 6  # internal ticks per visual frame for smooth movement

# Tritium cyberpunk colors
BG_COLOR = "#0a0a0f"
SURFACE_COLOR = "#0e0e14"
PANEL_BG = "#12121a"
GRID_COLOR = "#1a1a2e"
CYAN = "#00f0ff"
MAGENTA = "#ff2a6d"
GREEN = "#05ffa1"
YELLOW = "#fcee0a"
ORANGE = "#ff8c00"
PURPLE = "#b030ff"
DARK_GREY = "#333344"
ROAD_COLOR = "#2a2a3e"
TEXT_DIM = "#556677"
TEXT_BRIGHT = "#aaccdd"

# Activity state -> (color, marker, size, label)
_ACTIVITY_STYLE: dict[str, tuple[str, str, float, str]] = {
    # Walking variants
    ActivityState.WALKING:             (CYAN,    "o", 4.0, "Walking"),
    ActivityState.WALKING_TO_CAR:      (CYAN,    "o", 3.5, "Walking"),
    ActivityState.WALKING_TO_BUILDING: (CYAN,    "o", 3.5, "Walking"),
    ActivityState.ENTERING_BUILDING:   (CYAN,    "o", 3.0, "Walking"),
    ActivityState.EXITING_BUILDING:    (CYAN,    "o", 3.0, "Walking"),
    ActivityState.RETURNING_HOME:      (CYAN,    "o", 4.0, "Walking"),
    ActivityState.WALKING_TO_TRANSIT:  (CYAN,    "o", 3.5, "Walking"),
    ActivityState.LUNCH_BREAK:         (CYAN,    "o", 3.5, "Walking"),
    ActivityState.CHECKING_PHONE:      (CYAN,    "o", 3.5, "Phone"),
    ActivityState.SOCIALIZING:         (CYAN,    "o", 4.0, "Socializing"),
    # Driving
    ActivityState.DRIVING:             (YELLOW,  "s", 6.0, "Driving"),
    ActivityState.GETTING_IN_CAR:      (YELLOW,  "o", 3.5, "In car"),
    ActivityState.GETTING_OUT_OF_CAR:  (YELLOW,  "o", 3.5, "Out car"),
    ActivityState.PARKING:             (YELLOW,  "s", 5.0, "Parking"),
    ActivityState.DELIVERING:          (YELLOW,  "s", 6.0, "Delivering"),
    ActivityState.DELIVERY_STOP:       (ORANGE,  "o", 4.0, "Delivery"),
    # Working / school (inside building, dim)
    ActivityState.WORKING:             (GREEN,   "o", 2.5, "Working"),
    ActivityState.AT_SCHOOL:           (GREEN,   "o", 2.5, "At school"),
    ActivityState.INSIDE_BUILDING:     (GREEN,   "o", 2.0, "Inside"),
    # Outdoor activities
    ActivityState.JOGGING:             (ORANGE,  "o", 4.5, "Jogging"),
    ActivityState.WALKING_DOG:         (PURPLE,  "o", 4.0, "Dog walk"),
    ActivityState.PLAYING:             (GREEN,   "o", 4.0, "Playing"),
    ActivityState.GARDENING:           (GREEN,   "o", 3.5, "Gardening"),
    # Shopping / errands
    ActivityState.SHOPPING:            (MAGENTA, "o", 4.0, "Shopping"),
    ActivityState.DINING:              (MAGENTA, "o", 3.5, "Dining"),
    ActivityState.AT_GAS_STATION:      (MAGENTA, "o", 3.5, "Gas stn"),
    ActivityState.AT_DOCTOR:           (MAGENTA, "o", 3.0, "Doctor"),
    ActivityState.GETTING_COFFEE:      (MAGENTA, "o", 3.5, "Coffee"),
    # Sleeping / resting (not shown or dim)
    ActivityState.SLEEPING:            (DARK_GREY, "o", 0.0, "Sleeping"),
    ActivityState.NAPPING:             (DARK_GREY, "o", 0.0, "Napping"),
    ActivityState.WAKING_UP:           (DARK_GREY, "o", 0.0, "Waking"),
    ActivityState.RELAXING:            (DARK_GREY, "o", 0.0, "Relaxing"),
}

# Building type -> (color, label char)
_BUILDING_STYLE: dict[str, tuple[str, str]] = {
    BuildingType.HOME:       ("#1a2a3a", "H"),
    BuildingType.OFFICE:     ("#1a3a2a", "O"),
    BuildingType.SCHOOL:     ("#2a2a3a", "S"),
    BuildingType.GROCERY:    ("#3a2a1a", "G"),
    BuildingType.PARK:       ("#0a2a0a", "P"),
    BuildingType.RESTAURANT: ("#3a1a2a", "R"),
    BuildingType.GAS_STATION: ("#3a3a1a", "F"),
    BuildingType.COFFEE_SHOP: ("#2a1a1a", "C"),
    BuildingType.DOCTOR:     ("#1a1a3a", "D"),
}

# Consolidated activity categories for the stats panel
_CATEGORY_MAP = {
    "Walking": (CYAN, [
        ActivityState.WALKING, ActivityState.WALKING_TO_CAR,
        ActivityState.WALKING_TO_BUILDING, ActivityState.ENTERING_BUILDING,
        ActivityState.EXITING_BUILDING, ActivityState.RETURNING_HOME,
        ActivityState.WALKING_TO_TRANSIT, ActivityState.LUNCH_BREAK,
        ActivityState.CHECKING_PHONE, ActivityState.SOCIALIZING,
    ]),
    "Driving": (YELLOW, [
        ActivityState.DRIVING, ActivityState.GETTING_IN_CAR,
        ActivityState.GETTING_OUT_OF_CAR, ActivityState.PARKING,
        ActivityState.DELIVERING, ActivityState.DELIVERY_STOP,
    ]),
    "Working": (GREEN, [
        ActivityState.WORKING, ActivityState.AT_SCHOOL,
        ActivityState.INSIDE_BUILDING,
    ]),
    "Outdoor": (ORANGE, [
        ActivityState.JOGGING, ActivityState.WALKING_DOG,
        ActivityState.PLAYING, ActivityState.GARDENING,
    ]),
    "Errands": (MAGENTA, [
        ActivityState.SHOPPING, ActivityState.DINING,
        ActivityState.AT_GAS_STATION, ActivityState.AT_DOCTOR,
        ActivityState.GETTING_COFFEE,
    ]),
    "Sleeping": (DARK_GREY, [
        ActivityState.SLEEPING, ActivityState.NAPPING,
        ActivityState.WAKING_UP, ActivityState.RELAXING,
    ]),
}


# ---------------------------------------------------------------------------
# RF signature attachment
# ---------------------------------------------------------------------------

def attach_rf_profiles(
    residents: list[Resident],
    vehicles: list[SimVehicle],
) -> tuple[dict[str, PersonRFProfile], dict[str, VehicleRFProfile]]:
    """Generate and attach RF profiles to all residents and vehicles."""
    person_rf: dict[str, PersonRFProfile] = {}
    vehicle_rf: dict[str, VehicleRFProfile] = {}

    for r in residents:
        person_rf[r.resident_id] = RFSignatureGenerator.random_person()

    for v in vehicles:
        vehicle_rf[v.vehicle_id] = RFSignatureGenerator.random_vehicle()

    return person_rf, vehicle_rf


def count_rf_emissions(
    residents: list[Resident],
    vehicles: list[SimVehicle],
    person_rf: dict[str, PersonRFProfile],
    vehicle_rf: dict[str, VehicleRFProfile],
) -> tuple[int, int, int]:
    """Count BLE, WiFi, and TPMS emissions this tick."""
    ble_count = 0
    wifi_count = 0
    tpms_count = 0

    for r in residents:
        rf_level = state_rf_emission(r.activity_state)
        if rf_level == "none":
            continue
        prof = person_rf.get(r.resident_id)
        if prof is None:
            continue
        # Count BLE devices this person emits
        if prof.has_phone:
            ble_count += 1
        if prof.has_smartwatch:
            ble_count += 1
        if prof.has_earbuds:
            ble_count += 1
        # WiFi probes (only in full emission mode)
        if rf_level == "full" and prof.has_phone and prof.phone_wifi_probes:
            wifi_count += 1

    for v in vehicles:
        prof = vehicle_rf.get(v.vehicle_id)
        if prof is None:
            continue
        if v.driving:
            tpms_count += 4  # 4 tires
            if prof.has_keyfob:
                ble_count += 1
            if prof.has_dashcam_wifi:
                wifi_count += 1

    return ble_count, wifi_count, tpms_count


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------

def run_visual_demo(num_residents: int = NUM_RESIDENTS) -> None:
    """Run the full GTA-style demo with matplotlib TkAgg."""
    if not HAS_MPL:
        print("matplotlib with TkAgg backend required. Install: pip install matplotlib")
        print("Falling back to terminal output...")
        run_terminal_demo(num_residents)
        return

    # Create simulation
    sim = NeighborhoodSim(
        num_residents=num_residents,
        bounds=((0.0, 0.0), (WORLD_SIZE, WORLD_SIZE)),
        seed=42,
    )
    sim.populate()

    # Attach RF profiles
    person_rf, vehicle_rf = attach_rf_profiles(sim.residents, sim.vehicles)

    # State
    current_time = [START_HOUR]
    frame_count = [0]
    fps_time = [time.time()]
    fps_val = [0.0]

    # --- Set up figure ---
    fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title("TRITIUM — City Simulation")

    # Map axes (left 70%)
    ax_map = fig.add_axes([0.02, 0.02, 0.63, 0.96])
    ax_map.set_facecolor(BG_COLOR)
    ax_map.set_xlim(-10, WORLD_SIZE + 10)
    ax_map.set_ylim(-10, WORLD_SIZE + 10)
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors=TEXT_DIM, labelsize=6)
    for spine in ax_map.spines.values():
        spine.set_color(GRID_COLOR)

    # Stats panel (right 32%)
    ax_stats = fig.add_axes([0.68, 0.02, 0.30, 0.96])
    ax_stats.set_facecolor(PANEL_BG)
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    # Draw static buildings once
    building_patches = []
    for b in sim.buildings:
        style = _BUILDING_STYLE.get(b.building_type, ("#1a1a2a", "?"))
        color, label = style
        bx, by = b.position
        # Building rectangle (approx 20x15 for homes, bigger for offices)
        bw = 18 if b.building_type == BuildingType.HOME else 25
        bh = 12 if b.building_type == BuildingType.HOME else 18
        if b.building_type == BuildingType.PARK:
            bw, bh = 30, 30
        rect = Rectangle(
            (bx - bw / 2, by - bh / 2), bw, bh,
            facecolor=color, edgecolor="#334455", linewidth=0.5, alpha=0.8,
            zorder=1,
        )
        ax_map.add_patch(rect)
        ax_map.text(
            bx, by, label, fontsize=5, color=TEXT_DIM,
            ha="center", va="center", zorder=2, fontfamily="monospace",
        )
        building_patches.append(rect)

    # Pre-create scatter and text objects for animation updates
    person_scatter = ax_map.scatter([], [], s=[], c=[], marker="o", zorder=5, linewidths=0)
    vehicle_scatter = ax_map.scatter([], [], s=[], c=[], marker="s", zorder=4, linewidths=0)
    parked_scatter = ax_map.scatter([], [], s=10, c=DARK_GREY, marker="s", zorder=3,
                                    alpha=0.4, linewidths=0)
    rf_texts: list = []

    # Title
    title_text = ax_map.set_title(
        "TRITIUM NEIGHBORHOOD SIM",
        color=CYAN, fontsize=11, fontfamily="monospace", fontweight="bold",
        pad=8,
    )

    def _format_time(h: float) -> str:
        hours = int(h) % 24
        minutes = int((h % 1.0) * 60)
        ampm = "AM" if hours < 12 else "PM"
        display_h = hours % 12 or 12
        return f"{display_h:2d}:{minutes:02d} {ampm}"

    def _update(frame_num: int):
        nonlocal rf_texts

        # --- Advance simulation ---
        for _ in range(TICKS_PER_FRAME):
            sim.tick(SIM_DT, current_time[0])
            current_time[0] += (SIM_DT * TIME_SCALE) / TICKS_PER_FRAME
            if current_time[0] >= 24.0:
                current_time[0] -= 24.0

        # --- FPS ---
        frame_count[0] += 1
        now = time.time()
        elapsed = now - fps_time[0]
        if elapsed >= 1.0:
            fps_val[0] = frame_count[0] / elapsed
            frame_count[0] = 0
            fps_time[0] = now

        # --- Collect entity data ---
        # People
        px, py, ps, pc = [], [], [], []
        vx, vy, vs, vc = [], [], [], []
        parked_x, parked_y = [], []
        rf_mac_data: list[tuple[float, float, str]] = []

        activity_counts: dict[str, int] = defaultdict(int)
        visible_count = 0
        inside_count = 0

        for r in sim.residents:
            state = r.activity_state
            style = _ACTIVITY_STYLE.get(state)
            if style is None:
                style = (DARK_GREY, "o", 0.0, "Unknown")
            color, marker, size, cat_label = style

            # Count activities
            activity_counts[state] += 1

            if not r.visible:
                inside_count += 1
                continue

            visible_count += 1

            if size <= 0:
                continue

            if state in (ActivityState.DRIVING, ActivityState.DELIVERING,
                         ActivityState.PARKING, ActivityState.GETTING_IN_CAR,
                         ActivityState.GETTING_OUT_OF_CAR):
                vx.append(r.position[0])
                vy.append(r.position[1])
                vs.append(size ** 2)
                vc.append(color)
            else:
                px.append(r.position[0])
                py.append(r.position[1])
                ps.append(size ** 2)
                pc.append(color)

            # RF MAC fragment for visible entities
            prof = person_rf.get(r.resident_id)
            rf_level = state_rf_emission(r.activity_state)
            if prof and prof.has_phone and rf_level in ("full", "reduced"):
                mac_frag = prof.phone_mac[-5:]  # last 2 octets
                rf_mac_data.append((r.position[0] + 3, r.position[1] + 3, mac_frag))

        # Parked vehicles
        for v in sim.vehicles:
            if not v.driving and v.parked_at:
                parked_x.append(v.position[0])
                parked_y.append(v.position[1])

        # --- Update scatter plots ---
        if px:
            person_scatter.set_offsets(list(zip(px, py)))
            person_scatter.set_sizes(ps)
            person_scatter.set_color(pc)
        else:
            person_scatter.set_offsets([(0, 0)])
            person_scatter.set_sizes([0])

        if vx:
            vehicle_scatter.set_offsets(list(zip(vx, vy)))
            vehicle_scatter.set_sizes(vs)
            vehicle_scatter.set_color(vc)
        else:
            vehicle_scatter.set_offsets([(0, 0)])
            vehicle_scatter.set_sizes([0])

        if parked_x:
            parked_scatter.set_offsets(list(zip(parked_x, parked_y)))
            parked_scatter.set_sizes([8] * len(parked_x))
        else:
            parked_scatter.set_offsets([(0, 0)])
            parked_scatter.set_sizes([0])

        # --- RF text labels (clear old, draw new) ---
        for t in rf_texts:
            t.remove()
        rf_texts = []
        # Only show a sample to avoid clutter (max 15 labels)
        for i, (rx, ry, mac) in enumerate(rf_mac_data[:15]):
            t = ax_map.text(
                rx, ry, mac, fontsize=4, color=GREEN, alpha=0.5,
                fontfamily="monospace", zorder=6,
            )
            rf_texts.append(t)

        # --- Title with clock ---
        title_text.set_text(
            f"TRITIUM NEIGHBORHOOD SIM  \u2502  {_format_time(current_time[0])}"
        )

        # --- RF emission counts ---
        ble_count, wifi_count, tpms_count = count_rf_emissions(
            sim.residents, sim.vehicles, person_rf, vehicle_rf,
        )

        # --- Stats panel ---
        ax_stats.clear()
        ax_stats.set_facecolor(PANEL_BG)
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis("off")

        y = 0.96
        dy = 0.032

        # Clock
        ax_stats.text(
            0.5, y, _format_time(current_time[0]),
            fontsize=20, color=CYAN, ha="center", fontfamily="monospace",
            fontweight="bold",
        )
        y -= dy * 1.8

        # Day indicator
        day_progress = current_time[0] / 24.0
        ax_stats.add_patch(Rectangle(
            (0.05, y), 0.9, 0.012,
            facecolor=GRID_COLOR, edgecolor=TEXT_DIM, linewidth=0.5,
        ))
        ax_stats.add_patch(Rectangle(
            (0.05, y), 0.9 * day_progress, 0.012,
            facecolor=CYAN, alpha=0.6,
        ))
        # Sun/moon indicator
        sun_x = 0.05 + 0.9 * day_progress
        is_day = 6.0 <= current_time[0] <= 20.0
        ax_stats.plot(sun_x, y + 0.006, "o",
                      color=YELLOW if is_day else "#556688",
                      markersize=5, zorder=10)
        y -= dy * 1.5

        # Separator
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.5

        # Activity breakdown
        ax_stats.text(
            0.5, y, "ACTIVITY BREAKDOWN",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.2

        total_people = len(sim.residents)
        bar_max = total_people if total_people > 0 else 1

        for cat_name, (cat_color, states) in _CATEGORY_MAP.items():
            count = sum(activity_counts.get(s, 0) for s in states)
            bar_width = (count / bar_max) * 0.55

            # Label
            ax_stats.text(
                0.05, y, cat_name, fontsize=7, color=cat_color,
                fontfamily="monospace", va="center",
            )
            # Count
            ax_stats.text(
                0.95, y, str(count), fontsize=7, color=cat_color,
                fontfamily="monospace", va="center", ha="right",
            )
            # Bar background
            ax_stats.add_patch(Rectangle(
                (0.35, y - 0.008), 0.55, 0.016,
                facecolor=GRID_COLOR, linewidth=0,
            ))
            # Bar fill
            if bar_width > 0.001:
                ax_stats.add_patch(Rectangle(
                    (0.35, y - 0.008), bar_width, 0.016,
                    facecolor=cat_color, alpha=0.6, linewidth=0,
                ))
            y -= dy

        y -= dy * 0.5
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.8

        # Visibility stats
        ax_stats.text(
            0.5, y, "VISIBILITY",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.2

        stats = sim.get_statistics()
        vis_items = [
            ("Visible", stats["visible_on_map"], CYAN),
            ("Inside", stats["inside_buildings"], GREEN),
            ("Vehicles", stats["vehicles_driving"], YELLOW),
            ("Parked", stats["vehicles_parked"], DARK_GREY),
        ]
        for label, val, col in vis_items:
            ax_stats.text(
                0.08, y, label, fontsize=7, color=col,
                fontfamily="monospace", va="center",
            )
            ax_stats.text(
                0.92, y, str(val), fontsize=7, color=col,
                fontfamily="monospace", va="center", ha="right",
            )
            y -= dy

        y -= dy * 0.3
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.8

        # RF emissions
        ax_stats.text(
            0.5, y, "RF EMISSIONS",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.2

        rf_items = [
            ("BLE Ads", ble_count, CYAN),
            ("WiFi Probes", wifi_count, MAGENTA),
            ("TPMS", tpms_count, YELLOW),
            ("Total", ble_count + wifi_count + tpms_count, GREEN),
        ]
        for label, val, col in rf_items:
            ax_stats.text(
                0.08, y, label, fontsize=7, color=col,
                fontfamily="monospace", va="center",
            )
            ax_stats.text(
                0.92, y, str(val), fontsize=7, color=col,
                fontfamily="monospace", va="center", ha="right",
            )
            y -= dy

        y -= dy * 0.3
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.8

        # FPS and totals
        ax_stats.text(
            0.08, y, f"FPS: {fps_val[0]:.1f}", fontsize=7, color=TEXT_DIM,
            fontfamily="monospace", va="center",
        )
        ax_stats.text(
            0.92, y, f"Pop: {total_people}", fontsize=7, color=TEXT_DIM,
            fontfamily="monospace", va="center", ha="right",
        )
        y -= dy

        ax_stats.text(
            0.08, y, f"Buildings: {len(sim.buildings)}", fontsize=7,
            color=TEXT_DIM, fontfamily="monospace", va="center",
        )
        ax_stats.text(
            0.92, y, f"Vehicles: {len(sim.vehicles)}", fontsize=7,
            color=TEXT_DIM, fontfamily="monospace", va="center", ha="right",
        )
        y -= dy * 1.5

        # Legend
        ax_stats.text(
            0.5, y, "LEGEND",
            fontsize=7, color=TEXT_DIM, ha="center", fontfamily="monospace",
        )
        y -= dy
        legend_items = [
            ("o", CYAN, "Walking / On foot"),
            ("s", YELLOW, "Driving / Vehicle"),
            ("o", GREEN, "Working / Inside"),
            ("o", ORANGE, "Jogging / Outdoor"),
            ("o", MAGENTA, "Shopping / Errands"),
            ("o", PURPLE, "Walking dog"),
        ]
        for marker, color, desc in legend_items:
            ax_stats.plot(
                0.08, y, marker, color=color, markersize=4,
            )
            ax_stats.text(
                0.15, y, desc, fontsize=6, color=color,
                fontfamily="monospace", va="center",
            )
            y -= dy * 0.85

        return [person_scatter, vehicle_scatter, parked_scatter, title_text]

    # --- Animation ---
    anim = FuncAnimation(
        fig, _update, interval=100, blit=False, cache_frame_data=False,
    )

    plt.show()


# ---------------------------------------------------------------------------
# Terminal fallback
# ---------------------------------------------------------------------------

def run_terminal_demo(num_residents: int = NUM_RESIDENTS) -> None:
    """Run text-only demo if matplotlib not available."""
    sim = NeighborhoodSim(
        num_residents=num_residents,
        bounds=((0.0, 0.0), (WORLD_SIZE, WORLD_SIZE)),
        seed=42,
    )
    sim.populate()
    person_rf, vehicle_rf = attach_rf_profiles(sim.residents, sim.vehicles)

    current_time = START_HOUR
    print(f"\n{'='*60}")
    print("  TRITIUM NEIGHBORHOOD SIM — Terminal Mode")
    print(f"  {num_residents} residents | 10x speed | Ctrl+C to exit")
    print(f"{'='*60}\n")

    try:
        while True:
            for _ in range(TICKS_PER_FRAME):
                sim.tick(SIM_DT, current_time)
                current_time += (SIM_DT * TIME_SCALE) / TICKS_PER_FRAME
                if current_time >= 24.0:
                    current_time -= 24.0

            stats = sim.get_statistics()
            ble, wifi, tpms = count_rf_emissions(
                sim.residents, sim.vehicles, person_rf, vehicle_rf,
            )

            hours = int(current_time) % 24
            minutes = int((current_time % 1.0) * 60)
            ampm = "AM" if hours < 12 else "PM"
            dh = hours % 12 or 12

            # Build activity line
            act_parts = []
            for cat_name, (_, states) in _CATEGORY_MAP.items():
                count = sum(stats["activity_states"].get(s, 0) for s in states)
                if count > 0:
                    act_parts.append(f"{cat_name}:{count}")

            print(
                f"  [{dh:2d}:{minutes:02d}{ampm}] "
                f"Vis:{stats['visible_on_map']:2d} "
                f"In:{stats['inside_buildings']:2d} "
                f"Drive:{stats['vehicles_driving']:2d} "
                f"Park:{stats['vehicles_parked']:2d} "
                f"| BLE:{ble:2d} WiFi:{wifi:2d} TPMS:{tpms:2d} "
                f"| {' '.join(act_parts)}"
            )
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRITIUM Full City Simulation Demo"
    )
    parser.add_argument(
        "-n", "--residents", type=int, default=NUM_RESIDENTS,
        help=f"Number of residents (default {NUM_RESIDENTS})",
    )
    parser.add_argument(
        "--terminal", action="store_true",
        help="Force terminal-only output (no matplotlib)",
    )
    args = parser.parse_args()

    if args.terminal:
        run_terminal_demo(args.residents)
    else:
        run_visual_demo(args.residents)


if __name__ == "__main__":
    main()
