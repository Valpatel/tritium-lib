# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""RF signature visualization demo.

Run: python3 -m tritium_lib.sim_engine.ai.demos.demo_rf

Shows a neighborhood where every person/car emits BLE/WiFi/TPMS.
Demonstrates MAC rotation and persistent TPMS IDs.

Falls back to ASCII terminal output if matplotlib unavailable.
"""
from __future__ import annotations

import argparse
import math
import os
import random
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
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from tritium_lib.sim_engine.ai.city_sim import NeighborhoodSim, BuildingType
from tritium_lib.sim_engine.ai.rf_signatures import (
    RFSignatureGenerator,
    PersonRFProfile,
    VehicleRFProfile,
    BuildingRFProfile,
    MAC_ROTATION_INTERVAL_S,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE = 400.0
NUM_RESIDENTS = 30  # Fewer for readability with RF labels
DT = 1.0
TIME_SCALE = 10.0 / 60.0  # hours per real second
START_HOUR = 8.0  # 8 AM -- busy time
MAC_ROTATION_SIM_S = 90.0  # Rotate MAC every 90 sim-seconds for demo visibility


# ---------------------------------------------------------------------------
# RF-augmented city simulation
# ---------------------------------------------------------------------------

class RFCity:
    """City sim with RF profiles attached to every entity."""

    def __init__(self, seed: int = 42):
        self.sim = NeighborhoodSim(
            num_residents=NUM_RESIDENTS,
            bounds=((0, 0), (WORLD_SIZE, WORLD_SIZE)),
            seed=seed,
        )
        self.sim.populate()

        self.rng = random.Random(seed)
        self.person_rf: dict[str, PersonRFProfile] = {}
        self.vehicle_rf: dict[str, VehicleRFProfile] = {}
        self.building_rf: dict[str, BuildingRFProfile] = {}

        # Generate RF profiles
        for r in self.sim.residents:
            self.person_rf[r.resident_id] = RFSignatureGenerator.random_person(rng=self.rng)

        for v in self.sim.vehicles:
            self.vehicle_rf[v.vehicle_id] = RFSignatureGenerator.random_vehicle(rng=self.rng)

        for b in self.sim.buildings:
            btype = "commercial" if b.building_type in (BuildingType.OFFICE, BuildingType.RESTAURANT) else "residential"
            self.building_rf[b.building_id] = RFSignatureGenerator.random_building(btype, rng=self.rng)

        self.sim_time_s = 0.0
        self.mac_rotation_count = 0
        self.last_rotation_s = 0.0

    def tick(self, dt: float, current_hour: float) -> None:
        self.sim.tick(dt, current_hour)
        self.sim_time_s += dt

        # MAC rotation check (use sim time, not real time, for demo speed)
        if self.sim_time_s - self.last_rotation_s >= MAC_ROTATION_SIM_S:
            for rf in self.person_rf.values():
                rf.rotate_mac()
            self.mac_rotation_count += 1
            self.last_rotation_s = self.sim_time_s

    def get_all_rf_emissions(self) -> dict:
        """Collect all current RF emissions from the simulation."""
        result = {
            "ble_ads": [],
            "wifi_probes": [],
            "tpms": [],
            "wifi_beacons": [],
        }

        for r in self.sim.residents:
            rf = self.person_rf.get(r.resident_id)
            if not rf:
                continue
            pos = r.position
            result["ble_ads"].extend(rf.emit_ble_advertisements(pos))
            result["wifi_probes"].extend(rf.emit_wifi_probes(pos))

        for v in self.sim.vehicles:
            rf = self.vehicle_rf.get(v.vehicle_id)
            if not rf:
                continue
            pos = v.position
            result["tpms"].extend(rf.emit_tpms(pos))

        return result

    def get_rf_summary(self) -> dict:
        """Summary stats about RF emissions."""
        emissions = self.get_all_rf_emissions()
        unique_macs = set()
        unique_tpms = set()
        ssids_seen = set()

        for ad in emissions["ble_ads"]:
            unique_macs.add(ad["mac"])
        for probe in emissions["wifi_probes"]:
            unique_macs.add(probe["mac"])
            ssids_seen.add(probe["ssid"])
        for tpms in emissions["tpms"]:
            unique_tpms.add(tpms["metadata"]["device_id"])

        return {
            "ble_advertisements": len(emissions["ble_ads"]),
            "wifi_probes": len(emissions["wifi_probes"]),
            "tpms_readings": len(emissions["tpms"]),
            "unique_macs": len(unique_macs),
            "unique_tpms_ids": len(unique_tpms),
            "unique_ssids_probed": len(ssids_seen),
            "mac_rotations": self.mac_rotation_count,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hour_to_string(hour: float) -> str:
    h = int(hour) % 24
    m = int((hour % 1.0) * 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12:2d}:{m:02d} {ampm}"


def short_mac(mac: str) -> str:
    """Show last 4 chars of MAC for compact display."""
    return mac[-5:] if len(mac) >= 5 else mac


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------

def run_matplotlib(duration: float) -> None:
    """Run RF demo with matplotlib animation."""
    city = RFCity(seed=42)

    fig, (ax_map, ax_info) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("Tritium RF Signature Visualization", color="#00f0ff", fontsize=16)

    # Map
    ax_map.set_facecolor("#0e0e14")
    ax_map.set_xlim(0, WORLD_SIZE)
    ax_map.set_ylim(0, WORLD_SIZE)
    ax_map.set_aspect("equal")
    ax_map.set_title("RF Emissions Map", color="#aaaaaa", fontsize=12)
    ax_map.tick_params(colors="#555555")
    for spine in ax_map.spines.values():
        spine.set_color("#333333")

    # Draw buildings (static)
    for b in city.sim.buildings:
        sz = 8
        rect = Rectangle(
            (b.position[0] - sz / 2, b.position[1] - sz / 2),
            sz, sz,
            facecolor="#1a1a2e", edgecolor="#333333", linewidth=0.5,
            alpha=0.5, zorder=1,
        )
        ax_map.add_patch(rect)

    # People scatter (BLE emitters)
    people_scatter = ax_map.scatter([], [], s=30, c="#00f0ff", zorder=3, label="People (BLE)")
    # Vehicle scatter (TPMS emitters)
    vehicle_scatter = ax_map.scatter([], [], s=60, marker="s", c="#fcee0a",
                                     edgecolors="#ff8c00", linewidth=0.5,
                                     zorder=3, label="Vehicles (TPMS)")

    # RF label annotations (will be updated each frame)
    rf_labels: list = []

    # Info panel
    ax_info.set_facecolor("#0e0e14")
    ax_info.set_title("RF Statistics", color="#aaaaaa", fontsize=12)
    ax_info.axis("off")

    clock_text = ax_map.text(
        10, WORLD_SIZE - 12, "", color="#fcee0a", fontsize=13,
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0f", edgecolor="#333333"),
        zorder=10,
    )

    sim_hour = [START_HOUR]
    frame_count = [0]
    t_start = [time.time()]
    t_end = time.time() + duration

    # Show labels for a subset of entities to avoid clutter
    LABEL_LIMIT = 12

    def update(frame_num):
        now = time.time()
        if now >= t_end:
            plt.close(fig)
            return ()

        # Advance time
        real_dt = 0.05
        sim_hour[0] += real_dt * TIME_SCALE
        if sim_hour[0] >= 24.0:
            sim_hour[0] -= 24.0

        for _ in range(3):
            city.tick(DT, sim_hour[0])

        frame_count[0] += 1
        elapsed = now - t_start[0]
        fps = frame_count[0] / max(elapsed, 0.001)

        # Clear old RF labels
        for label in rf_labels:
            label.remove()
        rf_labels.clear()

        # People positions and RF labels
        px, py = [], []
        label_idx = 0
        for r in city.sim.residents:
            px.append(r.position[0])
            py.append(r.position[1])

            if label_idx < LABEL_LIMIT:
                rf = city.person_rf.get(r.resident_id)
                if rf and rf.has_phone:
                    label_text = f"BLE:{short_mac(rf.phone_mac)}"
                    if rf.has_smartwatch:
                        label_text += f"\nWatch:{short_mac(rf.watch_mac)}"
                    lbl = ax_map.text(
                        r.position[0] + 5, r.position[1] + 5,
                        label_text, color="#00f0ff", fontsize=5,
                        fontfamily="monospace", alpha=0.8, zorder=5,
                    )
                    rf_labels.append(lbl)
                    label_idx += 1

        if px:
            people_scatter.set_offsets(list(zip(px, py)))

        # Vehicle positions and TPMS labels
        vx, vy = [], []
        label_idx = 0
        for v in city.sim.vehicles:
            vx.append(v.position[0])
            vy.append(v.position[1])

            if label_idx < LABEL_LIMIT // 2:
                rf = city.vehicle_rf.get(v.vehicle_id)
                if rf:
                    tpms_str = ",".join(tid[:4] for tid in rf.tpms_ids[:2])
                    label_text = f"TPMS:{tpms_str}"
                    if rf.license_plate:
                        label_text += f"\n{rf.license_plate}"
                    lbl = ax_map.text(
                        v.position[0] + 5, v.position[1] - 8,
                        label_text, color="#fcee0a", fontsize=5,
                        fontfamily="monospace", alpha=0.8, zorder=5,
                    )
                    rf_labels.append(lbl)
                    label_idx += 1

        if vx:
            vehicle_scatter.set_offsets(list(zip(vx, vy)))

        # Clock
        clock_text.set_text(f"  {hour_to_string(sim_hour[0])}  ")

        # Update info panel
        ax_info.clear()
        ax_info.set_facecolor("#0e0e14")
        ax_info.axis("off")

        summary = city.get_rf_summary()
        info_lines = [
            ("BLE Advertisements", f"{summary['ble_advertisements']}", "#00f0ff"),
            ("WiFi Probes", f"{summary['wifi_probes']}", "#05ffa1"),
            ("TPMS Readings", f"{summary['tpms_readings']}", "#fcee0a"),
            ("", "", ""),
            ("Unique MACs (now)", f"{summary['unique_macs']}", "#ff2a6d"),
            ("Unique TPMS IDs", f"{summary['unique_tpms_ids']}", "#fcee0a"),
            ("SSIDs Probed", f"{summary['unique_ssids_probed']}", "#05ffa1"),
            ("", "", ""),
            ("MAC Rotations", f"{summary['mac_rotations']}", "#ff2a6d"),
            ("", "", ""),
            ("FPS", f"{fps:.0f}", "#888888"),
            ("Elapsed", f"{elapsed:.0f}s / {duration:.0f}s", "#888888"),
        ]

        y_pos = 0.95
        for label, value, color in info_lines:
            if not label:
                y_pos -= 0.03
                continue
            ax_info.text(0.05, y_pos, label, transform=ax_info.transAxes,
                        color="#888888", fontsize=10, fontfamily="monospace",
                        verticalalignment="top")
            ax_info.text(0.95, y_pos, value, transform=ax_info.transAxes,
                        color=color, fontsize=10, fontfamily="monospace",
                        verticalalignment="top", horizontalalignment="right")
            y_pos -= 0.06

        # MAC rotation warning
        if summary['mac_rotations'] > 0:
            ax_info.text(
                0.05, 0.08,
                f"MACs rotated {summary['mac_rotations']}x\n"
                f"TPMS IDs: NEVER change\n"
                f"Company IDs: persist",
                transform=ax_info.transAxes,
                color="#ff2a6d", fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0a0f",
                          edgecolor="#ff2a6d", alpha=0.8),
            )

        return ()

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    ax_map.legend(loc="lower right", facecolor="#12121a", edgecolor="#333333",
                  labelcolor="#cccccc", fontsize=9)
    plt.tight_layout()
    plt.show()

    elapsed = time.time() - t_start[0]
    summary = city.get_rf_summary()
    print(f"\nRF demo ran for {elapsed:.1f}s")
    print(f"MAC rotations: {summary['mac_rotations']}")
    print(f"Final unique MACs: {summary['unique_macs']}, TPMS IDs: {summary['unique_tpms_ids']}")


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------

def run_terminal(duration: float) -> None:
    """Run RF demo with ASCII terminal output."""
    city = RFCity(seed=42)

    sim_hour = START_HOUR
    frame = 0
    t_start = time.time()
    t_end = t_start + duration
    cols, rows = 60, 20

    try:
        while time.time() < t_end:
            real_dt = 0.2
            sim_hour += real_dt * TIME_SCALE
            if sim_hour >= 24.0:
                sim_hour -= 24.0

            for _ in range(3):
                city.tick(DT, sim_hour)

            frame += 1
            elapsed = time.time() - t_start
            fps = frame / max(elapsed, 0.001)

            if frame % 5 == 0:
                grid = [["." for _ in range(cols)] for _ in range(rows)]

                # Draw people with MAC hints
                for r in city.sim.residents:
                    gc = int(r.position[0] / WORLD_SIZE * cols) % cols
                    gr = int(r.position[1] / WORLD_SIZE * rows) % rows
                    rf = city.person_rf.get(r.resident_id)
                    if rf and rf.has_phone:
                        grid[gr][gc] = "@"  # person with phone
                    else:
                        grid[gr][gc] = "o"  # person without phone

                # Draw vehicles
                for v in city.sim.vehicles:
                    gc = int(v.position[0] / WORLD_SIZE * cols) % cols
                    gr = int(v.position[1] / WORLD_SIZE * rows) % rows
                    grid[gr][gc] = "V"

                summary = city.get_rf_summary()

                lines = []
                lines.append(f"  {hour_to_string(sim_hour)}  |  FPS: {fps:.0f}  |  "
                             f"MAC rotations: {summary['mac_rotations']}")
                lines.append("-" * cols)
                for row in grid:
                    lines.append("".join(row))
                lines.append("-" * cols)
                lines.append(f"  BLE ads: {summary['ble_advertisements']:3d}  |  "
                             f"WiFi probes: {summary['wifi_probes']:3d}  |  "
                             f"TPMS: {summary['tpms_readings']:3d}")
                lines.append(f"  Unique MACs: {summary['unique_macs']:3d}  |  "
                             f"TPMS IDs: {summary['unique_tpms_ids']:3d}  |  "
                             f"SSIDs: {summary['unique_ssids_probed']:3d}")
                lines.append("")

                # Show a few example RF profiles
                lines.append("  Example RF signatures (first 3 people):")
                for i, r in enumerate(city.sim.residents[:3]):
                    rf = city.person_rf.get(r.resident_id)
                    if rf and rf.has_phone:
                        lines.append(f"    {r.name}: MAC={rf.phone_mac} "
                                     f"company=0x{rf.phone_ble_company_id:04X} "
                                     f"eco={rf.phone_ecosystem}")
                        if rf.phone_wifi_probes:
                            lines.append(f"      probes: {', '.join(rf.phone_wifi_probes[:3])}")

                lines.append("")
                lines.append("  Example TPMS (first 2 vehicles):")
                for i, v in enumerate(city.sim.vehicles[:2]):
                    rf = city.vehicle_rf.get(v.vehicle_id)
                    if rf:
                        tids = ", ".join(rf.tpms_ids)
                        lines.append(f"    {rf.make_model} [{rf.license_plate}]: "
                                     f"TPMS={tids}")

                sys.stdout.write("\033[H\033[J")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            time.sleep(max(0, 0.2 - (time.time() - t_start - (frame - 1) * 0.2)))
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    summary = city.get_rf_summary()
    print(f"\nRF demo ran for {elapsed:.1f}s")
    print(f"MAC rotations: {summary['mac_rotations']}")
    print(f"Final unique MACs: {summary['unique_macs']}, TPMS IDs: {summary['unique_tpms_ids']}")


# ---------------------------------------------------------------------------
# Headless
# ---------------------------------------------------------------------------

def run_headless(duration: float) -> None:
    """Run RF simulation headless, print summary."""
    city = RFCity(seed=42)

    sim_hour = START_HOUR
    frame = 0
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        sim_hour += 0.05 * TIME_SCALE
        if sim_hour >= 24.0:
            sim_hour -= 24.0
        city.tick(DT, sim_hour)
        frame += 1

    elapsed = time.time() - t_start
    fps = frame / max(elapsed, 0.001)
    summary = city.get_rf_summary()

    print("=" * 60)
    print("  Tritium RF Signature Demo -- Headless Results")
    print("=" * 60)
    print(f"  Residents:         {NUM_RESIDENTS}")
    print(f"  Vehicles:          {len(city.sim.vehicles)}")
    print(f"  Buildings:         {len(city.sim.buildings)}")
    print(f"  Duration:          {elapsed:.1f}s")
    print(f"  Frames:            {frame}")
    print(f"  FPS:               {fps:.1f}")
    print(f"  Tick time:         {elapsed / max(frame, 1) * 1000:.2f} ms")
    print()
    print(f"  BLE advertisements: {summary['ble_advertisements']}")
    print(f"  WiFi probes:        {summary['wifi_probes']}")
    print(f"  TPMS readings:      {summary['tpms_readings']}")
    print(f"  Unique MACs (now):  {summary['unique_macs']}")
    print(f"  Unique TPMS IDs:    {summary['unique_tpms_ids']} (PERSISTENT)")
    print(f"  Unique SSIDs:       {summary['unique_ssids_probed']}")
    print(f"  MAC rotations:      {summary['mac_rotations']}")
    print()

    # Show company ID persistence
    print("  MAC rotation demo (company IDs persist across rotations):")
    for i, r in enumerate(city.sim.residents[:5]):
        rf = city.person_rf.get(r.resident_id)
        if rf and rf.has_phone:
            print(f"    {r.name}: MAC={rf.phone_mac} "
                  f"company=0x{rf.phone_ble_company_id:04X} "
                  f"eco={rf.phone_ecosystem}")

    print()
    print("  TPMS persistence (IDs never change):")
    for v in city.sim.vehicles[:3]:
        rf = city.vehicle_rf.get(v.vehicle_id)
        if rf:
            print(f"    {rf.make_model} [{rf.license_plate}]: "
                  f"TPMS={','.join(rf.tpms_ids)}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tritium RF signature visualization demo")
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
