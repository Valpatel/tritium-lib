# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GTA-style full city simulation — real streets, buildings, traffic, pedestrians.

Run: python3 -m tritium_lib.game_ai.demos.demo_full

A proper small town seen from above:
- Grid of named streets with main roads and residential streets
- Intersections with stop signs where vehicles pause
- Buildings placed ALONG streets (houses on residential, shops on main)
- Cars driving ON roads using RoadNetwork pathfinding
- Pedestrians walking on sidewalks offset from roads
- Lane markings, sidewalks, trees, parks, driveways
- RF emissions (BLE / WiFi / TPMS) from every entity
- Stats panel with clock, counts, activity breakdown, FPS

10x accelerated time: 1 real second = 10 sim minutes.
Cyberpunk color scheme matching Tritium.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import (
        Rectangle, FancyBboxPatch, Circle, RegularPolygon, Polygon,
    )
    from matplotlib.collections import LineCollection, PatchCollection
    import matplotlib.patheffects as pe
    import numpy as np
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
from tritium_lib.game_ai.pathfinding import RoadNetwork
from tritium_lib.game_ai.rf_signatures import (
    PersonRFProfile,
    RFSignatureGenerator,
    VehicleRFProfile,
)
from tritium_lib.game_ai.steering import Vec2


# ---------------------------------------------------------------------------
# Constants — cyberpunk palette
# ---------------------------------------------------------------------------

WORLD_W = 720.0
WORLD_H = 520.0
NUM_RESIDENTS = 50
TIME_SCALE = 10.0 / 60.0  # 10 sim minutes per real second (hours/sec)
START_HOUR = 6.0
SIM_DT = 1.0
TICKS_PER_FRAME = 6

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
TEXT_DIM = "#556677"
TEXT_BRIGHT = "#aaccdd"

# Road / city colors
ROAD_MAIN = "#2e2e3a"
ROAD_RESIDENTIAL = "#222230"
SIDEWALK_COLOR = "#28283a"
LANE_MARKING = "#555530"
GRASS_COLOR = "#0a1a0a"
PARK_FILL = "#0c260c"
BUILDING_OUTLINE = "#334455"
STOP_SIGN_COLOR = "#cc2222"

# Building type -> (fill, outline, label, width, height)
_BLDG_STYLE: dict[str, tuple[str, str, str, float, float]] = {
    "home":        ("#1a2a3a", "#2a4a5a", "H", 14, 12),
    "office":      ("#1a3a2a", "#2a5a3a", "O", 28, 22),
    "school":      ("#2a2a4a", "#3a3a6a", "S", 35, 28),
    "grocery":     ("#3a2a1a", "#5a3a2a", "G", 30, 20),
    "park":        (PARK_FILL, "#1a4a1a", "P", 50, 40),
    "restaurant":  ("#3a1a2a", "#5a2a3a", "R", 18, 14),
    "gas_station": ("#3a3a1a", "#5a5a2a", "F", 22, 16),
    "coffee_shop": ("#2a1a1a", "#4a2a2a", "C", 14, 12),
    "doctor":      ("#1a1a3a", "#2a2a5a", "D", 20, 16),
}

# Activity state -> (color, marker, size, label)
_ACTIVITY_STYLE: dict[str, tuple[str, str, float, str]] = {
    ActivityState.WALKING:             (CYAN,    "o", 4.0, "Walking"),
    ActivityState.WALKING_TO_CAR:      (CYAN,    "o", 3.5, "Walking"),
    ActivityState.WALKING_TO_BUILDING: (CYAN,    "o", 3.5, "Walking"),
    ActivityState.ENTERING_BUILDING:   (CYAN,    "o", 3.0, "Walking"),
    ActivityState.EXITING_BUILDING:    (CYAN,    "o", 3.0, "Walking"),
    ActivityState.RETURNING_HOME:      (CYAN,    "o", 4.0, "Walking"),
    ActivityState.WALKING_TO_TRANSIT:  (CYAN,    "o", 3.5, "Walking"),
    ActivityState.LUNCH_BREAK:         (CYAN,    "o", 3.5, "Walking"),
    ActivityState.CHECKING_PHONE:      (CYAN,    "o", 3.5, "Phone"),
    ActivityState.SOCIALIZING:         (CYAN,    "o", 4.0, "Social"),
    ActivityState.DRIVING:             (YELLOW,  "s", 6.0, "Driving"),
    ActivityState.GETTING_IN_CAR:      (YELLOW,  "o", 3.5, "In car"),
    ActivityState.GETTING_OUT_OF_CAR:  (YELLOW,  "o", 3.5, "Out car"),
    ActivityState.PARKING:             (YELLOW,  "s", 5.0, "Parking"),
    ActivityState.DELIVERING:          (YELLOW,  "s", 6.0, "Delivering"),
    ActivityState.DELIVERY_STOP:       (ORANGE,  "o", 4.0, "Delivery"),
    ActivityState.WORKING:             (GREEN,   "o", 2.5, "Working"),
    ActivityState.AT_SCHOOL:           (GREEN,   "o", 2.5, "At school"),
    ActivityState.INSIDE_BUILDING:     (GREEN,   "o", 2.0, "Inside"),
    ActivityState.JOGGING:             (ORANGE,  "o", 4.5, "Jogging"),
    ActivityState.WALKING_DOG:         (PURPLE,  "o", 4.0, "Dog walk"),
    ActivityState.PLAYING:             (GREEN,   "o", 4.0, "Playing"),
    ActivityState.GARDENING:           (GREEN,   "o", 3.5, "Gardening"),
    ActivityState.SHOPPING:            (MAGENTA, "o", 4.0, "Shopping"),
    ActivityState.DINING:              (MAGENTA, "o", 3.5, "Dining"),
    ActivityState.AT_GAS_STATION:      (MAGENTA, "o", 3.5, "Gas stn"),
    ActivityState.AT_DOCTOR:           (MAGENTA, "o", 3.0, "Doctor"),
    ActivityState.GETTING_COFFEE:      (MAGENTA, "o", 3.5, "Coffee"),
    ActivityState.SLEEPING:            (DARK_GREY, "o", 0.0, "Sleeping"),
    ActivityState.NAPPING:             (DARK_GREY, "o", 0.0, "Napping"),
    ActivityState.WAKING_UP:           (DARK_GREY, "o", 0.0, "Waking"),
    ActivityState.RELAXING:            (DARK_GREY, "o", 0.0, "Relaxing"),
}

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
# Street grid — the town layout
# ---------------------------------------------------------------------------

# Street names
_EW_STREET_NAMES = [
    "Oak St", "Main St", "Elm St", "Cedar Ave", "Maple Dr",
    "Pine St", "Birch Ln",
]
_NS_STREET_NAMES = [
    "1st Ave", "2nd Ave", "3rd Ave", "4th Ave", "5th Ave",
    "6th Ave", "7th Ave", "8th Ave", "9th Ave",
]


@dataclass
class StreetSegment:
    """A road segment between two intersections."""
    start: Vec2
    end: Vec2
    name: str
    is_main: bool  # main road = wider, faster
    speed_limit: float  # m/s

    @property
    def width(self) -> float:
        return 10.0 if self.is_main else 6.0


@dataclass
class Intersection:
    """Where two or more streets meet."""
    position: Vec2
    has_stop_sign: bool = True


@dataclass
class PlacedBuilding:
    """A building placed along a street with orientation."""
    building: Building
    rect_pos: Vec2      # lower-left of visual rectangle
    width: float
    height: float
    angle: float        # rotation in degrees
    fill_color: str
    outline_color: str
    label: str
    driveway_start: Vec2
    driveway_end: Vec2
    parking_spots: list[Vec2]


@dataclass
class TreeCluster:
    """Decorative trees for parks and yards."""
    position: Vec2
    radius: float = 3.0


def _build_street_grid(rng: random.Random) -> tuple[
    list[StreetSegment],
    list[Intersection],
    RoadNetwork,
    list[Vec2],  # intersection positions for building placement
]:
    """Create a realistic small-town street grid.

    Layout:
    - 7 east-west streets spaced ~70m apart
    - 9 north-south streets spaced ~75m apart
    - Main St and 5th Ave are wider main roads
    - All other streets are narrower residential
    """
    streets: list[StreetSegment] = []
    intersections: list[Intersection] = []
    road_net = RoadNetwork()
    intersection_positions: list[Vec2] = []

    margin_x = 40.0
    margin_y = 30.0
    spacing_ew = 70.0  # east-west street vertical spacing
    spacing_ns = 75.0  # north-south street horizontal spacing

    num_ew = len(_EW_STREET_NAMES)  # 7
    num_ns = len(_NS_STREET_NAMES)  # 9

    # Calculate intersection grid positions
    ew_ys = [margin_y + i * spacing_ew for i in range(num_ew)]
    ns_xs = [margin_x + j * spacing_ns for j in range(num_ns)]

    # Which streets are main roads
    main_ew = {1}  # Main St (index 1)
    main_ns = {4}  # 5th Ave (index 4)

    # Build intersection positions
    for i, y in enumerate(ew_ys):
        for j, x in enumerate(ns_xs):
            pos = (x, y)
            intersection_positions.append(pos)
            intersections.append(Intersection(position=pos, has_stop_sign=True))

    # East-West streets (horizontal)
    for i, y in enumerate(ew_ys):
        is_main = i in main_ew
        speed = 15.6 if is_main else 11.2  # ~35 mph vs ~25 mph
        name = _EW_STREET_NAMES[i]
        for j in range(num_ns - 1):
            x0 = ns_xs[j]
            x1 = ns_xs[j + 1]
            seg = StreetSegment(
                start=(x0, y), end=(x1, y),
                name=name, is_main=is_main, speed_limit=speed,
            )
            streets.append(seg)
            road_net.add_road((x0, y), (x1, y), speed_limit=speed)

    # North-South streets (vertical)
    for j, x in enumerate(ns_xs):
        is_main = j in main_ns
        speed = 15.6 if is_main else 11.2
        name = _NS_STREET_NAMES[j]
        for i in range(num_ew - 1):
            y0 = ew_ys[i]
            y1 = ew_ys[i + 1]
            seg = StreetSegment(
                start=(x, y0), end=(x, y1),
                name=name, is_main=is_main, speed_limit=speed,
            )
            streets.append(seg)
            road_net.add_road((x, y0), (x, y1), speed_limit=speed)

    return streets, intersections, road_net, intersection_positions


def _place_buildings_along_streets(
    streets: list[StreetSegment],
    intersection_positions: list[Vec2],
    rng: random.Random,
) -> tuple[list[PlacedBuilding], list[Building], list[TreeCluster]]:
    """Place buildings along streets like a real town.

    Rules:
    - Main roads get commercial buildings (offices, shops, restaurants)
    - Residential streets get houses
    - Buildings are set back from the road edge
    - Each building has a driveway connecting to the nearest road
    - Parks get trees
    """
    placed: list[PlacedBuilding] = []
    sim_buildings: list[Building] = []
    trees: list[TreeCluster] = []
    occupied_rects: list[tuple[float, float, float, float]] = []  # x, y, w, h

    def _overlaps(x: float, y: float, w: float, h: float) -> bool:
        for ox, oy, ow, oh in occupied_rects:
            if not (x + w < ox or ox + ow < x or y + h < oy or oy + oh < y):
                return True
        return False

    def _too_close_to_intersection(x: float, y: float, min_dist: float = 18.0) -> bool:
        for ix, iy in intersection_positions:
            if math.hypot(x - ix, y - iy) < min_dist:
                return True
        return False

    bid_counter = [0]

    def _make_building(
        btype: str, pos: Vec2, road_point: Vec2, side: str, seg: StreetSegment,
    ) -> PlacedBuilding | None:
        style = _BLDG_STYLE.get(btype, ("#1a1a2a", "#334455", "?", 14, 12))
        fill, outline, label, bw, bh = style

        # Determine setback from road
        setback = 14.0 if seg.is_main else 10.0
        if btype == "park":
            setback = 5.0

        # Calculate building position (offset from road)
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1.0:
            return None

        # Normal vector (perpendicular to road)
        nx, ny = -dy / seg_len, dx / seg_len
        if side == "south" or side == "west":
            nx, ny = -nx, -ny

        bx = pos[0] + nx * (setback + bw / 2)
        by = pos[1] + ny * (setback + bh / 2)

        # Check bounds
        if bx < 10 or bx > WORLD_W - 10 or by < 10 or by > WORLD_H - 10:
            return None

        # Check overlap
        if _overlaps(bx - bw / 2, by - bh / 2, bw, bh):
            return None
        if _too_close_to_intersection(pos[0], pos[1]):
            return None

        occupied_rects.append((bx - bw / 2 - 2, by - bh / 2 - 2, bw + 4, bh + 4))

        # Driveway from building to road
        driveway_start = (bx, by)
        driveway_end = road_point

        # Parking spot near building
        parking = (bx + rng.uniform(-bw / 3, bw / 3),
                   by + rng.uniform(-bh / 3, bh / 3))

        bid_counter[0] += 1
        sim_b = Building(
            building_id=f"b{bid_counter[0]:03d}",
            building_type=btype,
            position=(bx, by),
            name=f"{label}{bid_counter[0]}",
            parking_pos=parking,
        )
        sim_buildings.append(sim_b)

        pb = PlacedBuilding(
            building=sim_b,
            rect_pos=(bx - bw / 2, by - bh / 2),
            width=bw, height=bh, angle=0.0,
            fill_color=fill, outline_color=outline, label=label,
            driveway_start=driveway_start, driveway_end=driveway_end,
            parking_spots=[parking],
        )

        # Add trees around houses and parks
        if btype == "home":
            for _ in range(rng.randint(1, 3)):
                tx = bx + rng.uniform(-bw / 2 - 5, bw / 2 + 5)
                ty = by + rng.uniform(-bh / 2 - 5, bh / 2 + 5)
                trees.append(TreeCluster((tx, ty), radius=rng.uniform(2.0, 4.0)))
        elif btype == "park":
            for _ in range(rng.randint(8, 15)):
                tx = bx + rng.uniform(-bw / 2, bw / 2)
                ty = by + rng.uniform(-bh / 2, bh / 2)
                trees.append(TreeCluster((tx, ty), radius=rng.uniform(2.5, 5.0)))

        return pb

    # Place buildings along each street segment
    for seg in streets:
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 30:
            continue

        is_horizontal = abs(dx) > abs(dy)

        # Decide building types based on road type
        if seg.is_main:
            type_pool = ["office", "restaurant", "grocery", "coffee_shop",
                         "gas_station", "doctor"]
        else:
            type_pool = ["home", "home", "home", "home", "home"]

        # Place buildings on both sides of the street
        building_spacing = 28.0 if seg.is_main else 22.0
        num_slots = max(1, int((seg_len - 20) / building_spacing))

        for side_idx, side in enumerate(["north", "south"] if is_horizontal
                                         else ["east", "west"]):
            for slot in range(num_slots):
                # Skip some slots randomly for variety
                if rng.random() < 0.25:
                    continue

                t = (slot + 0.5 + rng.uniform(-0.15, 0.15)) / max(1, num_slots)
                t = max(0.15, min(0.85, t))  # keep away from intersections
                road_x = seg.start[0] + dx * t
                road_y = seg.start[1] + dy * t

                btype = rng.choice(type_pool)

                # Occasionally place a park instead of a house
                if btype == "home" and rng.random() < 0.08:
                    btype = "park"

                s = "south" if side_idx == 1 else "north"
                if not is_horizontal:
                    s = "west" if side_idx == 1 else "east"

                pb = _make_building(
                    btype, (road_x, road_y), (road_x, road_y), s, seg,
                )
                if pb:
                    placed.append(pb)

    return placed, sim_buildings, trees


# ---------------------------------------------------------------------------
# RF signature attachment (same as before)
# ---------------------------------------------------------------------------

def attach_rf_profiles(
    residents: list[Resident],
    vehicles: list[SimVehicle],
) -> tuple[dict[str, PersonRFProfile], dict[str, VehicleRFProfile]]:
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
        if prof.has_phone:
            ble_count += 1
        if prof.has_smartwatch:
            ble_count += 1
        if prof.has_earbuds:
            ble_count += 1
        if rf_level == "full" and prof.has_phone and prof.phone_wifi_probes:
            wifi_count += 1
    for v in vehicles:
        prof = vehicle_rf.get(v.vehicle_id)
        if prof is None:
            continue
        if v.driving:
            tpms_count += 4
            if prof.has_keyfob:
                ble_count += 1
            if prof.has_dashcam_wifi:
                wifi_count += 1
    return ble_count, wifi_count, tpms_count


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------

def run_visual_demo(num_residents: int = NUM_RESIDENTS) -> None:
    """Run the full city demo with real streets, buildings, and traffic."""
    if not HAS_MPL:
        print("matplotlib with TkAgg backend required. Install: pip install matplotlib")
        print("Falling back to terminal output...")
        run_terminal_demo(num_residents)
        return

    rng = random.Random(42)

    # --- Build the town ---
    streets, intersections, road_net, isect_positions = _build_street_grid(rng)
    placed_buildings, sim_buildings, trees = _place_buildings_along_streets(
        streets, isect_positions, rng,
    )

    # --- Create simulation using the placed buildings ---
    sim = NeighborhoodSim(
        num_residents=num_residents,
        bounds=((0.0, 0.0), (WORLD_W, WORLD_H)),
        seed=42,
    )

    # Pre-set our street-placed buildings and index them
    sim.buildings = list(sim_buildings)
    sim._buildings_by_type = {}
    for b in sim_buildings:
        sim._buildings_by_type.setdefault(b.building_type, []).append(b)

    # Ensure minimum building counts for each required type
    _ensure_buildings = [
        (BuildingType.HOME, 20, "House"),
        (BuildingType.OFFICE, 3, "Office"),
        (BuildingType.SCHOOL, 1, "School"),
        (BuildingType.GROCERY, 1, "Grocery"),
        (BuildingType.RESTAURANT, 2, "Restaurant"),
        (BuildingType.PARK, 1, "Park"),
        (BuildingType.GAS_STATION, 1, "Gas Station"),
        (BuildingType.COFFEE_SHOP, 1, "Cafe"),
    ]
    for btype, min_count, prefix in _ensure_buildings:
        existing = len(sim._buildings_by_type.get(btype, []))
        if existing < min_count:
            for i in range(min_count - existing):
                fb = Building(
                    building_id=f"f{btype.value[:2]}{i:03d}",
                    building_type=btype,
                    position=(rng.uniform(60, WORLD_W - 60),
                              rng.uniform(40, WORLD_H - 40)),
                    name=f"{prefix} F{i}",
                )
                sim.buildings.append(fb)
                sim._buildings_by_type.setdefault(btype, []).append(fb)

    # Monkey-patch _generate_buildings to no-op so populate() uses our layout
    sim._generate_buildings = lambda: None

    # Now populate residents (will skip building generation, use ours)
    sim.populate()

    # Attach RF profiles
    person_rf, vehicle_rf = attach_rf_profiles(sim.residents, sim.vehicles)

    # State
    current_time = [START_HOUR]
    frame_count = [0]
    fps_time = [time.time()]
    fps_val = [0.0]

    # --- Set up figure ---
    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title("TRITIUM — City Simulation")

    # Map axes (left 68%)
    ax_map = fig.add_axes([0.01, 0.01, 0.66, 0.98])
    ax_map.set_facecolor(BG_COLOR)
    ax_map.set_xlim(-5, WORLD_W + 5)
    ax_map.set_ylim(-5, WORLD_H + 5)
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors=TEXT_DIM, labelsize=5)
    for spine in ax_map.spines.values():
        spine.set_color(GRID_COLOR)

    # Stats panel (right 30%)
    ax_stats = fig.add_axes([0.69, 0.01, 0.30, 0.98])
    ax_stats.set_facecolor(PANEL_BG)
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    # ===== DRAW STATIC ELEMENTS (once) =====

    # 1. Draw road surfaces
    for seg in streets:
        hw = seg.width / 2  # half-width
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1:
            continue
        nx, ny = -dy / seg_len, dx / seg_len  # normal

        # Road polygon (rectangle along segment)
        color = ROAD_MAIN if seg.is_main else ROAD_RESIDENTIAL
        corners = [
            (seg.start[0] + nx * hw, seg.start[1] + ny * hw),
            (seg.end[0] + nx * hw, seg.end[1] + ny * hw),
            (seg.end[0] - nx * hw, seg.end[1] - ny * hw),
            (seg.start[0] - nx * hw, seg.start[1] - ny * hw),
        ]
        road_patch = Polygon(corners, closed=True, facecolor=color,
                             edgecolor="none", zorder=1)
        ax_map.add_patch(road_patch)

        # Lane markings (dashed center line on main roads)
        if seg.is_main:
            num_dashes = max(1, int(seg_len / 8))
            for d in range(num_dashes):
                t0 = (d * 2) / (num_dashes * 2)
                t1 = (d * 2 + 1) / (num_dashes * 2)
                if t1 > 1:
                    break
                x0 = seg.start[0] + dx * t0
                y0 = seg.start[1] + dy * t0
                x1 = seg.start[0] + dx * t1
                y1 = seg.start[1] + dy * t1
                ax_map.plot([x0, x1], [y0, y1], color=LANE_MARKING,
                           linewidth=0.8, zorder=2)

    # 2. Draw sidewalks (thin lines parallel to roads)
    for seg in streets:
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1:
            continue
        nx, ny = -dy / seg_len, dx / seg_len
        sw_offset = seg.width / 2 + 2.0  # sidewalk 2m from road edge

        for sign in [1, -1]:
            sx0 = seg.start[0] + nx * sw_offset * sign
            sy0 = seg.start[1] + ny * sw_offset * sign
            sx1 = seg.end[0] + nx * sw_offset * sign
            sy1 = seg.end[1] + ny * sw_offset * sign
            ax_map.plot([sx0, sx1], [sy0, sy1], color=SIDEWALK_COLOR,
                       linewidth=1.5, zorder=1, alpha=0.6)

    # 3. Draw intersection areas and stop signs
    for isect in intersections:
        ix, iy = isect.position
        # Intersection square (filled)
        isect_patch = Rectangle(
            (ix - 6, iy - 6), 12, 12,
            facecolor=ROAD_MAIN, edgecolor="none", zorder=1,
        )
        ax_map.add_patch(isect_patch)

        if isect.has_stop_sign:
            # Small red octagon
            stop = RegularPolygon(
                (ix + 7, iy + 7), numVertices=8, radius=2.0,
                facecolor=STOP_SIGN_COLOR, edgecolor="#881111",
                linewidth=0.5, zorder=8,
            )
            ax_map.add_patch(stop)

    # 4. Draw buildings
    for pb in placed_buildings:
        rect = Rectangle(
            pb.rect_pos, pb.width, pb.height,
            facecolor=pb.fill_color, edgecolor=pb.outline_color,
            linewidth=0.7, alpha=0.85, zorder=3,
        )
        ax_map.add_patch(rect)

        # Building label
        cx = pb.rect_pos[0] + pb.width / 2
        cy = pb.rect_pos[1] + pb.height / 2
        ax_map.text(
            cx, cy, pb.label, fontsize=5, color=TEXT_DIM,
            ha="center", va="center", zorder=4, fontfamily="monospace",
        )

        # Driveway (thin line from building to road)
        ax_map.plot(
            [pb.driveway_start[0], pb.driveway_end[0]],
            [pb.driveway_start[1], pb.driveway_end[1]],
            color="#1a1a2a", linewidth=1.0, zorder=2, alpha=0.5,
        )

    # 5. Draw trees
    for tree in trees:
        tc = Circle(
            tree.position, tree.radius,
            facecolor="#0a3a0a", edgecolor="#0a4a0a",
            linewidth=0.3, alpha=0.6, zorder=3,
        )
        ax_map.add_patch(tc)

    # 6. Street labels at intersections (sparse — only at edges)
    label_set = set()
    for seg in streets:
        # Label at start of each unique street
        if seg.name not in label_set:
            label_set.add(seg.name)
            mx = seg.start[0]
            my = seg.start[1]
            is_horiz = abs(seg.end[0] - seg.start[0]) > abs(seg.end[1] - seg.start[1])
            rot = 0 if is_horiz else 90
            offset_x = -15 if is_horiz else 8
            offset_y = 8 if is_horiz else -5
            ax_map.text(
                mx + offset_x, my + offset_y, seg.name,
                fontsize=4, color="#445566", fontfamily="monospace",
                rotation=rot, zorder=9, alpha=0.7,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG_COLOR)],
            )

    # Pre-create scatter objects for animation
    person_scatter = ax_map.scatter([], [], s=[], c=[], marker="o",
                                     zorder=6, linewidths=0)
    vehicle_scatter = ax_map.scatter([], [], s=[], c=[], marker="s",
                                      zorder=5, linewidths=0.3,
                                      edgecolors=YELLOW)
    parked_scatter = ax_map.scatter([], [], s=10, c=DARK_GREY, marker="s",
                                     zorder=4, alpha=0.4, linewidths=0)
    rf_texts: list = []

    # Title
    title_text = ax_map.set_title(
        "TRITIUM CITY SIM",
        color=CYAN, fontsize=12, fontfamily="monospace", fontweight="bold",
        pad=6,
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
        px, py, ps, pc = [], [], [], []
        vx, vy, vs, vc = [], [], [], []
        parked_x, parked_y = [], []
        rf_mac_data: list[tuple[float, float, str]] = []
        activity_counts: dict[str, int] = defaultdict(int)

        for r in sim.residents:
            state = r.activity_state
            style = _ACTIVITY_STYLE.get(state, (DARK_GREY, "o", 0.0, "Unknown"))
            color, marker, size, cat_label = style
            activity_counts[state] += 1

            if not r.visible:
                continue
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

            # RF MAC fragment
            prof = person_rf.get(r.resident_id)
            rf_level = state_rf_emission(r.activity_state)
            if prof and prof.has_phone and rf_level in ("full", "reduced"):
                mac_frag = prof.phone_mac[-5:]
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

        # --- RF labels ---
        for t in rf_texts:
            t.remove()
        rf_texts = []
        for rx, ry, mac in rf_mac_data[:12]:
            t = ax_map.text(
                rx, ry, mac, fontsize=3.5, color=GREEN, alpha=0.45,
                fontfamily="monospace", zorder=7,
            )
            rf_texts.append(t)

        # --- Title with clock ---
        title_text.set_text(
            f"TRITIUM CITY SIM  \u2502  {_format_time(current_time[0])}"
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
        dy = 0.030

        # Clock
        ax_stats.text(
            0.5, y, _format_time(current_time[0]),
            fontsize=22, color=CYAN, ha="center", fontfamily="monospace",
            fontweight="bold",
        )
        y -= dy * 2.0

        # Day progress bar
        day_progress = current_time[0] / 24.0
        ax_stats.add_patch(Rectangle(
            (0.05, y), 0.9, 0.012,
            facecolor=GRID_COLOR, edgecolor=TEXT_DIM, linewidth=0.5,
        ))
        ax_stats.add_patch(Rectangle(
            (0.05, y), 0.9 * day_progress, 0.012,
            facecolor=CYAN, alpha=0.6,
        ))
        is_day = 6.0 <= current_time[0] <= 20.0
        sun_x = 0.05 + 0.9 * day_progress
        ax_stats.plot(sun_x, y + 0.006, "o",
                      color=YELLOW if is_day else "#556688",
                      markersize=5, zorder=10)
        y -= dy * 1.5

        # Separator
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.5

        # Vehicle / Pedestrian counts
        stats = sim.get_statistics()
        ax_stats.text(
            0.5, y, "POPULATION",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.0

        pop_items = [
            (f"Vehicles: {stats['vehicles_driving']} driving / "
             f"{stats['vehicles_parked']} parked", YELLOW),
            (f"Visible: {stats['visible_on_map']}  |  "
             f"Inside: {stats['inside_buildings']}", CYAN),
            (f"Total: {len(sim.residents)} residents  |  "
             f"{len(sim.buildings)} bldgs", TEXT_DIM),
        ]
        for text, col in pop_items:
            ax_stats.text(
                0.08, y, text, fontsize=6.5, color=col,
                fontfamily="monospace", va="center",
            )
            y -= dy

        y -= dy * 0.3
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.5

        # Activity breakdown
        ax_stats.text(
            0.5, y, "ACTIVITY BREAKDOWN",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.0

        total_people = len(sim.residents)
        bar_max = total_people if total_people > 0 else 1

        for cat_name, (cat_color, states) in _CATEGORY_MAP.items():
            count = sum(activity_counts.get(s, 0) for s in states)
            bar_width = (count / bar_max) * 0.50

            ax_stats.text(
                0.05, y, cat_name, fontsize=6.5, color=cat_color,
                fontfamily="monospace", va="center",
            )
            ax_stats.text(
                0.95, y, str(count), fontsize=6.5, color=cat_color,
                fontfamily="monospace", va="center", ha="right",
            )
            ax_stats.add_patch(Rectangle(
                (0.35, y - 0.007), 0.50, 0.014,
                facecolor=GRID_COLOR, linewidth=0,
            ))
            if bar_width > 0.001:
                ax_stats.add_patch(Rectangle(
                    (0.35, y - 0.007), bar_width, 0.014,
                    facecolor=cat_color, alpha=0.6, linewidth=0,
                ))
            y -= dy

        y -= dy * 0.3
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.5

        # RF emissions
        ax_stats.text(
            0.5, y, "RF EMISSIONS",
            fontsize=8, color=TEXT_BRIGHT, ha="center", fontfamily="monospace",
        )
        y -= dy * 1.0

        rf_items = [
            ("BLE Advertisements", ble_count, CYAN),
            ("WiFi Probes", wifi_count, MAGENTA),
            ("TPMS Sensors", tpms_count, YELLOW),
            ("Total RF", ble_count + wifi_count + tpms_count, GREEN),
        ]
        for label, val, col in rf_items:
            ax_stats.text(
                0.08, y, label, fontsize=6.5, color=col,
                fontfamily="monospace", va="center",
            )
            ax_stats.text(
                0.92, y, str(val), fontsize=6.5, color=col,
                fontfamily="monospace", va="center", ha="right",
            )
            y -= dy

        y -= dy * 0.3
        ax_stats.plot([0.05, 0.95], [y, y], color=GRID_COLOR, linewidth=0.5)
        y -= dy * 0.5

        # FPS
        ax_stats.text(
            0.08, y, f"FPS: {fps_val[0]:.1f}", fontsize=6.5, color=TEXT_DIM,
            fontfamily="monospace", va="center",
        )
        ax_stats.text(
            0.92, y, f"Streets: {len(streets)}", fontsize=6.5, color=TEXT_DIM,
            fontfamily="monospace", va="center", ha="right",
        )
        y -= dy * 1.2

        # Legend
        ax_stats.text(
            0.5, y, "LEGEND",
            fontsize=7, color=TEXT_DIM, ha="center", fontfamily="monospace",
        )
        y -= dy * 0.8
        legend_items = [
            ("o", CYAN, "Walking / Pedestrian"),
            ("s", YELLOW, "Driving / Vehicle"),
            ("o", GREEN, "Working / Inside"),
            ("o", ORANGE, "Jogging / Outdoor"),
            ("o", MAGENTA, "Shopping / Errands"),
            ("o", PURPLE, "Walking dog"),
        ]
        for marker, color, desc in legend_items:
            ax_stats.plot(0.08, y, marker, color=color, markersize=4)
            ax_stats.text(
                0.15, y, desc, fontsize=5.5, color=color,
                fontfamily="monospace", va="center",
            )
            y -= dy * 0.8

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
        bounds=((0.0, 0.0), (WORLD_W, WORLD_H)),
        seed=42,
    )
    sim.populate()
    person_rf, vehicle_rf = attach_rf_profiles(sim.residents, sim.vehicles)

    current_time = START_HOUR
    print(f"\n{'='*60}")
    print("  TRITIUM CITY SIM — Terminal Mode")
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
        description="TRITIUM City Simulation Demo"
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
