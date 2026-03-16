# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GTA-style city life simulation — deep daily routines for a living neighborhood.

People LIVE here. They sleep, wake up, walk to their car, drive on roads, park
at work, walk from the parking lot to the building entrance, work for hours,
walk back to the car, drive to the grocery store, park, walk inside, shop for
10 minutes, walk back, drive home, park in the driveway, walk inside, relax,
walk the dog, come back, and go to sleep.

Every micro-transition is modeled: getting in/out of the car, walking between
parking spots and building entrances, being inside a building (invisible on
the tactical map), stopping at intersections, checking phones on the sidewalk.

Vehicles stay on roads. People walk on sidewalks. RF signatures change based
on what the person is doing (phone in pocket while driving emits differently
than phone held while walking).

This is pure civilian ambient life — no combat. The combat simulation in
tritium-sc layers hostiles/defenders on top of this.

Usage::

    from tritium_lib.game_ai.city_sim import NeighborhoodSim

    sim = NeighborhoodSim(num_residents=50, bounds=((0, 0), (500, 500)))
    sim.populate()
    sim.tick(dt=1.0, current_time=8.5)  # 8:30 AM
    entities = sim.get_all_entities()    # TargetTracker-compatible dicts
    stats = sim.get_statistics()
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tritium_lib.game_ai.steering import Vec2, distance, normalize, magnitude


# ---------------------------------------------------------------------------
# Vec2 helpers
# ---------------------------------------------------------------------------

def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _scale(v: Vec2, s: float) -> Vec2:
    return (v[0] * s, v[1] * s)


def _clamp(p: Vec2, lo: Vec2, hi: Vec2) -> Vec2:
    return (
        max(lo[0], min(hi[0], p[0])),
        max(lo[1], min(hi[1], p[1])),
    )


def _lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ---------------------------------------------------------------------------
# Activity states — the 25-state lifecycle
# ---------------------------------------------------------------------------

class ActivityState(str, Enum):
    """Micro-states for the deep resident lifecycle.

    Each state defines exactly what the person is doing, where they are,
    how they move, whether they are visible on the tactical map, and what
    RF signals they emit.
    """
    # Sleep cycle
    SLEEPING = "sleeping"
    WAKING_UP = "waking_up"

    # Home activities
    RELAXING = "relaxing"
    GARDENING = "gardening"
    NAPPING = "napping"

    # Car transitions
    WALKING_TO_CAR = "walking_to_car"
    GETTING_IN_CAR = "getting_in_car"
    DRIVING = "driving"
    PARKING = "parking"
    GETTING_OUT_OF_CAR = "getting_out_of_car"

    # Building transitions
    WALKING_TO_BUILDING = "walking_to_building"
    ENTERING_BUILDING = "entering_building"
    INSIDE_BUILDING = "inside_building"
    EXITING_BUILDING = "exiting_building"

    # Work
    WORKING = "working"
    AT_SCHOOL = "at_school"
    LUNCH_BREAK = "lunch_break"

    # Errands
    SHOPPING = "shopping"
    DINING = "dining"
    AT_GAS_STATION = "at_gas_station"
    AT_DOCTOR = "at_doctor"
    GETTING_COFFEE = "getting_coffee"

    # Outdoor activities
    WALKING = "walking"
    JOGGING = "jogging"
    WALKING_DOG = "walking_dog"
    PLAYING = "playing"
    CHECKING_PHONE = "checking_phone"
    SOCIALIZING = "socializing"

    # Delivery specific
    DELIVERING = "delivering"
    DELIVERY_STOP = "delivery_stop"

    # Commuting (walking to/from transit, waiting)
    WALKING_TO_TRANSIT = "walking_to_transit"
    RETURNING_HOME = "returning_home"


# State metadata: (movement_type, visible_on_map, rf_emission_level)
# movement_type: "stationary", "walking", "driving"
# visible_on_map: whether the person shows up on the tactical map
# rf_emission_level: "full" (phone+watch+earbuds), "reduced" (phone in pocket),
#                    "minimal" (phone on silent), "none" (airplane mode / sleeping)
_STATE_META: dict[str, tuple[str, bool, str]] = {
    ActivityState.SLEEPING:            ("stationary", False, "minimal"),
    ActivityState.WAKING_UP:           ("stationary", False, "reduced"),
    ActivityState.RELAXING:            ("stationary", False, "full"),
    ActivityState.GARDENING:           ("stationary", True,  "reduced"),
    ActivityState.NAPPING:             ("stationary", False, "minimal"),
    ActivityState.WALKING_TO_CAR:      ("walking",    True,  "full"),
    ActivityState.GETTING_IN_CAR:      ("stationary", True,  "full"),
    ActivityState.DRIVING:             ("driving",    True,  "reduced"),
    ActivityState.PARKING:             ("stationary", True,  "reduced"),
    ActivityState.GETTING_OUT_OF_CAR:  ("stationary", True,  "full"),
    ActivityState.WALKING_TO_BUILDING: ("walking",    True,  "full"),
    ActivityState.ENTERING_BUILDING:   ("walking",    True,  "full"),
    ActivityState.INSIDE_BUILDING:     ("stationary", False, "reduced"),
    ActivityState.EXITING_BUILDING:    ("walking",    True,  "full"),
    ActivityState.WORKING:             ("stationary", False, "reduced"),
    ActivityState.AT_SCHOOL:           ("stationary", False, "reduced"),
    ActivityState.LUNCH_BREAK:         ("walking",    True,  "full"),
    ActivityState.SHOPPING:            ("walking",    True,  "full"),
    ActivityState.DINING:              ("stationary", False, "full"),
    ActivityState.AT_GAS_STATION:      ("stationary", True,  "full"),
    ActivityState.AT_DOCTOR:           ("stationary", False, "minimal"),
    ActivityState.GETTING_COFFEE:      ("stationary", True,  "full"),
    ActivityState.WALKING:             ("walking",    True,  "full"),
    ActivityState.JOGGING:             ("walking",    True,  "full"),
    ActivityState.WALKING_DOG:         ("walking",    True,  "full"),
    ActivityState.PLAYING:             ("walking",    True,  "full"),
    ActivityState.CHECKING_PHONE:      ("stationary", True,  "full"),
    ActivityState.SOCIALIZING:         ("stationary", True,  "full"),
    ActivityState.DELIVERING:          ("driving",    True,  "reduced"),
    ActivityState.DELIVERY_STOP:       ("walking",    True,  "full"),
    ActivityState.WALKING_TO_TRANSIT:  ("walking",    True,  "full"),
    ActivityState.RETURNING_HOME:      ("walking",    True,  "full"),
}


def state_movement_type(state: str) -> str:
    """Return 'stationary', 'walking', or 'driving' for a given state."""
    meta = _STATE_META.get(state)
    return meta[0] if meta else "stationary"


def state_visible_on_map(state: str) -> bool:
    """Return whether a person in this state is visible on the tactical map."""
    meta = _STATE_META.get(state)
    return meta[1] if meta else True


def state_rf_emission(state: str) -> str:
    """Return RF emission level: 'full', 'reduced', 'minimal', 'none'."""
    meta = _STATE_META.get(state)
    return meta[2] if meta else "full"


# ---------------------------------------------------------------------------
# Errand types with duration ranges (in seconds)
# ---------------------------------------------------------------------------

class ErrandType(str, Enum):
    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    COFFEE = "coffee"
    GAS_STATION = "gas_station"
    DOCTOR = "doctor"
    DELIVERY = "delivery"


# (min_seconds, max_seconds) for each errand
ERRAND_DURATIONS: dict[str, tuple[float, float]] = {
    ErrandType.GROCERY:     (300.0, 1200.0),   # 5-20 min
    ErrandType.RESTAURANT:  (1200.0, 3600.0),  # 20-60 min
    ErrandType.COFFEE:      (300.0, 600.0),     # 5-10 min
    ErrandType.GAS_STATION: (180.0, 300.0),     # 3-5 min
    ErrandType.DOCTOR:      (1800.0, 3600.0),   # 30-60 min
    ErrandType.DELIVERY:    (60.0, 180.0),       # 1-3 min per stop
}


# ---------------------------------------------------------------------------
# Vehicle types
# ---------------------------------------------------------------------------

class VehicleType(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    DELIVERY_VAN = "delivery_van"


# (speed_min, speed_max, size_description)
_VEHICLE_SPECS: dict[str, tuple[float, float, str]] = {
    VehicleType.CAR:          (8.0, 15.0, "sedan"),
    VehicleType.TRUCK:        (6.0, 13.0, "pickup"),
    VehicleType.MOTORCYCLE:   (10.0, 18.0, "motorcycle"),
    VehicleType.BICYCLE:      (3.0, 6.0,  "bicycle"),
    VehicleType.DELIVERY_VAN: (5.0, 12.0, "van"),
}


# ---------------------------------------------------------------------------
# Building types
# ---------------------------------------------------------------------------

class BuildingType(str, Enum):
    HOME = "home"
    OFFICE = "office"
    SCHOOL = "school"
    GROCERY = "grocery"
    PARK = "park"
    RESTAURANT = "restaurant"
    GAS_STATION = "gas_station"
    COFFEE_SHOP = "coffee_shop"
    DOCTOR = "doctor"


# Parking offset from building — where the parking lot is relative to entrance
_PARKING_OFFSET: dict[str, float] = {
    BuildingType.HOME:       8.0,    # driveway
    BuildingType.OFFICE:     25.0,   # parking lot
    BuildingType.SCHOOL:     20.0,
    BuildingType.GROCERY:    30.0,   # big parking lot
    BuildingType.PARK:       15.0,
    BuildingType.RESTAURANT: 15.0,
    BuildingType.GAS_STATION: 10.0,
    BuildingType.COFFEE_SHOP: 12.0,
    BuildingType.DOCTOR:     20.0,
}


@dataclass
class Building:
    """A location in the neighborhood with entrance and parking."""
    building_id: str
    building_type: BuildingType
    position: Vec2            # building entrance position
    name: str = ""
    capacity: int = 10
    parking_pos: Vec2 = (0.0, 0.0)  # where cars park

    def __post_init__(self) -> None:
        if self.parking_pos == (0.0, 0.0) and self.position != (0.0, 0.0):
            # Default parking is offset from entrance
            offset = _PARKING_OFFSET.get(self.building_type, 15.0)
            self.parking_pos = (self.position[0] + offset, self.position[1])


# ---------------------------------------------------------------------------
# Schedule system
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    """A single activity in a daily schedule.

    hour: fractional hour (0-24) when this activity starts.
    activity: what the resident is doing (maps to a high-level goal).
    location_type: where to go (BuildingType or 'home', 'outdoors').
    duration_hours: how long before checking next entry.
    errand_type: optional errand type for shopping/dining activities.
    """
    hour: float
    activity: str
    location_type: str = "home"
    duration_hours: float = 1.0
    errand_type: str = ""


class DailySchedule:
    """Time-based activity schedule for a resident.

    Entries are sorted by hour. The schedule wraps around midnight.
    The schedule defines HIGH-LEVEL goals (go to work, go shopping).
    The ActivityState machine handles the micro-transitions (walk to car,
    drive, park, walk to building, enter, etc.).
    """

    def __init__(self, entries: list[ScheduleEntry] | None = None) -> None:
        self.entries: list[ScheduleEntry] = sorted(
            entries or [], key=lambda e: e.hour
        )

    def activity_at(self, hour: float) -> ScheduleEntry:
        """Return the active schedule entry at a given hour (0-24)."""
        hour = hour % 24.0
        if not self.entries:
            return ScheduleEntry(hour=0, activity="sleeping", location_type="home")
        # Find the last entry whose start hour is <= current hour
        active = self.entries[0]
        for entry in self.entries:
            if entry.hour <= hour:
                active = entry
            else:
                break
        return active

    # -- factory class methods ------------------------------------------------

    @classmethod
    def office_worker(cls) -> DailySchedule:
        """Wake 6:30, commute 7:15, work 7:45, lunch 12:00,
        work 1:00, leave 5:15, grocery 5:30, home 6:15, walk dog 7:00,
        relax 7:45, sleep 10:30."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 6.5),
            ScheduleEntry(6.5, "waking_up", "home", 0.75),
            ScheduleEntry(7.25, "commuting", "office", 0.5),
            ScheduleEntry(7.75, "working", "office", 4.25),
            ScheduleEntry(12.0, "lunch", "restaurant", 1.0, ErrandType.RESTAURANT),
            ScheduleEntry(13.0, "working", "office", 4.25),
            ScheduleEntry(17.25, "commuting", "grocery", 0.5,
                          ErrandType.GROCERY),
            ScheduleEntry(17.75, "shopping", "grocery", 0.5,
                          ErrandType.GROCERY),
            ScheduleEntry(18.25, "commuting", "home", 0.75),
            ScheduleEntry(19.0, "walking_dog", "outdoors", 0.75),
            ScheduleEntry(19.75, "relaxing", "home", 2.75),
            ScheduleEntry(22.5, "sleeping", "home", 8.0),
        ])

    @classmethod
    def school_kid(cls) -> DailySchedule:
        """Wake 7:00, bus 7:30, school 8:00, lunch 12:00, school ends 3:00,
        play outside 3:30, home 5:30, sleep 9:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 7.0),
            ScheduleEntry(7.0, "waking_up", "home", 0.5),
            ScheduleEntry(7.5, "commuting", "school", 0.5),
            ScheduleEntry(8.0, "at_school", "school", 4.0),
            ScheduleEntry(12.0, "lunch", "school", 0.5),
            ScheduleEntry(12.5, "at_school", "school", 2.5),
            ScheduleEntry(15.0, "commuting", "home", 0.5),
            ScheduleEntry(15.5, "playing", "outdoors", 2.0),
            ScheduleEntry(17.5, "relaxing", "home", 3.5),
            ScheduleEntry(21.0, "sleeping", "home", 10.0),
        ])

    @classmethod
    def retired(cls) -> DailySchedule:
        """Wake 6:00, morning walk 6:30, home 7:30, garden 9:00,
        lunch 12:00, nap 1:00, coffee 3:00, afternoon walk 3:30,
        home 4:30, sleep 9:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 6.0),
            ScheduleEntry(6.0, "waking_up", "home", 0.5),
            ScheduleEntry(6.5, "walking", "outdoors", 1.0),
            ScheduleEntry(7.5, "relaxing", "home", 1.5),
            ScheduleEntry(9.0, "gardening", "home", 3.0),
            ScheduleEntry(12.0, "lunch", "home", 1.0),
            ScheduleEntry(13.0, "napping", "home", 2.0),
            ScheduleEntry(15.0, "getting_coffee", "coffee_shop", 0.5,
                          ErrandType.COFFEE),
            ScheduleEntry(15.5, "walking", "outdoors", 1.0),
            ScheduleEntry(16.5, "relaxing", "home", 4.5),
            ScheduleEntry(21.0, "sleeping", "home", 9.0),
        ])

    @classmethod
    def delivery_driver(cls) -> DailySchedule:
        """Start 8:00, drive route with stops, end 5:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 7.0),
            ScheduleEntry(7.0, "waking_up", "home", 1.0),
            ScheduleEntry(8.0, "delivering", "outdoors", 2.0,
                          ErrandType.DELIVERY),
            ScheduleEntry(10.0, "delivering", "outdoors", 2.0,
                          ErrandType.DELIVERY),
            ScheduleEntry(12.0, "lunch", "restaurant", 0.5,
                          ErrandType.RESTAURANT),
            ScheduleEntry(12.5, "delivering", "outdoors", 2.0,
                          ErrandType.DELIVERY),
            ScheduleEntry(14.5, "delivering", "outdoors", 2.5,
                          ErrandType.DELIVERY),
            ScheduleEntry(17.0, "commuting", "home", 0.5),
            ScheduleEntry(17.5, "relaxing", "home", 4.5),
            ScheduleEntry(22.0, "sleeping", "home", 9.0),
        ])

    @classmethod
    def work_from_home(cls) -> DailySchedule:
        """Wake 7:30, work 8:30, lunch walk 12:00, work 1:00, done 5:30,
        jog 6:00, coffee 6:45, dinner 7:00, sleep 11:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 7.5),
            ScheduleEntry(7.5, "waking_up", "home", 1.0),
            ScheduleEntry(8.5, "working", "home", 3.5),
            ScheduleEntry(12.0, "walking", "outdoors", 1.0),
            ScheduleEntry(13.0, "working", "home", 4.5),
            ScheduleEntry(17.5, "relaxing", "home", 0.5),
            ScheduleEntry(18.0, "jogging", "outdoors", 0.75),
            ScheduleEntry(18.75, "getting_coffee", "coffee_shop", 0.25,
                          ErrandType.COFFEE),
            ScheduleEntry(19.0, "relaxing", "home", 4.0),
            ScheduleEntry(23.0, "sleeping", "home", 8.5),
        ])

    @classmethod
    def service_worker(cls) -> DailySchedule:
        """Wake 5:30, commute 6:00, work 6:30, lunch 11:30, work 12:00,
        off 2:30, errands 3:00, home 4:00, sleep 9:30."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 5.5),
            ScheduleEntry(5.5, "waking_up", "home", 0.5),
            ScheduleEntry(6.0, "commuting", "office", 0.5),
            ScheduleEntry(6.5, "working", "office", 5.0),
            ScheduleEntry(11.5, "lunch", "restaurant", 0.5,
                          ErrandType.RESTAURANT),
            ScheduleEntry(12.0, "working", "office", 2.5),
            ScheduleEntry(14.5, "commuting", "home", 0.5),
            ScheduleEntry(15.0, "shopping", "grocery", 1.0,
                          ErrandType.GROCERY),
            ScheduleEntry(16.0, "relaxing", "home", 5.5),
            ScheduleEntry(21.5, "sleeping", "home", 8.0),
        ])


# ---------------------------------------------------------------------------
# Resident role enum
# ---------------------------------------------------------------------------

class ResidentRole(str, Enum):
    OFFICE_WORKER = "office_worker"
    SCHOOL_KID = "school_kid"
    RETIRED = "retired"
    DELIVERY_DRIVER = "delivery_driver"
    WORK_FROM_HOME = "work_from_home"
    SERVICE_WORKER = "service_worker"


# Schedule factory per role
_SCHEDULE_FACTORIES: dict[str, type] = {
    ResidentRole.OFFICE_WORKER: DailySchedule.office_worker,
    ResidentRole.SCHOOL_KID: DailySchedule.school_kid,
    ResidentRole.RETIRED: DailySchedule.retired,
    ResidentRole.DELIVERY_DRIVER: DailySchedule.delivery_driver,
    ResidentRole.WORK_FROM_HOME: DailySchedule.work_from_home,
    ResidentRole.SERVICE_WORKER: DailySchedule.service_worker,
}

# Default population mix
DEFAULT_MIX: dict[str, float] = {
    ResidentRole.OFFICE_WORKER: 0.40,
    ResidentRole.SCHOOL_KID: 0.20,
    ResidentRole.RETIRED: 0.15,
    ResidentRole.WORK_FROM_HOME: 0.10,
    ResidentRole.SERVICE_WORKER: 0.10,
    ResidentRole.DELIVERY_DRIVER: 0.05,
}

# Roles that own a car
_VEHICLE_ROLES = {
    ResidentRole.OFFICE_WORKER,
    ResidentRole.DELIVERY_DRIVER,
    ResidentRole.SERVICE_WORKER,
}

# Speed in m/s for different movement modes
_MOVE_SPEEDS: dict[str, tuple[float, float]] = {
    "walking": (0.8, 1.6),
    "walking_slow": (0.4, 0.8),     # elderly, phone-checking
    "walking_fast": (1.4, 2.0),     # late for work
    "jogging": (2.5, 3.5),
    "driving": (5.0, 15.0),
    "biking": (3.0, 6.0),
    "playing": (1.0, 2.5),
    "child": (1.2, 2.0),
    "elderly": (0.5, 1.0),
}


# ---------------------------------------------------------------------------
# SimVehicle
# ---------------------------------------------------------------------------

@dataclass
class SimVehicle:
    """A simulated vehicle parked or driving."""
    vehicle_id: str
    owner_id: str
    vehicle_type: str = VehicleType.CAR
    position: Vec2 = (0.0, 0.0)
    velocity: Vec2 = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    parked_at: Vec2 | None = None
    driving: bool = False
    path: list[Vec2] = field(default_factory=list)
    path_index: int = 0
    _intersection_pause: float = 0.0  # seconds to pause at intersection

    def tick(self, dt: float, bounds: tuple[Vec2, Vec2] | None = None) -> None:
        """Advance vehicle position along its path."""
        if not self.driving or not self.path:
            self.velocity = (0.0, 0.0)
            return

        # Pause at intersections (turns)
        if self._intersection_pause > 0:
            self._intersection_pause -= dt
            self.velocity = (0.0, 0.0)
            return

        if self.path_index >= len(self.path):
            # Arrived — park
            self.driving = False
            self.parked_at = self.position
            self.velocity = (0.0, 0.0)
            self.speed = 0.0
            return

        target = self.path[self.path_index]
        direction = _sub(target, self.position)
        dist = magnitude(direction)

        if dist < self.speed * dt * 1.5:
            self.position = target
            old_index = self.path_index
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.driving = False
                self.parked_at = self.position
                self.velocity = (0.0, 0.0)
                self.speed = 0.0
            elif self.path_index < len(self.path):
                # Check if this is a turn (direction changes significantly)
                next_target = self.path[self.path_index]
                new_dir = _sub(next_target, self.position)
                new_mag = magnitude(new_dir)
                if new_mag > 1.0 and dist > 0.1:
                    old_norm = normalize(direction)
                    new_norm = normalize(new_dir)
                    # Dot product — values near 0 mean a sharp turn
                    dot = old_norm[0] * new_norm[0] + old_norm[1] * new_norm[1]
                    if dot < 0.7:  # > ~45 degree turn
                        self._intersection_pause = 1.5  # 1.5 sec stop
        else:
            norm = normalize(direction)
            self.velocity = _scale(norm, self.speed)
            self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
            self.position = _add(self.position, _scale(self.velocity, dt))

        if bounds:
            self.position = _clamp(self.position, bounds[0], bounds[1])

    def start_driving(self, path: list[Vec2], speed: float) -> None:
        """Begin driving along a path."""
        self.path = path
        self.path_index = 0
        self.driving = True
        self.parked_at = None
        self.speed = speed
        self._intersection_pause = 0.0

    def park(self, location: Vec2) -> None:
        """Park the vehicle at a location."""
        self.driving = False
        self.parked_at = location
        self.position = location
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.path = []
        self.path_index = 0
        self._intersection_pause = 0.0

    def to_dict(self) -> dict:
        """Export as TargetTracker-compatible dict."""
        return {
            "target_id": f"veh_{self.vehicle_id}",
            "name": f"Vehicle {self.vehicle_id[:6]}",
            "source": "city_sim",
            "asset_type": "vehicle",
            "alliance": "neutral",
            "classification": "vehicle",
            "position_x": self.position[0],
            "position_y": self.position[1],
            "heading": self.heading,
            "speed": self.speed if self.driving else 0.0,
            "state": "moving" if self.driving else "parked",
            "metadata": {
                "owner_id": self.owner_id,
                "parked": not self.driving,
                "vehicle_type": self.vehicle_type,
                "simulated": True,
            },
        }


# ---------------------------------------------------------------------------
# Resident
# ---------------------------------------------------------------------------

# Randomized first names for generating residents
_FIRST_NAMES = [
    "Alex", "Blake", "Casey", "Dana", "Ellis", "Frankie", "Gray",
    "Harper", "Indigo", "Jordan", "Kelly", "Lane", "Morgan", "Nico",
    "Parker", "Quinn", "Riley", "Sage", "Taylor", "Val", "Wren",
    "Avery", "Bailey", "Cameron", "Drew", "Emery", "Finley", "Greer",
    "Hayden", "Isa", "Jamie", "Kit", "Logan", "Micah", "Noel",
    "Oakley", "Peyton", "Reese", "Skyler", "Tatum", "Uma", "Winter",
]

_LAST_NAMES = [
    "Smith", "Chen", "Patel", "Garcia", "Kim", "Brown", "Silva",
    "Nguyen", "Martinez", "Anderson", "Taylor", "Wilson", "Moore",
    "Clark", "Lopez", "Lee", "Walker", "Hall", "Allen", "Young",
]


@dataclass
class Resident:
    """A simulated person with a deep daily routine and micro-state machine.

    The schedule defines high-level goals (go to work, go shopping).
    The activity_state tracks exactly what the person is doing right now
    (walking to car, driving, parking, walking to building entrance, etc.).
    """
    resident_id: str
    name: str
    role: str  # ResidentRole value
    home_location: Vec2
    work_location: Vec2 | None = None
    vehicle: SimVehicle | None = None
    schedule: DailySchedule = field(default_factory=DailySchedule)

    # High-level activity from schedule (for backward compat)
    current_activity: str = "sleeping"
    # Deep micro-state
    activity_state: str = ActivityState.SLEEPING

    position: Vec2 = (0.0, 0.0)
    velocity: Vec2 = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0

    # Is the person visible on the tactical map?
    visible: bool = False

    # Internal state machine
    _target_location: Vec2 | None = None
    _target_building: Building | None = field(default=None, repr=False)
    _walk_target: Vec2 | None = None       # immediate walk-to point
    _transition_timer: float = 0.0         # seconds remaining in timed state
    _outdoor_path: list[Vec2] = field(default_factory=list)
    _path_index: int = 0
    _grocery_day: int = 0  # day of week for grocery run (0=Mon)
    _has_dog: bool = False
    _errand_type: str = ""
    _pending_schedule_activity: str = ""   # next schedule activity to process
    _last_schedule_activity: str = ""      # to detect schedule transitions
    _car_parked_at: Vec2 | None = None     # where the car is parked right now
    _phone_check_cooldown: float = 0.0     # seconds until next phone check

    def tick(
        self,
        dt: float,
        current_time: float,
        bounds: tuple[Vec2, Vec2],
        buildings: dict[str, list[Building]],
        rng: random.Random,
    ) -> None:
        """Advance resident simulation by dt seconds.

        current_time: hours since midnight (0-24).
        buildings: dict mapping BuildingType -> list of Building.
        """
        # Check schedule for high-level activity transitions
        entry = self.schedule.activity_at(current_time)
        if entry.activity != self._last_schedule_activity:
            self._last_schedule_activity = entry.activity
            self._begin_goal(entry, bounds, buildings, rng)

        # Tick the micro-state machine
        self._tick_state(dt, bounds, rng)

        # Update visibility
        self.visible = state_visible_on_map(self.activity_state)

        # Random phone checks while walking
        if self._phone_check_cooldown > 0:
            self._phone_check_cooldown -= dt

    def _begin_goal(
        self,
        entry: ScheduleEntry,
        bounds: tuple[Vec2, Vec2],
        buildings: dict[str, list[Building]],
        rng: random.Random,
    ) -> None:
        """Begin a new high-level goal from the schedule.

        This resolves WHERE to go and sets up the micro-state sequence.
        The actual movement happens in _tick_state().
        """
        self.current_activity = entry.activity
        self._errand_type = entry.errand_type
        self._outdoor_path = []
        self._path_index = 0

        loc_type = entry.location_type
        target_building: Building | None = None

        # Resolve destination
        if loc_type == "home":
            self._target_location = self.home_location
        elif loc_type == "office" and self.work_location:
            self._target_location = self.work_location
            # Find the actual building for parking info
            offices = buildings.get(BuildingType.OFFICE, [])
            if offices:
                target_building = min(
                    offices,
                    key=lambda b: distance(b.position, self.work_location),
                )
        elif loc_type == "school":
            schools = buildings.get(BuildingType.SCHOOL, [])
            if schools:
                target_building = rng.choice(schools)
                self._target_location = target_building.position
            else:
                self._target_location = self.work_location or self.home_location
        elif loc_type == "grocery":
            stores = buildings.get(BuildingType.GROCERY, [])
            if stores:
                target_building = rng.choice(stores)
                self._target_location = target_building.position
            else:
                self._target_location = self.home_location
        elif loc_type == "restaurant":
            restaurants = buildings.get(BuildingType.RESTAURANT, [])
            if restaurants:
                target_building = rng.choice(restaurants)
                self._target_location = target_building.position
            else:
                self._target_location = self.home_location
        elif loc_type == "coffee_shop":
            # Use restaurants as coffee shops if no dedicated ones
            shops = buildings.get(BuildingType.COFFEE_SHOP, [])
            if not shops:
                shops = buildings.get(BuildingType.RESTAURANT, [])
            if shops:
                target_building = rng.choice(shops)
                self._target_location = target_building.position
            else:
                self._target_location = self.home_location
        elif loc_type == "outdoors":
            parks = buildings.get(BuildingType.PARK, [])
            if parks:
                park = rng.choice(parks)
                self._target_location = park.position
            else:
                self._target_location = self._random_nearby(
                    self.home_location, 80.0, bounds, rng
                )
        else:
            self._target_location = self.home_location

        self._target_building = target_building

        # Determine the initial micro-state transition
        activity = entry.activity

        if activity in ("sleeping",):
            self._enter_state(ActivityState.SLEEPING, rng)
        elif activity in ("waking_up",):
            self._enter_state(ActivityState.WAKING_UP, rng)
            self._transition_timer = rng.uniform(300.0, 600.0)  # 5-10 min
        elif activity in ("relaxing",):
            self._enter_state(ActivityState.RELAXING, rng)
        elif activity in ("gardening",):
            self._enter_state(ActivityState.GARDENING, rng)
        elif activity in ("napping",):
            self._enter_state(ActivityState.NAPPING, rng)
        elif activity in ("working",):
            if loc_type == "home":
                # Work from home — just stay put
                self._enter_state(ActivityState.WORKING, rng)
            else:
                # Need to travel to office
                self._begin_travel_to(self._target_location, target_building,
                                      bounds, rng, arrival_state=ActivityState.WORKING)
        elif activity in ("at_school",):
            self._begin_travel_to(self._target_location, target_building,
                                  bounds, rng, arrival_state=ActivityState.AT_SCHOOL)
        elif activity in ("commuting",):
            self._begin_travel_to(self._target_location, target_building,
                                  bounds, rng)
        elif activity in ("lunch",):
            if loc_type == "home" or loc_type == "school":
                self._enter_state(ActivityState.LUNCH_BREAK, rng)
                self._transition_timer = rng.uniform(1200.0, 2400.0)
            else:
                self._begin_travel_to(self._target_location, target_building,
                                      bounds, rng, arrival_state=ActivityState.DINING)
        elif activity in ("shopping",):
            self._begin_travel_to(self._target_location, target_building,
                                  bounds, rng, arrival_state=ActivityState.SHOPPING)
        elif activity in ("getting_coffee",):
            self._begin_travel_to(self._target_location, target_building,
                                  bounds, rng, arrival_state=ActivityState.GETTING_COFFEE)
        elif activity in ("walking", "walking_dog", "jogging", "playing"):
            self._begin_outdoor_activity(activity, bounds, rng)
        elif activity in ("delivering",):
            self._begin_delivery(bounds, rng)
        else:
            # Fallback: treat as walking to target
            if self._target_location and distance(self.position, self._target_location) > 5.0:
                self._begin_travel_to(self._target_location, target_building,
                                      bounds, rng)
            else:
                self._enter_state(ActivityState.RELAXING, rng)

    def _begin_travel_to(
        self,
        destination: Vec2 | None,
        building: Building | None,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
        arrival_state: str = "",
    ) -> None:
        """Start the full travel sequence: walk-to-car, drive, park, walk-to-building."""
        if destination is None:
            destination = self.home_location

        self._target_location = destination
        self._target_building = building
        self._pending_schedule_activity = arrival_state

        dist_to_dest = distance(self.position, destination)

        # Can we drive?
        if self.vehicle and dist_to_dest > 50.0:
            # Need to walk to the car first
            car_pos = self.vehicle.parked_at or self.vehicle.position
            dist_to_car = distance(self.position, car_pos)
            if dist_to_car > 3.0:
                self._walk_target = car_pos
                self._enter_state(ActivityState.WALKING_TO_CAR, rng)
            else:
                # Already at the car
                self._enter_state(ActivityState.GETTING_IN_CAR, rng)
                self._transition_timer = rng.uniform(3.0, 8.0)
        else:
            # Walk the whole way
            if dist_to_dest > 3.0:
                self._walk_target = destination
                self._enter_state(ActivityState.WALKING_TO_BUILDING, rng)
            elif arrival_state:
                self._enter_state(arrival_state, rng)
                if arrival_state == ActivityState.SHOPPING:
                    dur = ERRAND_DURATIONS.get(self._errand_type,
                                               (300.0, 1200.0))
                    self._transition_timer = rng.uniform(*dur)
            else:
                self._enter_state(ActivityState.INSIDE_BUILDING, rng)

    def _begin_outdoor_activity(
        self,
        activity: str,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Start an outdoor activity (walking, jogging, playing, walking_dog)."""
        state_map = {
            "walking": ActivityState.WALKING,
            "walking_dog": ActivityState.WALKING_DOG,
            "jogging": ActivityState.JOGGING,
            "playing": ActivityState.PLAYING,
        }
        state = state_map.get(activity, ActivityState.WALKING)

        # If inside, walk outside first
        if self.activity_state in (ActivityState.INSIDE_BUILDING,
                                   ActivityState.RELAXING,
                                   ActivityState.WORKING,
                                   ActivityState.SLEEPING,
                                   ActivityState.WAKING_UP):
            self._enter_state(ActivityState.EXITING_BUILDING, rng)
            self._transition_timer = rng.uniform(5.0, 15.0)
            self._pending_schedule_activity = state
            return

        # Build outdoor path
        if self._target_location:
            waypoints = 5 if activity == "walking_dog" else 3
            self._outdoor_path = self._make_outdoor_path(
                self.position, self._target_location, bounds, rng,
                waypoints=waypoints,
            )
            self._path_index = 0

        self._enter_state(state, rng)

    def _begin_delivery(
        self,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Start a delivery run — drive to random locations, stop briefly."""
        if self.vehicle:
            # Pick a random delivery destination
            dest = self._random_nearby(self.position, 150.0, bounds, rng)
            self._target_location = dest
            self._pending_schedule_activity = ActivityState.DELIVERY_STOP

            car_pos = self.vehicle.parked_at or self.vehicle.position
            dist_to_car = distance(self.position, car_pos)
            if dist_to_car > 3.0:
                self._walk_target = car_pos
                self._enter_state(ActivityState.WALKING_TO_CAR, rng)
            else:
                self._enter_state(ActivityState.GETTING_IN_CAR, rng)
                self._transition_timer = rng.uniform(2.0, 5.0)
        else:
            # Walk delivery
            self._enter_state(ActivityState.DELIVERING, rng)

    def _enter_state(self, state: str, rng: random.Random) -> None:
        """Transition to a new activity micro-state."""
        self.activity_state = state
        meta = _STATE_META.get(state, ("stationary", True, "full"))
        move_type = meta[0]

        if move_type == "stationary":
            self.speed = 0.0
            self.velocity = (0.0, 0.0)
        elif move_type == "walking":
            if self.role == ResidentRole.SCHOOL_KID:
                self.speed = rng.uniform(*_MOVE_SPEEDS["child"])
            elif self.role == ResidentRole.RETIRED:
                self.speed = rng.uniform(*_MOVE_SPEEDS["elderly"])
            elif state == ActivityState.JOGGING:
                self.speed = rng.uniform(*_MOVE_SPEEDS["jogging"])
            elif state == ActivityState.PLAYING:
                self.speed = rng.uniform(*_MOVE_SPEEDS["playing"])
            elif state == ActivityState.SHOPPING:
                self.speed = rng.uniform(*_MOVE_SPEEDS["walking_slow"])
            else:
                self.speed = rng.uniform(*_MOVE_SPEEDS["walking"])

    def _tick_state(
        self,
        dt: float,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Tick the micro-state machine. Handles transitions and movement."""
        state = self.activity_state

        # ----- Timed states (wait then transition) -----
        if self._transition_timer > 0:
            self._transition_timer -= dt
            if self._transition_timer > 0:
                # Still in timed state — handle movement if applicable
                if state in (ActivityState.SHOPPING,):
                    # Wander slightly inside the store
                    self._wander_inside(dt, bounds, rng, radius=10.0)
                return
            # Timer expired — advance to next state
            self._advance_from_timed_state(bounds, rng)
            return

        # ----- Walking to car -----
        if state == ActivityState.WALKING_TO_CAR:
            if self._walk_target:
                arrived = self._walk_toward(self._walk_target, dt, bounds)
                if arrived:
                    self._enter_state(ActivityState.GETTING_IN_CAR, rng)
                    self._transition_timer = rng.uniform(3.0, 8.0)
            else:
                self._enter_state(ActivityState.GETTING_IN_CAR, rng)
                self._transition_timer = rng.uniform(3.0, 8.0)
            return

        # ----- Getting in car (timed via transition_timer above) -----
        if state == ActivityState.GETTING_IN_CAR:
            # Timer already handled above — this means timer hit 0
            # Start driving
            if self.vehicle and self._target_location:
                self._start_driving(self._target_location, bounds, rng)
                self._enter_state(ActivityState.DRIVING, rng)
            else:
                # No vehicle somehow — walk
                self._walk_target = self._target_location
                self._enter_state(ActivityState.WALKING_TO_BUILDING, rng)
            return

        # ----- Driving -----
        if state == ActivityState.DRIVING:
            if self.vehicle and self.vehicle.driving:
                self.vehicle.tick(dt, bounds)
                self.position = self.vehicle.position
                self.heading = self.vehicle.heading
                self.velocity = self.vehicle.velocity
                if not self.vehicle.driving:
                    # Arrived — park
                    parking_pos = self._get_parking_pos(rng)
                    self.vehicle.park(parking_pos)
                    self.position = parking_pos
                    self._car_parked_at = parking_pos
                    self._enter_state(ActivityState.PARKING, rng)
                    self._transition_timer = rng.uniform(2.0, 5.0)
            else:
                # Vehicle not driving anymore
                self._enter_state(ActivityState.PARKING, rng)
                self._transition_timer = rng.uniform(2.0, 5.0)
            return

        # ----- Parking (timed) -----
        # Handled by transition_timer -> _advance_from_timed_state

        # ----- Getting out of car (timed) -----
        # Handled by transition_timer -> _advance_from_timed_state

        # ----- Walking to building -----
        if state == ActivityState.WALKING_TO_BUILDING:
            target = self._walk_target or self._target_location
            if target:
                arrived = self._walk_toward(target, dt, bounds)
                # Random phone check while walking
                if not arrived and self._phone_check_cooldown <= 0:
                    if rng.random() < 0.005:  # ~0.5% chance per tick
                        self._enter_state(ActivityState.CHECKING_PHONE, rng)
                        self._transition_timer = rng.uniform(5.0, 20.0)
                        return
                if arrived:
                    self._enter_state(ActivityState.ENTERING_BUILDING, rng)
                    self._transition_timer = rng.uniform(3.0, 8.0)
            else:
                self._enter_state(ActivityState.ENTERING_BUILDING, rng)
                self._transition_timer = rng.uniform(3.0, 8.0)
            return

        # ----- Entering building (timed) -----
        # Handled by transition_timer -> _advance_from_timed_state

        # ----- Exiting building (timed) -----
        # Handled by transition_timer -> _advance_from_timed_state

        # ----- Inside building (wait for schedule change) -----
        if state in (ActivityState.INSIDE_BUILDING,
                     ActivityState.WORKING,
                     ActivityState.AT_SCHOOL,
                     ActivityState.SLEEPING,
                     ActivityState.WAKING_UP,
                     ActivityState.RELAXING,
                     ActivityState.NAPPING,
                     ActivityState.GARDENING,
                     ActivityState.DINING,
                     ActivityState.AT_DOCTOR,
                     ActivityState.GETTING_COFFEE):
            self.velocity = (0.0, 0.0)
            return

        # ----- Shopping (walking slowly inside) -----
        if state == ActivityState.SHOPPING:
            self._wander_inside(dt, bounds, rng, radius=10.0)
            return

        # ----- Outdoor activities: walking, jogging, walking_dog, playing -----
        if state in (ActivityState.WALKING, ActivityState.JOGGING,
                     ActivityState.WALKING_DOG, ActivityState.PLAYING):
            self._tick_outdoor(dt, bounds, rng)
            return

        # ----- Delivery stop -----
        if state == ActivityState.DELIVERY_STOP:
            # Brief stop then back to driving
            self._transition_timer = rng.uniform(60.0, 180.0)
            return

        # ----- Delivering (driving) -----
        if state == ActivityState.DELIVERING:
            if self.vehicle and self.vehicle.driving:
                self.vehicle.tick(dt, bounds)
                self.position = self.vehicle.position
                self.heading = self.vehicle.heading
                self.velocity = self.vehicle.velocity
                if not self.vehicle.driving:
                    parking_pos = self._get_parking_pos(rng)
                    self.vehicle.park(parking_pos)
                    self.position = parking_pos
                    self._car_parked_at = parking_pos
                    self._enter_state(ActivityState.DELIVERY_STOP, rng)
                    self._transition_timer = rng.uniform(60.0, 180.0)
            return

        # ----- Checking phone (timed) -----
        # Handled by transition_timer

        # ----- Socializing (timed) -----
        # Handled by transition_timer

        # ----- Returning home (walking) -----
        if state == ActivityState.RETURNING_HOME:
            arrived = self._walk_toward(self.home_location, dt, bounds)
            if arrived:
                self._enter_state(ActivityState.RELAXING, rng)
            return

        # ----- Lunch break (walking near workplace) -----
        if state == ActivityState.LUNCH_BREAK:
            self._tick_outdoor(dt, bounds, rng)
            return

    def _advance_from_timed_state(
        self,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Called when a timed state's timer reaches 0. Advance to next state."""
        state = self.activity_state

        if state == ActivityState.WAKING_UP:
            self._enter_state(ActivityState.RELAXING, rng)

        elif state == ActivityState.GETTING_IN_CAR:
            # Start driving
            if self.vehicle and self._target_location:
                self._start_driving(self._target_location, bounds, rng)
                self._enter_state(ActivityState.DRIVING, rng)
            else:
                self._walk_target = self._target_location
                self._enter_state(ActivityState.WALKING_TO_BUILDING, rng)

        elif state == ActivityState.PARKING:
            self._enter_state(ActivityState.GETTING_OUT_OF_CAR, rng)
            self._transition_timer = rng.uniform(3.0, 8.0)

        elif state == ActivityState.GETTING_OUT_OF_CAR:
            # Walk from parking to building entrance
            if self._target_building:
                self._walk_target = self._target_building.position
            elif self._target_location:
                self._walk_target = self._target_location
            self._enter_state(ActivityState.WALKING_TO_BUILDING, rng)

        elif state == ActivityState.ENTERING_BUILDING:
            # Now inside — what are we doing here?
            pending = self._pending_schedule_activity
            self._pending_schedule_activity = ""
            if pending == ActivityState.WORKING:
                self._enter_state(ActivityState.WORKING, rng)
            elif pending == ActivityState.AT_SCHOOL:
                self._enter_state(ActivityState.AT_SCHOOL, rng)
            elif pending == ActivityState.SHOPPING:
                dur = ERRAND_DURATIONS.get(self._errand_type, (300.0, 1200.0))
                self._transition_timer = rng.uniform(*dur)
                self._enter_state(ActivityState.SHOPPING, rng)
            elif pending == ActivityState.DINING:
                dur = ERRAND_DURATIONS.get(ErrandType.RESTAURANT, (1200.0, 3600.0))
                self._transition_timer = rng.uniform(*dur)
                self._enter_state(ActivityState.DINING, rng)
            elif pending == ActivityState.GETTING_COFFEE:
                dur = ERRAND_DURATIONS.get(ErrandType.COFFEE, (300.0, 600.0))
                self._transition_timer = rng.uniform(*dur)
                self._enter_state(ActivityState.GETTING_COFFEE, rng)
            elif pending == ActivityState.AT_DOCTOR:
                dur = ERRAND_DURATIONS.get(ErrandType.DOCTOR, (1800.0, 3600.0))
                self._transition_timer = rng.uniform(*dur)
                self._enter_state(ActivityState.AT_DOCTOR, rng)
            elif pending == ActivityState.DELIVERY_STOP:
                dur = ERRAND_DURATIONS.get(ErrandType.DELIVERY, (60.0, 180.0))
                self._transition_timer = rng.uniform(*dur)
                self._enter_state(ActivityState.DELIVERY_STOP, rng)
            elif self._target_location == self.home_location:
                self._enter_state(ActivityState.RELAXING, rng)
            else:
                self._enter_state(ActivityState.INSIDE_BUILDING, rng)

        elif state == ActivityState.EXITING_BUILDING:
            # Exited — start the pending outdoor activity or travel
            pending = self._pending_schedule_activity
            self._pending_schedule_activity = ""
            if pending in (ActivityState.WALKING, ActivityState.JOGGING,
                           ActivityState.WALKING_DOG, ActivityState.PLAYING):
                self._enter_state(pending, rng)
            elif pending:
                self._enter_state(pending, rng)
            else:
                self._enter_state(ActivityState.WALKING, rng)

        elif state == ActivityState.CHECKING_PHONE:
            # Resume walking
            self._phone_check_cooldown = rng.uniform(60.0, 300.0)
            self._enter_state(ActivityState.WALKING_TO_BUILDING, rng)

        elif state == ActivityState.SHOPPING:
            # Done shopping — exit building
            self._enter_state(ActivityState.EXITING_BUILDING, rng)
            self._transition_timer = rng.uniform(5.0, 15.0)
            # After exiting, need to walk to car or next destination
            self._pending_schedule_activity = ""

        elif state == ActivityState.DINING:
            self._enter_state(ActivityState.EXITING_BUILDING, rng)
            self._transition_timer = rng.uniform(5.0, 10.0)

        elif state == ActivityState.GETTING_COFFEE:
            self._enter_state(ActivityState.EXITING_BUILDING, rng)
            self._transition_timer = rng.uniform(3.0, 8.0)

        elif state == ActivityState.AT_DOCTOR:
            self._enter_state(ActivityState.EXITING_BUILDING, rng)
            self._transition_timer = rng.uniform(5.0, 15.0)

        elif state == ActivityState.DELIVERY_STOP:
            # Back in the vehicle for next delivery
            if self.vehicle:
                car_pos = self.vehicle.parked_at or self.vehicle.position
                dist_to_car = distance(self.position, car_pos)
                if dist_to_car > 3.0:
                    self._walk_target = car_pos
                    self._enter_state(ActivityState.WALKING_TO_CAR, rng)
                else:
                    self._enter_state(ActivityState.GETTING_IN_CAR, rng)
                    self._transition_timer = rng.uniform(2.0, 5.0)
                # Pick next delivery point
                lo, hi = bounds if bounds else ((0.0, 0.0), (500.0, 500.0))
                self._target_location = self._random_nearby(
                    self.position, 150.0, bounds, rng)
                self._pending_schedule_activity = ActivityState.DELIVERY_STOP
            else:
                self._enter_state(ActivityState.WALKING, rng)

        elif state == ActivityState.SOCIALIZING:
            # Resume previous activity (usually walking)
            self._enter_state(ActivityState.WALKING, rng)

        elif state == ActivityState.LUNCH_BREAK:
            # Go back to work
            pass  # schedule will drive next transition

    def _start_driving(
        self,
        destination: Vec2,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Start driving the vehicle to a destination."""
        if not self.vehicle:
            return
        specs = _VEHICLE_SPECS.get(self.vehicle.vehicle_type,
                                   (8.0, 15.0, "car"))
        speed = rng.uniform(specs[0], specs[1])
        path = self._make_road_path(self.position, destination, rng)
        self.vehicle.start_driving(path, speed)
        self.speed = 0.0  # person speed 0 while in car

    def _walk_toward(
        self,
        target: Vec2,
        dt: float,
        bounds: tuple[Vec2, Vec2],
    ) -> bool:
        """Walk toward a target point. Returns True if arrived."""
        direction = _sub(target, self.position)
        dist = magnitude(direction)

        if dist < max(self.speed * dt * 2.0, 2.0):
            self.position = target
            self.velocity = (0.0, 0.0)
            return True

        norm = normalize(direction)
        self.velocity = _scale(norm, self.speed)
        self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
        self.position = _add(self.position, _scale(self.velocity, dt))
        self.position = _clamp(self.position, bounds[0], bounds[1])
        return False

    def _wander_inside(
        self,
        dt: float,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
        radius: float = 10.0,
    ) -> None:
        """Slow random movement inside a building (e.g., browsing aisles)."""
        if self.speed <= 0:
            self.speed = rng.uniform(*_MOVE_SPEEDS["walking_slow"])

        if not self._walk_target or distance(self.position, self._walk_target) < 3.0:
            # Pick a new wander point nearby
            base = self._target_location or self.position
            self._walk_target = self._random_nearby(
                base, radius, bounds, rng
            )

        self._walk_toward(self._walk_target, dt, bounds)

    def _tick_outdoor(
        self,
        dt: float,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Tick outdoor activities (walking, jogging, playing, dog walking)."""
        # Follow outdoor path if one exists
        if self._outdoor_path and self._path_index < len(self._outdoor_path):
            target = self._outdoor_path[self._path_index]
            direction = _sub(target, self.position)
            dist = magnitude(direction)

            if dist < max(self.speed * dt * 2.0, 2.0):
                self.position = target
                self._path_index += 1
                # Dog walkers pause at waypoints
                if self.activity_state == ActivityState.WALKING_DOG:
                    self._transition_timer = rng.uniform(2.0, 8.0)
                if self._path_index >= len(self._outdoor_path):
                    if self.activity_state == ActivityState.JOGGING:
                        self._path_index = 0  # loop
                    # Others just stop
                return
            else:
                norm = normalize(direction)
                self.velocity = _scale(norm, self.speed)
                self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
                self.position = _add(self.position, _scale(self.velocity, dt))
                self.position = _clamp(self.position, bounds[0], bounds[1])

                # Random phone check
                if self._phone_check_cooldown <= 0 and rng.random() < 0.003:
                    self._phone_check_cooldown = rng.uniform(60.0, 300.0)
                    old_state = self.activity_state
                    self._enter_state(ActivityState.CHECKING_PHONE, rng)
                    self._transition_timer = rng.uniform(5.0, 15.0)
                    self._pending_schedule_activity = old_state
                return

        # No path or path exhausted — wander toward target
        if self._target_location:
            direction = _sub(self._target_location, self.position)
            dist = magnitude(direction)
            if dist < max(self.speed * dt * 2.0, 2.0):
                self.position = self._target_location
                self.velocity = (0.0, 0.0)
            else:
                norm = normalize(direction)
                self.velocity = _scale(norm, self.speed)
                self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
                self.position = _add(self.position, _scale(self.velocity, dt))
                self.position = _clamp(self.position, bounds[0], bounds[1])

    def _get_parking_pos(self, rng: random.Random) -> Vec2:
        """Get the parking position near the target building."""
        if self._target_building:
            # Park near the building's parking area with slight randomness
            pp = self._target_building.parking_pos
            return (pp[0] + rng.uniform(-3.0, 3.0),
                    pp[1] + rng.uniform(-3.0, 3.0))
        elif self._target_location:
            offset = rng.uniform(8.0, 20.0)
            angle = rng.uniform(0, 2 * math.pi)
            return (self._target_location[0] + offset * math.cos(angle),
                    self._target_location[1] + offset * math.sin(angle))
        return self.position

    def to_dict(self) -> dict:
        """Export as TargetTracker-compatible dict."""
        classification = "person"
        asset_type = "pedestrian"
        if self.role == ResidentRole.SCHOOL_KID:
            asset_type = "child"
        elif self.activity_state == ActivityState.JOGGING:
            asset_type = "jogger"
        elif self.activity_state == ActivityState.WALKING_DOG:
            asset_type = "dog_walker"
        elif self.activity_state == ActivityState.DRIVING:
            asset_type = "driver"

        move_type = state_movement_type(self.activity_state)
        is_moving = move_type in ("walking", "driving") and self.speed > 0.0

        return {
            "target_id": f"res_{self.resident_id}",
            "name": self.name,
            "source": "city_sim",
            "asset_type": asset_type,
            "alliance": "neutral",
            "classification": classification,
            "position_x": self.position[0],
            "position_y": self.position[1],
            "heading": self.heading,
            "speed": self.speed if is_moving else 0.0,
            "state": self.activity_state,
            "visible": self.visible,
            "metadata": {
                "role": self.role,
                "activity": self.current_activity,
                "activity_state": self.activity_state,
                "has_vehicle": self.vehicle is not None,
                "visible_on_map": self.visible,
                "rf_emission": state_rf_emission(self.activity_state),
                "simulated": True,
            },
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _random_nearby(
        center: Vec2, radius: float, bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> Vec2:
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(20.0, radius)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        return _clamp((x, y), bounds[0], bounds[1])

    @staticmethod
    def _make_outdoor_path(
        start: Vec2, end: Vec2, bounds: tuple[Vec2, Vec2],
        rng: random.Random, waypoints: int = 3,
    ) -> list[Vec2]:
        path: list[Vec2] = [start]
        for i in range(1, waypoints + 1):
            frac = i / (waypoints + 1)
            mx = start[0] + (end[0] - start[0]) * frac + rng.gauss(0, 15)
            my = start[1] + (end[1] - start[1]) * frac + rng.gauss(0, 15)
            path.append(_clamp((mx, my), bounds[0], bounds[1]))
        path.append(end)
        return path

    @staticmethod
    def _make_road_path(
        start: Vec2, end: Vec2, rng: random.Random,
    ) -> list[Vec2]:
        """L-shaped or Z-shaped road path for driving."""
        path: list[Vec2] = [start]
        if rng.random() < 0.5:
            path.append((end[0], start[1]))
        else:
            mid_y = start[1] + (end[1] - start[1]) * rng.uniform(0.3, 0.7)
            path.append((start[0], mid_y))
            path.append((end[0], mid_y))
        path.append(end)
        return path


# ---------------------------------------------------------------------------
# NeighborhoodSim
# ---------------------------------------------------------------------------

class NeighborhoodSim:
    """Simulates a neighborhood with N residents following deep daily routines.

    Each resident goes through micro-states: SLEEPING -> WAKING_UP ->
    WALKING_TO_CAR -> GETTING_IN_CAR -> DRIVING -> PARKING ->
    GETTING_OUT_OF_CAR -> WALKING_TO_BUILDING -> ENTERING_BUILDING ->
    WORKING -> ... and back.

    Parameters
    ----------
    num_residents : int
        Target number of residents to generate.
    bounds : tuple[Vec2, Vec2]
        (min_corner, max_corner) in local meters.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_residents: int = 50,
        bounds: tuple[Vec2, Vec2] = ((0.0, 0.0), (500.0, 500.0)),
        seed: int | None = None,
    ) -> None:
        self.num_residents = num_residents
        self.bounds = bounds
        self.residents: list[Resident] = []
        self.vehicles: list[SimVehicle] = []
        self.buildings: list[Building] = []
        self._buildings_by_type: dict[str, list[Building]] = {}
        self._rng = random.Random(seed)
        self._populated = False

    # -- public API -----------------------------------------------------------

    def populate(self, mix: dict[str, float] | None = None) -> None:
        """Generate residents with a realistic mix.

        Default: 40% office workers, 20% school kids, 15% retired,
        10% work-from-home, 10% service workers, 5% delivery.
        """
        mix = mix or dict(DEFAULT_MIX)
        # Normalize mix
        total = sum(mix.values())
        if total <= 0:
            return
        mix = {k: v / total for k, v in mix.items()}

        # Generate buildings first
        self._generate_buildings()

        # Generate residents per role
        for role, fraction in mix.items():
            count = max(1, round(self.num_residents * fraction))
            schedule_factory = _SCHEDULE_FACTORIES.get(role)
            if not schedule_factory:
                continue
            for _ in range(count):
                self._create_resident(role, schedule_factory)

        self._populated = True

    def tick(self, dt: float, current_time: float) -> None:
        """Advance simulation. current_time is hours since midnight (0-24)."""
        for resident in self.residents:
            resident.tick(
                dt, current_time, self.bounds,
                self._buildings_by_type, self._rng,
            )

    def get_all_entities(self) -> list[dict]:
        """Export all people + vehicles as TargetTracker-compatible dicts."""
        entities: list[dict] = []
        for resident in self.residents:
            entities.append(resident.to_dict())
        for vehicle in self.vehicles:
            entities.append(vehicle.to_dict())
        return entities

    def get_visible_entities(self) -> list[dict]:
        """Export only entities visible on the tactical map.

        People inside buildings are NOT visible. Parked cars are visible.
        """
        entities: list[dict] = []
        for resident in self.residents:
            if resident.visible:
                entities.append(resident.to_dict())
        for vehicle in self.vehicles:
            entities.append(vehicle.to_dict())
        return entities

    def get_statistics(self) -> dict:
        """How many sleeping, commuting, working, etc."""
        activity_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        vehicles_driving = 0
        vehicles_parked = 0
        visible_count = 0
        inside_building_count = 0

        for r in self.residents:
            activity_counts[r.current_activity] = (
                activity_counts.get(r.current_activity, 0) + 1
            )
            state_counts[r.activity_state] = (
                state_counts.get(r.activity_state, 0) + 1
            )
            role_counts[r.role] = role_counts.get(r.role, 0) + 1
            if r.visible:
                visible_count += 1
            if not state_visible_on_map(r.activity_state):
                inside_building_count += 1

        for v in self.vehicles:
            if v.driving:
                vehicles_driving += 1
            else:
                vehicles_parked += 1

        return {
            "total_residents": len(self.residents),
            "total_vehicles": len(self.vehicles),
            "vehicles_driving": vehicles_driving,
            "vehicles_parked": vehicles_parked,
            "activities": activity_counts,
            "activity_states": state_counts,
            "roles": role_counts,
            "total_buildings": len(self.buildings),
            "visible_on_map": visible_count,
            "inside_buildings": inside_building_count,
        }

    # -- building generation --------------------------------------------------

    def _generate_buildings(self) -> None:
        """Generate a neighborhood layout with homes, offices, etc."""
        lo, hi = self.bounds
        w = hi[0] - lo[0]
        h = hi[1] - lo[1]

        # Place building clusters by type
        self._place_buildings(
            BuildingType.HOME, count=max(20, self.num_residents // 2),
            region=(lo, hi), name_prefix="House",
        )
        self._place_buildings(
            BuildingType.OFFICE, count=max(3, self.num_residents // 15),
            region=((lo[0] + w * 0.5, lo[1]), (hi[0], lo[1] + h * 0.4)),
            name_prefix="Office",
        )
        self._place_buildings(
            BuildingType.SCHOOL, count=max(1, self.num_residents // 40),
            region=((lo[0], lo[1] + h * 0.3), (lo[0] + w * 0.4, lo[1] + h * 0.6)),
            name_prefix="School",
        )
        self._place_buildings(
            BuildingType.GROCERY, count=max(1, self.num_residents // 25),
            region=((lo[0] + w * 0.2, lo[1] + h * 0.6), (lo[0] + w * 0.6, hi[1])),
            name_prefix="Grocery",
        )
        self._place_buildings(
            BuildingType.PARK, count=max(2, self.num_residents // 20),
            region=(lo, hi), name_prefix="Park",
        )
        self._place_buildings(
            BuildingType.RESTAURANT, count=max(2, self.num_residents // 20),
            region=((lo[0] + w * 0.3, lo[1] + h * 0.2), (lo[0] + w * 0.8, lo[1] + h * 0.8)),
            name_prefix="Restaurant",
        )
        self._place_buildings(
            BuildingType.GAS_STATION, count=max(1, self.num_residents // 50),
            region=((lo[0] + w * 0.6, lo[1] + h * 0.7), (hi[0], hi[1])),
            name_prefix="Gas Station",
        )

    def _place_buildings(
        self,
        btype: BuildingType,
        count: int,
        region: tuple[Vec2, Vec2],
        name_prefix: str,
    ) -> None:
        """Place buildings of a type within a region."""
        rlo, rhi = region
        for i in range(count):
            pos = (
                self._rng.uniform(rlo[0], rhi[0]),
                self._rng.uniform(rlo[1], rhi[1]),
            )
            # Parking offset direction varies
            offset = _PARKING_OFFSET.get(btype, 15.0)
            angle = self._rng.uniform(0, 2 * math.pi)
            parking = (
                pos[0] + offset * math.cos(angle),
                pos[1] + offset * math.sin(angle),
            )
            b = Building(
                building_id=uuid.uuid4().hex[:8],
                building_type=btype,
                position=pos,
                name=f"{name_prefix} {i + 1}",
                parking_pos=parking,
            )
            self.buildings.append(b)
            self._buildings_by_type.setdefault(btype, []).append(b)

    # -- resident generation --------------------------------------------------

    def _create_resident(
        self, role: str, schedule_factory: callable,
    ) -> Resident:
        """Create a single resident with home, work, vehicle, schedule."""
        rid = uuid.uuid4().hex[:8]
        name = (
            f"{self._rng.choice(_FIRST_NAMES)} "
            f"{self._rng.choice(_LAST_NAMES)}"
        )

        # Assign home
        homes = self._buildings_by_type.get(BuildingType.HOME, [])
        home_building = self._rng.choice(homes) if homes else None
        home_loc = home_building.position if home_building else self._random_pos()

        # Assign work location
        work_loc: Vec2 | None = None
        if role in (ResidentRole.OFFICE_WORKER, ResidentRole.SERVICE_WORKER):
            offices = self._buildings_by_type.get(BuildingType.OFFICE, [])
            if offices:
                work_loc = self._rng.choice(offices).position
            else:
                work_loc = self._random_pos()
        elif role == ResidentRole.SCHOOL_KID:
            schools = self._buildings_by_type.get(BuildingType.SCHOOL, [])
            if schools:
                work_loc = self._rng.choice(schools).position

        # Create schedule with slight randomization (+-15min jitter)
        schedule = schedule_factory()
        jitter = self._rng.uniform(-0.25, 0.25)
        for entry in schedule.entries:
            entry.hour = max(0.0, min(23.99, entry.hour + jitter))
        schedule.entries.sort(key=lambda e: e.hour)

        # Create vehicle for roles that drive
        vehicle: SimVehicle | None = None
        if role in _VEHICLE_ROLES:
            vid = uuid.uuid4().hex[:8]
            # Vary vehicle types
            if role == ResidentRole.DELIVERY_DRIVER:
                vtype = VehicleType.DELIVERY_VAN
            elif self._rng.random() < 0.1:
                vtype = VehicleType.TRUCK
            elif self._rng.random() < 0.05:
                vtype = VehicleType.MOTORCYCLE
            else:
                vtype = VehicleType.CAR

            vehicle = SimVehicle(
                vehicle_id=vid,
                owner_id=rid,
                vehicle_type=vtype,
                position=home_loc,
                parked_at=home_loc,
            )
            self.vehicles.append(vehicle)

        resident = Resident(
            resident_id=rid,
            name=name,
            role=role,
            home_location=home_loc,
            work_location=work_loc,
            vehicle=vehicle,
            schedule=schedule,
            current_activity="sleeping",
            activity_state=ActivityState.SLEEPING,
            position=home_loc,
            visible=False,
            _has_dog=role == ResidentRole.RETIRED or self._rng.random() < 0.15,
            _grocery_day=self._rng.randint(0, 6),
            _car_parked_at=home_loc if vehicle else None,
        )
        self.residents.append(resident)
        return resident

    def _random_pos(self) -> Vec2:
        lo, hi = self.bounds
        return (
            self._rng.uniform(lo[0], hi[0]),
            self._rng.uniform(lo[1], hi[1]),
        )
