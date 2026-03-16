# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GTA-style city life simulation — daily routines for a living neighborhood.

Generates residents (office workers, school kids, retirees, delivery drivers,
service workers, work-from-home) who follow realistic daily schedules. People
wake up, commute, work, shop, walk dogs, jog, and sleep. Vehicles park at
destinations and drive between them.

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


@dataclass
class Building:
    """A location in the neighborhood."""
    building_id: str
    building_type: BuildingType
    position: Vec2
    name: str = ""
    capacity: int = 10


# ---------------------------------------------------------------------------
# Schedule system
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    """A single activity in a daily schedule.

    hour: fractional hour (0-24) when this activity starts.
    activity: what the resident is doing.
    location_type: where to go (BuildingType or 'home', 'outdoors').
    duration_hours: how long before checking next entry.
    """
    hour: float
    activity: str
    location_type: str = "home"
    duration_hours: float = 1.0


class DailySchedule:
    """Time-based activity schedule for a resident.

    Entries are sorted by hour. The schedule wraps around midnight.
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
        """Wake 6:30, leave home 7:15, arrive work 7:45, lunch 12:00,
        leave work 5:15, grocery 5:30 (random), home 6:00, walk dog 7:00,
        sleep 10:30."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 6.5),
            ScheduleEntry(6.5, "waking_up", "home", 0.75),
            ScheduleEntry(7.25, "commuting", "office", 0.5),
            ScheduleEntry(7.75, "working", "office", 4.25),
            ScheduleEntry(12.0, "lunch", "restaurant", 1.0),
            ScheduleEntry(13.0, "working", "office", 4.25),
            ScheduleEntry(17.25, "commuting", "home", 0.75),
            ScheduleEntry(18.0, "relaxing", "home", 1.0),
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
        lunch 12:00, nap 1:00, afternoon walk 3:00, home 4:00, sleep 9:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 6.0),
            ScheduleEntry(6.0, "waking_up", "home", 0.5),
            ScheduleEntry(6.5, "walking", "outdoors", 1.0),
            ScheduleEntry(7.5, "relaxing", "home", 1.5),
            ScheduleEntry(9.0, "gardening", "home", 3.0),
            ScheduleEntry(12.0, "lunch", "home", 1.0),
            ScheduleEntry(13.0, "napping", "home", 2.0),
            ScheduleEntry(15.0, "walking", "outdoors", 1.0),
            ScheduleEntry(16.0, "relaxing", "home", 5.0),
            ScheduleEntry(21.0, "sleeping", "home", 9.0),
        ])

    @classmethod
    def delivery_driver(cls) -> DailySchedule:
        """Start 8:00, drive route with stops, end 5:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 7.0),
            ScheduleEntry(7.0, "waking_up", "home", 1.0),
            ScheduleEntry(8.0, "delivering", "outdoors", 2.0),
            ScheduleEntry(10.0, "delivering", "outdoors", 2.0),
            ScheduleEntry(12.0, "lunch", "restaurant", 0.5),
            ScheduleEntry(12.5, "delivering", "outdoors", 2.0),
            ScheduleEntry(14.5, "delivering", "outdoors", 2.5),
            ScheduleEntry(17.0, "commuting", "home", 0.5),
            ScheduleEntry(17.5, "relaxing", "home", 4.5),
            ScheduleEntry(22.0, "sleeping", "home", 9.0),
        ])

    @classmethod
    def work_from_home(cls) -> DailySchedule:
        """Wake 7:30, work 8:30, lunch walk 12:00, work 1:00, done 5:30,
        jog 6:00, dinner 7:00, sleep 11:00."""
        return cls([
            ScheduleEntry(0.0, "sleeping", "home", 7.5),
            ScheduleEntry(7.5, "waking_up", "home", 1.0),
            ScheduleEntry(8.5, "working", "home", 3.5),
            ScheduleEntry(12.0, "walking", "outdoors", 1.0),
            ScheduleEntry(13.0, "working", "home", 4.5),
            ScheduleEntry(17.5, "relaxing", "home", 0.5),
            ScheduleEntry(18.0, "jogging", "outdoors", 0.75),
            ScheduleEntry(18.75, "relaxing", "home", 4.25),
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
            ScheduleEntry(11.5, "lunch", "restaurant", 0.5),
            ScheduleEntry(12.0, "working", "office", 2.5),
            ScheduleEntry(14.5, "commuting", "home", 0.5),
            ScheduleEntry(15.0, "shopping", "grocery", 1.0),
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
    "jogging": (2.5, 3.5),
    "driving": (5.0, 15.0),
    "biking": (3.0, 6.0),
    "playing": (1.0, 2.5),
}


# ---------------------------------------------------------------------------
# SimVehicle
# ---------------------------------------------------------------------------

@dataclass
class SimVehicle:
    """A simulated vehicle parked or driving."""
    vehicle_id: str
    owner_id: str
    position: Vec2 = (0.0, 0.0)
    velocity: Vec2 = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    parked_at: Vec2 | None = None
    driving: bool = False
    path: list[Vec2] = field(default_factory=list)
    path_index: int = 0

    def tick(self, dt: float, bounds: tuple[Vec2, Vec2] | None = None) -> None:
        """Advance vehicle position along its path."""
        if not self.driving or not self.path:
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
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.driving = False
                self.parked_at = self.position
                self.velocity = (0.0, 0.0)
                self.speed = 0.0
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

    def park(self, location: Vec2) -> None:
        """Park the vehicle at a location."""
        self.driving = False
        self.parked_at = location
        self.position = location
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.path = []
        self.path_index = 0

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
    """A simulated person with a daily routine."""
    resident_id: str
    name: str
    role: str  # ResidentRole value
    home_location: Vec2
    work_location: Vec2 | None = None
    vehicle: SimVehicle | None = None
    schedule: DailySchedule = field(default_factory=DailySchedule)
    current_activity: str = "sleeping"
    position: Vec2 = (0.0, 0.0)
    velocity: Vec2 = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    _target_location: Vec2 | None = None
    _transition_timer: float = 0.0
    _outdoor_path: list[Vec2] = field(default_factory=list)
    _path_index: int = 0
    _grocery_day: int = 0  # day of week for grocery run (0=Mon)
    _has_dog: bool = False

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
        # Check schedule for activity transitions
        entry = self.schedule.activity_at(current_time)
        if entry.activity != self.current_activity:
            self._begin_activity(entry, bounds, buildings, rng)

        # Move toward target location
        self._move(dt, bounds, rng)

    def _begin_activity(
        self,
        entry: ScheduleEntry,
        bounds: tuple[Vec2, Vec2],
        buildings: dict[str, list[Building]],
        rng: random.Random,
    ) -> None:
        """Transition to a new activity."""
        old_activity = self.current_activity
        self.current_activity = entry.activity
        self._outdoor_path = []
        self._path_index = 0

        loc_type = entry.location_type

        if loc_type == "home":
            self._target_location = self.home_location
        elif loc_type == "office" and self.work_location:
            self._target_location = self.work_location
        elif loc_type == "school":
            schools = buildings.get(BuildingType.SCHOOL, [])
            if schools:
                self._target_location = rng.choice(schools).position
            else:
                self._target_location = self.work_location or self.home_location
        elif loc_type == "grocery":
            stores = buildings.get(BuildingType.GROCERY, [])
            if stores:
                self._target_location = rng.choice(stores).position
            else:
                self._target_location = self.home_location
        elif loc_type == "restaurant":
            restaurants = buildings.get(BuildingType.RESTAURANT, [])
            if restaurants:
                self._target_location = rng.choice(restaurants).position
            else:
                self._target_location = self.home_location
        elif loc_type == "outdoors":
            # Generate an outdoor path for walking/jogging/playing
            parks = buildings.get(BuildingType.PARK, [])
            if parks:
                park = rng.choice(parks)
                self._target_location = park.position
            else:
                self._target_location = self._random_nearby(
                    self.home_location, 80.0, bounds, rng
                )
            # Build a wander path for outdoor activities
            if entry.activity in ("walking", "walking_dog", "jogging", "playing"):
                self._outdoor_path = self._make_outdoor_path(
                    self.position, self._target_location, bounds, rng,
                    waypoints=5 if entry.activity == "walking_dog" else 3,
                )
                self._path_index = 0
        else:
            self._target_location = self.home_location

        # Determine movement mode
        if entry.activity in ("commuting", "delivering"):
            # Drive if has vehicle and distance > 50m
            if self.vehicle and self._target_location:
                dist = distance(self.position, self._target_location)
                if dist > 50.0:
                    self._start_driving(self._target_location, bounds, rng)
                    return
            # Otherwise walk
            self.speed = rng.uniform(*_MOVE_SPEEDS["walking"])
        elif entry.activity == "jogging":
            self.speed = rng.uniform(*_MOVE_SPEEDS["jogging"])
        elif entry.activity == "playing":
            self.speed = rng.uniform(*_MOVE_SPEEDS["playing"])
        elif entry.activity in ("sleeping", "napping", "working", "at_school",
                                "relaxing", "gardening", "waking_up"):
            self.speed = 0.0
            self.velocity = (0.0, 0.0)
        else:
            self.speed = rng.uniform(*_MOVE_SPEEDS["walking"])

    def _start_driving(
        self,
        destination: Vec2,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Start driving the vehicle to a destination."""
        if not self.vehicle:
            return
        speed = rng.uniform(*_MOVE_SPEEDS["driving"])
        # Simple L-shaped road path
        path = self._make_road_path(self.position, destination, rng)
        self.vehicle.start_driving(path, speed)
        # Resident rides in vehicle
        self.speed = 0.0

    def _move(
        self,
        dt: float,
        bounds: tuple[Vec2, Vec2],
        rng: random.Random,
    ) -> None:
        """Move the resident toward their target or along a path."""
        # If riding in a vehicle, follow vehicle position
        if self.vehicle and self.vehicle.driving:
            self.vehicle.tick(dt, bounds)
            self.position = self.vehicle.position
            self.heading = self.vehicle.heading
            self.velocity = self.vehicle.velocity
            # Vehicle arrived?
            if not self.vehicle.driving:
                self.vehicle.park(self.position)
            return

        # Stationary activities
        if self.speed <= 0.0 or self.current_activity in (
            "sleeping", "napping", "working", "at_school",
            "relaxing", "gardening", "waking_up",
        ):
            self.velocity = (0.0, 0.0)
            return

        # Follow outdoor path if one exists
        if self._outdoor_path and self._path_index < len(self._outdoor_path):
            target = self._outdoor_path[self._path_index]
            direction = _sub(target, self.position)
            dist = magnitude(direction)

            if dist < self.speed * dt * 2.0:
                self.position = target
                self._path_index += 1
                # Dog walkers pause at waypoints
                if self.current_activity == "walking_dog":
                    self._transition_timer = rng.uniform(2.0, 8.0)
                if self._path_index >= len(self._outdoor_path):
                    # Loop joggers, stop others
                    if self.current_activity == "jogging":
                        self._path_index = 0
                return
            else:
                norm = normalize(direction)
                self.velocity = _scale(norm, self.speed)
                self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
                self.position = _add(self.position, _scale(self.velocity, dt))
                self.position = _clamp(self.position, bounds[0], bounds[1])
                return

        # Move toward target location
        if self._target_location:
            direction = _sub(self._target_location, self.position)
            dist = magnitude(direction)
            if dist < self.speed * dt * 2.0:
                self.position = self._target_location
                self.velocity = (0.0, 0.0)
            else:
                norm = normalize(direction)
                self.velocity = _scale(norm, self.speed)
                self.heading = math.degrees(math.atan2(norm[0], norm[1])) % 360
                self.position = _add(self.position, _scale(self.velocity, dt))
                self.position = _clamp(self.position, bounds[0], bounds[1])

    def to_dict(self) -> dict:
        """Export as TargetTracker-compatible dict."""
        classification = "person"
        asset_type = "pedestrian"
        if self.role == ResidentRole.SCHOOL_KID:
            asset_type = "child"
        elif self.current_activity == "jogging":
            asset_type = "jogger"
        elif self.current_activity == "walking_dog":
            asset_type = "dog_walker"

        is_moving = self.speed > 0.0 and self.current_activity not in (
            "sleeping", "napping", "working", "at_school",
            "relaxing", "gardening", "waking_up",
        )

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
            "state": self.current_activity,
            "metadata": {
                "role": self.role,
                "activity": self.current_activity,
                "has_vehicle": self.vehicle is not None,
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
    """Simulates a neighborhood with N residents following daily routines.

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

    def get_statistics(self) -> dict:
        """How many sleeping, commuting, working, etc."""
        activity_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        vehicles_driving = 0
        vehicles_parked = 0

        for r in self.residents:
            activity_counts[r.current_activity] = (
                activity_counts.get(r.current_activity, 0) + 1
            )
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

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
            "roles": role_counts,
            "total_buildings": len(self.buildings),
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
            b = Building(
                building_id=uuid.uuid4().hex[:8],
                building_type=btype,
                position=pos,
                name=f"{name_prefix} {i + 1}",
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

        # Create schedule with slight randomization (±15min jitter)
        schedule = schedule_factory()
        jitter = self._rng.uniform(-0.25, 0.25)
        for entry in schedule.entries:
            entry.hour = max(0.0, min(23.99, entry.hour + jitter))
        schedule.entries.sort(key=lambda e: e.hour)

        # Create vehicle for roles that drive
        vehicle: SimVehicle | None = None
        if role in _VEHICLE_ROLES:
            vid = uuid.uuid4().hex[:8]
            vehicle = SimVehicle(
                vehicle_id=vid,
                owner_id=rid,
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
            position=home_loc,
            _has_dog=role == ResidentRole.RETIRED or self._rng.random() < 0.15,
            _grocery_day=self._rng.randint(0, 6),
        )
        self.residents.append(resident)
        return resident

    def _random_pos(self) -> Vec2:
        lo, hi = self.bounds
        return (
            self._rng.uniform(lo[0], hi[0]),
            self._rng.uniform(lo[1], hi[1]),
        )
