# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Traffic vehicle simulation using IDM + MOBIL.

Edge-based city traffic model where vehicles drive along road edges,
follow leaders using IDM car-following physics, and change lanes using
MOBIL.  This is the Python equivalent of ``web/sim/vehicle.js``.

The model works as follows:
1. Each vehicle tracks its position ``u`` along a road edge.
2. IDM controls longitudinal acceleration (speed).
3. MOBIL evaluates lane changes on multi-lane roads.
4. At edge boundaries, vehicles transition to the next edge in their route.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from tritium_lib.sim_engine.idm import (
    IDMParams,
    IDM_DEFAULTS,
    ROAD_SPEEDS,
    VEHICLE_IDM_PROFILES,
    idm_acceleration,
    idm_free_flow,
    idm_step,
)
from tritium_lib.sim_engine.mobil import (
    MOBILParams,
    MOBIL_DEFAULTS,
    decide_lane_change,
)


# ---------------------------------------------------------------------------
# Road edge (simplified for traffic simulation)
# ---------------------------------------------------------------------------


@dataclass
class RoadEdge:
    """A directed road segment between two intersections.

    Attributes
    ----------
    edge_id : str
        Unique identifier.
    from_node : str
        Source intersection ID.
    to_node : str
        Destination intersection ID.
    length : float
        Length in meters.
    ax, az : float
        Start point (x, z) in world coordinates.
    bx, bz : float
        End point (x, z) in world coordinates.
    lanes_per_dir : int
        Number of lanes per travel direction.
    lane_width : float
        Width of each lane in meters.
    road_class : str
        Road classification (residential, primary, etc.).
    speed_limit : float | None
        Speed limit in m/s.  Falls back to ROAD_SPEEDS lookup.
    """

    edge_id: str
    from_node: str
    to_node: str
    length: float
    ax: float = 0.0
    az: float = 0.0
    bx: float = 0.0
    bz: float = 0.0
    lanes_per_dir: int = 1
    lane_width: float = 3.0
    road_class: str = "residential"
    speed_limit: float | None = None

    @property
    def effective_speed_limit(self) -> float:
        """Speed limit in m/s, falling back to road class default."""
        if self.speed_limit is not None:
            return self.speed_limit
        return ROAD_SPEEDS.get(self.road_class, 10.0)


# ---------------------------------------------------------------------------
# Route step
# ---------------------------------------------------------------------------


@dataclass
class RouteStep:
    """A step in a vehicle's route."""

    edge: RoadEdge
    node_id: str   # the intersection node we're heading toward


# ---------------------------------------------------------------------------
# Vehicle subtypes
# ---------------------------------------------------------------------------


class VehicleSubtype(str, Enum):
    """Vehicle subtypes with distinct physical profiles."""

    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    VAN = "van"


@dataclass
class VehicleProfile:
    """Physical profile for a vehicle subtype."""

    length: float     # meters
    width: float      # meters
    height: float     # meters
    mass: float       # kg
    idm: IDMParams


VEHICLE_PROFILES: dict[str, VehicleProfile] = {
    "sedan": VehicleProfile(
        length=4.5, width=1.8, height=1.4, mass=1400,
        idm=VEHICLE_IDM_PROFILES["sedan"],
    ),
    "suv": VehicleProfile(
        length=5.0, width=2.0, height=1.7, mass=2000,
        idm=VEHICLE_IDM_PROFILES["suv"],
    ),
    "truck": VehicleProfile(
        length=7.0, width=2.5, height=2.5, mass=5000,
        idm=VEHICLE_IDM_PROFILES["truck"],
    ),
    "motorcycle": VehicleProfile(
        length=2.2, width=0.8, height=1.2, mass=250,
        idm=VEHICLE_IDM_PROFILES["motorcycle"],
    ),
    "van": VehicleProfile(
        length=5.5, width=2.0, height=2.0, mass=2500,
        idm=VEHICLE_IDM_PROFILES["van"],
    ),
}

# Weighted subtype distribution (sedan most common)
_SUBTYPE_WEIGHTS = [
    ("sedan", 3),
    ("suv", 2),
    ("truck", 1),
    ("motorcycle", 1),
    ("van", 1),
]
_SUBTYPE_POOL = [s for s, w in _SUBTYPE_WEIGHTS for _ in range(w)]


def _random_subtype() -> str:
    return random.choice(_SUBTYPE_POOL)


# ---------------------------------------------------------------------------
# Vehicle purpose
# ---------------------------------------------------------------------------


class VehiclePurpose(str, Enum):
    """Why a vehicle exists in the simulation."""

    RANDOM = "random"
    COMMUTE = "commute"
    DELIVERY = "delivery"
    TAXI = "taxi"
    PATROL = "patrol"
    EMERGENCY = "emergency"


# ---------------------------------------------------------------------------
# Traffic vehicle
# ---------------------------------------------------------------------------

_next_id = 0


@dataclass
class TrafficVehicle:
    """A city-traffic vehicle using IDM + MOBIL physics.

    This is the Python equivalent of the JS SimVehicle in ``web/sim/vehicle.js``.
    Vehicles drive along road edges, follow leaders via IDM, and change lanes
    via MOBIL.

    Attributes
    ----------
    vehicle_id : str
        Unique identifier (e.g. ``car_0``).
    edge_id : str
        Current road edge ID.
    u : float
        Distance along current edge (0..edge.length).
    direction : int
        +1 = from->to, -1 = to->from.
    speed : float
        Current speed in m/s.
    acc : float
        Current acceleration in m/s^2.
    heading : float
        Current heading angle in radians.
    x, z : float
        World-space position.
    """

    vehicle_id: str = ""
    edge_id: str = ""
    u: float = 0.0
    direction: int = 1
    speed: float = 0.0
    acc: float = 0.0
    heading: float = 0.0
    x: float = 0.0
    z: float = 0.0
    alive: bool = True

    # Vehicle profile
    subtype: str = "sedan"
    length: float = 4.5
    width: float = 1.8
    height: float = 1.4
    mass: float = 1400.0

    # IDM parameters (per-vehicle, adjusted for subtype + road)
    idm: IDMParams = field(default_factory=IDMParams)

    # Purpose
    purpose: VehiclePurpose = VehiclePurpose.RANDOM

    # Route
    route: list[RouteStep] = field(default_factory=list)
    route_idx: int = 0

    # Lane state
    lane_idx: int = 0
    _mobil_timer: float = 0.0
    _lane_change_state: dict | None = None  # {from_lane, to_lane, t, duration}

    # Parking / accident
    parked: bool = False
    park_timer: float = 0.0
    in_accident: bool = False
    accident_timer: float = 0.0

    # Red light virtual obstacle
    _red_light_active: bool = False
    _red_light_gap: float = 0.0

    # Emergency
    is_emergency: bool = False


def create_traffic_vehicle(
    edge: RoadEdge,
    u: float = 0.0,
    direction: int = 1,
    subtype: str | None = None,
    purpose: VehiclePurpose = VehiclePurpose.RANDOM,
) -> TrafficVehicle:
    """Create a new traffic vehicle on a road edge.

    Parameters
    ----------
    edge : RoadEdge
        Starting road edge.
    u : float
        Starting position along the edge.
    direction : int
        Travel direction (+1 or -1).
    subtype : str, optional
        Vehicle subtype.  Random if not specified.
    purpose : VehiclePurpose
        Vehicle purpose / intent.

    Returns
    -------
    TrafficVehicle
    """
    global _next_id

    if subtype is None:
        subtype = _random_subtype()

    profile = VEHICLE_PROFILES.get(subtype, VEHICLE_PROFILES["sedan"])

    # Create IDM params adjusted for this road
    base_speed = edge.effective_speed_limit
    # Add +/-10% speed variation per vehicle
    variation = 0.9 + random.random() * 0.2
    idm_params = IDMParams(
        v0=base_speed * variation,
        a=profile.idm.a,
        b=profile.idm.b,
        s0=profile.idm.s0,
        T=profile.idm.T,
        delta=profile.idm.delta,
    )

    vehicle_id = f"car_{_next_id}"
    _next_id += 1

    car = TrafficVehicle(
        vehicle_id=vehicle_id,
        edge_id=edge.edge_id,
        u=u,
        direction=direction,
        subtype=subtype,
        length=profile.length,
        width=profile.width,
        height=profile.height,
        mass=profile.mass,
        idm=idm_params,
        purpose=purpose,
        lane_idx=random.randint(0, max(0, edge.lanes_per_dir - 1)),
        _mobil_timer=random.random() * 2.0,
    )

    _update_position(car, edge)
    return car


# ---------------------------------------------------------------------------
# Tick logic
# ---------------------------------------------------------------------------


def tick_vehicle(
    car: TrafficVehicle,
    edge: RoadEdge,
    dt: float,
    nearby_vehicles: list[TrafficVehicle],
) -> str | None:
    """Advance a traffic vehicle by one timestep.

    Implements the full IDM + MOBIL tick cycle:
    1. Find leader (nearest vehicle ahead in lane)
    2. MOBIL lane change evaluation (multi-lane roads)
    3. IDM acceleration
    4. Integration (speed + position)
    5. Edge transition detection

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle to update.
    edge : RoadEdge
        Current road edge the vehicle is on.
    dt : float
        Timestep in seconds.
    nearby_vehicles : list[TrafficVehicle]
        Vehicles on the same or adjacent edges for leader detection.

    Returns
    -------
    str or None
        If the vehicle needs an edge transition, returns the node ID at the
        edge boundary.  None if the vehicle stays on its current edge.
    """
    if not car.alive:
        return None

    # Parked vehicles
    if car.parked:
        car.park_timer -= dt
        if car.park_timer <= 0:
            car.parked = False
        return None

    # Accident recovery
    if car.in_accident:
        car.accident_timer -= dt
        if car.accident_timer <= 0:
            car.in_accident = False
            car.accident_timer = 0.0
        return None

    # --- 1. Find leader in current lane ---
    effective_lane = car.lane_idx
    if car._lane_change_state is not None:
        effective_lane = car._lane_change_state["to_lane"]

    leader_gap = float("inf")
    leader_speed = car.idm.v0

    for other in nearby_vehicles:
        if other is car or other.edge_id != car.edge_id:
            continue
        if other.direction != car.direction:
            continue
        if other.lane_idx != effective_lane:
            continue

        gap = (other.u - car.u) * car.direction
        if gap > 0 and gap < leader_gap:
            leader_gap = gap - (car.length + other.length) / 2
            leader_speed = other.speed

    # Check red light
    if car._red_light_active and car._red_light_gap > 0 and not car.is_emergency:
        if car._red_light_gap < leader_gap:
            leader_gap = car._red_light_gap
            leader_speed = 0.0

    # --- 2. MOBIL lane change ---
    if edge.lanes_per_dir > 1 and car._lane_change_state is None:
        car._mobil_timer -= dt
        if car._mobil_timer <= 0:
            car._mobil_timer = 1.5 + random.random()
            decision = decide_lane_change(
                car, nearby_vehicles, edge.lanes_per_dir,
            )
            if decision.target_lane is not None:
                car._lane_change_state = {
                    "from_lane": car.lane_idx,
                    "to_lane": decision.target_lane,
                    "t": 0.0,
                    "duration": 2.0,
                }

    # Animate lane change
    if car._lane_change_state is not None:
        lcs = car._lane_change_state
        lcs["t"] += dt / lcs["duration"]
        if lcs["t"] >= 1.0:
            car.lane_idx = lcs["to_lane"]
            car._lane_change_state = None

    # --- 3. IDM acceleration ---
    if leader_gap < float("inf") and leader_gap > 0:
        car.acc = idm_acceleration(car.speed, leader_gap, leader_speed, car.idm)
    else:
        car.acc = idm_free_flow(car.speed, car.idm)

    # Brake near edge end if route is exhausted
    remaining = (edge.length - car.u) if car.direction > 0 else car.u
    if remaining < 5 and remaining > 0:
        brake_acc = -(car.speed * car.speed) / (2.0 * max(remaining, 0.5))
        car.acc = min(car.acc, max(brake_acc, -4.0))

    # --- 4. Integration ---
    result = idm_step(car.speed, car.acc, dt)
    car.speed = result.v
    car.u += result.ds * car.direction

    # --- 5. Edge transition check ---
    transition_node = None
    if car.direction > 0 and car.u >= edge.length:
        transition_node = edge.to_node
    elif car.direction < 0 and car.u <= 0:
        transition_node = edge.from_node

    # Update position
    _update_position(car, edge)

    return transition_node


def advance_to_next_edge(
    car: TrafficVehicle,
    next_edge: RoadEdge,
    arrival_node: str,
) -> None:
    """Transition a vehicle to the next edge in its route.

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle.
    next_edge : RoadEdge
        The new road edge.
    arrival_node : str
        The node where we arrived (determines travel direction on new edge).
    """
    # Determine direction on new edge based on which end we arrived at
    if next_edge.from_node == arrival_node:
        car.direction = 1
        car.u = 0.0
    elif next_edge.to_node == arrival_node:
        car.direction = -1
        car.u = next_edge.length
    else:
        # Edge doesn't connect -- reverse direction as fallback
        car.direction *= -1
        car.u = max(0.0, min(car.u, next_edge.length))

    car.edge_id = next_edge.edge_id

    # Adjust speed for new road class
    base_speed = next_edge.effective_speed_limit
    variation = 0.9 + random.random() * 0.2
    car.idm = IDMParams(
        v0=base_speed * variation,
        a=car.idm.a,
        b=car.idm.b,
        s0=car.idm.s0,
        T=car.idm.T,
        delta=car.idm.delta,
    )

    # Clamp lane index
    car.lane_idx = min(car.lane_idx, max(0, next_edge.lanes_per_dir - 1))
    car._lane_change_state = None

    _update_position(car, next_edge)


def set_red_light(car: TrafficVehicle, active: bool, gap: float = 0.0) -> None:
    """Set virtual red-light obstacle for IDM.

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle.
    active : bool
        Whether a red light is ahead.
    gap : float
        Distance to the stop line in meters.
    """
    car._red_light_active = active
    car._red_light_gap = gap


def park_vehicle(car: TrafficVehicle, duration: float = 60.0) -> None:
    """Park a vehicle for a specified duration.

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle to park.
    duration : float
        How long to stay parked in seconds.
    """
    car.parked = True
    car.park_timer = duration
    car.speed = 0.0
    car.acc = 0.0


# ---------------------------------------------------------------------------
# Position update
# ---------------------------------------------------------------------------


def _update_position(car: TrafficVehicle, edge: RoadEdge) -> None:
    """Update world-space x, z, heading from edge position.

    Uses linear interpolation along the edge endpoints (straight road).

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle.
    edge : RoadEdge
        The road edge.
    """
    t = max(0.0, min(1.0, car.u / edge.length)) if edge.length > 0 else 0.0
    car.x = edge.ax + t * (edge.bx - edge.ax)
    car.z = edge.az + t * (edge.bz - edge.az)

    dx = edge.bx - edge.ax
    dz = edge.bz - edge.az
    car.heading = math.atan2(dx, dz)
    if car.direction < 0:
        car.heading += math.pi


# ---------------------------------------------------------------------------
# Traffic manager
# ---------------------------------------------------------------------------


class TrafficManager:
    """Manages a collection of traffic vehicles using IDM + MOBIL.

    Provides a high-level tick method that updates all vehicles,
    handles edge transitions, and manages the vehicle lifecycle.

    Parameters
    ----------
    edges : dict[str, RoadEdge]
        Map of edge_id to RoadEdge.
    """

    def __init__(self, edges: dict[str, RoadEdge] | None = None) -> None:
        self.edges: dict[str, RoadEdge] = edges or {}
        self.vehicles: dict[str, TrafficVehicle] = {}
        self._adjacency: dict[str, list[str]] = {}  # node -> [edge_ids]

    def add_edge(self, edge: RoadEdge) -> None:
        """Register a road edge."""
        self.edges[edge.edge_id] = edge
        # Build adjacency index
        self._adjacency.setdefault(edge.from_node, []).append(edge.edge_id)
        self._adjacency.setdefault(edge.to_node, []).append(edge.edge_id)

    def spawn_vehicle(
        self,
        edge_id: str,
        u: float = 0.0,
        direction: int = 1,
        subtype: str | None = None,
        purpose: VehiclePurpose = VehiclePurpose.RANDOM,
    ) -> TrafficVehicle:
        """Spawn a new vehicle on a road edge.

        Parameters
        ----------
        edge_id : str
            Road edge to spawn on.
        u : float
            Position along edge.
        direction : int
            Travel direction.
        subtype : str, optional
            Vehicle subtype.
        purpose : VehiclePurpose
            Vehicle purpose.

        Returns
        -------
        TrafficVehicle

        Raises
        ------
        KeyError
            If edge_id is not registered.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Unknown edge: {edge_id!r}")

        edge = self.edges[edge_id]
        car = create_traffic_vehicle(edge, u, direction, subtype, purpose)
        self.vehicles[car.vehicle_id] = car
        return car

    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle."""
        self.vehicles.pop(vehicle_id, None)

    def tick(self, dt: float) -> list[str]:
        """Advance all vehicles by one timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds.

        Returns
        -------
        list[str]
            Vehicle IDs that completed edge transitions.
        """
        transitioned: list[str] = []

        # Collect all vehicles as a list for neighbor queries
        all_vehicles = list(self.vehicles.values())

        for car in all_vehicles:
            if car.edge_id not in self.edges:
                continue

            edge = self.edges[car.edge_id]

            # Gather nearby vehicles (same edge for now)
            nearby = [
                v for v in all_vehicles
                if v.edge_id == car.edge_id and v is not car
            ]

            transition_node = tick_vehicle(car, edge, dt, nearby)

            if transition_node is not None:
                # Find next edge from route or pick randomly
                next_edge = self._get_next_edge(car, transition_node)
                if next_edge is not None:
                    advance_to_next_edge(car, next_edge, transition_node)
                    transitioned.append(car.vehicle_id)
                else:
                    # No exit -- reverse direction
                    car.direction *= -1
                    car.u = max(0.0, min(car.u, edge.length))

        return transitioned

    def _get_next_edge(
        self,
        car: TrafficVehicle,
        node_id: str,
    ) -> RoadEdge | None:
        """Get the next edge for a vehicle from its route or randomly.

        Parameters
        ----------
        car : TrafficVehicle
            The vehicle.
        node_id : str
            The intersection node we arrived at.

        Returns
        -------
        RoadEdge or None
        """
        # Check route
        if car.route_idx < len(car.route):
            step = car.route[car.route_idx]
            car.route_idx += 1
            return step.edge

        # Random exit: pick a connected edge that isn't the one we came from
        connected_edge_ids = self._adjacency.get(node_id, [])
        candidates = [
            eid for eid in connected_edge_ids
            if eid != car.edge_id and eid in self.edges
        ]

        if not candidates:
            return None

        next_edge_id = random.choice(candidates)
        return self.edges[next_edge_id]

    @property
    def vehicle_count(self) -> int:
        """Number of active vehicles."""
        return len(self.vehicles)

    def get_vehicles_on_edge(self, edge_id: str) -> list[TrafficVehicle]:
        """Get all vehicles on a specific edge."""
        return [v for v in self.vehicles.values() if v.edge_id == edge_id]

    def to_dict(self) -> dict:
        """Export traffic state as a dict for JSON serialization."""
        return {
            "vehicle_count": self.vehicle_count,
            "vehicles": [
                {
                    "id": v.vehicle_id,
                    "x": v.x,
                    "z": v.z,
                    "heading": v.heading,
                    "speed": v.speed,
                    "acc": v.acc,
                    "subtype": v.subtype,
                    "lane": v.lane_idx,
                    "edge": v.edge_id,
                    "parked": v.parked,
                    "purpose": v.purpose.value if isinstance(v.purpose, Enum) else v.purpose,
                }
                for v in self.vehicles.values()
            ],
        }
