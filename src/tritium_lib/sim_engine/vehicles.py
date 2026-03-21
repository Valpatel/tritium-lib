# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Vehicle and drone combat simulation with Three.js-compatible output.

Simulates military vehicles and drones with realistic movement, weapons,
damage, convoy behavior, and autonomous drone control. All output dicts
are structured for direct consumption by a Three.js frontend renderer.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    normalize,
    magnitude,
    _sub,
    _add,
    _scale,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VehicleClass(Enum):
    """Categories of vehicles in the simulation."""

    CAR = "car"
    TRUCK = "truck"
    APC = "apc"
    TANK = "tank"
    HELICOPTER = "helicopter"
    DRONE_QUAD = "drone_quad"
    DRONE_FIXED_WING = "drone_fixed_wing"
    BOAT = "boat"
    MOTORCYCLE = "motorcycle"


# Aircraft classes that can change altitude
_AIRCRAFT = {VehicleClass.HELICOPTER, VehicleClass.DRONE_QUAD, VehicleClass.DRONE_FIXED_WING}

# Fixed-wing aircraft need minimum speed to stay airborne
_FIXED_WING = {VehicleClass.DRONE_FIXED_WING}

# Classes that can hover (zero forward speed while airborne)
_HOVER_CAPABLE = {VehicleClass.HELICOPTER, VehicleClass.DRONE_QUAD}


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------


@dataclass
class VehicleState:
    """Mutable state of a single vehicle in the simulation."""

    vehicle_id: str
    name: str
    vehicle_class: VehicleClass
    alliance: str
    position: Vec2
    altitude: float = 0.0  # meters above ground, >0 for aircraft
    heading: float = 0.0  # radians, 0 = +x (east)
    speed: float = 0.0  # m/s current forward speed
    max_speed: float = 30.0  # m/s
    acceleration: float = 5.0  # m/s^2
    turn_rate: float = 1.0  # rad/s max
    health: float = 100.0
    max_health: float = 100.0
    armor: float = 0.0  # 0-1 damage reduction fraction
    fuel: float = 1.0  # 0-1 fraction remaining
    fuel_consumption: float = 0.01  # per second at max speed
    crew: int = 1
    passengers: list[str] = field(default_factory=list)  # unit IDs riding inside
    weapons: list[str] = field(default_factory=list)  # weapon_ids from arsenal
    is_destroyed: bool = False

    # Critical hit flags (internal state)
    engine_disabled: bool = False
    weapons_disabled: bool = False
    fuel_leak: bool = False

    def is_aircraft(self) -> bool:
        """True if this vehicle can fly."""
        return self.vehicle_class in _AIRCRAFT

    def is_hover_capable(self) -> bool:
        """True if this vehicle can hover in place while airborne."""
        return self.vehicle_class in _HOVER_CAPABLE

    def is_fixed_wing(self) -> bool:
        """True if this vehicle needs minimum speed to stay airborne."""
        return self.vehicle_class in _FIXED_WING

    @property
    def min_flight_speed(self) -> float:
        """Minimum speed for fixed-wing aircraft to stay airborne."""
        if self.is_fixed_wing():
            return self.max_speed * 0.25
        return 0.0


# ---------------------------------------------------------------------------
# Vehicle physics engine
# ---------------------------------------------------------------------------


class VehiclePhysicsEngine:
    """Stateless physics engine for updating vehicle state each tick.

    Handles acceleration, turning, fuel consumption, altitude changes,
    and minimum speed constraints for fixed-wing aircraft.
    """

    @staticmethod
    def update(
        vehicle: VehicleState,
        throttle: float,
        steering: float,
        dt: float,
        altitude_input: float = 0.0,
    ) -> VehicleState:
        """Advance vehicle physics by *dt* seconds.

        Parameters
        ----------
        vehicle : VehicleState
            Current vehicle state (mutated in place and returned).
        throttle : float
            -1 (full reverse) to 1 (full forward).
        steering : float
            -1 (full left) to 1 (full right).
        dt : float
            Time step in seconds.
        altitude_input : float
            -1 (descend) to 1 (ascend), only applies to aircraft.

        Returns
        -------
        VehicleState
            The same object, mutated.
        """
        if vehicle.is_destroyed:
            vehicle.speed = 0.0
            return vehicle

        # Clamp inputs
        throttle = max(-1.0, min(1.0, throttle))
        steering = max(-1.0, min(1.0, steering))
        altitude_input = max(-1.0, min(1.0, altitude_input))

        # Engine disabled -- no acceleration, speed decays
        if vehicle.engine_disabled:
            vehicle.speed *= max(0.0, 1.0 - 2.0 * dt)  # rapid decel
            if abs(vehicle.speed) < 0.01:
                vehicle.speed = 0.0
        else:
            # Apply throttle
            if throttle >= 0:
                vehicle.speed += throttle * vehicle.acceleration * dt
            else:
                # Braking / reverse
                if vehicle.speed > 0:
                    vehicle.speed -= abs(throttle) * vehicle.acceleration * 1.5 * dt
                    vehicle.speed = max(vehicle.speed, 0.0)
                else:
                    # Reverse at half acceleration
                    vehicle.speed += throttle * vehicle.acceleration * 0.5 * dt

        # Clamp speed
        max_reverse = vehicle.max_speed * 0.3
        vehicle.speed = max(-max_reverse, min(vehicle.speed, vehicle.max_speed))

        # Fixed-wing stall check -- if airborne and too slow, lose altitude
        if vehicle.is_fixed_wing() and vehicle.altitude > 0:
            if vehicle.speed < vehicle.min_flight_speed and not vehicle.engine_disabled:
                # Force minimum speed to stay airborne
                vehicle.speed = max(vehicle.speed, vehicle.min_flight_speed * 0.5)
                # Lose altitude when stalling
                vehicle.altitude = max(0.0, vehicle.altitude - 5.0 * dt)

        # Steering -- apply turn rate
        if abs(vehicle.speed) > 0.01:
            turn = steering * vehicle.turn_rate * dt
            vehicle.heading += turn
            # Normalize heading to [0, 2*pi)
            vehicle.heading = vehicle.heading % (2 * math.pi)

        # Altitude changes (aircraft only)
        if vehicle.is_aircraft() and not vehicle.engine_disabled:
            climb_rate = 10.0  # m/s max climb/descend rate
            vehicle.altitude += altitude_input * climb_rate * dt
            vehicle.altitude = max(0.0, vehicle.altitude)

        # Update position
        dx = vehicle.speed * math.cos(vehicle.heading) * dt
        dy = vehicle.speed * math.sin(vehicle.heading) * dt
        vehicle.position = (vehicle.position[0] + dx, vehicle.position[1] + dy)

        # Fuel consumption (proportional to speed)
        if vehicle.fuel > 0 and abs(vehicle.speed) > 0.01:
            speed_fraction = abs(vehicle.speed) / vehicle.max_speed
            consumption = vehicle.fuel_consumption * speed_fraction * dt
            if vehicle.fuel_leak:
                consumption *= 3.0  # fuel leak triples consumption
            vehicle.fuel = max(0.0, vehicle.fuel - consumption)
            if vehicle.fuel <= 0:
                vehicle.engine_disabled = True

        return vehicle

    @staticmethod
    def apply_damage(
        vehicle: VehicleState,
        damage: float,
        hit_pos: Vec2 | None = None,
    ) -> dict:
        """Apply damage to a vehicle, accounting for armor.

        Returns a damage report dict suitable for Three.js effect rendering.
        """
        if vehicle.is_destroyed or damage <= 0:
            return {"effects": [], "damage_dealt": 0.0, "destroyed": False}

        # Armor reduces damage
        actual_damage = damage * (1.0 - vehicle.armor)
        actual_damage = max(0.0, actual_damage)
        vehicle.health -= actual_damage

        effects: list[dict] = []
        critical_hits: list[str] = []

        # Sparks on any hit
        effects.append({
            "type": "sparks",
            "position": hit_pos or vehicle.position,
            "intensity": min(1.0, actual_damage / 50.0),
        })

        # Critical hit check -- 15% base chance, higher with more damage
        crit_chance = 0.15 + (actual_damage / vehicle.max_health) * 0.3
        if random.random() < crit_chance and actual_damage > 0:
            crit_roll = random.random()
            if crit_roll < 0.33 and not vehicle.engine_disabled:
                vehicle.engine_disabled = True
                critical_hits.append("engine")
                effects.append({
                    "type": "smoke",
                    "position": vehicle.position,
                    "duration": 10.0,
                    "color": "black",
                })
            elif crit_roll < 0.66 and not vehicle.weapons_disabled:
                vehicle.weapons_disabled = True
                critical_hits.append("weapons")
            else:
                if not vehicle.fuel_leak:
                    vehicle.fuel_leak = True
                    critical_hits.append("fuel")
                    effects.append({
                        "type": "fire",
                        "position": vehicle.position,
                        "duration": 15.0,
                        "intensity": 0.6,
                    })

        # Destruction check
        destroyed = False
        if vehicle.health <= 0:
            vehicle.health = 0.0
            vehicle.is_destroyed = True
            vehicle.speed = 0.0
            destroyed = True
            effects.append({
                "type": "explosion",
                "position": vehicle.position,
                "radius": 5.0 + vehicle.max_health / 100.0,
                "duration": 2.0,
            })
            # Fire effect on wreckage
            effects.append({
                "type": "fire",
                "position": vehicle.position,
                "duration": 30.0,
                "intensity": 1.0,
            })

        return {
            "vehicle_id": vehicle.vehicle_id,
            "damage_dealt": actual_damage,
            "armor_absorbed": damage - actual_damage,
            "health_remaining": vehicle.health,
            "critical_hits": critical_hits,
            "destroyed": destroyed,
            "effects": effects,
        }


# ---------------------------------------------------------------------------
# Drone controller
# ---------------------------------------------------------------------------


class DroneController:
    """Autonomous controller for drone vehicles.

    Provides patrol, pursuit, orbit, and return-to-base behaviors.
    Each tick produces throttle/steering commands for the physics engine.
    """

    def __init__(self, drone: VehicleState) -> None:
        if drone.vehicle_class not in (VehicleClass.DRONE_QUAD, VehicleClass.DRONE_FIXED_WING):
            raise ValueError(f"DroneController requires a drone, got {drone.vehicle_class}")
        self.drone = drone
        self.waypoints: list[Vec2] = []
        self.mode: str = "idle"  # patrol, pursue, orbit, rtb, idle
        self.target_id: str | None = None
        self._waypoint_idx: int = 0
        self._orbit_center: Vec2 = (0.0, 0.0)
        self._orbit_radius: float = 50.0
        self._orbit_altitude: float = 30.0
        self._pursue_target: Vec2 = (0.0, 0.0)
        self._base_pos: Vec2 = (0.0, 0.0)
        self._waypoint_threshold: float = 5.0  # meters to consider waypoint reached

    def set_patrol(self, waypoints: list[Vec2]) -> None:
        """Set patrol route. Drone cycles through waypoints."""
        self.waypoints = list(waypoints)
        self._waypoint_idx = 0
        self.mode = "patrol"

    def pursue(self, target_pos: Vec2) -> None:
        """Set pursuit mode toward a target position."""
        self._pursue_target = target_pos
        self.mode = "pursue"

    def orbit(self, center: Vec2, radius: float, altitude: float) -> None:
        """Orbit around a center point at given radius and altitude."""
        self._orbit_center = center
        self._orbit_radius = radius
        self._orbit_altitude = altitude
        self.mode = "orbit"

    def return_to_base(self, base_pos: Vec2) -> None:
        """Return to base position."""
        self._base_pos = base_pos
        self.mode = "rtb"

    def tick(self, dt: float) -> tuple[float, float, float]:
        """Compute control inputs for this tick.

        Returns
        -------
        tuple[float, float, float]
            (throttle, steering, altitude_input) commands.
        """
        if self.drone.is_destroyed:
            return (0.0, 0.0, 0.0)

        if self.mode == "idle":
            # Hover/loiter in place
            if self.drone.is_hover_capable():
                return (0.0, 0.0, 0.0)
            else:
                # Fixed-wing must keep moving -- gentle circle
                return (0.5, 0.3, 0.0)

        if self.mode == "patrol":
            return self._tick_patrol(dt)
        elif self.mode == "pursue":
            return self._tick_seek(self._pursue_target, dt)
        elif self.mode == "orbit":
            return self._tick_orbit(dt)
        elif self.mode == "rtb":
            return self._tick_seek(self._base_pos, dt)

        return (0.0, 0.0, 0.0)

    def _tick_patrol(self, dt: float) -> tuple[float, float, float]:
        """Follow waypoints in order, cycling back to start."""
        if not self.waypoints:
            self.mode = "idle"
            return (0.0, 0.0, 0.0)

        target = self.waypoints[self._waypoint_idx]
        dist = distance(self.drone.position, target)

        if dist < self._waypoint_threshold:
            self._waypoint_idx = (self._waypoint_idx + 1) % len(self.waypoints)
            target = self.waypoints[self._waypoint_idx]

        return self._steer_toward(target, dt)

    def _tick_seek(self, target: Vec2, dt: float) -> tuple[float, float, float]:
        """Fly directly toward a target position."""
        return self._steer_toward(target, dt)

    def _tick_orbit(self, dt: float) -> tuple[float, float, float]:
        """Orbit around center point at desired radius."""
        # Compute tangent point on orbit circle
        to_center = _sub(self._orbit_center, self.drone.position)
        dist_to_center = magnitude(to_center)

        if dist_to_center < 1e-6:
            # At center -- move outward first
            return (0.8, 0.0, 0.0)

        # Normalized direction to center
        dir_to_center = normalize(to_center)

        # Perpendicular (tangent to orbit, counter-clockwise)
        tangent = (-dir_to_center[1], dir_to_center[0])

        # Blend: tangent force + radial correction
        radial_error = dist_to_center - self._orbit_radius
        correction_strength = max(-1.0, min(1.0, radial_error / self._orbit_radius))

        # Target direction: mostly tangent, with radial correction
        target_dir = _add(
            _scale(tangent, 0.8),
            _scale(dir_to_center, 0.2 * correction_strength),
        )
        target_dir = normalize(target_dir)

        # Convert to a world-space target point ahead
        look_ahead = 20.0
        target_pos = _add(self.drone.position, _scale(target_dir, look_ahead))

        throttle, steering, _ = self._steer_toward(target_pos, dt)

        # Altitude correction
        alt_error = self._orbit_altitude - self.drone.altitude
        alt_input = max(-1.0, min(1.0, alt_error * 0.1))

        return (throttle, steering, alt_input)

    def _steer_toward(self, target: Vec2, dt: float) -> tuple[float, float, float]:
        """Compute throttle and steering to drive toward a target point."""
        dx = target[0] - self.drone.position[0]
        dy = target[1] - self.drone.position[1]
        desired_heading = math.atan2(dy, dx)

        # Angle difference, wrapped to [-pi, pi]
        angle_diff = desired_heading - self.drone.heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Steering proportional to angle error
        steering = max(-1.0, min(1.0, angle_diff / (math.pi * 0.5)))

        # Throttle: full speed unless very close or turning sharply
        dist = distance(self.drone.position, target)
        if dist < 10.0:
            throttle = max(0.2, dist / 10.0)
        else:
            throttle = 1.0 - 0.3 * abs(steering)

        # Fixed-wing needs minimum throttle
        if self.drone.is_fixed_wing():
            throttle = max(0.4, throttle)

        # Default: maintain current altitude
        alt_input = 0.0
        if self.drone.altitude < 10.0 and self.drone.is_aircraft():
            alt_input = 0.5  # climb if too low

        return (throttle, steering, alt_input)

    def to_three_js(self) -> dict:
        """Export drone state and path for Three.js visualization."""
        return {
            "type": "drone",
            "id": self.drone.vehicle_id,
            "position": {
                "x": self.drone.position[0],
                "y": self.drone.position[1],
                "z": self.drone.altitude,
            },
            "heading": self.drone.heading,
            "speed": self.drone.speed,
            "mode": self.mode,
            "vehicle_class": self.drone.vehicle_class.value,
            "health": self.drone.health / self.drone.max_health,
            "fuel": self.drone.fuel,
            "waypoints": [{"x": w[0], "y": w[1]} for w in self.waypoints],
            "current_waypoint_idx": self._waypoint_idx,
            "orbit": {
                "center": {"x": self._orbit_center[0], "y": self._orbit_center[1]},
                "radius": self._orbit_radius,
                "altitude": self._orbit_altitude,
            } if self.mode == "orbit" else None,
            "destroyed": self.drone.is_destroyed,
        }


# ---------------------------------------------------------------------------
# Convoy simulator
# ---------------------------------------------------------------------------


class ConvoySimulator:
    """Simulates a column of vehicles following a route in formation.

    Vehicles maintain spacing and can react to ambush scenarios.
    """

    def __init__(self, vehicles: list[VehicleState], route: list[Vec2]) -> None:
        if not vehicles:
            raise ValueError("Convoy requires at least one vehicle")
        if len(route) < 2:
            raise ValueError("Convoy route requires at least 2 waypoints")
        self.vehicles = list(vehicles)
        self.route = list(route)
        self.spacing: float = 15.0  # meters between vehicles
        self._physics = VehiclePhysicsEngine()
        self._waypoint_indices: list[int] = [0] * len(vehicles)
        self._is_ambushed: bool = False
        self._scatter_mode: bool = False
        self._waypoint_threshold: float = 8.0

    def tick(self, dt: float) -> None:
        """Advance all vehicles along the route for one time step."""
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_destroyed:
                continue

            if self._scatter_mode:
                # Scatter: each vehicle picks a random direction away from convoy center
                self._tick_scatter(vehicle, dt)
                continue

            # Lead vehicle follows route directly
            if i == 0:
                throttle, steering = self._follow_route(vehicle, i, dt)
            else:
                # Following vehicles maintain spacing behind the vehicle ahead
                leader = self.vehicles[i - 1]
                dist_to_leader = distance(vehicle.position, leader.position)
                desired_dist = self.spacing

                if dist_to_leader > desired_dist * 2.0:
                    # Too far behind -- speed up
                    throttle, steering = self._steer_toward_vehicle(vehicle, leader)
                    throttle = min(1.0, throttle * 1.5)
                elif dist_to_leader < desired_dist * 0.5:
                    # Too close -- slow down
                    throttle = 0.1
                    steering = 0.0
                else:
                    # Normal following
                    throttle, steering = self._follow_route(vehicle, i, dt)
                    # Adjust speed based on spacing
                    speed_factor = dist_to_leader / desired_dist
                    throttle *= max(0.3, min(1.2, speed_factor))

            self._physics.update(vehicle, throttle, steering, dt)

    def ambush(self, attacker_positions: list[Vec2]) -> None:
        """React to an ambush from given positions.

        Convoy stops, then vehicles with weapons return fire direction.
        After a delay, convoy scatters if outnumbered.
        """
        self._is_ambushed = True

        if not attacker_positions:
            return

        # Calculate centroid of attackers
        ax = sum(p[0] for p in attacker_positions) / len(attacker_positions)
        ay = sum(p[1] for p in attacker_positions) / len(attacker_positions)
        threat_center = (ax, ay)

        # Each vehicle turns toward threat
        for vehicle in self.vehicles:
            if vehicle.is_destroyed:
                continue
            dx = threat_center[0] - vehicle.position[0]
            dy = threat_center[1] - vehicle.position[1]
            vehicle.heading = math.atan2(dy, dx)
            vehicle.speed = 0.0  # Stop

        # If heavily outnumbered, scatter
        active_vehicles = sum(1 for v in self.vehicles if not v.is_destroyed)
        if len(attacker_positions) > active_vehicles * 2:
            self._scatter_mode = True

    def to_three_js(self) -> dict:
        """Export convoy state for Three.js visualization."""
        return {
            "type": "convoy",
            "vehicles": [
                {
                    "id": v.vehicle_id,
                    "position": {"x": v.position[0], "y": v.position[1], "z": v.altitude},
                    "heading": v.heading,
                    "speed": v.speed,
                    "vehicle_class": v.vehicle_class.value,
                    "health": v.health / v.max_health,
                    "destroyed": v.is_destroyed,
                    "alliance": v.alliance,
                }
                for v in self.vehicles
            ],
            "route": [{"x": p[0], "y": p[1]} for p in self.route],
            "spacing": self.spacing,
            "is_ambushed": self._is_ambushed,
            "scatter_mode": self._scatter_mode,
        }

    # -- internal helpers ---------------------------------------------------

    def _follow_route(
        self, vehicle: VehicleState, index: int, dt: float
    ) -> tuple[float, float]:
        """Steer vehicle along the route waypoints."""
        wi = self._waypoint_indices[index]
        if wi >= len(self.route):
            return (0.0, 0.0)  # reached end

        target = self.route[wi]
        dist = distance(vehicle.position, target)

        if dist < self._waypoint_threshold and wi < len(self.route) - 1:
            self._waypoint_indices[index] += 1
            target = self.route[self._waypoint_indices[index]]

        return self._compute_steering(vehicle, target)

    def _steer_toward_vehicle(
        self, vehicle: VehicleState, leader: VehicleState
    ) -> tuple[float, float]:
        """Steer toward another vehicle's position."""
        return self._compute_steering(vehicle, leader.position)

    def _compute_steering(
        self, vehicle: VehicleState, target: Vec2
    ) -> tuple[float, float]:
        """Compute throttle and steering toward a target point."""
        dx = target[0] - vehicle.position[0]
        dy = target[1] - vehicle.position[1]
        desired_heading = math.atan2(dy, dx)

        angle_diff = desired_heading - vehicle.heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        steering = max(-1.0, min(1.0, angle_diff / (math.pi * 0.5)))
        throttle = 0.7 - 0.2 * abs(steering)

        return (throttle, steering)

    def _tick_scatter(self, vehicle: VehicleState, dt: float) -> None:
        """Move vehicle away from convoy center in a random-ish direction."""
        # Compute convoy center
        active = [v for v in self.vehicles if not v.is_destroyed]
        if not active:
            return
        cx = sum(v.position[0] for v in active) / len(active)
        cy = sum(v.position[1] for v in active) / len(active)

        # Move away from center
        dx = vehicle.position[0] - cx
        dy = vehicle.position[1] - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx, dy = 1.0, 0.0

        away_heading = math.atan2(dy, dx)
        target = (
            vehicle.position[0] + math.cos(away_heading) * 50.0,
            vehicle.position[1] + math.sin(away_heading) * 50.0,
        )

        throttle, steering = self._compute_steering(vehicle, target)
        self._physics.update(vehicle, max(throttle, 0.8), steering, dt)


# ---------------------------------------------------------------------------
# Vehicle templates
# ---------------------------------------------------------------------------

VEHICLE_TEMPLATES: dict[str, dict] = {
    "humvee": {
        "vehicle_class": VehicleClass.CAR,
        "max_speed": 100.0 / 3.6,  # 100 km/h -> m/s
        "acceleration": 4.0,
        "turn_rate": 1.2,
        "max_health": 500.0,
        "armor": 0.2,
        "fuel_consumption": 0.008,
        "crew": 2,
        "weapons": ["m2_turret"],
    },
    "technical": {
        "vehicle_class": VehicleClass.TRUCK,
        "max_speed": 80.0 / 3.6,
        "acceleration": 3.0,
        "turn_rate": 0.9,
        "max_health": 200.0,
        "armor": 0.0,
        "fuel_consumption": 0.01,
        "crew": 2,
        "weapons": ["pkm_mounted"],
    },
    "btr80": {
        "vehicle_class": VehicleClass.APC,
        "max_speed": 80.0 / 3.6,
        "acceleration": 3.5,
        "turn_rate": 0.7,
        "max_health": 1000.0,
        "armor": 0.5,
        "fuel_consumption": 0.015,
        "crew": 3,
        "weapons": ["30mm_cannon"],
    },
    "t72": {
        "vehicle_class": VehicleClass.TANK,
        "max_speed": 60.0 / 3.6,
        "acceleration": 2.5,
        "turn_rate": 0.5,
        "max_health": 2000.0,
        "armor": 0.8,
        "fuel_consumption": 0.025,
        "crew": 3,
        "weapons": ["125mm_cannon", "coax_mg"],
    },
    "blackhawk": {
        "vehicle_class": VehicleClass.HELICOPTER,
        "max_speed": 250.0 / 3.6,
        "acceleration": 6.0,
        "turn_rate": 1.5,
        "max_health": 500.0,
        "armor": 0.3,
        "fuel_consumption": 0.02,
        "crew": 4,
        "weapons": ["minigun_left", "minigun_right"],
    },
    "apache": {
        "vehicle_class": VehicleClass.HELICOPTER,
        "max_speed": 280.0 / 3.6,
        "acceleration": 7.0,
        "turn_rate": 1.8,
        "max_health": 800.0,
        "armor": 0.4,
        "fuel_consumption": 0.018,
        "crew": 2,
        "weapons": ["chain_gun", "hydra_rockets", "hellfire"],
    },
    "quadcopter": {
        "vehicle_class": VehicleClass.DRONE_QUAD,
        "max_speed": 60.0 / 3.6,
        "acceleration": 8.0,
        "turn_rate": 3.0,
        "max_health": 50.0,
        "armor": 0.0,
        "fuel_consumption": 0.005,
        "crew": 0,
        "weapons": ["camera"],
    },
    "reaper": {
        "vehicle_class": VehicleClass.DRONE_FIXED_WING,
        "max_speed": 400.0 / 3.6,
        "acceleration": 5.0,
        "turn_rate": 0.6,
        "max_health": 300.0,
        "armor": 0.1,
        "fuel_consumption": 0.012,
        "crew": 0,
        "weapons": ["hellfire_agm", "gbu_12"],
    },
    "motorcycle": {
        "vehicle_class": VehicleClass.MOTORCYCLE,
        "max_speed": 120.0 / 3.6,
        "acceleration": 6.0,
        "turn_rate": 2.0,
        "max_health": 80.0,
        "armor": 0.0,
        "fuel_consumption": 0.004,
        "crew": 1,
        "weapons": [],
    },
    "zodiac": {
        "vehicle_class": VehicleClass.BOAT,
        "max_speed": 60.0 / 3.6,
        "acceleration": 3.5,
        "turn_rate": 1.0,
        "max_health": 200.0,
        "armor": 0.0,
        "fuel_consumption": 0.01,
        "crew": 2,
        "weapons": [],
    },
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vehicle(
    template: str,
    vehicle_id: str,
    alliance: str,
    position: Vec2,
    name: str | None = None,
) -> VehicleState:
    """Create a vehicle from a template name.

    Parameters
    ----------
    template : str
        Key into :data:`VEHICLE_TEMPLATES`.
    vehicle_id : str
        Unique identifier for this vehicle instance.
    alliance : str
        Alliance string (e.g. "friendly", "hostile").
    position : Vec2
        Starting position (x, y) in meters.
    name : str, optional
        Display name. Defaults to *template* title-cased.

    Returns
    -------
    VehicleState

    Raises
    ------
    KeyError
        If *template* is not in :data:`VEHICLE_TEMPLATES`.
    """
    if template not in VEHICLE_TEMPLATES:
        raise KeyError(f"Unknown vehicle template: {template!r}")

    t = VEHICLE_TEMPLATES[template]
    return VehicleState(
        vehicle_id=vehicle_id,
        name=name or template.replace("_", " ").title(),
        vehicle_class=t["vehicle_class"],
        alliance=alliance,
        position=position,
        max_speed=t["max_speed"],
        acceleration=t["acceleration"],
        turn_rate=t["turn_rate"],
        health=t["max_health"],
        max_health=t["max_health"],
        armor=t["armor"],
        fuel_consumption=t["fuel_consumption"],
        crew=t["crew"],
        weapons=list(t["weapons"]),
    )
