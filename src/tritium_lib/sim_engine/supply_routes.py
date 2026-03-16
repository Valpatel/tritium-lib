# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Supply line simulation for the Tritium sim engine.

Simulates supply routes connecting caches to frontline units, with convoy
movement, route interdiction, and per-unit supply tracking for ammo and
food depletion.

Usage::

    from tritium_lib.sim_engine.supply_routes import (
        SupplyRouteEngine, SupplyLine, SupplyConvoy, UnitSupplyState,
    )

    engine = SupplyRouteEngine()
    engine.add_supply_line(SupplyLine(
        line_id="main_supply",
        waypoints=[(0.0, 0.0), (100.0, 50.0), (200.0, 100.0)],
        source_cache_id="depot_rear",
        alliance="friendly",
    ))
    engine.register_unit("alpha-1", alliance="friendly")
    engine.dispatch_convoy("main_supply", payload_ammo=200.0, payload_food=50.0)
    engine.tick(0.1, unit_positions={"alpha-1": (200.0, 100.0)}, enemy_positions={})
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConvoyStatus(Enum):
    """Status of a supply convoy."""
    EN_ROUTE = "en_route"
    DELIVERING = "delivering"
    RETURNING = "returning"
    DESTROYED = "destroyed"
    ARRIVED = "arrived"


class RouteStatus(Enum):
    """Status of a supply route."""
    OPEN = "open"
    CONTESTED = "contested"     # enemy near the route
    INTERDICTED = "interdicted"  # blocked by enemy
    DESTROYED = "destroyed"      # permanently cut


class SupplyLevel(Enum):
    """Qualitative supply level for a unit."""
    FULL = "full"           # >80%
    ADEQUATE = "adequate"   # 50-80%
    LOW = "low"             # 20-50%
    CRITICAL = "critical"   # 5-20%
    EMPTY = "empty"         # <5%


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SupplyLine:
    """A route connecting a supply source to a delivery area."""
    line_id: str
    waypoints: list[Vec2]
    source_cache_id: str = ""
    alliance: str = "friendly"
    status: RouteStatus = RouteStatus.OPEN
    interdiction_radius: float = 30.0  # enemy within this of any waypoint = contested
    capacity_per_trip: float = 500.0   # max payload per convoy

    def total_length(self) -> float:
        """Total distance along the route in meters."""
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += distance(self.waypoints[i], self.waypoints[i + 1])
        return total


@dataclass
class SupplyConvoy:
    """A convoy moving along a supply line."""
    convoy_id: str
    line_id: str
    alliance: str = "friendly"
    status: ConvoyStatus = ConvoyStatus.EN_ROUTE
    position: Vec2 = (0.0, 0.0)
    speed: float = 8.0             # m/s
    health: float = 100.0
    armor: float = 0.1             # damage reduction 0-1
    # Payload
    ammo: float = 0.0
    food: float = 0.0
    # Route progress
    current_waypoint_index: int = 0
    distance_along_segment: float = 0.0


@dataclass
class UnitSupplyState:
    """Per-unit supply tracking."""
    unit_id: str
    alliance: str = "friendly"
    ammo: float = 100.0
    max_ammo: float = 100.0
    food: float = 100.0
    max_food: float = 100.0
    ammo_consumption_rate: float = 0.5   # per second in combat
    food_consumption_rate: float = 0.02  # per second always
    in_combat: bool = False
    is_supplied: bool = True    # has received supply recently
    resupply_range: float = 20.0  # range to receive from convoy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Delivery range: convoy delivers to units within this distance
DELIVERY_RANGE: float = 30.0

# Ammo/food levels for qualitative assessment
def _supply_level(current: float, maximum: float) -> SupplyLevel:
    """Determine the qualitative supply level."""
    if maximum <= 0:
        return SupplyLevel.EMPTY
    ratio = current / maximum
    if ratio > 0.8:
        return SupplyLevel.FULL
    if ratio > 0.5:
        return SupplyLevel.ADEQUATE
    if ratio > 0.2:
        return SupplyLevel.LOW
    if ratio > 0.05:
        return SupplyLevel.CRITICAL
    return SupplyLevel.EMPTY


# ---------------------------------------------------------------------------
# SupplyRouteEngine
# ---------------------------------------------------------------------------

class SupplyRouteEngine:
    """Manages supply lines, convoys, and per-unit supply state.

    Each tick:
    1. Updates route status based on enemy proximity
    2. Moves convoys along their routes
    3. Delivers supplies when convoys reach endpoints
    4. Consumes unit ammo/food
    5. Reports supply warnings
    """

    def __init__(self) -> None:
        self.supply_lines: dict[str, SupplyLine] = {}
        self.convoys: list[SupplyConvoy] = []
        self.unit_states: dict[str, UnitSupplyState] = {}
        self._event_log: list[dict[str, Any]] = []

    # -- supply lines -------------------------------------------------------

    def add_supply_line(self, line: SupplyLine) -> None:
        """Register a supply line."""
        self.supply_lines[line.line_id] = line

    def remove_supply_line(self, line_id: str) -> None:
        """Remove a supply line."""
        self.supply_lines.pop(line_id, None)

    # -- unit registration --------------------------------------------------

    def register_unit(
        self,
        unit_id: str,
        alliance: str = "friendly",
        max_ammo: float = 100.0,
        max_food: float = 100.0,
    ) -> UnitSupplyState:
        """Register a unit for supply tracking."""
        uss = UnitSupplyState(
            unit_id=unit_id,
            alliance=alliance,
            ammo=max_ammo,
            max_ammo=max_ammo,
            food=max_food,
            max_food=max_food,
        )
        self.unit_states[unit_id] = uss
        return uss

    def remove_unit(self, unit_id: str) -> None:
        """Remove unit from supply tracking."""
        self.unit_states.pop(unit_id, None)

    def set_combat_status(self, unit_id: str, in_combat: bool) -> None:
        """Set whether a unit is in combat (affects ammo consumption)."""
        uss = self.unit_states.get(unit_id)
        if uss is not None:
            uss.in_combat = in_combat

    # -- convoys ------------------------------------------------------------

    def dispatch_convoy(
        self,
        line_id: str,
        payload_ammo: float = 100.0,
        payload_food: float = 50.0,
        speed: float = 8.0,
    ) -> SupplyConvoy | None:
        """Dispatch a convoy along a supply line.

        Returns the convoy if the line exists and is not destroyed, else None.
        """
        line = self.supply_lines.get(line_id)
        if line is None or line.status == RouteStatus.DESTROYED:
            return None
        if len(line.waypoints) < 2:
            return None

        convoy = SupplyConvoy(
            convoy_id=f"convoy_{uuid.uuid4().hex[:8]}",
            line_id=line_id,
            alliance=line.alliance,
            position=line.waypoints[0],
            speed=speed,
            ammo=min(payload_ammo, line.capacity_per_trip),
            food=min(payload_food, line.capacity_per_trip),
        )
        self.convoys.append(convoy)
        self._event_log.append({
            "type": "convoy_dispatched",
            "convoy_id": convoy.convoy_id,
            "line_id": line_id,
            "ammo": convoy.ammo,
            "food": convoy.food,
        })
        return convoy

    def attack_convoy(self, convoy_id: str, damage: float) -> bool:
        """Attack a convoy, reducing its health. Returns True if destroyed."""
        for convoy in self.convoys:
            if convoy.convoy_id == convoy_id:
                effective_damage = damage * (1.0 - convoy.armor)
                convoy.health -= effective_damage
                if convoy.health <= 0.0:
                    convoy.status = ConvoyStatus.DESTROYED
                    convoy.health = 0.0
                    self._event_log.append({
                        "type": "convoy_destroyed",
                        "convoy_id": convoy_id,
                    })
                    return True
                return False
        return False

    # -- queries ------------------------------------------------------------

    def get_unit_supply_level(self, unit_id: str) -> dict[str, str]:
        """Return qualitative supply levels for a unit."""
        uss = self.unit_states.get(unit_id)
        if uss is None:
            return {"ammo": "unknown", "food": "unknown"}
        return {
            "ammo": _supply_level(uss.ammo, uss.max_ammo).value,
            "food": _supply_level(uss.food, uss.max_food).value,
        }

    def get_unit_ammo(self, unit_id: str) -> float:
        """Return current ammo for a unit."""
        uss = self.unit_states.get(unit_id)
        return uss.ammo if uss is not None else 0.0

    def get_unit_food(self, unit_id: str) -> float:
        """Return current food for a unit."""
        uss = self.unit_states.get(unit_id)
        return uss.food if uss is not None else 0.0

    def get_route_status(self, line_id: str) -> RouteStatus:
        """Return the current status of a supply route."""
        line = self.supply_lines.get(line_id)
        return line.status if line is not None else RouteStatus.DESTROYED

    def drain_event_log(self) -> list[dict[str, Any]]:
        """Return and clear the event log."""
        log = self._event_log.copy()
        self._event_log.clear()
        return log

    # -- tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, Vec2] | None = None,
        enemy_positions: dict[str, Vec2] | None = None,
    ) -> dict[str, Any]:
        """Advance the supply simulation by *dt* seconds.

        1. Checks route safety based on enemy positions
        2. Moves convoys along waypoints
        3. Delivers supplies at endpoints
        4. Consumes unit ammo/food
        5. Returns events and supply status

        Args:
            dt: time step in seconds
            unit_positions: {unit_id: (x, y)} for delivery range checks
            enemy_positions: {enemy_id: (x, y)} for route interdiction
        """
        if unit_positions is None:
            unit_positions = {}
        if enemy_positions is None:
            enemy_positions = {}

        events: list[dict[str, Any]] = []

        # 1. Update route status
        for line in self.supply_lines.values():
            if line.status == RouteStatus.DESTROYED:
                continue
            contested = False
            interdicted = False
            for wp in line.waypoints:
                for epos in enemy_positions.values():
                    d = distance(wp, epos)
                    if d <= line.interdiction_radius:
                        interdicted = True
                        break
                    elif d <= line.interdiction_radius * 2:
                        contested = True
                if interdicted:
                    break

            old_status = line.status
            if interdicted:
                line.status = RouteStatus.INTERDICTED
            elif contested:
                line.status = RouteStatus.CONTESTED
            else:
                line.status = RouteStatus.OPEN

            if line.status != old_status:
                events.append({
                    "type": "route_status_changed",
                    "line_id": line.line_id,
                    "old_status": old_status.value,
                    "new_status": line.status.value,
                })

        # 2. Move convoys
        for convoy in self.convoys:
            if convoy.status in (ConvoyStatus.DESTROYED, ConvoyStatus.ARRIVED):
                continue

            line = self.supply_lines.get(convoy.line_id)
            if line is None or len(line.waypoints) < 2:
                continue

            # If route is interdicted, convoy halts
            if line.status == RouteStatus.INTERDICTED:
                continue

            # Speed penalty on contested routes
            effective_speed = convoy.speed
            if line.status == RouteStatus.CONTESTED:
                effective_speed *= 0.5

            # Move toward next waypoint
            remaining_move = effective_speed * dt
            while remaining_move > 0.0 and convoy.current_waypoint_index < len(line.waypoints) - 1:
                target_wp = line.waypoints[convoy.current_waypoint_index + 1]
                d_to_wp = distance(convoy.position, target_wp)

                if d_to_wp <= remaining_move:
                    # Arrive at waypoint
                    convoy.position = target_wp
                    remaining_move -= d_to_wp
                    convoy.current_waypoint_index += 1
                    convoy.distance_along_segment = 0.0
                else:
                    # Move partially toward waypoint
                    dx = target_wp[0] - convoy.position[0]
                    dy = target_wp[1] - convoy.position[1]
                    if d_to_wp > 0:
                        ratio = remaining_move / d_to_wp
                        convoy.position = (
                            convoy.position[0] + dx * ratio,
                            convoy.position[1] + dy * ratio,
                        )
                    convoy.distance_along_segment += remaining_move
                    remaining_move = 0.0

            # Check if convoy reached the end
            if convoy.current_waypoint_index >= len(line.waypoints) - 1:
                convoy.status = ConvoyStatus.ARRIVED
                events.append({
                    "type": "convoy_arrived",
                    "convoy_id": convoy.convoy_id,
                    "line_id": convoy.line_id,
                })

        # 3. Deliver supplies from arrived convoys to nearby units
        for convoy in self.convoys:
            if convoy.status != ConvoyStatus.ARRIVED:
                continue
            if convoy.ammo <= 0 and convoy.food <= 0:
                continue

            for uid, pos in unit_positions.items():
                uss = self.unit_states.get(uid)
                if uss is None or uss.alliance != convoy.alliance:
                    continue
                if distance(pos, convoy.position) > DELIVERY_RANGE:
                    continue

                # Deliver ammo
                ammo_needed = uss.max_ammo - uss.ammo
                if ammo_needed > 0 and convoy.ammo > 0:
                    delivered = min(ammo_needed, convoy.ammo)
                    uss.ammo += delivered
                    convoy.ammo -= delivered

                # Deliver food
                food_needed = uss.max_food - uss.food
                if food_needed > 0 and convoy.food > 0:
                    delivered = min(food_needed, convoy.food)
                    uss.food += delivered
                    convoy.food -= delivered

                if convoy.ammo <= 0 and convoy.food <= 0:
                    break

        # 4. Consume unit supplies
        warnings: list[dict[str, Any]] = []
        for uid, uss in self.unit_states.items():
            # Food always consumed
            uss.food = max(0.0, uss.food - uss.food_consumption_rate * dt)

            # Ammo consumed only in combat
            if uss.in_combat:
                uss.ammo = max(0.0, uss.ammo - uss.ammo_consumption_rate * dt)

            # Check supply level
            ammo_level = _supply_level(uss.ammo, uss.max_ammo)
            food_level = _supply_level(uss.food, uss.max_food)

            if ammo_level in (SupplyLevel.CRITICAL, SupplyLevel.EMPTY):
                warnings.append({
                    "type": "low_ammo",
                    "unit_id": uid,
                    "level": ammo_level.value,
                    "remaining": round(uss.ammo, 1),
                })

            if food_level in (SupplyLevel.CRITICAL, SupplyLevel.EMPTY):
                warnings.append({
                    "type": "low_food",
                    "unit_id": uid,
                    "level": food_level.value,
                    "remaining": round(uss.food, 1),
                })

        # 5. Cleanup old destroyed convoys
        self.convoys = [
            c for c in self.convoys
            if c.status != ConvoyStatus.DESTROYED
        ]

        return {
            "events": events,
            "warnings": warnings,
            "active_convoys": sum(
                1 for c in self.convoys
                if c.status in (ConvoyStatus.EN_ROUTE, ConvoyStatus.DELIVERING)
            ),
            "arrived_convoys": sum(
                1 for c in self.convoys if c.status == ConvoyStatus.ARRIVED
            ),
        }

    # -- Three.js visualization ---------------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export supply route state for Three.js visualization.

        Returns route lines, convoy positions, and supply status per unit.
        """
        _STATUS_COLORS: dict[RouteStatus, str] = {
            RouteStatus.OPEN: "#05ffa1",
            RouteStatus.CONTESTED: "#fcee0a",
            RouteStatus.INTERDICTED: "#ff2a6d",
            RouteStatus.DESTROYED: "#333333",
        }

        routes_out: list[dict[str, Any]] = []
        for line in self.supply_lines.values():
            routes_out.append({
                "id": line.line_id,
                "waypoints": [
                    [wp[0], 0.0, wp[1]] for wp in line.waypoints
                ],
                "status": line.status.value,
                "alliance": line.alliance,
                "color": _STATUS_COLORS.get(line.status, "#ffffff"),
                "dashed": line.status != RouteStatus.OPEN,
            })

        convoys_out: list[dict[str, Any]] = []
        for c in self.convoys:
            if c.status == ConvoyStatus.DESTROYED:
                continue
            convoys_out.append({
                "id": c.convoy_id,
                "position": [c.position[0], 0.5, c.position[1]],
                "status": c.status.value,
                "health": round(c.health, 1),
                "ammo": round(c.ammo, 1),
                "food": round(c.food, 1),
                "alliance": c.alliance,
                "color": "#05ffa1" if c.alliance == "friendly" else "#ff2a6d",
            })

        units_out: list[dict[str, Any]] = []
        for uid, uss in self.unit_states.items():
            ammo_level = _supply_level(uss.ammo, uss.max_ammo)
            food_level = _supply_level(uss.food, uss.max_food)

            _LEVEL_COLORS = {
                SupplyLevel.FULL: "#05ffa1",
                SupplyLevel.ADEQUATE: "#00f0ff",
                SupplyLevel.LOW: "#fcee0a",
                SupplyLevel.CRITICAL: "#ff8800",
                SupplyLevel.EMPTY: "#ff2a6d",
            }

            # Use worst level for color
            worst = min(ammo_level.value, food_level.value, key=lambda x: [
                "full", "adequate", "low", "critical", "empty",
            ].index(x))

            units_out.append({
                "id": uid,
                "ammo_ratio": round(uss.ammo / uss.max_ammo, 3) if uss.max_ammo > 0 else 0.0,
                "food_ratio": round(uss.food / uss.max_food, 3) if uss.max_food > 0 else 0.0,
                "ammo_level": ammo_level.value,
                "food_level": food_level.value,
                "in_combat": uss.in_combat,
                "supply_color": _LEVEL_COLORS.get(
                    _supply_level(uss.ammo, uss.max_ammo), "#ffffff",
                ),
            })

        return {
            "routes": routes_out,
            "convoys": convoys_out,
            "unit_supply": units_out,
        }
