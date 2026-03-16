"""Supply and logistics system for the simulation engine.

Simulates ammo, fuel, food, medical supplies, supply lines, and resupply.
Units consume supplies over time; caches store them; routes connect caches;
the LogisticsEngine ticks consumption and auto-resupply each frame.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SupplyType(Enum):
    """Categories of supplies tracked by the logistics system."""

    AMMO = "ammo"
    FUEL = "fuel"
    FOOD = "food"
    MEDICAL = "medical"
    PARTS = "parts"
    WATER = "water"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SupplyCache:
    """A fixed supply depot or forward cache."""

    cache_id: str
    position: Vec2
    supplies: dict[SupplyType, float] = field(default_factory=dict)
    capacity: dict[SupplyType, float] = field(default_factory=dict)
    is_destroyed: bool = False
    alliance: str = "neutral"

    # -- helpers --

    def available(self, st: SupplyType) -> float:
        """How much of *st* is currently available."""
        if self.is_destroyed:
            return 0.0
        return self.supplies.get(st, 0.0)

    def withdraw(self, st: SupplyType, amount: float) -> float:
        """Withdraw up to *amount* of *st*, returning actual withdrawn."""
        if self.is_destroyed or amount <= 0:
            return 0.0
        have = self.supplies.get(st, 0.0)
        taken = min(have, amount)
        self.supplies[st] = have - taken
        return taken

    def deposit(self, st: SupplyType, amount: float) -> float:
        """Deposit up to *amount* respecting capacity. Returns actual deposited."""
        if self.is_destroyed or amount <= 0:
            return 0.0
        cap = self.capacity.get(st, float("inf"))
        cur = self.supplies.get(st, 0.0)
        room = max(cap - cur, 0.0)
        added = min(amount, room)
        self.supplies[st] = cur + added
        return added

    def fill_ratio(self, st: SupplyType) -> float:
        """Return 0..1 fill ratio for a supply type (0 if no capacity)."""
        cap = self.capacity.get(st, 0.0)
        if cap <= 0:
            return 0.0
        return min(self.supplies.get(st, 0.0) / cap, 1.0)

    def total_fill_ratio(self) -> float:
        """Average fill ratio across all capacity entries."""
        if not self.capacity:
            return 0.0
        ratios = [self.fill_ratio(st) for st in self.capacity]
        return sum(ratios) / len(ratios)


@dataclass
class SupplyRequest:
    """A unit's request for resupply."""

    requester_id: str
    supply_type: SupplyType
    amount: float
    priority: int = 1
    position: Vec2 = (0.0, 0.0)
    timestamp: float = 0.0
    fulfilled: bool = False

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SupplyRoute:
    """A logistics route between two caches."""

    route_id: str
    waypoints: list[Vec2] = field(default_factory=list)
    source_cache_id: str = ""
    dest_cache_id: str = ""
    is_active: bool = True
    risk_level: float = 0.0  # 0=safe, 1=through enemy territory


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

SUPPLY_PRESETS: dict[str, dict[SupplyType, float]] = {
    "infantry_fob": {
        SupplyType.AMMO: 1000.0,
        SupplyType.MEDICAL: 200.0,
        SupplyType.FOOD: 500.0,
        SupplyType.WATER: 500.0,
    },
    "vehicle_depot": {
        SupplyType.FUEL: 5000.0,
        SupplyType.AMMO: 2000.0,
        SupplyType.PARTS: 500.0,
    },
    "field_hospital": {
        SupplyType.MEDICAL: 1000.0,
        SupplyType.WATER: 500.0,
    },
    "ammo_dump": {
        SupplyType.AMMO: 5000.0,
    },
    "forward_cache": {
        SupplyType.AMMO: 200.0,
        SupplyType.MEDICAL: 50.0,
        SupplyType.WATER: 100.0,
    },
}


def cache_from_preset(
    preset_name: str,
    cache_id: str,
    position: Vec2,
    alliance: str = "friendly",
    fill: float = 1.0,
) -> SupplyCache:
    """Create a SupplyCache from a named preset.

    *fill* (0..1) controls how full the cache starts.
    """
    caps = SUPPLY_PRESETS[preset_name]
    supplies = {st: amt * max(0.0, min(fill, 1.0)) for st, amt in caps.items()}
    return SupplyCache(
        cache_id=cache_id,
        position=position,
        supplies=supplies,
        capacity=dict(caps),
        alliance=alliance,
    )


# ---------------------------------------------------------------------------
# Low-supply warning
# ---------------------------------------------------------------------------

LOW_SUPPLY_THRESHOLD = 0.2  # 20% remaining triggers warning


@dataclass
class LowSupplyWarning:
    """Emitted when a cache drops below threshold."""

    cache_id: str
    supply_type: SupplyType
    remaining_ratio: float
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Logistics engine
# ---------------------------------------------------------------------------

# Default auto-resupply range (meters)
DEFAULT_RESUPPLY_RANGE = 50.0


class LogisticsEngine:
    """Tick-driven supply and logistics simulation."""

    def __init__(self, resupply_range: float = DEFAULT_RESUPPLY_RANGE) -> None:
        self.caches: dict[str, SupplyCache] = {}
        self.routes: list[SupplyRoute] = []
        self.requests: list[SupplyRequest] = []
        self.consumption_rates: dict[str, dict[SupplyType, float]] = {}
        self.resupply_range: float = resupply_range
        # Accumulated per-unit supplies (what units are carrying)
        self.unit_supplies: dict[str, dict[SupplyType, float]] = {}
        # Warnings generated during last tick
        self.warnings: list[LowSupplyWarning] = []

    # -- mutation --

    def add_cache(self, cache: SupplyCache) -> None:
        """Register a supply cache."""
        self.caches[cache.cache_id] = cache

    def remove_cache(self, cache_id: str) -> SupplyCache | None:
        """Remove and return a cache, or None."""
        return self.caches.pop(cache_id, None)

    def add_route(self, route: SupplyRoute) -> None:
        """Register a supply route."""
        self.routes.append(route)

    def set_consumption_rate(
        self, unit_id: str, rates: dict[SupplyType, float]
    ) -> None:
        """Set per-second consumption rates for a unit."""
        self.consumption_rates[unit_id] = dict(rates)

    def set_unit_supplies(
        self, unit_id: str, supplies: dict[SupplyType, float]
    ) -> None:
        """Set current supply levels for a unit."""
        self.unit_supplies[unit_id] = dict(supplies)

    # -- requests --

    def request_supply(
        self,
        requester_id: str,
        supply_type: SupplyType,
        amount: float,
        priority: int = 1,
        position: Vec2 = (0.0, 0.0),
    ) -> SupplyRequest:
        """Create and register a supply request."""
        req = SupplyRequest(
            requester_id=requester_id,
            supply_type=supply_type,
            amount=amount,
            priority=priority,
            position=position,
        )
        self.requests.append(req)
        return req

    # -- resupply --

    def resupply_unit(
        self, unit_id: str, cache_id: str, supply_type: SupplyType, amount: float
    ) -> float:
        """Withdraw *amount* from a cache and give to a unit.

        Returns actual amount supplied (limited by cache stock).
        """
        cache = self.caches.get(cache_id)
        if cache is None or cache.is_destroyed:
            return 0.0
        taken = cache.withdraw(supply_type, amount)
        if taken > 0:
            unit_sup = self.unit_supplies.setdefault(unit_id, {})
            unit_sup[supply_type] = unit_sup.get(supply_type, 0.0) + taken
        return taken

    # -- queries --

    def find_nearest_cache(
        self,
        position: Vec2,
        alliance: str,
        supply_type: SupplyType | None = None,
    ) -> SupplyCache | None:
        """Find the closest non-destroyed allied cache.

        If *supply_type* is given, only consider caches that have stock.
        """
        best: SupplyCache | None = None
        best_dist = float("inf")
        for cache in self.caches.values():
            if cache.is_destroyed or cache.alliance != alliance:
                continue
            if supply_type is not None and cache.available(supply_type) <= 0:
                continue
            d = distance(position, cache.position)
            if d < best_dist:
                best_dist = d
                best = cache
        return best

    def get_supply_status(self, alliance: str) -> dict[str, Any]:
        """Overall supply situation for an alliance.

        Returns dict with totals, cache count, low-supply list.
        """
        totals: dict[str, float] = {}
        capacities: dict[str, float] = {}
        low: list[dict[str, Any]] = []
        cache_count = 0
        destroyed_count = 0

        for cache in self.caches.values():
            if cache.alliance != alliance:
                continue
            if cache.is_destroyed:
                destroyed_count += 1
                continue
            cache_count += 1
            for st in SupplyType:
                totals[st.value] = totals.get(st.value, 0.0) + cache.available(st)
                capacities[st.value] = capacities.get(st.value, 0.0) + cache.capacity.get(st, 0.0)
            for st in cache.capacity:
                ratio = cache.fill_ratio(st)
                if ratio < LOW_SUPPLY_THRESHOLD:
                    low.append(
                        {
                            "cache_id": cache.cache_id,
                            "supply_type": st.value,
                            "ratio": round(ratio, 3),
                        }
                    )

        ratios: dict[str, float] = {}
        for key in totals:
            cap = capacities.get(key, 0.0)
            ratios[key] = round(totals[key] / cap, 3) if cap > 0 else 0.0

        return {
            "alliance": alliance,
            "cache_count": cache_count,
            "destroyed_count": destroyed_count,
            "totals": totals,
            "capacities": capacities,
            "ratios": ratios,
            "low_supply": low,
        }

    def pending_requests(self, fulfilled: bool = False) -> list[SupplyRequest]:
        """Return unfulfilled (or fulfilled) requests sorted by priority desc."""
        return sorted(
            [r for r in self.requests if r.fulfilled == fulfilled],
            key=lambda r: r.priority,
            reverse=True,
        )

    # -- tick --

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, Vec2],
        unit_alliances: dict[str, str] | None = None,
    ) -> list[LowSupplyWarning]:
        """Advance the logistics simulation by *dt* seconds.

        1. Units consume supplies based on consumption_rates.
        2. Units near an allied cache auto-resupply.
        3. Low-supply warnings are generated.

        *unit_alliances* maps unit_id -> alliance string; defaults to
        "friendly" for all units if not provided.

        Returns list of LowSupplyWarning generated this tick.
        """
        if unit_alliances is None:
            unit_alliances = {}

        self.warnings = []

        # 1. Consumption
        for unit_id, rates in self.consumption_rates.items():
            unit_sup = self.unit_supplies.get(unit_id)
            if unit_sup is None:
                continue
            for st, rate in rates.items():
                cur = unit_sup.get(st, 0.0)
                consumed = min(cur, rate * dt)
                unit_sup[st] = cur - consumed

        # 2. Auto-resupply from nearest allied cache within range
        for unit_id, pos in unit_positions.items():
            alliance = unit_alliances.get(unit_id, "friendly")
            unit_sup = self.unit_supplies.get(unit_id, {})
            rates = self.consumption_rates.get(unit_id, {})
            for st in rates:
                cur = unit_sup.get(st, 0.0)
                # Resupply if below 50% of a "standard load" (rate * 60s)
                standard_load = rates[st] * 60.0
                if standard_load > 0 and cur < standard_load * 0.5:
                    cache = self.find_nearest_cache(pos, alliance, st)
                    if cache is not None and distance(pos, cache.position) <= self.resupply_range:
                        needed = standard_load - cur
                        taken = cache.withdraw(st, needed)
                        if taken > 0:
                            unit_sup[st] = cur + taken
                            self.unit_supplies[unit_id] = unit_sup

        # 3. Fulfill pending requests
        for req in self.requests:
            if req.fulfilled:
                continue
            alliance = unit_alliances.get(req.requester_id, "friendly")
            cache = self.find_nearest_cache(req.position, alliance, req.supply_type)
            if cache is not None and distance(req.position, cache.position) <= self.resupply_range:
                taken = cache.withdraw(req.supply_type, req.amount)
                if taken > 0:
                    unit_sup = self.unit_supplies.setdefault(req.requester_id, {})
                    unit_sup[req.supply_type] = unit_sup.get(req.supply_type, 0.0) + taken
                    if taken >= req.amount * 0.9:  # 90% fulfilled counts
                        req.fulfilled = True

        # 4. Low-supply warnings
        now = time.time()
        for cache in self.caches.values():
            if cache.is_destroyed:
                continue
            for st in cache.capacity:
                ratio = cache.fill_ratio(st)
                if ratio < LOW_SUPPLY_THRESHOLD:
                    w = LowSupplyWarning(
                        cache_id=cache.cache_id,
                        supply_type=st,
                        remaining_ratio=ratio,
                        timestamp=now,
                    )
                    self.warnings.append(w)

        return self.warnings

    # -- serialization --

    def to_three_js(self) -> dict[str, Any]:
        """Export state for Three.js / frontend rendering."""
        alliance_colors = {
            "friendly": "#05ffa1",
            "hostile": "#ff2a6d",
            "neutral": "#fcee0a",
        }

        caches_out: list[dict[str, Any]] = []
        for c in self.caches.values():
            supplies_ratios = {}
            for st in SupplyType:
                cap = c.capacity.get(st, 0.0)
                if cap > 0:
                    supplies_ratios[st.value] = round(c.available(st) / cap, 3)
            caches_out.append(
                {
                    "id": c.cache_id,
                    "x": c.position[0],
                    "y": c.position[1],
                    "alliance": c.alliance,
                    "destroyed": c.is_destroyed,
                    "supplies": supplies_ratios,
                    "color": alliance_colors.get(c.alliance, "#ffffff"),
                }
            )

        routes_out: list[dict[str, Any]] = []
        for r in self.routes:
            routes_out.append(
                {
                    "id": r.route_id,
                    "waypoints": [list(w) for w in r.waypoints],
                    "source": r.source_cache_id,
                    "dest": r.dest_cache_id,
                    "active": r.is_active,
                    "risk": round(r.risk_level, 3),
                }
            )

        requests_out: list[dict[str, Any]] = []
        for req in self.requests:
            if not req.fulfilled:
                requests_out.append(
                    {
                        "id": req.requester_id,
                        "x": req.position[0],
                        "y": req.position[1],
                        "type": req.supply_type.value,
                        "priority": req.priority,
                    }
                )

        return {
            "caches": caches_out,
            "routes": routes_out,
            "requests": requests_out,
        }
