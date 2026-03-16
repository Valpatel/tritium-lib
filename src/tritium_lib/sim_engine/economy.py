# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Economy and resource management system — the RTS layer.

Provides per-faction resource pools with income, capacity, and upkeep;
a build queue for unit production; a multi-tier tech tree with prerequisites;
and an EconomyEngine that ties them all together with a tick-driven loop.

Usage::

    from tritium_lib.sim_engine.economy import (
        EconomyEngine, ResourceType, ResourcePool, UnitCost,
        BuildQueue, TechTree, UNIT_COSTS, TECH_TREE, ECONOMY_PRESETS,
    )

    engine = EconomyEngine()
    engine.setup_faction("alpha", ECONOMY_PRESETS["standard"])
    engine.register_unit_costs(UNIT_COSTS)
    engine.register_tech_tree("alpha", TECH_TREE)

    ok = engine.purchase_unit("alpha", "infantry")
    engine.tick(1.0)
    completed = engine.build_queues["alpha"].pop_completed()

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResourceType(Enum):
    """Economy resource categories."""

    CREDITS = "credits"
    MANPOWER = "manpower"
    FUEL = "fuel"
    STEEL = "steel"
    ELECTRONICS = "electronics"
    FOOD = "food"


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------


@dataclass
class ResourcePool:
    """Per-faction resource stockpile with income and capacity limits.

    *resources* holds current amounts; *income* is added per tick; *capacity*
    caps each resource type.  Resources that are not present in *capacity*
    have no upper limit.
    """

    resources: dict[ResourceType, float] = field(default_factory=dict)
    income: dict[ResourceType, float] = field(default_factory=dict)
    capacity: dict[ResourceType, float] = field(default_factory=dict)

    # -- mutation -----------------------------------------------------------

    def add(self, rt: ResourceType, amount: float) -> float:
        """Add *amount* of *rt*, capped by capacity.  Returns actual added."""
        if amount <= 0:
            return 0.0
        cur = self.resources.get(rt, 0.0)
        cap = self.capacity.get(rt, float("inf"))
        room = max(cap - cur, 0.0)
        actual = min(amount, room)
        self.resources[rt] = cur + actual
        return actual

    def spend(self, rt: ResourceType, amount: float) -> float:
        """Spend up to *amount* of *rt*.  Returns actual spent."""
        if amount <= 0:
            return 0.0
        cur = self.resources.get(rt, 0.0)
        actual = min(cur, amount)
        self.resources[rt] = cur - actual
        return actual

    def spend_exact(self, cost: dict[ResourceType, float]) -> bool:
        """Spend all resources in *cost* atomically.  Returns False and
        spends nothing if the pool cannot cover every entry."""
        if not self.can_afford(cost):
            return False
        for rt, amt in cost.items():
            self.resources[rt] = self.resources.get(rt, 0.0) - amt
        return True

    def can_afford(self, cost: dict[ResourceType, float]) -> bool:
        """Return True if the pool has at least *cost* of every resource."""
        for rt, amt in cost.items():
            if self.resources.get(rt, 0.0) < amt:
                return False
        return True

    def tick(self, dt: float = 1.0) -> None:
        """Apply income for *dt* seconds, capping at capacity."""
        for rt, rate in self.income.items():
            gained = rate * dt
            if gained <= 0:
                continue
            cur = self.resources.get(rt, 0.0)
            cap = self.capacity.get(rt, float("inf"))
            self.resources[rt] = min(cur + gained, cap)

    def get(self, rt: ResourceType) -> float:
        """Return the current amount of *rt*."""
        return self.resources.get(rt, 0.0)

    def snapshot(self) -> dict[str, float]:
        """Return ``{resource_name: amount}`` for all tracked resources."""
        return {rt.value: round(amt, 2) for rt, amt in self.resources.items()}


# ---------------------------------------------------------------------------
# UnitCost
# ---------------------------------------------------------------------------


@dataclass
class UnitCost:
    """Build cost and upkeep for a unit template."""

    unit_template: str
    cost: dict[ResourceType, float] = field(default_factory=dict)
    build_time: float = 10.0  # seconds
    upkeep: dict[ResourceType, float] = field(default_factory=dict)  # per minute


# ---------------------------------------------------------------------------
# BuildQueue
# ---------------------------------------------------------------------------


class BuildQueue:
    """Production queue: units are added, progress toward completion, and
    get popped when done."""

    def __init__(self) -> None:
        self.queue: list[dict[str, Any]] = []

    def add(self, template: str, cost: dict[ResourceType, float],
            build_time: float) -> None:
        """Enqueue a new build order."""
        self.queue.append({
            "unit_template": template,
            "progress": 0.0,
            "build_time": build_time,
            "cost": {rt.value: amt for rt, amt in cost.items()},
        })

    def tick(self, dt: float) -> None:
        """Advance build progress for the *first* item in the queue."""
        if not self.queue:
            return
        # Only the head of the queue builds (serial production).
        self.queue[0]["progress"] += dt

    def pop_completed(self) -> list[str]:
        """Remove and return template names of all completed builds."""
        completed: list[str] = []
        remaining: list[dict[str, Any]] = []
        for entry in self.queue:
            if entry["progress"] >= entry["build_time"]:
                completed.append(entry["unit_template"])
            else:
                remaining.append(entry)
        self.queue = remaining
        return completed

    def peek(self) -> dict[str, Any] | None:
        """Return the current build item without removing it, or None."""
        return self.queue[0] if self.queue else None

    def cancel(self, index: int = 0) -> dict[str, Any] | None:
        """Cancel and return the build at *index*, or None if out of range."""
        if 0 <= index < len(self.queue):
            return self.queue.pop(index)
        return None

    def __len__(self) -> int:
        return len(self.queue)

    def to_list(self) -> list[dict[str, Any]]:
        """Serializable snapshot of the queue."""
        out: list[dict[str, Any]] = []
        for entry in self.queue:
            pct = 0.0
            if entry["build_time"] > 0:
                pct = min(entry["progress"] / entry["build_time"], 1.0)
            out.append({
                "unit_template": entry["unit_template"],
                "progress_pct": round(pct, 3),
                "remaining_s": round(max(entry["build_time"] - entry["progress"], 0.0), 2),
            })
        return out


# ---------------------------------------------------------------------------
# TechTree
# ---------------------------------------------------------------------------


class TechTree:
    """Technology research tree with prerequisites and unit unlocks.

    Each tech is a dict with keys: ``name``, ``cost`` (dict[ResourceType, float]),
    ``prerequisites`` (list[str] of tech names), ``unlocks`` (list[str] of unit
    template names or ability identifiers).
    """

    def __init__(self, techs: dict[str, dict[str, Any]] | None = None) -> None:
        self.techs: dict[str, dict[str, Any]] = dict(techs) if techs else {}
        self.researched: set[str] = set()

    def add_tech(self, name: str, cost: dict[ResourceType, float],
                 prerequisites: list[str] | None = None,
                 unlocks: list[str] | None = None) -> None:
        """Register a technology."""
        self.techs[name] = {
            "name": name,
            "cost": cost,
            "prerequisites": prerequisites or [],
            "unlocks": unlocks or [],
        }

    def is_available(self, tech_name: str) -> bool:
        """True if all prerequisites are researched and tech is not yet done."""
        if tech_name in self.researched:
            return False
        tech = self.techs.get(tech_name)
        if tech is None:
            return False
        return all(p in self.researched for p in tech.get("prerequisites", []))

    def research(self, tech_name: str, pool: ResourcePool) -> bool:
        """Research *tech_name*, spending resources from *pool*.

        Returns True on success, False if unavailable or cannot afford.
        """
        if not self.is_available(tech_name):
            return False
        tech = self.techs[tech_name]
        cost = tech.get("cost", {})
        if not pool.spend_exact(cost):
            return False
        self.researched.add(tech_name)
        return True

    def get_unlocked_units(self) -> list[str]:
        """Return all unit templates unlocked by researched techs."""
        units: list[str] = []
        for tech_name in self.researched:
            tech = self.techs.get(tech_name, {})
            units.extend(tech.get("unlocks", []))
        return units

    def available_techs(self) -> list[str]:
        """Return names of all techs that can be researched right now."""
        return [name for name in self.techs if self.is_available(name)]

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot."""
        out: dict[str, Any] = {}
        for name, tech in self.techs.items():
            cost_ser = {rt.value: amt for rt, amt in tech.get("cost", {}).items()}
            out[name] = {
                "name": tech["name"],
                "cost": cost_ser,
                "prerequisites": tech.get("prerequisites", []),
                "unlocks": tech.get("unlocks", []),
                "researched": name in self.researched,
                "available": self.is_available(name),
            }
        return out


# ---------------------------------------------------------------------------
# EconomyEngine
# ---------------------------------------------------------------------------


class EconomyEngine:
    """Central economy manager: resource pools, build queues, tech trees,
    and unit costs across all factions.

    Tick-driven: call ``tick(dt)`` each frame to apply income, upkeep, and
    build-queue progress.
    """

    def __init__(self) -> None:
        self.pools: dict[str, ResourcePool] = {}
        self.build_queues: dict[str, BuildQueue] = {}
        self.tech_trees: dict[str, TechTree] = {}
        self.unit_costs: dict[str, UnitCost] = {}
        # Track active units per faction for upkeep
        self._active_units: dict[str, list[str]] = {}  # faction -> [template, ...]

    # -- setup ---------------------------------------------------------------

    def setup_faction(self, faction_id: str,
                      preset: dict[str, Any] | None = None) -> None:
        """Initialise a faction's economy.  If *preset* is given it should
        contain ``resources``, ``income``, and ``capacity`` dicts keyed by
        :class:`ResourceType`."""
        if preset:
            self.pools[faction_id] = ResourcePool(
                resources=dict(preset.get("resources", {})),
                income=dict(preset.get("income", {})),
                capacity=dict(preset.get("capacity", {})),
            )
        else:
            self.pools[faction_id] = ResourcePool()
        self.build_queues[faction_id] = BuildQueue()
        self._active_units.setdefault(faction_id, [])

    def register_unit_costs(self, costs: dict[str, UnitCost]) -> None:
        """Register unit cost definitions (shared across factions)."""
        self.unit_costs.update(costs)

    def register_tech_tree(self, faction_id: str, tree: TechTree) -> None:
        """Assign a tech tree to a faction."""
        self.tech_trees[faction_id] = tree

    def add_active_unit(self, faction_id: str, template: str) -> None:
        """Record that a unit was produced (affects upkeep)."""
        self._active_units.setdefault(faction_id, []).append(template)

    def remove_active_unit(self, faction_id: str, template: str) -> bool:
        """Remove one active unit (e.g. destroyed).  Returns True if found."""
        units = self._active_units.get(faction_id, [])
        if template in units:
            units.remove(template)
            return True
        return False

    # -- actions -------------------------------------------------------------

    def purchase_unit(self, faction_id: str, template: str) -> bool:
        """Attempt to purchase a unit: check cost, spend resources, enqueue.

        Returns True on success, False if insufficient resources, unknown
        template, or unknown faction.
        """
        pool = self.pools.get(faction_id)
        bq = self.build_queues.get(faction_id)
        uc = self.unit_costs.get(template)
        if pool is None or bq is None or uc is None:
            return False
        if not pool.spend_exact(uc.cost):
            return False
        bq.add(template, uc.cost, uc.build_time)
        return True

    def research_tech(self, faction_id: str, tech_name: str) -> bool:
        """Research a tech for a faction.  Returns True on success."""
        tree = self.tech_trees.get(faction_id)
        pool = self.pools.get(faction_id)
        if tree is None or pool is None:
            return False
        return tree.research(tech_name, pool)

    # -- tick ----------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """Advance the economy by *dt* seconds.

        1. Apply income to each faction's pool.
        2. Deduct upkeep for active units (prorated per-minute -> per-second).
        3. Advance build queues.
        4. Pop completed builds and record them as active units.

        Returns list of ``"faction:template"`` strings for newly completed units.
        """
        completed_all: list[str] = []

        for faction_id, pool in self.pools.items():
            # 1. Income
            pool.tick(dt)

            # 2. Upkeep (per minute -> per second)
            for tmpl in self._active_units.get(faction_id, []):
                uc = self.unit_costs.get(tmpl)
                if uc is None:
                    continue
                for rt, rate_per_min in uc.upkeep.items():
                    cost = rate_per_min * dt / 60.0
                    pool.spend(rt, cost)

            # 3. Build progress
            bq = self.build_queues.get(faction_id)
            if bq:
                bq.tick(dt)
                # 4. Pop completed
                done = bq.pop_completed()
                for tmpl in done:
                    self.add_active_unit(faction_id, tmpl)
                    completed_all.append(f"{faction_id}:{tmpl}")

        return completed_all

    # -- queries -------------------------------------------------------------

    def get_economy_status(self, faction_id: str) -> dict[str, Any]:
        """Full economy snapshot for a faction."""
        pool = self.pools.get(faction_id)
        bq = self.build_queues.get(faction_id)
        tree = self.tech_trees.get(faction_id)

        return {
            "faction": faction_id,
            "resources": pool.snapshot() if pool else {},
            "income": {rt.value: round(r, 2) for rt, r in pool.income.items()} if pool else {},
            "capacity": {rt.value: round(c, 2) for rt, c in pool.capacity.items()} if pool else {},
            "build_queue": bq.to_list() if bq else [],
            "build_queue_length": len(bq) if bq else 0,
            "active_units": list(self._active_units.get(faction_id, [])),
            "active_unit_count": len(self._active_units.get(faction_id, [])),
            "tech_researched": list(tree.researched) if tree else [],
            "tech_available": tree.available_techs() if tree else [],
            "unlocked_units": tree.get_unlocked_units() if tree else [],
        }

    def to_three_js(self, faction_id: str) -> dict[str, Any]:
        """Export for Three.js / frontend rendering — resource bars,
        build queue, and tech tree display data."""
        pool = self.pools.get(faction_id)
        bq = self.build_queues.get(faction_id)
        tree = self.tech_trees.get(faction_id)

        # Resource bars: value, capacity, fill ratio, income
        bars: list[dict[str, Any]] = []
        if pool:
            for rt in ResourceType:
                cur = pool.get(rt)
                cap = pool.capacity.get(rt, 0.0)
                inc = pool.income.get(rt, 0.0)
                if cap > 0 or cur > 0:
                    bars.append({
                        "resource": rt.value,
                        "value": round(cur, 1),
                        "capacity": round(cap, 1),
                        "fill": round(cur / cap, 3) if cap > 0 else 0.0,
                        "income": round(inc, 2),
                        "color": _RESOURCE_COLORS.get(rt, "#ffffff"),
                    })

        # Upkeep totals
        total_upkeep: dict[str, float] = {}
        for tmpl in self._active_units.get(faction_id, []):
            uc = self.unit_costs.get(tmpl)
            if uc:
                for rt, rate in uc.upkeep.items():
                    total_upkeep[rt.value] = total_upkeep.get(rt.value, 0.0) + rate

        return {
            "faction": faction_id,
            "resource_bars": bars,
            "upkeep_per_min": {k: round(v, 2) for k, v in total_upkeep.items()},
            "build_queue": bq.to_list() if bq else [],
            "tech_tree": tree.to_dict() if tree else {},
        }


# ---------------------------------------------------------------------------
# Resource display colours (cyberpunk palette)
# ---------------------------------------------------------------------------

_RESOURCE_COLORS: dict[ResourceType, str] = {
    ResourceType.CREDITS: "#fcee0a",      # yellow
    ResourceType.MANPOWER: "#05ffa1",     # green
    ResourceType.FUEL: "#ff8800",         # orange
    ResourceType.STEEL: "#888888",        # grey
    ResourceType.ELECTRONICS: "#00f0ff",  # cyan
    ResourceType.FOOD: "#66bb6a",         # light green
}


# ---------------------------------------------------------------------------
# UNIT_COSTS — costs for all unit templates in units.py
# ---------------------------------------------------------------------------

UNIT_COSTS: dict[str, UnitCost] = {
    "infantry": UnitCost(
        unit_template="infantry",
        cost={ResourceType.CREDITS: 50, ResourceType.MANPOWER: 1, ResourceType.FOOD: 5},
        build_time=5.0,
        upkeep={ResourceType.CREDITS: 2, ResourceType.FOOD: 1},
    ),
    "sniper": UnitCost(
        unit_template="sniper",
        cost={ResourceType.CREDITS: 120, ResourceType.MANPOWER: 1, ResourceType.ELECTRONICS: 10},
        build_time=10.0,
        upkeep={ResourceType.CREDITS: 5, ResourceType.FOOD: 1},
    ),
    "heavy": UnitCost(
        unit_template="heavy",
        cost={ResourceType.CREDITS: 150, ResourceType.MANPOWER: 1, ResourceType.STEEL: 30},
        build_time=12.0,
        upkeep={ResourceType.CREDITS: 6, ResourceType.FOOD: 2},
    ),
    "medic": UnitCost(
        unit_template="medic",
        cost={ResourceType.CREDITS: 80, ResourceType.MANPOWER: 1},
        build_time=8.0,
        upkeep={ResourceType.CREDITS: 3, ResourceType.FOOD: 1},
    ),
    "engineer": UnitCost(
        unit_template="engineer",
        cost={ResourceType.CREDITS: 100, ResourceType.MANPOWER: 1, ResourceType.STEEL: 10, ResourceType.ELECTRONICS: 5},
        build_time=10.0,
        upkeep={ResourceType.CREDITS: 4, ResourceType.FOOD: 1},
    ),
    "scout": UnitCost(
        unit_template="scout",
        cost={ResourceType.CREDITS: 60, ResourceType.MANPOWER: 1},
        build_time=4.0,
        upkeep={ResourceType.CREDITS: 2, ResourceType.FOOD: 1},
    ),
    "drone": UnitCost(
        unit_template="drone",
        cost={ResourceType.CREDITS: 200, ResourceType.ELECTRONICS: 40, ResourceType.FUEL: 10},
        build_time=15.0,
        upkeep={ResourceType.CREDITS: 8, ResourceType.FUEL: 3, ResourceType.ELECTRONICS: 2},
    ),
    "turret": UnitCost(
        unit_template="turret",
        cost={ResourceType.CREDITS: 300, ResourceType.STEEL: 80, ResourceType.ELECTRONICS: 20},
        build_time=20.0,
        upkeep={ResourceType.CREDITS: 5, ResourceType.ELECTRONICS: 1},
    ),
    # -- vehicles (if unlocked via tech) --
    "apc": UnitCost(
        unit_template="apc",
        cost={ResourceType.CREDITS: 400, ResourceType.STEEL: 100, ResourceType.FUEL: 30},
        build_time=25.0,
        upkeep={ResourceType.CREDITS: 10, ResourceType.FUEL: 8},
    ),
    "tank": UnitCost(
        unit_template="tank",
        cost={ResourceType.CREDITS: 800, ResourceType.STEEL: 200, ResourceType.FUEL: 50, ResourceType.ELECTRONICS: 30},
        build_time=40.0,
        upkeep={ResourceType.CREDITS: 20, ResourceType.FUEL: 15},
    ),
    "attack_helicopter": UnitCost(
        unit_template="attack_helicopter",
        cost={ResourceType.CREDITS: 1200, ResourceType.STEEL: 100, ResourceType.FUEL: 80, ResourceType.ELECTRONICS: 60},
        build_time=50.0,
        upkeep={ResourceType.CREDITS: 30, ResourceType.FUEL: 25, ResourceType.ELECTRONICS: 5},
    ),
    "patrol_boat": UnitCost(
        unit_template="patrol_boat",
        cost={ResourceType.CREDITS: 500, ResourceType.STEEL: 120, ResourceType.FUEL: 40},
        build_time=30.0,
        upkeep={ResourceType.CREDITS: 12, ResourceType.FUEL: 10},
    ),
}


# ---------------------------------------------------------------------------
# TECH_TREE — 15+ techs in 3 tiers
# ---------------------------------------------------------------------------


def _build_default_tech_tree() -> TechTree:
    """Construct the default tech tree (15 techs, 3 tiers)."""
    tree = TechTree()

    # -- Tier 1: no prerequisites ------------------------------------------
    tree.add_tech("basic_training", {ResourceType.CREDITS: 100},
                  unlocks=["infantry", "scout", "medic"])
    tree.add_tech("steel_works", {ResourceType.CREDITS: 150, ResourceType.STEEL: 20},
                  unlocks=["heavy", "turret"])
    tree.add_tech("electronics_lab", {ResourceType.CREDITS: 200, ResourceType.ELECTRONICS: 10},
                  unlocks=["sniper", "engineer"])
    tree.add_tech("logistics_hub", {ResourceType.CREDITS: 120},
                  unlocks=[])
    tree.add_tech("field_rations", {ResourceType.CREDITS: 80, ResourceType.FOOD: 20},
                  unlocks=[])

    # -- Tier 2: require one or more Tier 1 --------------------------------
    tree.add_tech("drone_warfare", {ResourceType.CREDITS: 400, ResourceType.ELECTRONICS: 30},
                  prerequisites=["electronics_lab"],
                  unlocks=["drone"])
    tree.add_tech("mechanized_infantry",
                  {ResourceType.CREDITS: 500, ResourceType.STEEL: 60, ResourceType.FUEL: 20},
                  prerequisites=["steel_works", "logistics_hub"],
                  unlocks=["apc"])
    tree.add_tech("advanced_optics", {ResourceType.CREDITS: 300, ResourceType.ELECTRONICS: 20},
                  prerequisites=["electronics_lab"],
                  unlocks=[])
    tree.add_tech("fortification_engineering",
                  {ResourceType.CREDITS: 250, ResourceType.STEEL: 40},
                  prerequisites=["steel_works"],
                  unlocks=[])
    tree.add_tech("supply_chain_optimization",
                  {ResourceType.CREDITS: 200},
                  prerequisites=["logistics_hub"],
                  unlocks=[])
    tree.add_tech("combat_medicine",
                  {ResourceType.CREDITS: 180},
                  prerequisites=["basic_training", "field_rations"],
                  unlocks=[])

    # -- Tier 3: require Tier 2 prerequisites ------------------------------
    tree.add_tech("armored_warfare",
                  {ResourceType.CREDITS: 800, ResourceType.STEEL: 120, ResourceType.FUEL: 40},
                  prerequisites=["mechanized_infantry"],
                  unlocks=["tank"])
    tree.add_tech("air_superiority",
                  {ResourceType.CREDITS: 1000, ResourceType.ELECTRONICS: 50, ResourceType.FUEL: 60},
                  prerequisites=["drone_warfare", "advanced_optics"],
                  unlocks=["attack_helicopter"])
    tree.add_tech("naval_operations",
                  {ResourceType.CREDITS: 600, ResourceType.STEEL: 80, ResourceType.FUEL: 30},
                  prerequisites=["mechanized_infantry"],
                  unlocks=["patrol_boat"])
    tree.add_tech("electronic_warfare",
                  {ResourceType.CREDITS: 700, ResourceType.ELECTRONICS: 60},
                  prerequisites=["advanced_optics", "drone_warfare"],
                  unlocks=[])

    return tree


TECH_TREE: TechTree = _build_default_tech_tree()


# ---------------------------------------------------------------------------
# ECONOMY_PRESETS — starting resources for different game modes
# ---------------------------------------------------------------------------

ECONOMY_PRESETS: dict[str, dict[str, dict[ResourceType, float]]] = {
    "standard": {
        "resources": {
            ResourceType.CREDITS: 1000,
            ResourceType.MANPOWER: 10,
            ResourceType.FUEL: 200,
            ResourceType.STEEL: 150,
            ResourceType.ELECTRONICS: 50,
            ResourceType.FOOD: 300,
        },
        "income": {
            ResourceType.CREDITS: 10,
            ResourceType.MANPOWER: 0.1,
            ResourceType.FUEL: 2,
            ResourceType.STEEL: 1.5,
            ResourceType.ELECTRONICS: 0.5,
            ResourceType.FOOD: 3,
        },
        "capacity": {
            ResourceType.CREDITS: 5000,
            ResourceType.MANPOWER: 50,
            ResourceType.FUEL: 1000,
            ResourceType.STEEL: 800,
            ResourceType.ELECTRONICS: 300,
            ResourceType.FOOD: 1500,
        },
    },
    "skirmish": {
        "resources": {
            ResourceType.CREDITS: 500,
            ResourceType.MANPOWER: 5,
            ResourceType.FUEL: 100,
            ResourceType.STEEL: 80,
            ResourceType.ELECTRONICS: 20,
            ResourceType.FOOD: 150,
        },
        "income": {
            ResourceType.CREDITS: 5,
            ResourceType.MANPOWER: 0.05,
            ResourceType.FUEL: 1,
            ResourceType.STEEL: 0.8,
            ResourceType.ELECTRONICS: 0.3,
            ResourceType.FOOD: 1.5,
        },
        "capacity": {
            ResourceType.CREDITS: 2500,
            ResourceType.MANPOWER: 25,
            ResourceType.FUEL: 500,
            ResourceType.STEEL: 400,
            ResourceType.ELECTRONICS: 150,
            ResourceType.FOOD: 800,
        },
    },
    "rich": {
        "resources": {
            ResourceType.CREDITS: 5000,
            ResourceType.MANPOWER: 30,
            ResourceType.FUEL: 800,
            ResourceType.STEEL: 600,
            ResourceType.ELECTRONICS: 200,
            ResourceType.FOOD: 1000,
        },
        "income": {
            ResourceType.CREDITS: 25,
            ResourceType.MANPOWER: 0.3,
            ResourceType.FUEL: 5,
            ResourceType.STEEL: 4,
            ResourceType.ELECTRONICS: 2,
            ResourceType.FOOD: 8,
        },
        "capacity": {
            ResourceType.CREDITS: 20000,
            ResourceType.MANPOWER: 100,
            ResourceType.FUEL: 5000,
            ResourceType.STEEL: 3000,
            ResourceType.ELECTRONICS: 1000,
            ResourceType.FOOD: 5000,
        },
    },
    "insurgent": {
        "resources": {
            ResourceType.CREDITS: 300,
            ResourceType.MANPOWER: 15,
            ResourceType.FUEL: 50,
            ResourceType.STEEL: 30,
            ResourceType.ELECTRONICS: 5,
            ResourceType.FOOD: 200,
        },
        "income": {
            ResourceType.CREDITS: 3,
            ResourceType.MANPOWER: 0.2,
            ResourceType.FUEL: 0.5,
            ResourceType.STEEL: 0.3,
            ResourceType.ELECTRONICS: 0.1,
            ResourceType.FOOD: 2,
        },
        "capacity": {
            ResourceType.CREDITS: 1500,
            ResourceType.MANPOWER: 40,
            ResourceType.FUEL: 300,
            ResourceType.STEEL: 200,
            ResourceType.ELECTRONICS: 50,
            ResourceType.FOOD: 800,
        },
    },
    "unlimited": {
        "resources": {
            ResourceType.CREDITS: 99999,
            ResourceType.MANPOWER: 999,
            ResourceType.FUEL: 99999,
            ResourceType.STEEL: 99999,
            ResourceType.ELECTRONICS: 99999,
            ResourceType.FOOD: 99999,
        },
        "income": {
            ResourceType.CREDITS: 100,
            ResourceType.MANPOWER: 1,
            ResourceType.FUEL: 50,
            ResourceType.STEEL: 50,
            ResourceType.ELECTRONICS: 50,
            ResourceType.FOOD: 50,
        },
        "capacity": {
            ResourceType.CREDITS: 999999,
            ResourceType.MANPOWER: 9999,
            ResourceType.FUEL: 999999,
            ResourceType.STEEL: 999999,
            ResourceType.ELECTRONICS: 999999,
            ResourceType.FOOD: 999999,
        },
    },
}
