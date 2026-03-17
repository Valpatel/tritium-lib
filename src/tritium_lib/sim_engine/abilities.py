# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Ability and special powers system for simulation units.

Each unit class gets a set of abilities — active skills, passive buffs,
toggles, and channeled powers.  The AbilityEngine manages cooldowns,
resource costs, range checks, and channeled-ability processing.  Abilities
produce structured event dicts that downstream systems (damage, medical,
status_effects, etc.) can consume.

Usage::

    from tritium_lib.sim_engine.abilities import (
        AbilityEngine, Ability, AbilityType, TargetType, ABILITIES,
    )

    engine = AbilityEngine()
    engine.grant_ability("alpha-1", ABILITIES["frag_grenade"])
    result = engine.activate("alpha-1", "frag_grenade", target_pos=(50, 30))
    events = engine.tick(0.5)

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import copy
import math
import enum
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AbilityType(enum.Enum):
    """How the ability is activated and sustained."""
    ACTIVE = "active"           # instant activation, goes on cooldown
    PASSIVE = "passive"         # always-on stat modifier
    TOGGLE = "toggle"           # on/off state, may drain resources while on
    CHANNELED = "channeled"     # must be maintained for duration, interruptible


class TargetType(enum.Enum):
    """What the ability can be aimed at."""
    SELF = "self"
    SINGLE_ALLY = "single_ally"
    SINGLE_ENEMY = "single_enemy"
    AREA_ALLY = "area_ally"
    AREA_ENEMY = "area_enemy"
    AREA_ALL = "area_all"
    NONE = "none"


# ---------------------------------------------------------------------------
# Ability dataclass
# ---------------------------------------------------------------------------

@dataclass
class Ability:
    """A single ability that can be granted to a unit.

    Attributes:
        ability_id:       Unique string key for this ability definition.
        name:             Human-readable display name.
        description:      Tooltip / flavour text.
        ability_type:     ACTIVE, PASSIVE, TOGGLE, or CHANNELED.
        target_type:      What the ability targets.
        cooldown:         Cooldown duration in seconds after activation.
        current_cooldown: Remaining cooldown (decremented by tick).
        cost:             Resource cost dict, e.g. {"ammo": 1, "energy": 20}.
        range:            Maximum activation range in meters (0 = self only).
        radius:           Blast / effect radius for area abilities.
        duration:         How long the effect lasts (buffs, channels).
        effects:          List of effect descriptors applied on activation.
        icon:             Icon identifier for Three.js UI.
        color:            Hex color for UI rendering.
        toggled_on:       Whether a TOGGLE ability is currently active.
    """
    ability_id: str
    name: str
    description: str
    ability_type: AbilityType
    target_type: TargetType
    cooldown: float
    current_cooldown: float = 0.0
    cost: dict[str, float] = field(default_factory=dict)
    range: float = 0.0
    radius: float = 0.0
    duration: float = 0.0
    effects: list[dict[str, Any]] = field(default_factory=list)
    icon: str = ""
    color: str = "#ffffff"
    toggled_on: bool = False

    def clone(self) -> Ability:
        """Return a deep copy suitable for granting to a unit."""
        return copy.deepcopy(self)

    @property
    def is_ready(self) -> bool:
        """True if the ability is off cooldown."""
        return self.current_cooldown <= 0.0


# ---------------------------------------------------------------------------
# AbilityEngine
# ---------------------------------------------------------------------------

class AbilityEngine:
    """Manages abilities across all simulation units.

    Handles granting, activation (with cooldown/cost/range validation),
    tick processing for cooldowns and channeled abilities, and serialisation
    for the Three.js frontend.
    """

    def __init__(self) -> None:
        # unit_id -> list of granted abilities
        self.unit_abilities: dict[str, list[Ability]] = {}
        # Active channeled abilities: {unit_id, ability_id, remaining, elapsed, target_pos, target_id}
        self.active_channels: list[dict[str, Any]] = []
        # Unit resource pools (ammo, energy, health, etc.)
        self.unit_resources: dict[str, dict[str, float]] = {}

    # -- resource management ------------------------------------------------

    def set_resources(self, unit_id: str, resources: dict[str, float]) -> None:
        """Set the resource pool for a unit."""
        self.unit_resources[unit_id] = dict(resources)

    def get_resources(self, unit_id: str) -> dict[str, float]:
        """Get current resource pool for a unit."""
        return dict(self.unit_resources.get(unit_id, {}))

    def _check_cost(self, unit_id: str, cost: dict[str, float]) -> bool:
        """Return True if the unit can afford the cost."""
        if not cost:
            return True
        resources = self.unit_resources.get(unit_id, {})
        for resource, amount in cost.items():
            if resources.get(resource, 0.0) < amount:
                return False
        return True

    def _pay_cost(self, unit_id: str, cost: dict[str, float]) -> None:
        """Deduct cost from the unit's resource pool."""
        if not cost:
            return
        resources = self.unit_resources.setdefault(unit_id, {})
        for resource, amount in cost.items():
            resources[resource] = resources.get(resource, 0.0) - amount

    # -- grant / revoke -----------------------------------------------------

    def grant_ability(self, unit_id: str, ability: Ability) -> Ability:
        """Grant a copy of *ability* to *unit_id*.

        If the unit already has an ability with the same ``ability_id``,
        the existing one is replaced.  Returns the granted ability instance.
        """
        abilities = self.unit_abilities.setdefault(unit_id, [])
        # Replace if already present
        for i, existing in enumerate(abilities):
            if existing.ability_id == ability.ability_id:
                abilities[i] = ability.clone()
                return abilities[i]
        new_ability = ability.clone()
        abilities.append(new_ability)
        return new_ability

    def revoke_ability(self, unit_id: str, ability_id: str) -> bool:
        """Remove an ability from a unit.  Returns True if found and removed."""
        abilities = self.unit_abilities.get(unit_id, [])
        for i, ab in enumerate(abilities):
            if ab.ability_id == ability_id:
                abilities.pop(i)
                # Also cancel any active channel
                self.active_channels = [
                    ch for ch in self.active_channels
                    if not (ch["unit_id"] == unit_id and ch["ability_id"] == ability_id)
                ]
                return True
        return False

    def get_ability(self, unit_id: str, ability_id: str) -> Ability | None:
        """Look up a specific ability on a unit."""
        for ab in self.unit_abilities.get(unit_id, []):
            if ab.ability_id == ability_id:
                return ab
        return None

    # -- activation ---------------------------------------------------------

    def activate(
        self,
        unit_id: str,
        ability_id: str,
        target_pos: Vec2 | None = None,
        target_id: str | None = None,
        unit_pos: Vec2 | None = None,
    ) -> dict[str, Any]:
        """Activate an ability on a unit.

        Validates cooldown, cost, and range before applying.

        Args:
            unit_id:    The unit activating the ability.
            ability_id: Which ability to activate.
            target_pos: World position for targeted/area abilities.
            target_id:  Target unit ID for single-target abilities.
            unit_pos:   Position of the activating unit (for range checks).

        Returns:
            A result dict with at minimum ``{"success": bool, "reason": str}``.
            On success, includes ``"effects"`` and other activation data.
        """
        ability = self.get_ability(unit_id, ability_id)
        if ability is None:
            return {"success": False, "reason": "ability_not_found",
                    "unit_id": unit_id, "ability_id": ability_id}

        # Passive abilities cannot be "activated"
        if ability.ability_type == AbilityType.PASSIVE:
            return {"success": False, "reason": "passive_cannot_activate",
                    "unit_id": unit_id, "ability_id": ability_id}

        # Toggle abilities flip state
        if ability.ability_type == AbilityType.TOGGLE:
            return self._toggle(unit_id, ability)

        # Cooldown check
        if ability.current_cooldown > 0:
            return {"success": False, "reason": "on_cooldown",
                    "unit_id": unit_id, "ability_id": ability_id,
                    "remaining": round(ability.current_cooldown, 2)}

        # Cost check
        if not self._check_cost(unit_id, ability.cost):
            return {"success": False, "reason": "insufficient_resources",
                    "unit_id": unit_id, "ability_id": ability_id,
                    "cost": ability.cost}

        # Range check (only if target_pos and unit_pos are provided and range > 0)
        if ability.range > 0 and target_pos is not None and unit_pos is not None:
            dist = distance(unit_pos, target_pos)
            if dist > ability.range:
                return {"success": False, "reason": "out_of_range",
                        "unit_id": unit_id, "ability_id": ability_id,
                        "distance": round(dist, 2), "max_range": ability.range}

        # Pay cost
        self._pay_cost(unit_id, ability.cost)

        # Set cooldown
        ability.current_cooldown = ability.cooldown

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "unit_id": unit_id,
            "ability_id": ability_id,
            "ability_name": ability.name,
            "ability_type": ability.ability_type.value,
            "target_type": ability.target_type.value,
            "effects": list(ability.effects),
        }

        if target_pos is not None:
            result["target_pos"] = target_pos
        if target_id is not None:
            result["target_id"] = target_id
        if ability.radius > 0:
            result["radius"] = ability.radius
        if ability.duration > 0:
            result["duration"] = ability.duration

        # If channeled, start tracking
        if ability.ability_type == AbilityType.CHANNELED:
            channel = {
                "unit_id": unit_id,
                "ability_id": ability_id,
                "remaining": ability.duration,
                "elapsed": 0.0,
                "target_pos": target_pos,
                "target_id": target_id,
                "effects": list(ability.effects),
            }
            self.active_channels.append(channel)
            result["channeling"] = True

        return result

    def _toggle(self, unit_id: str, ability: Ability) -> dict[str, Any]:
        """Handle toggle ability activation (flip on/off)."""
        if ability.toggled_on:
            # Turning off — no cost or cooldown
            ability.toggled_on = False
            return {
                "success": True,
                "unit_id": unit_id,
                "ability_id": ability.ability_id,
                "ability_name": ability.name,
                "toggled": False,
                "effects": [],
            }
        else:
            # Turning on — check cost
            if not self._check_cost(unit_id, ability.cost):
                return {"success": False, "reason": "insufficient_resources",
                        "unit_id": unit_id, "ability_id": ability.ability_id,
                        "cost": ability.cost}
            self._pay_cost(unit_id, ability.cost)
            ability.toggled_on = True
            return {
                "success": True,
                "unit_id": unit_id,
                "ability_id": ability.ability_id,
                "ability_name": ability.name,
                "toggled": True,
                "effects": list(ability.effects),
            }

    # -- interrupt ----------------------------------------------------------

    def interrupt_channel(self, unit_id: str, ability_id: str | None = None) -> bool:
        """Interrupt a channeled ability.

        If *ability_id* is None, interrupts all channels for the unit.
        Returns True if anything was interrupted.
        """
        before = len(self.active_channels)
        if ability_id is None:
            self.active_channels = [
                ch for ch in self.active_channels if ch["unit_id"] != unit_id
            ]
        else:
            self.active_channels = [
                ch for ch in self.active_channels
                if not (ch["unit_id"] == unit_id and ch["ability_id"] == ability_id)
            ]
        return len(self.active_channels) < before

    # -- tick ---------------------------------------------------------------

    def tick(self, dt: float) -> list[dict[str, Any]]:
        """Advance cooldowns and process channeled abilities.

        Returns a list of event dicts:
        - ``{"type": "channel_tick", "unit_id": ..., "ability_id": ..., "elapsed": ..., "remaining": ...}``
        - ``{"type": "channel_complete", "unit_id": ..., "ability_id": ..., "effects": [...]}``
        - ``{"type": "cooldown_ready", "unit_id": ..., "ability_id": ...}``
        """
        events: list[dict[str, Any]] = []

        # Reduce cooldowns on all abilities
        for unit_id, abilities in self.unit_abilities.items():
            for ab in abilities:
                if ab.current_cooldown > 0:
                    old_cd = ab.current_cooldown
                    ab.current_cooldown = max(0.0, ab.current_cooldown - dt)
                    if old_cd > 0 and ab.current_cooldown <= 0:
                        events.append({
                            "type": "cooldown_ready",
                            "unit_id": unit_id,
                            "ability_id": ab.ability_id,
                        })

        # Process channeled abilities
        completed: list[int] = []
        for idx, ch in enumerate(self.active_channels):
            ch["elapsed"] += dt
            ch["remaining"] -= dt

            if ch["remaining"] <= 0:
                # Channel complete
                completed.append(idx)
                events.append({
                    "type": "channel_complete",
                    "unit_id": ch["unit_id"],
                    "ability_id": ch["ability_id"],
                    "effects": ch["effects"],
                    "target_pos": ch.get("target_pos"),
                    "target_id": ch.get("target_id"),
                })
            else:
                events.append({
                    "type": "channel_tick",
                    "unit_id": ch["unit_id"],
                    "ability_id": ch["ability_id"],
                    "elapsed": round(ch["elapsed"], 3),
                    "remaining": round(ch["remaining"], 3),
                })

        # Remove completed channels (reverse order)
        for idx in reversed(completed):
            self.active_channels.pop(idx)

        return events

    # -- queries ------------------------------------------------------------

    def get_available(self, unit_id: str) -> list[Ability]:
        """Return abilities that are off cooldown and not passive for *unit_id*."""
        result: list[Ability] = []
        for ab in self.unit_abilities.get(unit_id, []):
            if ab.ability_type == AbilityType.PASSIVE:
                continue
            if ab.current_cooldown <= 0:
                result.append(ab)
        return result

    def get_all(self, unit_id: str) -> list[Ability]:
        """Return all abilities for *unit_id*."""
        return list(self.unit_abilities.get(unit_id, []))

    def get_passives(self, unit_id: str) -> list[Ability]:
        """Return only passive abilities for *unit_id*."""
        return [
            ab for ab in self.unit_abilities.get(unit_id, [])
            if ab.ability_type == AbilityType.PASSIVE
        ]

    def get_active_toggles(self, unit_id: str) -> list[Ability]:
        """Return toggle abilities that are currently switched on."""
        return [
            ab for ab in self.unit_abilities.get(unit_id, [])
            if ab.ability_type == AbilityType.TOGGLE and ab.toggled_on
        ]

    def is_channeling(self, unit_id: str) -> bool:
        """Return True if the unit is currently channeling any ability."""
        return any(ch["unit_id"] == unit_id for ch in self.active_channels)

    def get_passive_modifiers(self, unit_id: str) -> dict[str, float]:
        """Aggregate stat modifiers from all passive and active-toggle abilities.

        Effects with a ``"stat"`` and ``"value"`` key are summed.
        """
        mods: dict[str, float] = {}
        for ab in self.unit_abilities.get(unit_id, []):
            active = (
                ab.ability_type == AbilityType.PASSIVE
                or (ab.ability_type == AbilityType.TOGGLE and ab.toggled_on)
            )
            if not active:
                continue
            for eff in ab.effects:
                if "stat" in eff and "value" in eff:
                    stat = eff["stat"]
                    mods[stat] = mods.get(stat, 0.0) + eff["value"]
        return mods

    # -- Three.js serialisation ---------------------------------------------

    def to_three_js(self, unit_id: str) -> list[dict[str, Any]]:
        """Return ability display data for the frontend.

        Each dict contains: ability_id, name, icon, color, ability_type,
        target_type, cooldown, current_cooldown, progress (0-1 cooldown bar),
        toggled_on, is_channeling.
        """
        result: list[dict[str, Any]] = []
        is_ch = {ch["ability_id"] for ch in self.active_channels if ch["unit_id"] == unit_id}

        for ab in self.unit_abilities.get(unit_id, []):
            if ab.cooldown > 0:
                progress = max(0.0, min(1.0, 1.0 - ab.current_cooldown / ab.cooldown))
            else:
                progress = 1.0

            result.append({
                "ability_id": ab.ability_id,
                "name": ab.name,
                "description": ab.description,
                "icon": ab.icon,
                "color": ab.color,
                "ability_type": ab.ability_type.value,
                "target_type": ab.target_type.value,
                "cooldown": ab.cooldown,
                "current_cooldown": round(ab.current_cooldown, 2),
                "progress": round(progress, 3),
                "ready": ab.is_ready,
                "toggled_on": ab.toggled_on,
                "is_channeling": ab.ability_id in is_ch,
                "cost": ab.cost,
                "range": ab.range,
                "radius": ab.radius,
            })
        return result


# ---------------------------------------------------------------------------
# Helper to build abilities concisely
# ---------------------------------------------------------------------------

def _ab(
    ability_id: str,
    name: str,
    description: str,
    ability_type: AbilityType,
    target_type: TargetType,
    cooldown: float,
    *,
    cost: dict[str, float] | None = None,
    range_: float = 0.0,
    radius: float = 0.0,
    duration: float = 0.0,
    effects: list[dict[str, Any]] | None = None,
    icon: str = "",
    color: str = "#ffffff",
) -> Ability:
    return Ability(
        ability_id=ability_id,
        name=name,
        description=description,
        ability_type=ability_type,
        target_type=target_type,
        cooldown=cooldown,
        cost=cost or {},
        range=range_,
        radius=radius,
        duration=duration,
        effects=effects or [],
        icon=icon,
        color=color,
    )


# ---------------------------------------------------------------------------
# ABILITIES catalog — 27 predefined abilities across 7 unit classes
# ---------------------------------------------------------------------------

ABILITIES: dict[str, Ability] = {}

# -- Infantry (6) ----------------------------------------------------------

ABILITIES["frag_grenade"] = _ab(
    "frag_grenade", "Frag Grenade",
    "Throw a fragmentation grenade dealing explosive damage in an area.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=15.0,
    cost={"grenades": 1}, range_=30.0, radius=8.0,
    effects=[{"type": "damage", "damage_type": "explosive", "base_damage": 60.0},
             {"type": "status", "name": "concussed", "duration": 5.0}],
    icon="bomb", color="#ff4400",
)

ABILITIES["smoke_grenade"] = _ab(
    "smoke_grenade", "Smoke Grenade",
    "Deploy a smoke screen that obscures line of sight.",
    AbilityType.ACTIVE, TargetType.AREA_ALL, cooldown=20.0,
    cost={"grenades": 1}, range_=25.0, radius=10.0, duration=15.0,
    effects=[{"type": "area_effect", "effect_type": "smoke",
              "stat": "detection", "value": -0.8}],
    icon="cloud", color="#999999",
)

ABILITIES["flashbang"] = _ab(
    "flashbang", "Flashbang",
    "Stun and blind enemies in a small radius.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=18.0,
    cost={"grenades": 1}, range_=20.0, radius=6.0,
    effects=[{"type": "status", "name": "blinded", "duration": 5.0},
             {"type": "status", "name": "deafened", "duration": 8.0}],
    icon="zap", color="#ffff00",
)

ABILITIES["sprint"] = _ab(
    "sprint", "Sprint",
    "Burst of speed for a short duration.",
    AbilityType.ACTIVE, TargetType.SELF, cooldown=12.0,
    duration=4.0,
    effects=[{"type": "buff", "stat": "speed", "value": 0.5},
             {"type": "debuff", "stat": "accuracy", "value": -0.3}],
    icon="fast-forward", color="#05ffa1",
)

ABILITIES["go_prone"] = _ab(
    "go_prone", "Go Prone",
    "Drop to the ground for better accuracy and reduced visibility.",
    AbilityType.TOGGLE, TargetType.SELF, cooldown=1.0,
    effects=[{"type": "buff", "stat": "accuracy", "value": 0.2},
             {"type": "buff", "stat": "stealth", "value": 0.3},
             {"type": "debuff", "stat": "speed", "value": -0.7}],
    icon="arrow-down", color="#336633",
)

ABILITIES["rally_cry"] = _ab(
    "rally_cry", "Rally Cry",
    "Boost morale of all nearby allies.",
    AbilityType.ACTIVE, TargetType.AREA_ALLY, cooldown=30.0,
    radius=15.0, duration=10.0,
    effects=[{"type": "buff", "stat": "morale", "value": 0.3},
             {"type": "buff", "stat": "accuracy", "value": 0.1}],
    icon="megaphone", color="#fcee0a",
)

# -- Medic (4) -------------------------------------------------------------

ABILITIES["first_aid"] = _ab(
    "first_aid", "First Aid",
    "Heal a single ally for a moderate amount.",
    AbilityType.ACTIVE, TargetType.SINGLE_ALLY, cooldown=8.0,
    cost={"medical_supplies": 1}, range_=3.0,
    effects=[{"type": "heal", "amount": 35.0}],
    icon="heart", color="#ff2a6d",
)

ABILITIES["triage_scan"] = _ab(
    "triage_scan", "Triage Scan",
    "Reveal injury status of all nearby units.",
    AbilityType.ACTIVE, TargetType.AREA_ALL, cooldown=10.0,
    radius=20.0,
    effects=[{"type": "reveal", "reveal_type": "injuries"}],
    icon="search", color="#00f0ff",
)

ABILITIES["morphine_shot"] = _ab(
    "morphine_shot", "Morphine Shot",
    "Suppress pain and slowly heal a wounded ally.",
    AbilityType.ACTIVE, TargetType.SINGLE_ALLY, cooldown=25.0,
    cost={"medical_supplies": 1}, range_=3.0, duration=30.0,
    effects=[{"type": "status", "name": "morphine", "duration": 60.0},
             {"type": "heal_over_time", "heal_per_second": 1.0}],
    icon="syringe", color="#66ccff",
)

ABILITIES["evac_call"] = _ab(
    "evac_call", "Evacuation Call",
    "Request medical evacuation for a critical casualty.",
    AbilityType.ACTIVE, TargetType.SINGLE_ALLY, cooldown=60.0,
    range_=5.0,
    effects=[{"type": "evac_request", "priority": "immediate"}],
    icon="phone", color="#ff0000",
)

# -- Engineer (5) ----------------------------------------------------------

ABILITIES["build_cover"] = _ab(
    "build_cover", "Build Cover",
    "Construct a defensive barrier at target location.",
    AbilityType.CHANNELED, TargetType.AREA_ALL, cooldown=20.0,
    cost={"building_materials": 2}, range_=5.0, duration=5.0,
    effects=[{"type": "spawn", "spawn_type": "cover", "defense": 0.4}],
    icon="shield", color="#05ffa1",
)

ABILITIES["plant_mine"] = _ab(
    "plant_mine", "Plant Mine",
    "Place an anti-personnel mine at target location.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=15.0,
    cost={"explosives": 1}, range_=3.0, radius=5.0,
    effects=[{"type": "spawn", "spawn_type": "mine",
              "damage_type": "explosive", "base_damage": 80.0}],
    icon="alert-triangle", color="#ff6600",
)

ABILITIES["repair_vehicle"] = _ab(
    "repair_vehicle", "Repair Vehicle",
    "Repair a damaged vehicle over time.",
    AbilityType.CHANNELED, TargetType.SINGLE_ALLY, cooldown=30.0,
    cost={"building_materials": 3}, range_=4.0, duration=10.0,
    effects=[{"type": "repair", "amount": 50.0},
             {"type": "remove_status", "name": "engine_damage"}],
    icon="wrench", color="#ffcc00",
)

ABILITIES["breach_wall"] = _ab(
    "breach_wall", "Breach Wall",
    "Blow a hole in a wall or fortification.",
    AbilityType.ACTIVE, TargetType.AREA_ALL, cooldown=25.0,
    cost={"explosives": 2}, range_=5.0, radius=3.0,
    effects=[{"type": "destruction", "structure_damage": 100.0},
             {"type": "damage", "damage_type": "explosive", "base_damage": 30.0}],
    icon="x-circle", color="#ff4400",
)

ABILITIES["defuse_bomb"] = _ab(
    "defuse_bomb", "Defuse Bomb",
    "Carefully defuse an explosive device.",
    AbilityType.CHANNELED, TargetType.NONE, cooldown=10.0,
    range_=2.0, duration=8.0,
    effects=[{"type": "defuse"}],
    icon="scissors", color="#00ff00",
)

# -- Sniper (3) ------------------------------------------------------------

ABILITIES["hold_breath"] = _ab(
    "hold_breath", "Hold Breath",
    "Steady aim for maximum accuracy on the next shot.",
    AbilityType.ACTIVE, TargetType.SELF, cooldown=8.0,
    duration=5.0,
    effects=[{"type": "buff", "stat": "accuracy", "value": 0.3},
             {"type": "buff", "stat": "critical_chance", "value": 0.15}],
    icon="crosshair", color="#00f0ff",
)

ABILITIES["mark_target"] = _ab(
    "mark_target", "Mark Target",
    "Tag an enemy, making them visible to all allies and increasing damage taken.",
    AbilityType.ACTIVE, TargetType.SINGLE_ENEMY, cooldown=15.0,
    range_=100.0, duration=15.0,
    effects=[{"type": "status", "name": "marked", "duration": 15.0},
             {"type": "reveal", "reveal_type": "position"}],
    icon="target", color="#ff2a6d",
)

ABILITIES["ghillie_deploy"] = _ab(
    "ghillie_deploy", "Ghillie Deploy",
    "Activate ghillie camouflage — nearly invisible while stationary.",
    AbilityType.TOGGLE, TargetType.SELF, cooldown=2.0,
    effects=[{"type": "buff", "stat": "stealth", "value": 0.7},
             {"type": "debuff", "stat": "speed", "value": -0.8}],
    icon="eye-off", color="#336633",
)

# -- Heavy (3) -------------------------------------------------------------

ABILITIES["suppressive_fire"] = _ab(
    "suppressive_fire", "Suppressive Fire",
    "Lay down heavy suppressive fire in a cone, pinning enemies.",
    AbilityType.CHANNELED, TargetType.AREA_ENEMY, cooldown=20.0,
    cost={"ammo": 30}, range_=40.0, radius=12.0, duration=6.0,
    effects=[{"type": "damage", "damage_type": "kinetic", "base_damage": 5.0,
              "rate": 10.0},
             {"type": "status", "name": "suppressed", "duration": 8.0}],
    icon="volume-2", color="#ff6600",
)

ABILITIES["deploy_bipod"] = _ab(
    "deploy_bipod", "Deploy Bipod",
    "Set up the weapon on a bipod for improved accuracy and fire rate.",
    AbilityType.TOGGLE, TargetType.SELF, cooldown=2.0,
    effects=[{"type": "buff", "stat": "accuracy", "value": 0.3},
             {"type": "buff", "stat": "fire_rate", "value": 0.2},
             {"type": "debuff", "stat": "speed", "value": -0.9}],
    icon="anchor", color="#00f0ff",
)

ABILITIES["rocket_barrage"] = _ab(
    "rocket_barrage", "Rocket Barrage",
    "Fire a salvo of rockets at a target area.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=45.0,
    cost={"rockets": 3}, range_=60.0, radius=15.0,
    effects=[{"type": "damage", "damage_type": "explosive", "base_damage": 80.0},
             {"type": "status", "name": "burning", "duration": 6.0}],
    icon="rocket", color="#ff0000",
)

# -- Commander (4) ---------------------------------------------------------

ABILITIES["call_airstrike"] = _ab(
    "call_airstrike", "Call Airstrike",
    "Request an airstrike on a target area. Massive damage after a delay.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=120.0,
    range_=200.0, radius=25.0,
    effects=[{"type": "damage", "damage_type": "explosive", "base_damage": 200.0,
              "delay": 8.0},
             {"type": "status", "name": "burning", "duration": 10.0}],
    icon="cloud-lightning", color="#ff2a6d",
)

ABILITIES["call_artillery"] = _ab(
    "call_artillery", "Call Artillery",
    "Request artillery bombardment on a target zone.",
    AbilityType.ACTIVE, TargetType.AREA_ENEMY, cooldown=90.0,
    range_=300.0, radius=20.0, duration=10.0,
    effects=[{"type": "damage", "damage_type": "explosive", "base_damage": 100.0,
              "rounds": 6, "interval": 1.5},
             {"type": "status", "name": "shell_shocked", "duration": 10.0}],
    icon="target", color="#ff4400",
)

ABILITIES["radar_sweep"] = _ab(
    "radar_sweep", "Radar Sweep",
    "Reveal all enemy positions in a large area for a short time.",
    AbilityType.ACTIVE, TargetType.AREA_ALL, cooldown=45.0,
    radius=100.0, duration=8.0,
    effects=[{"type": "reveal", "reveal_type": "all_enemies"},
             {"type": "status", "name": "spotted", "duration": 10.0}],
    icon="radio", color="#00f0ff",
)

ABILITIES["reinforce"] = _ab(
    "reinforce", "Reinforce",
    "Call in reinforcement units at a rally point.",
    AbilityType.ACTIVE, TargetType.AREA_ALLY, cooldown=180.0,
    range_=50.0, radius=10.0,
    effects=[{"type": "spawn", "spawn_type": "reinforcements", "count": 4,
              "unit_type": "infantry"}],
    icon="users", color="#05ffa1",
)

# -- Scout (3) -------------------------------------------------------------

ABILITIES["binoculars"] = _ab(
    "binoculars", "Binoculars",
    "Scan a distant area to reveal enemy positions.",
    AbilityType.ACTIVE, TargetType.AREA_ALL, cooldown=10.0,
    range_=120.0, radius=20.0, duration=5.0,
    effects=[{"type": "reveal", "reveal_type": "area"},
             {"type": "buff", "stat": "detection_range", "value": 0.5}],
    icon="eye", color="#00f0ff",
)

ABILITIES["tag_enemy"] = _ab(
    "tag_enemy", "Tag Enemy",
    "Electronically tag an enemy, tracking them through fog of war.",
    AbilityType.ACTIVE, TargetType.SINGLE_ENEMY, cooldown=20.0,
    range_=60.0, duration=30.0,
    effects=[{"type": "status", "name": "spotted", "duration": 30.0},
             {"type": "reveal", "reveal_type": "continuous_tracking"}],
    icon="tag", color="#ff2a6d",
)

ABILITIES["silent_move"] = _ab(
    "silent_move", "Silent Move",
    "Move silently, invisible to enemy detection for a short time.",
    AbilityType.ACTIVE, TargetType.SELF, cooldown=25.0,
    duration=8.0,
    effects=[{"type": "buff", "stat": "stealth", "value": 0.9},
             {"type": "debuff", "stat": "speed", "value": -0.2}],
    icon="volume-x", color="#336633",
)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "AbilityType",
    "TargetType",
    "Ability",
    "AbilityEngine",
    "ABILITIES",
]
