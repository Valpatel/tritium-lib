# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Status effects and buffs/debuffs system for the Tritium sim engine.

Manages temporary and permanent modifiers on units: buffs, debuffs,
damage-over-time, heal-over-time, and crowd control effects.  Each effect
can stack, expire, and emit tick events for DOT/HOT processing.

Usage::

    from tritium_lib.sim_engine.status_effects import (
        StatusEffectEngine, StatusEffect, EffectType, EFFECTS_CATALOG,
    )

    engine = StatusEffectEngine()
    engine.apply("alpha-1", EFFECTS_CATALOG["suppressed"])
    mod = engine.get_modifier("alpha-1", "accuracy")   # e.g. -0.4
    events = engine.tick(0.5)                            # advance 0.5s

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EffectType(enum.Enum):
    """Category of status effect."""
    BUFF = "buff"
    DEBUFF = "debuff"
    DOT = "dot"           # damage over time
    HOT = "hot"           # heal over time
    CROWD_CONTROL = "crowd_control"


# ---------------------------------------------------------------------------
# StatusEffect dataclass
# ---------------------------------------------------------------------------

@dataclass
class StatusEffect:
    """A single status effect that can be applied to a unit.

    Attributes:
        effect_id:         Unique identifier for this effect instance.
        name:              Human-readable name (also used as lookup key).
        effect_type:       Category — buff, debuff, dot, hot, crowd_control.
        duration:          Total duration in seconds.  -1 = permanent.
        remaining:         Seconds remaining.  Decremented by ``tick()``.
        stacks:            Current stack count.
        max_stacks:        Maximum allowed stacks.
        stat_modifiers:    Additive modifiers keyed by stat name,
                           e.g. ``{"speed": -0.3, "accuracy": 0.1}``.
        damage_per_second: DOT damage applied each tick (scaled by dt).
        heal_per_second:   HOT healing applied each tick (scaled by dt).
        icon:              Icon identifier for Three.js display.
        color:             Hex color string for UI rendering.
    """
    effect_id: str
    name: str
    effect_type: EffectType
    duration: float
    remaining: float
    stacks: int = 1
    max_stacks: int = 1
    stat_modifiers: dict[str, float] = field(default_factory=dict)
    damage_per_second: float = 0.0
    heal_per_second: float = 0.0
    icon: str = ""
    color: str = "#ffffff"

    def clone(self) -> StatusEffect:
        """Return a deep copy suitable for applying to a unit."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# StatusEffectEngine
# ---------------------------------------------------------------------------

class StatusEffectEngine:
    """Manages active status effects across all units.

    Effects are stored per-unit and processed each tick.  The engine handles
    stacking, expiration, DOT/HOT damage, and returns structured events.
    """

    def __init__(self) -> None:
        self.active_effects: dict[str, list[StatusEffect]] = {}

    # -- mutators -----------------------------------------------------------

    def apply(self, unit_id: str, effect: StatusEffect) -> StatusEffect:
        """Apply (or stack) *effect* on *unit_id*.

        If the unit already has an effect with the same ``name``:
          - If below ``max_stacks``, increment ``stacks`` and refresh
            ``remaining`` to the longer of old remaining vs new duration.
          - If at ``max_stacks``, just refresh ``remaining``.

        Returns the active effect instance on the unit.
        """
        effects = self.active_effects.setdefault(unit_id, [])

        for existing in effects:
            if existing.name == effect.name:
                if existing.stacks < existing.max_stacks:
                    existing.stacks += 1
                # Refresh duration to whichever is longer
                existing.remaining = max(existing.remaining, effect.duration)
                return existing

        new_effect = effect.clone()
        new_effect.remaining = new_effect.duration if new_effect.duration > 0 else float("inf")
        effects.append(new_effect)
        return new_effect

    def remove(self, unit_id: str, effect_name: str) -> bool:
        """Remove all instances of *effect_name* from *unit_id*.

        Returns ``True`` if anything was removed.
        """
        effects = self.active_effects.get(unit_id)
        if not effects:
            return False
        before = len(effects)
        self.active_effects[unit_id] = [e for e in effects if e.name != effect_name]
        if not self.active_effects[unit_id]:
            del self.active_effects[unit_id]
        return len(self.active_effects.get(unit_id, [])) < before

    def clear_all(self, unit_id: str) -> int:
        """Remove every effect from *unit_id*.  Returns count removed."""
        effects = self.active_effects.pop(unit_id, [])
        return len(effects)

    # -- queries ------------------------------------------------------------

    def has_effect(self, unit_id: str, name: str) -> bool:
        """Return ``True`` if *unit_id* currently has an effect named *name*."""
        for e in self.active_effects.get(unit_id, []):
            if e.name == name:
                return True
        return False

    def get_effects(self, unit_id: str) -> list[StatusEffect]:
        """Return a copy of the active effects list for *unit_id*."""
        return list(self.active_effects.get(unit_id, []))

    def get_modifier(self, unit_id: str, stat: str) -> float:
        """Compute the total additive modifier for *stat* on *unit_id*.

        Each effect contributes ``modifier_value * stacks``.  The values are
        summed additively — a +0.2 and a -0.3 yield -0.1.
        """
        total = 0.0
        for e in self.active_effects.get(unit_id, []):
            if stat in e.stat_modifiers:
                total += e.stat_modifiers[stat] * e.stacks
        return total

    # -- tick ---------------------------------------------------------------

    def tick(self, dt: float) -> list[dict[str, Any]]:
        """Advance all effects by *dt* seconds.

        Returns a list of event dicts with the following shapes:

        - ``{"type": "dot_tick",     "unit_id": ..., "effect": ..., "damage": ...}``
        - ``{"type": "hot_tick",     "unit_id": ..., "effect": ..., "healing": ...}``
        - ``{"type": "effect_expired", "unit_id": ..., "effect": ...}``
        """
        events: list[dict[str, Any]] = []
        expired_units: list[str] = []

        for unit_id, effects in self.active_effects.items():
            to_remove: list[int] = []
            for idx, eff in enumerate(effects):
                # DOT damage
                if eff.damage_per_second > 0:
                    dmg = eff.damage_per_second * eff.stacks * dt
                    events.append({
                        "type": "dot_tick",
                        "unit_id": unit_id,
                        "effect": eff.name,
                        "damage": round(dmg, 4),
                    })

                # HOT healing
                if eff.heal_per_second > 0:
                    heal = eff.heal_per_second * eff.stacks * dt
                    events.append({
                        "type": "hot_tick",
                        "unit_id": unit_id,
                        "effect": eff.name,
                        "healing": round(heal, 4),
                    })

                # Decrement remaining (permanent effects have inf remaining)
                if eff.duration > 0:
                    eff.remaining -= dt
                    if eff.remaining <= 0:
                        to_remove.append(idx)
                        events.append({
                            "type": "effect_expired",
                            "unit_id": unit_id,
                            "effect": eff.name,
                        })

            # Remove expired (reverse order to preserve indices)
            for idx in reversed(to_remove):
                effects.pop(idx)

            if not effects:
                expired_units.append(unit_id)

        for uid in expired_units:
            del self.active_effects[uid]

        return events

    # -- serialisation for Three.js -----------------------------------------

    def to_three_js(self, unit_id: str) -> list[dict[str, Any]]:
        """Return a list of dicts describing active effects for frontend rendering.

        Each dict contains: name, icon, color, remaining, duration, stacks,
        max_stacks, effect_type, and a ``progress`` bar value (0.0-1.0).
        """
        result: list[dict[str, Any]] = []
        for eff in self.active_effects.get(unit_id, []):
            if eff.duration > 0:
                progress = max(0.0, min(1.0, eff.remaining / eff.duration))
            else:
                progress = 1.0  # permanent
            result.append({
                "name": eff.name,
                "icon": eff.icon,
                "color": eff.color,
                "remaining": round(eff.remaining, 2) if eff.remaining != float("inf") else -1,
                "duration": eff.duration,
                "stacks": eff.stacks,
                "max_stacks": eff.max_stacks,
                "effect_type": eff.effect_type.value,
                "progress": round(progress, 3),
            })
        return result


# ---------------------------------------------------------------------------
# EFFECTS_CATALOG — 30+ predefined effects
# ---------------------------------------------------------------------------

def _make(
    effect_id: str,
    name: str,
    effect_type: EffectType,
    duration: float,
    *,
    max_stacks: int = 1,
    stat_modifiers: dict[str, float] | None = None,
    damage_per_second: float = 0.0,
    heal_per_second: float = 0.0,
    icon: str = "",
    color: str = "#ffffff",
) -> StatusEffect:
    return StatusEffect(
        effect_id=effect_id,
        name=name,
        effect_type=effect_type,
        duration=duration,
        remaining=duration if duration > 0 else float("inf"),
        max_stacks=max_stacks,
        stat_modifiers=stat_modifiers or {},
        damage_per_second=damage_per_second,
        heal_per_second=heal_per_second,
        icon=icon,
        color=color,
    )


EFFECTS_CATALOG: dict[str, StatusEffect] = {}

# -- Combat ----------------------------------------------------------------

EFFECTS_CATALOG["suppressed"] = _make(
    "combat_suppressed", "suppressed", EffectType.DEBUFF, 5.0,
    stat_modifiers={"accuracy": -0.4, "speed": -0.2},
    max_stacks=3, icon="shield-alert", color="#ff6600",
)
EFFECTS_CATALOG["pinned"] = _make(
    "combat_pinned", "pinned", EffectType.CROWD_CONTROL, 8.0,
    stat_modifiers={"speed": -0.9, "accuracy": -0.3},
    icon="pin", color="#ff3300",
)
EFFECTS_CATALOG["flanked"] = _make(
    "combat_flanked", "flanked", EffectType.DEBUFF, 4.0,
    stat_modifiers={"defense": -0.3, "morale": -0.15},
    icon="crosshair", color="#ff2a6d",
)
EFFECTS_CATALOG["entrenched"] = _make(
    "combat_entrenched", "entrenched", EffectType.BUFF, -1,
    stat_modifiers={"defense": 0.4, "speed": -0.5},
    icon="shield", color="#05ffa1",
)
EFFECTS_CATALOG["overwatch"] = _make(
    "combat_overwatch", "overwatch", EffectType.BUFF, -1,
    stat_modifiers={"accuracy": 0.2, "reaction": 0.3},
    icon="eye", color="#00f0ff",
)
EFFECTS_CATALOG["adrenaline"] = _make(
    "combat_adrenaline", "adrenaline", EffectType.BUFF, 10.0,
    stat_modifiers={"speed": 0.3, "damage": 0.15, "accuracy": -0.1},
    icon="zap", color="#fcee0a",
)

# -- Medical ---------------------------------------------------------------

EFFECTS_CATALOG["bleeding"] = _make(
    "med_bleeding", "bleeding", EffectType.DOT, 30.0,
    damage_per_second=2.0, max_stacks=3,
    icon="droplet", color="#cc0000",
)
EFFECTS_CATALOG["concussed"] = _make(
    "med_concussed", "concussed", EffectType.DEBUFF, 15.0,
    stat_modifiers={"accuracy": -0.5, "speed": -0.3, "reaction": -0.4},
    icon="brain", color="#ff9900",
)
EFFECTS_CATALOG["morphine"] = _make(
    "med_morphine", "morphine", EffectType.BUFF, 60.0,
    stat_modifiers={"pain_threshold": 0.8, "accuracy": -0.15, "reaction": -0.2},
    heal_per_second=0.5,
    icon="syringe", color="#66ccff",
)
EFFECTS_CATALOG["bandaged"] = _make(
    "med_bandaged", "bandaged", EffectType.HOT, 45.0,
    heal_per_second=1.0,
    icon="bandage", color="#ffffff",
)
EFFECTS_CATALOG["tourniquet"] = _make(
    "med_tourniquet", "tourniquet", EffectType.BUFF, -1,
    stat_modifiers={"speed": -0.4},
    icon="bandage", color="#cc3333",
)

# -- Environmental ---------------------------------------------------------

EFFECTS_CATALOG["burning"] = _make(
    "env_burning", "burning", EffectType.DOT, 8.0,
    damage_per_second=5.0, max_stacks=2,
    stat_modifiers={"accuracy": -0.3},
    icon="flame", color="#ff4400",
)
EFFECTS_CATALOG["frozen"] = _make(
    "env_frozen", "frozen", EffectType.CROWD_CONTROL, 6.0,
    stat_modifiers={"speed": -0.7, "reaction": -0.5},
    icon="snowflake", color="#aaddff",
)
EFFECTS_CATALOG["soaked"] = _make(
    "env_soaked", "soaked", EffectType.DEBUFF, 20.0,
    stat_modifiers={"speed": -0.15, "stealth": -0.2},
    icon="cloud-rain", color="#4488cc",
)
EFFECTS_CATALOG["blinded"] = _make(
    "env_blinded", "blinded", EffectType.CROWD_CONTROL, 5.0,
    stat_modifiers={"accuracy": -0.8, "detection": -0.9},
    icon="eye-off", color="#ffff00",
)
EFFECTS_CATALOG["deafened"] = _make(
    "env_deafened", "deafened", EffectType.DEBUFF, 10.0,
    stat_modifiers={"reaction": -0.4, "detection": -0.3},
    icon="volume-x", color="#888888",
)
EFFECTS_CATALOG["irradiated"] = _make(
    "env_irradiated", "irradiated", EffectType.DOT, 60.0,
    damage_per_second=1.5, max_stacks=5,
    stat_modifiers={"max_health": -0.1},
    icon="radiation", color="#33ff33",
)

# -- Tactical --------------------------------------------------------------

EFFECTS_CATALOG["camouflaged"] = _make(
    "tac_camouflaged", "camouflaged", EffectType.BUFF, -1,
    stat_modifiers={"stealth": 0.5, "detection_range": -0.4},
    icon="eye-off", color="#336633",
)
EFFECTS_CATALOG["spotted"] = _make(
    "tac_spotted", "spotted", EffectType.DEBUFF, 10.0,
    stat_modifiers={"stealth": -0.8},
    icon="target", color="#ff0000",
)
EFFECTS_CATALOG["marked"] = _make(
    "tac_marked", "marked", EffectType.DEBUFF, 15.0,
    stat_modifiers={"stealth": -1.0, "defense": -0.15},
    icon="crosshair", color="#ff2a6d",
)
EFFECTS_CATALOG["jammed"] = _make(
    "tac_jammed", "jammed", EffectType.DEBUFF, 12.0,
    stat_modifiers={"comms": -1.0, "detection": -0.5},
    icon="wifi-off", color="#ff6600",
)
EFFECTS_CATALOG["hacked"] = _make(
    "tac_hacked", "hacked", EffectType.DEBUFF, 20.0,
    stat_modifiers={"comms": -0.5, "accuracy": -0.2, "detection": -0.3},
    icon="terminal", color="#00ff00",
)

# -- Morale ----------------------------------------------------------------

EFFECTS_CATALOG["inspired"] = _make(
    "mor_inspired", "inspired", EffectType.BUFF, 30.0,
    stat_modifiers={"morale": 0.3, "accuracy": 0.1, "speed": 0.1},
    icon="star", color="#fcee0a",
)
EFFECTS_CATALOG["terrified"] = _make(
    "mor_terrified", "terrified", EffectType.DEBUFF, 15.0,
    stat_modifiers={"morale": -0.5, "accuracy": -0.3, "speed": 0.2},
    icon="alert-triangle", color="#cc00cc",
)
EFFECTS_CATALOG["berserk"] = _make(
    "mor_berserk", "berserk", EffectType.BUFF, 12.0,
    stat_modifiers={"damage": 0.4, "speed": 0.2, "defense": -0.3, "accuracy": -0.2},
    icon="skull", color="#ff0000",
)
EFFECTS_CATALOG["shell_shocked"] = _make(
    "mor_shell_shocked", "shell_shocked", EffectType.CROWD_CONTROL, 10.0,
    stat_modifiers={"accuracy": -0.6, "speed": -0.4, "reaction": -0.5, "morale": -0.3},
    icon="zap-off", color="#996633",
)
EFFECTS_CATALOG["rallied"] = _make(
    "mor_rallied", "rallied", EffectType.BUFF, 20.0,
    stat_modifiers={"morale": 0.4, "defense": 0.1, "accuracy": 0.15},
    icon="flag", color="#00f0ff",
)

# -- Vehicle ---------------------------------------------------------------

EFFECTS_CATALOG["engine_damage"] = _make(
    "veh_engine_damage", "engine_damage", EffectType.DEBUFF, -1,
    stat_modifiers={"speed": -0.5, "acceleration": -0.6},
    max_stacks=3, icon="settings", color="#ff6600",
)
EFFECTS_CATALOG["flat_tire"] = _make(
    "veh_flat_tire", "flat_tire", EffectType.DEBUFF, -1,
    stat_modifiers={"speed": -0.3, "handling": -0.4},
    max_stacks=4, icon="circle-slash", color="#996633",
)
EFFECTS_CATALOG["fuel_leak"] = _make(
    "veh_fuel_leak", "fuel_leak", EffectType.DOT, -1,
    damage_per_second=0.5,
    stat_modifiers={"range": -0.2},
    max_stacks=2, icon="droplet", color="#ffcc00",
)
EFFECTS_CATALOG["turret_jammed"] = _make(
    "veh_turret_jammed", "turret_jammed", EffectType.CROWD_CONTROL, -1,
    stat_modifiers={"fire_rate": -1.0, "accuracy": -0.5},
    icon="lock", color="#cc3333",
)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "EffectType",
    "StatusEffect",
    "StatusEffectEngine",
    "EFFECTS_CATALOG",
]
