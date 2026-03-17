# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AI behavior profile system — configurable archetypes that control unit decisions.

Each unit can be assigned a BehaviorProfile that determines how it reacts to
threats, manages ammo, seeks cover, retreats, and engages.  Profiles are
composable via BehaviorTrait enums and continuous 0-1 parameter sliders.

The BehaviorEngine centralizes profile storage and provides decision methods
that the sim tick loop can call per-unit.

Usage::

    from tritium_lib.sim_engine.ai.behavior_profiles import (
        BehaviorEngine, BehaviorTrait, PROFILES,
    )

    engine = BehaviorEngine(profiles=PROFILES)
    engine.assign_profile("unit_42", "elite_operator")
    action = engine.decide("unit_42", {"health": 0.8, "threats": 3, ...})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BehaviorTrait(Enum):
    """High-level behavioral archetype flags."""

    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    CAUTIOUS = "cautious"
    RECKLESS = "reckless"
    METHODICAL = "methodical"
    OPPORTUNISTIC = "opportunistic"
    SUPPORTIVE = "supportive"
    INDEPENDENT = "independent"


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------


@dataclass
class BehaviorProfile:
    """A configurable AI archetype that shapes unit decision-making.

    All float fields are 0-1 unless stated otherwise.  Higher values mean
    *more* of that quality (e.g. ``aggression=1.0`` is maximum aggression).
    """

    profile_id: str
    name: str
    traits: list[BehaviorTrait] = field(default_factory=list)

    # Core personality axes (0-1)
    aggression: float = 0.5
    caution: float = 0.5
    teamwork: float = 0.5
    discipline: float = 0.5
    initiative: float = 0.5
    morale_resilience: float = 0.5

    # Tactical preferences
    preferred_range: str = "medium"  # close, medium, long
    retreat_threshold: float = 0.25  # health ratio to trigger retreat
    suppression_tolerance: float = 0.5  # 0 = panics immediately, 1 = ignores
    cover_priority: float = 0.5  # how strongly unit seeks cover
    ammo_conservation: float = 0.5  # 0 = spray, 1 = single aimed shots

    def __post_init__(self) -> None:
        """Clamp all float parameters to valid ranges."""
        for attr in (
            "aggression", "caution", "teamwork", "discipline",
            "initiative", "morale_resilience", "suppression_tolerance",
            "cover_priority", "ammo_conservation",
        ):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))
        self.retreat_threshold = max(0.0, min(1.0, self.retreat_threshold))
        if self.preferred_range not in ("close", "medium", "long"):
            self.preferred_range = "medium"


# ---------------------------------------------------------------------------
# Behavior Engine
# ---------------------------------------------------------------------------


_RANGE_MULTIPLIER: dict[str, float] = {
    "close": 0.4,
    "medium": 0.7,
    "long": 1.0,
}


class BehaviorEngine:
    """Manages behavior profiles and produces per-unit decisions.

    Args:
        profiles: Optional initial mapping of profile_id -> BehaviorProfile.
    """

    def __init__(self, profiles: dict[str, BehaviorProfile] | None = None) -> None:
        self.profiles: dict[str, BehaviorProfile] = dict(profiles) if profiles else {}
        self.unit_profiles: dict[str, str] = {}

    # -- Profile management --------------------------------------------------

    def add_profile(self, profile: BehaviorProfile) -> None:
        """Register or replace a profile."""
        self.profiles[profile.profile_id] = profile

    def assign_profile(self, unit_id: str, profile_id: str) -> None:
        """Assign a profile to a unit.

        Raises:
            KeyError: If *profile_id* is not registered.
        """
        if profile_id not in self.profiles:
            raise KeyError(f"Unknown profile: {profile_id}")
        self.unit_profiles[unit_id] = profile_id

    def get_profile(self, unit_id: str) -> BehaviorProfile | None:
        """Return the profile assigned to *unit_id*, or ``None``."""
        pid = self.unit_profiles.get(unit_id)
        if pid is None:
            return None
        return self.profiles.get(pid)

    # -- Decision-making -----------------------------------------------------

    def decide(self, unit_id: str, situation: dict[str, Any]) -> dict[str, Any]:
        """Decide an action for *unit_id* given the current *situation*.

        Situation keys consumed:
            health (float 0-1), ammo (float 0-1), threats (int),
            allies (int), in_cover (bool), suppressed (bool),
            enemy_distance (float), has_objective (bool),
            morale (float 0-1).

        Returns a dict with:
            action (str), modifiers (dict), reasoning (str).
        """
        profile = self.get_profile(unit_id)
        if profile is None:
            return {
                "action": "hold",
                "modifiers": {},
                "reasoning": "No profile assigned — defaulting to hold.",
            }

        health = situation.get("health", 1.0)
        ammo = situation.get("ammo", 1.0)
        threats = situation.get("threats", 0)
        allies = situation.get("allies", 0)
        in_cover = situation.get("in_cover", False)
        suppressed = situation.get("suppressed", False)
        enemy_distance = situation.get("enemy_distance", 100.0)
        has_objective = situation.get("has_objective", False)
        morale = situation.get("morale", 1.0)

        # --- Retreat check ---
        effective_retreat = profile.retreat_threshold
        # Reckless units lower their retreat threshold
        if BehaviorTrait.RECKLESS in profile.traits:
            effective_retreat *= 0.5
        # Cautious units raise it
        if BehaviorTrait.CAUTIOUS in profile.traits:
            effective_retreat = min(effective_retreat * 1.5, 0.8)

        if health <= effective_retreat:
            return {
                "action": "retreat",
                "modifiers": {"urgency": 1.0 - health},
                "reasoning": f"Health {health:.0%} below retreat threshold {effective_retreat:.0%}.",
            }

        # --- Suppression response ---
        if suppressed and profile.suppression_tolerance < 0.5:
            if not in_cover:
                return {
                    "action": "seek_cover",
                    "modifiers": {"urgency": 1.0 - profile.suppression_tolerance},
                    "reasoning": "Under suppression with low tolerance — seeking cover.",
                }

        # --- Ammo conservation ---
        if ammo < 0.1 and profile.ammo_conservation > 0.5:
            return {
                "action": "conserve",
                "modifiers": {"fire_rate_mult": 0.3},
                "reasoning": "Low ammo with high conservation priority — conserving.",
            }
        if ammo < 0.05:
            return {
                "action": "retreat",
                "modifiers": {"urgency": 0.8},
                "reasoning": "Critically low ammo — retreating.",
            }

        # --- Morale break ---
        effective_morale_threshold = 0.3 * (1.0 - profile.morale_resilience)
        if morale < effective_morale_threshold and morale < 0.2:
            return {
                "action": "rout",
                "modifiers": {"panic": 1.0 - morale},
                "reasoning": "Morale broken — routing.",
            }

        # --- Supportive behavior ---
        if BehaviorTrait.SUPPORTIVE in profile.traits and profile.teamwork > 0.7:
            if allies > 0 and threats > 0:
                return {
                    "action": "support",
                    "modifiers": {"heal_priority": profile.teamwork},
                    "reasoning": "Supportive profile — prioritizing ally support.",
                }

        # --- No threats ---
        if threats == 0:
            if has_objective and profile.initiative > 0.5:
                return {
                    "action": "advance",
                    "modifiers": {"speed_mult": 0.5 + profile.initiative * 0.5},
                    "reasoning": "No threats, objective present — advancing.",
                }
            return {
                "action": "patrol",
                "modifiers": {"alertness": profile.caution},
                "reasoning": "No threats — patrolling.",
            }

        # --- Aggressive engagement ---
        if BehaviorTrait.AGGRESSIVE in profile.traits or profile.aggression > 0.7:
            if enemy_distance < 30.0 or not in_cover:
                return {
                    "action": "assault",
                    "modifiers": {
                        "aggression": profile.aggression,
                        "speed_mult": 1.0 + profile.aggression * 0.3,
                    },
                    "reasoning": "Aggressive profile — assaulting enemy position.",
                }

        # --- Defensive posture ---
        if BehaviorTrait.DEFENSIVE in profile.traits or profile.caution > 0.7:
            if in_cover:
                return {
                    "action": "hold_and_engage",
                    "modifiers": {
                        "accuracy_bonus": profile.discipline * 0.2,
                        "fire_rate_mult": 1.0 - profile.ammo_conservation * 0.3,
                    },
                    "reasoning": "Defensive profile in cover — holding and engaging.",
                }
            return {
                "action": "seek_cover",
                "modifiers": {"urgency": profile.cover_priority},
                "reasoning": "Defensive profile — seeking cover first.",
            }

        # --- Opportunistic ---
        if BehaviorTrait.OPPORTUNISTIC in profile.traits:
            if threats == 1 and health > 0.6:
                return {
                    "action": "flank",
                    "modifiers": {"stealth": 0.5 + profile.caution * 0.3},
                    "reasoning": "Opportunistic — flanking lone target.",
                }

        # --- Methodical advance ---
        if BehaviorTrait.METHODICAL in profile.traits:
            if in_cover:
                return {
                    "action": "overwatch",
                    "modifiers": {
                        "accuracy_bonus": profile.discipline * 0.15,
                        "patience": profile.discipline,
                    },
                    "reasoning": "Methodical profile — providing overwatch.",
                }
            return {
                "action": "bound_advance",
                "modifiers": {"cover_priority": profile.cover_priority},
                "reasoning": "Methodical profile — bounding to next cover.",
            }

        # --- Default: engage or seek cover ---
        if in_cover:
            return {
                "action": "engage",
                "modifiers": {
                    "fire_rate_mult": 1.0 - profile.ammo_conservation * 0.3,
                },
                "reasoning": "In cover with threats — engaging.",
            }
        return {
            "action": "seek_cover",
            "modifiers": {"urgency": profile.cover_priority},
            "reasoning": "Threats present, not in cover — seeking cover.",
        }

    # -- Stat modification ---------------------------------------------------

    def modify_stats(self, unit_id: str, base_stats: dict[str, float]) -> dict[str, float]:
        """Apply profile-based modifiers to *base_stats*.

        Modifiable keys: speed, accuracy, detection_range, attack_range,
        damage, armor.  Unknown keys pass through unchanged.
        """
        profile = self.get_profile(unit_id)
        if profile is None:
            return dict(base_stats)

        stats = dict(base_stats)

        # Speed: aggressive units move faster, cautious slower
        if "speed" in stats:
            speed_mult = 1.0 + (profile.aggression - 0.5) * 0.4
            # Cautious units slow down
            speed_mult -= (profile.caution - 0.5) * 0.2
            stats["speed"] = stats["speed"] * max(0.5, min(1.5, speed_mult))

        # Accuracy: discipline and caution improve accuracy
        if "accuracy" in stats:
            acc_mult = 1.0 + profile.discipline * 0.2 + profile.caution * 0.1
            # Reckless units lose accuracy
            if BehaviorTrait.RECKLESS in profile.traits:
                acc_mult -= 0.15
            stats["accuracy"] = min(1.0, stats["accuracy"] * acc_mult)

        # Detection range: cautious units see farther
        if "detection_range" in stats:
            det_mult = 1.0 + (profile.caution - 0.5) * 0.3
            stats["detection_range"] = stats["detection_range"] * det_mult

        # Attack range: preferred range influences effective range
        if "attack_range" in stats:
            range_mult = _RANGE_MULTIPLIER.get(profile.preferred_range, 0.7)
            # Blend toward preferred range (don't fully override)
            stats["attack_range"] = stats["attack_range"] * (0.5 + range_mult * 0.5)

        # Damage: aggression boosts, caution reduces
        if "damage" in stats:
            dmg_mult = 1.0 + (profile.aggression - 0.5) * 0.3
            stats["damage"] = stats["damage"] * max(0.7, min(1.3, dmg_mult))

        # Armor utilization: disciplined units use armor better
        if "armor" in stats:
            armor_mult = 1.0 + profile.discipline * 0.1
            stats["armor"] = min(1.0, stats["armor"] * armor_mult)

        return stats

    # -- Threat response -----------------------------------------------------

    def evaluate_threat_response(
        self, unit_id: str, threats: list[dict[str, Any]],
    ) -> str:
        """Return a profile-dependent threat reaction string.

        Threat dicts should have: distance (float), threat_level (float 0-1),
        is_flanking (bool).

        Returns one of: "engage", "take_cover", "flank", "retreat",
        "suppress", "hold", "charge", "evade".
        """
        profile = self.get_profile(unit_id)
        if profile is None:
            return "hold"

        if not threats:
            return "hold"

        # Aggregate threat picture
        max_threat = max(t.get("threat_level", 0.5) for t in threats)
        min_dist = min(t.get("distance", 100.0) for t in threats)
        any_flanking = any(t.get("is_flanking", False) for t in threats)
        num_threats = len(threats)

        # Overwhelming threat — even aggressive units may retreat
        if max_threat > 0.9 and num_threats > 3 and profile.discipline < 0.5:
            return "retreat"

        # Flanking threat — response depends on discipline
        if any_flanking:
            if profile.discipline > 0.7:
                return "suppress"
            if profile.aggression > 0.7:
                return "charge"
            return "evade"

        # Close range
        if min_dist < 10.0:
            if BehaviorTrait.RECKLESS in profile.traits:
                return "charge"
            if BehaviorTrait.AGGRESSIVE in profile.traits:
                return "engage"
            if profile.caution > 0.6:
                return "evade"
            return "engage"

        # Medium range
        if min_dist < 50.0:
            if BehaviorTrait.DEFENSIVE in profile.traits:
                return "take_cover"
            if profile.aggression > 0.6:
                return "engage"
            if BehaviorTrait.METHODICAL in profile.traits:
                return "suppress"
            return "engage"

        # Long range
        if profile.preferred_range == "long":
            return "engage"
        if BehaviorTrait.CAUTIOUS in profile.traits:
            return "take_cover"
        if profile.initiative > 0.6:
            return "flank"
        return "hold"


# ---------------------------------------------------------------------------
# 12 pre-built profiles
# ---------------------------------------------------------------------------


PROFILES: dict[str, BehaviorProfile] = {
    "elite_operator": BehaviorProfile(
        profile_id="elite_operator",
        name="Elite Operator",
        traits=[BehaviorTrait.METHODICAL, BehaviorTrait.INDEPENDENT],
        aggression=0.7,
        caution=0.6,
        teamwork=0.7,
        discipline=0.95,
        initiative=0.9,
        morale_resilience=0.9,
        preferred_range="medium",
        retreat_threshold=0.15,
        suppression_tolerance=0.8,
        cover_priority=0.7,
        ammo_conservation=0.6,
    ),
    "conscript": BehaviorProfile(
        profile_id="conscript",
        name="Conscript",
        traits=[BehaviorTrait.CAUTIOUS, BehaviorTrait.DEFENSIVE],
        aggression=0.2,
        caution=0.7,
        teamwork=0.4,
        discipline=0.2,
        initiative=0.2,
        morale_resilience=0.2,
        preferred_range="medium",
        retreat_threshold=0.4,
        suppression_tolerance=0.2,
        cover_priority=0.8,
        ammo_conservation=0.3,
    ),
    "guerrilla": BehaviorProfile(
        profile_id="guerrilla",
        name="Guerrilla Fighter",
        traits=[BehaviorTrait.OPPORTUNISTIC, BehaviorTrait.INDEPENDENT],
        aggression=0.5,
        caution=0.6,
        teamwork=0.3,
        discipline=0.5,
        initiative=0.8,
        morale_resilience=0.6,
        preferred_range="medium",
        retreat_threshold=0.3,
        suppression_tolerance=0.5,
        cover_priority=0.6,
        ammo_conservation=0.7,
    ),
    "sniper_patient": BehaviorProfile(
        profile_id="sniper_patient",
        name="Patient Sniper",
        traits=[BehaviorTrait.CAUTIOUS, BehaviorTrait.METHODICAL],
        aggression=0.2,
        caution=0.9,
        teamwork=0.2,
        discipline=0.95,
        initiative=0.4,
        morale_resilience=0.7,
        preferred_range="long",
        retreat_threshold=0.3,
        suppression_tolerance=0.6,
        cover_priority=0.9,
        ammo_conservation=0.95,
    ),
    "berserker": BehaviorProfile(
        profile_id="berserker",
        name="Berserker",
        traits=[BehaviorTrait.AGGRESSIVE, BehaviorTrait.RECKLESS],
        aggression=1.0,
        caution=0.05,
        teamwork=0.1,
        discipline=0.1,
        initiative=0.9,
        morale_resilience=0.8,
        preferred_range="close",
        retreat_threshold=0.05,
        suppression_tolerance=0.95,
        cover_priority=0.05,
        ammo_conservation=0.0,
    ),
    "medic_angel": BehaviorProfile(
        profile_id="medic_angel",
        name="Combat Medic",
        traits=[BehaviorTrait.SUPPORTIVE, BehaviorTrait.CAUTIOUS],
        aggression=0.1,
        caution=0.7,
        teamwork=0.95,
        discipline=0.7,
        initiative=0.6,
        morale_resilience=0.7,
        preferred_range="medium",
        retreat_threshold=0.3,
        suppression_tolerance=0.4,
        cover_priority=0.7,
        ammo_conservation=0.8,
    ),
    "engineer_builder": BehaviorProfile(
        profile_id="engineer_builder",
        name="Combat Engineer",
        traits=[BehaviorTrait.METHODICAL, BehaviorTrait.SUPPORTIVE],
        aggression=0.3,
        caution=0.6,
        teamwork=0.8,
        discipline=0.7,
        initiative=0.5,
        morale_resilience=0.6,
        preferred_range="medium",
        retreat_threshold=0.25,
        suppression_tolerance=0.5,
        cover_priority=0.7,
        ammo_conservation=0.6,
    ),
    "scout_ghost": BehaviorProfile(
        profile_id="scout_ghost",
        name="Ghost Scout",
        traits=[BehaviorTrait.CAUTIOUS, BehaviorTrait.INDEPENDENT],
        aggression=0.15,
        caution=0.9,
        teamwork=0.3,
        discipline=0.8,
        initiative=0.85,
        morale_resilience=0.6,
        preferred_range="long",
        retreat_threshold=0.35,
        suppression_tolerance=0.3,
        cover_priority=0.9,
        ammo_conservation=0.8,
    ),
    "commander_calm": BehaviorProfile(
        profile_id="commander_calm",
        name="Calm Commander",
        traits=[BehaviorTrait.METHODICAL, BehaviorTrait.SUPPORTIVE],
        aggression=0.4,
        caution=0.6,
        teamwork=0.9,
        discipline=0.9,
        initiative=0.7,
        morale_resilience=0.9,
        preferred_range="medium",
        retreat_threshold=0.2,
        suppression_tolerance=0.7,
        cover_priority=0.6,
        ammo_conservation=0.5,
    ),
    "civilian_panicked": BehaviorProfile(
        profile_id="civilian_panicked",
        name="Panicked Civilian",
        traits=[BehaviorTrait.CAUTIOUS, BehaviorTrait.DEFENSIVE],
        aggression=0.0,
        caution=1.0,
        teamwork=0.2,
        discipline=0.05,
        initiative=0.1,
        morale_resilience=0.05,
        preferred_range="long",
        retreat_threshold=0.9,
        suppression_tolerance=0.0,
        cover_priority=1.0,
        ammo_conservation=1.0,
    ),
    "robot_precise": BehaviorProfile(
        profile_id="robot_precise",
        name="Precision Robot",
        traits=[BehaviorTrait.METHODICAL, BehaviorTrait.INDEPENDENT],
        aggression=0.5,
        caution=0.5,
        teamwork=0.6,
        discipline=1.0,
        initiative=0.5,
        morale_resilience=1.0,
        preferred_range="medium",
        retreat_threshold=0.1,
        suppression_tolerance=1.0,
        cover_priority=0.5,
        ammo_conservation=0.5,
    ),
    "veteran_steady": BehaviorProfile(
        profile_id="veteran_steady",
        name="Steady Veteran",
        traits=[BehaviorTrait.DEFENSIVE, BehaviorTrait.METHODICAL],
        aggression=0.5,
        caution=0.6,
        teamwork=0.7,
        discipline=0.85,
        initiative=0.6,
        morale_resilience=0.85,
        preferred_range="medium",
        retreat_threshold=0.2,
        suppression_tolerance=0.7,
        cover_priority=0.7,
        ammo_conservation=0.6,
    ),
}
