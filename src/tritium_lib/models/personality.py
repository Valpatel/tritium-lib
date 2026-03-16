# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Commander personality traits — configurable AI behavior parameters.

Each commander (Amy or future commanders) has a personality that affects
decision-making, narration style, and escalation thresholds. Traits are
normalized floats [0.0, 1.0] where 0.5 is balanced/default.

Different operational contexts benefit from different personality profiles:
- Patrol mode: low aggression, high curiosity, medium verbosity
- Battle mode: high aggression, medium curiosity, low verbosity
- Stealth mode: low everything except caution
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommanderPersonality:
    """Configurable personality traits for an AI commander.

    All traits are floats in [0.0, 1.0] where 0.5 is the default/balanced
    value. Values below 0.5 reduce the trait's influence, above 0.5 increase it.

    Attributes:
        aggression: How quickly the commander escalates threats and engages.
                   Low = cautious/defensive. High = preemptive/aggressive.
        curiosity: Eagerness to investigate unknowns, scan new targets,
                  and explore unmonitored areas.
                  Low = ignores unknowns. High = actively investigates everything.
        verbosity: How much the commander narrates and explains decisions.
                  Low = terse/silent. High = constant commentary.
        caution: Risk awareness and tendency to hedge decisions.
                Low = reckless. High = overly careful, seeks confirmation.
        initiative: Willingness to take autonomous action without operator input.
                   Low = waits for orders. High = acts independently.
    """

    aggression: float = 0.5
    curiosity: float = 0.5
    verbosity: float = 0.5
    caution: float = 0.5
    initiative: float = 0.5

    def __post_init__(self) -> None:
        """Clamp all traits to [0.0, 1.0]."""
        self.aggression = max(0.0, min(1.0, self.aggression))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.verbosity = max(0.0, min(1.0, self.verbosity))
        self.caution = max(0.0, min(1.0, self.caution))
        self.initiative = max(0.0, min(1.0, self.initiative))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "aggression": round(self.aggression, 3),
            "curiosity": round(self.curiosity, 3),
            "verbosity": round(self.verbosity, 3),
            "caution": round(self.caution, 3),
            "initiative": round(self.initiative, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommanderPersonality:
        """Deserialize from dictionary."""
        return cls(
            aggression=data.get("aggression", 0.5),
            curiosity=data.get("curiosity", 0.5),
            verbosity=data.get("verbosity", 0.5),
            caution=data.get("caution", 0.5),
            initiative=data.get("initiative", 0.5),
        )

    @property
    def profile_label(self) -> str:
        """Human-readable label for the dominant personality trait."""
        traits = {
            "aggressive": self.aggression,
            "curious": self.curiosity,
            "verbose": self.verbosity,
            "cautious": self.caution,
            "proactive": self.initiative,
        }
        dominant = max(traits, key=traits.get)  # type: ignore[arg-type]
        dominant_val = traits[dominant]
        if dominant_val < 0.3:
            return "restrained"
        if dominant_val > 0.7:
            return dominant
        return "balanced"


# -- Preset profiles ----------------------------------------------------------

PATROL_PERSONALITY = CommanderPersonality(
    aggression=0.3,
    curiosity=0.7,
    verbosity=0.5,
    caution=0.6,
    initiative=0.5,
)

BATTLE_PERSONALITY = CommanderPersonality(
    aggression=0.8,
    curiosity=0.4,
    verbosity=0.3,
    caution=0.3,
    initiative=0.8,
)

STEALTH_PERSONALITY = CommanderPersonality(
    aggression=0.2,
    curiosity=0.3,
    verbosity=0.1,
    caution=0.8,
    initiative=0.3,
)

OBSERVER_PERSONALITY = CommanderPersonality(
    aggression=0.1,
    curiosity=0.9,
    verbosity=0.7,
    caution=0.7,
    initiative=0.2,
)

PRESET_PERSONALITIES: dict[str, CommanderPersonality] = {
    "default": CommanderPersonality(),
    "patrol": PATROL_PERSONALITY,
    "battle": BATTLE_PERSONALITY,
    "stealth": STEALTH_PERSONALITY,
    "observer": OBSERVER_PERSONALITY,
}
