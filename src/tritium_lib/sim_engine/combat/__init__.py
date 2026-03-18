# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Combat sub-package — projectiles, weapons, squads.

Moved from tritium-sc/src/engine/simulation/ during Phase 3 of sim engine
unification.
"""

from .combat import CombatSystem, Projectile, HIT_RADIUS, MISS_OVERSHOOT
from .weapons import Weapon, WeaponSystem, WEAPON_CATALOG
from .squads import (
    Squad,
    SquadManager,
    SQUAD_RADIUS,
    FORMATION_SPACING,
)

__all__ = [
    "CombatSystem",
    "Projectile",
    "HIT_RADIUS",
    "MISS_OVERSHOOT",
    "Weapon",
    "WeaponSystem",
    "WEAPON_CATALOG",
    "Squad",
    "SquadManager",
    "SQUAD_RADIUS",
    "FORMATION_SPACING",
]
