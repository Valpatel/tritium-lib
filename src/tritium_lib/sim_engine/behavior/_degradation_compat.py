# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Degradation compatibility shim for behavior module.

Tries to import from engine.simulation.degradation (SC-specific) first,
falls back to a standalone implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.sim_engine.core.entity import SimulationTarget

# Health fraction below which fire is disabled
_FIRE_DISABLED_THRESHOLD = 0.1


def can_fire_degraded(target: SimulationTarget) -> bool:
    """Return True if the unit is healthy enough to fire.

    Units below 10% health cannot fire.
    """
    if target.max_health <= 0:
        return False
    return (target.health / target.max_health) >= _FIRE_DISABLED_THRESHOLD
