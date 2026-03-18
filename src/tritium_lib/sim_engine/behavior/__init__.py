# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Behavior sub-package — unit AI, FSMs, missions, NPC population.

Moved from tritium-sc/src/engine/simulation/ during Phase 2 of sim engine
unification so that addons, runners, and other consumers can reuse the
behavior system without depending on the full Command Center.
"""

from .behaviors import UnitBehaviors
from .unit_states import (
    create_turret_fsm,
    create_rover_fsm,
    create_drone_fsm,
    create_hostile_fsm,
    create_fsm_for_type,
)
from .unit_missions import UnitMissionSystem
from .npc import NPCManager, NPCMission, NPC_VEHICLE_TYPES, traffic_density

__all__ = [
    "UnitBehaviors",
    "create_turret_fsm",
    "create_rover_fsm",
    "create_drone_fsm",
    "create_hostile_fsm",
    "create_fsm_for_type",
    "UnitMissionSystem",
    "NPCManager",
    "NPCMission",
    "NPC_VEHICLE_TYPES",
    "traffic_density",
]
