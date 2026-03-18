# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Core simulation entity types shared across Tritium components.

This package contains the fundamental entity classes that were originally
in tritium-sc/src/engine/simulation/ and have been extracted to tritium-lib
for reuse by addons, runners, and other consumers.
"""

from .entity import SimulationTarget, UnitIdentity
from .inventory import UnitInventory, InventoryItem
from .movement import MovementController
from .state_machine import StateMachine, State, Transition
from .spatial import SpatialGrid

__all__ = [
    "SimulationTarget",
    "UnitIdentity",
    "UnitInventory",
    "InventoryItem",
    "MovementController",
    "StateMachine",
    "State",
    "Transition",
    "SpatialGrid",
]
