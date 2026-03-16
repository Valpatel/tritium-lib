# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tritium Simulation Engine — combat, NPC AI, physics, effects, audio.

Sub-packages:
    sim_engine.ai       — Steering, pathfinding, ambient NPCs, combat AI, behavior trees
    sim_engine.audio    — Spatial audio math
    sim_engine.debug    — Debug data streams
    sim_engine.effects  — Particle systems, explosions, weapon fire
    sim_engine.physics  — Collision detection and vehicle dynamics
    sim_engine.demos    — Standalone demo scripts
"""

from .ai import *      # noqa: F401,F403
from .audio import *   # noqa: F401,F403
from .debug import *   # noqa: F401,F403
from .effects import * # noqa: F401,F403
from .physics import * # noqa: F401,F403
