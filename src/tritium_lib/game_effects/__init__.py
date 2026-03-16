# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Game effects: particle systems for combat visualization.

Produces particle data (positions, colors, sizes, lifetimes) that the
frontend renders via Canvas 2D or Three.js.  The backend computes physics,
the frontend draws.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from .particles import (
    EffectsManager,
    Particle,
    ParticleEmitter,
    blood_splatter,
    debris,
    explosion,
    fire,
    muzzle_flash,
    smoke,
    sparks,
    tracer,
)
from .weapons import (
    WEAPONS,
    FireMode,
    FiredRound,
    WeaponFirer,
    WeaponProfile,
    create_firer,
)

__all__ = [
    "EffectsManager",
    "Particle",
    "ParticleEmitter",
    "blood_splatter",
    "debris",
    "explosion",
    "fire",
    "muzzle_flash",
    "smoke",
    "sparks",
    "tracer",
    # Weapons
    "WEAPONS",
    "FireMode",
    "FiredRound",
    "WeaponFirer",
    "WeaponProfile",
    "create_firer",
]
