# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Game physics: collision detection, momentum transfer, vehicle dynamics.

NumPy-vectorized 2D physics for the Tritium simulation layer.
Works with the existing steering_np arrays — same SoA (struct-of-arrays)
pattern for cache-friendly vectorized computation.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from .collision import (
    CollisionEvent,
    PhysicsWorld,
    RigidBody,
)
from .vehicle import VehiclePhysics

__all__ = [
    "CollisionEvent",
    "PhysicsWorld",
    "RigidBody",
    "VehiclePhysics",
]
