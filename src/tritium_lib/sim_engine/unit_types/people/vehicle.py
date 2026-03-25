# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
from tritium_lib.sim_engine.unit_types.base import CombatStats, MovementCategory, UnitType


class Vehicle(UnitType):
    type_id = "vehicle"
    display_name = "Vehicle"
    icon = "V"
    cot_type = "a-n-G-E-V-C"
    category = MovementCategory.GROUND
    speed = 0.0
    drain_rate = 0.0
    vision_radius = 25.0
    ambient_radius = 10.0
    placeable = False
    combat = CombatStats(
        health=300, max_health=300,
        weapon_range=0.0, weapon_cooldown=0.0, weapon_damage=0,
        is_combatant=False,
    )
