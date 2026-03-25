# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
from tritium_lib.sim_engine.unit_types.base import CombatStats, MovementCategory, UnitType


class HostilePerson(UnitType):
    type_id = "hostile_person"
    display_name = "Hostile"
    icon = "H"
    cot_type = "a-h-G-U-C-I"
    category = MovementCategory.FOOT
    speed = 1.5
    drain_rate = 0.0
    vision_radius = 20.0
    ambient_radius = 12.0
    placeable = False
    combat = CombatStats(
        health=80, max_health=80,
        weapon_range=40.0, weapon_cooldown=2.5, weapon_damage=10,
        is_combatant=True,
    )
