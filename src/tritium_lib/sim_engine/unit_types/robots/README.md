# unit_types/robots/ — dispatchable machines and emplaced weapons

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

One file per `UnitType` archetype for friendly/controllable hardware -- the
units an operator dispatches plus the static weapons that defend a position.
Importing `sim_engine.unit_types` auto-discovers every class here; the parent
README covers the registry, base classes, and CoT symbol mapping.

## Archetypes (8)

| File | `type_id` | Display | Movement |
|------|-----------|---------|----------|
| `rover.py` | `rover` | Rover | GROUND |
| `tank.py` | `tank` | Tank | GROUND |
| `apc.py` | `apc` | APC | GROUND |
| `drone.py` | `drone` | Drone | AIR |
| `scout_drone.py` | `scout_drone` | Scout Drone | AIR |
| `turret.py` | `turret` | Turret | STATIONARY |
| `heavy_turret.py` | `heavy_turret` | Heavy Turret | STATIONARY |
| `missile_turret.py` | `missile_turret` | Missile Turret | STATIONARY |

Mobile ground/air units (`rover`, `tank`, `apc`, `drone`, `scout_drone`) are
`placeable` and appear in `dispatchable_type_ids()`; the `*turret` archetypes
are STATIONARY emplacements -- they fire but do not move. Each declares its
weapon profile via `CombatStats`, which `combat/` uses to resolve fire.

## Add one

Create `robots/my_bot.py`, subclass `UnitType`, set `type_id`, `category`, and
`combat=CombatStats(...)`. Discovery finds it on next import. Worked example in
[`../README.md`](../README.md).

**Parent:** [`../README.md`](../README.md)
