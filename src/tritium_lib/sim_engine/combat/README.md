# sim_engine/combat/

Combat resolution subsystem — projectile flight, weapons, and squad tactics.

## Files

| File | Purpose |
|------|---------|
| `combat.py` | `CombatSystem` — projectile lifecycle, hit detection, damage application |
| `weapons.py` | `WeaponSystem`, `Weapon`, `WEAPON_CATALOG` — per-unit ammo/accuracy/range |
| `squads.py` | `SquadManager` — leader-follower formation logic, focus-fire coordination |

## Key Classes

- `CombatSystem.fire(src, tgt)` — launches a projectile from one unit toward another
- `CombatSystem.tick(dt)` — advances all in-flight projectiles, resolves hits
- `WeaponSystem` — tracks reload, ammo, and accuracy modifiers per unit
- `SquadManager` — auto-clusters hostile units into squads with shared targeting

## Integration

`_world.py` owns a `CombatSystem` instance. Units call `fire()` during their AI step;
the world ticks the combat system each frame.
