# sim_engine/behavior/

Unit behavior layer — per-type combat AI, NPC daily routines, missions, and states.

## Files

| File | Purpose |
|------|---------|
| `behaviors.py` | `UnitBehaviors` — per-unit-type AI (infantry, sniper, medic, heavy) |
| `npc.py` | `NPCPopulation` — vehicles, pedestrians, road following, daily schedules |
| `unit_states.py` | `UnitState` dataclass — health, ammo, morale, suppression, kill count |
| `unit_missions.py` | `MissionController` — patrol, assault, defend, escort mission logic |
| `_degradation_compat.py` | Backward-compat shim for older degradation API |

## Key Concepts

- `UnitBehaviors` is called by `World.tick()` for each unit that needs an AI decision.
- Combat AI choices (attack, flee, suppress, heal) are weighted by health/morale/range.
- `NPCPopulation` runs the civilian simulation separately from the tactical unit layer.
- `unit_states.py` is the canonical mutable state container — never stored on the entity class directly.
