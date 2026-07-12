# unit_types/people/ — human, animal, and civilian/hostile archetypes

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

One file per `UnitType` archetype for living / driven units. Importing
`sim_engine.unit_types` auto-discovers every class here (no registration list) --
see the parent README for how the registry, base classes, and CoT mapping work.

## Archetypes (7)

| File | `type_id` | Display | Movement |
|------|-----------|---------|----------|
| `person.py` | `person` | Person | FOOT |
| `vehicle.py` | `vehicle` | Vehicle | GROUND |
| `animal.py` | `animal` | Animal | FOOT |
| `hostile_person.py` | `hostile_person` | Hostile | FOOT |
| `hostile_vehicle.py` | `hostile_vehicle` | Hostile Vehicle | GROUND |
| `hostile_leader.py` | `hostile_leader` | Hostile Leader | FOOT |
| `swarm_drone.py` | `swarm_drone` | Swarm Drone | AIR |

The `hostile_*` archetypes carry combatant `CombatStats` and map to hostile CoT
symbols; `person` / `vehicle` / `animal` are the neutral civilian population the
tracker and fusion pipeline classify. `hostile_leader` anchors the Epstein
protest/riot emergence model; `swarm_drone` is the low-cost aerial hostile.

## Add one

Create `people/my_type.py`, subclass `UnitType`, set `type_id` +
`combat=CombatStats(...)`. Discovery picks it up on next import -- no edit here.
See the worked example in [`../README.md`](../README.md).

**Parent:** [`../README.md`](../README.md)
