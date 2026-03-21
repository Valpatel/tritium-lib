# sim_engine/core/

Foundational building blocks shared across the entire sim engine.

## Files

| File | Purpose |
|------|---------|
| `entity.py` | `SimulationTarget` — base movable entity; `UnitIdentity` for unique ID generation |
| `movement.py` | `MovementController` — smooth position interpolation and heading updates |
| `inventory.py` | `UnitInventory`, `InventoryItem` — item slots, weights, equip/drop logic |
| `state_machine.py` | `StateMachine`, `State`, `Transition` — generic FSM for unit behavior |
| `spatial.py` | `SpatialGrid` — 2-D cell grid for fast proximity queries |
| `npc_thinker.py` | `NPCThinker` — async LLM-backed NPC decision making (llama-server) |

## Key Concepts

- All simulation entities ultimately inherit from or compose `SimulationTarget`.
- `SpatialGrid` is the performance backbone: O(1) neighbor lookups replace O(n) scans.
- `StateMachine` is reused by units, NPCs, and game-mode controllers.
- `NPCThinker` is optional and degrades gracefully when no LLM is available.
