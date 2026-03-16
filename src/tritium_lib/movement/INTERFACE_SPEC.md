# Movement Module Interface Specification

This module MUST interface with the existing tritium-sc simulation engine. Here's how:

## SimulationTarget (tritium-sc/src/engine/simulation/target.py)
```python
@dataclass
class SimulationTarget:
    target_id: str
    position: tuple[float, float]   # (x, y) in map coords (meters from geo center)
    heading: float = 0.0            # degrees, 0 = north, clockwise
    speed: float = 1.0              # meters/second
    fsm_state: str | None = None    # current behavior state
    alliance: str = "neutral"       # "friendly", "hostile", "neutral"
    asset_type: str = "person"      # "person", "vehicle", "turret", "drone", etc.
    waypoints: list = field(...)    # list of (x, y) waypoint tuples
    waypoint_index: int = 0
```

## UnitBehaviors (tritium-sc/src/engine/simulation/behaviors.py)
```python
class UnitBehaviors:
    def tick(self, dt: float, targets: dict[str, SimulationTarget], ...) -> None:
        # Dispatches to _turret_behavior, _drone_behavior, _rover_behavior, etc.
        # Sets target.fsm_state, target.waypoints, target.heading, target.speed
```

## How tritium-lib movement modules plug in:

1. **steering.py** — returns force vectors as `tuple[float, float]`. The SC engine applies these to position/velocity in SimulationTarget.tick().

2. **pathfinding.py** — returns waypoint lists as `list[tuple[float, float]]`. SC sets these on `target.waypoints`.

3. **behavior_tree.py** — `context` dict contains the SimulationTarget fields. Actions set `fsm_state`, `speed`, `waypoints` on the context.

4. **ambient.py** — spawns `AmbientEntity` objects whose fields map directly to SimulationTarget fields (position, heading, speed, asset_type).

## Key constraints:
- Coordinates are meters from geo center (NOT lat/lng)
- 10Hz tick rate (dt ≈ 0.1s)
- 100+ entities simultaneously
- No external dependencies (pure Python + stdlib only)
- All functions must be composable and stateless where possible

## NPC Behavior Types Needed:
- civilian_walk: random destinations, sidewalks, stops to check phone
- civilian_drive: follow roads, park, speed limits
- jogger: loop route at ~3m/s
- dog_walker: slow, frequent stops, wander
- patrol_guard: follow route, investigate anomalies
- hostile_infiltrator: approach target stealthily, use cover
- delivery_driver: visit multiple stops, park briefly
- crowd_member: social force model, follow group
