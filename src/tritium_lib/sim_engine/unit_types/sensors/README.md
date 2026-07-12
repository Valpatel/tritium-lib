# unit_types/sensors/ — passive detector archetypes

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

`UnitType` archetypes for emplaced, non-combatant sensors -- units that
*perceive* the field rather than move or fire. Importing
`sim_engine.unit_types` auto-discovers them; the parent README documents the
registry, the perception-cone fields, and CoT mapping.

## Archetypes (2)

| File | `type_id` | Display | Movement |
|------|-----------|---------|----------|
| `camera.py` | `camera` | Camera | STATIONARY |
| `motion_sensor.py` | `sensor` | Motion Sensor | STATIONARY |

Both are STATIONARY and non-combatant. Their value is the perception cone
(`cone_range`, `cone_angle`, `cone_sweeps`, `vision_radius`) that `world/vision.py`
reads to decide what each sensor can see -- the sim's stand-in for a real
camera or PIR node feeding the tracking pipeline. Note `motion_sensor.py`
registers under `type_id = "sensor"`.

## Add one

Create `sensors/my_sensor.py`, subclass `UnitType`, set `type_id` and the
perception-cone `ClassVar`s. Discovery finds it on next import. See
[`../README.md`](../README.md).

**Parent:** [`../README.md`](../README.md)
