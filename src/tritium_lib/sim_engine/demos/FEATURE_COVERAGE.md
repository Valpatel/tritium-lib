# City3D Demo — Feature Coverage Audit

**Status**: 13 fully demonstrated, 28 partial, 21 missing (out of 62 modules)

## NOT DEMONSTRATED (21 modules — need integration)

| Priority | Module | What to add |
|----------|--------|-------------|
| HIGH | weather_fx.py | Rain/snow particles, fog, dynamic sky |
| HIGH | territory.py | Influence heatmap overlay, control point capture |
| HIGH | commander.py | Amy narration text in kill feed / HUD |
| HIGH | objectives.py | Visible objective markers + progress bars |
| HIGH | buildings.py | Room clearing when police breach buildings |
| MEDIUM | artillery.py | Mortar barrage during riot escalation |
| MEDIUM | economy.py | Resource bars in HUD (police budget) |
| MEDIUM | fortifications.py | Police barricades + riot barriers |
| MEDIUM | asymmetric.py | IED/booby traps in protest area |
| MEDIUM | intel.py | Fog of war on crowd areas |
| MEDIUM | campaign.py | Multi-phase scenario progression |
| MEDIUM | replay.py | Record button + playback controls |
| MEDIUM | multiplayer.py | Split-screen or networked control |
| MEDIUM | soundtrack.py | Background music + reactive intensity |
| LOW | cyber.py | Drone camera feed + jamming |
| LOW | behavior_tree.py | Debug viz of AI decision tree |
| LOW | behavior_profiles.py | Personality display on units |
| LOW | rf_signatures.py | RF detection overlay |
| LOW | naval.py | River/harbor with patrol boats |
| LOW | air_combat.py | Police helicopter engaging |
| LOW | electronic_warfare.py | Comm jamming during riot |

## PARTIALLY DEMONSTRATED (28 modules — need enhancement)

Key gaps: steering (complex behaviors), combat_ai (flanking/suppression), squad (formation geometry), damage (armor/criticals), environment (weather), crowd (mood propagation), status_effects (buff/debuff display), scoring (achievements), animation (easing library).

## FULLY DEMONSTRATED (13 modules)

pathfinding, ambient, units, vehicles, terrain, medical, civilian, renderer, world, particles, mapgen (partial city gen), city_sim (partial schedules), debug (overlay).
