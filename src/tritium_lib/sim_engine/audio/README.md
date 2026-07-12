# sim_engine/audio/ — spatial audio math

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

One file of pure math that tells the browser's Web Audio how to place a sound:
how loud (distance attenuation), which ear (stereo pan), pitch-shifted by motion
(Doppler), delayed by travel time, muffled by buildings (occlusion), and how
much room echo (reverb). The backend computes the numbers; the frontend drives
the actual audio nodes. All 2-D (`Vec2`), stdlib only.

## Files

| File | Key objects | What it does |
|------|-------------|--------------|
| `spatial.py` | `SoundEvent` (`:211`), `distance_attenuation` (`:33`), `stereo_pan` (`:53`), `doppler_factor` (`:86`), `propagation_delay` (`:122`), `occlusion_factor` (`:164`), `reverb_level` (`:187`), `gunshot_layers` (`:285`), `explosion_parameters` (`:326`) | Spatial-audio parameters for one listener; `SoundEvent` bundles a positioned sound; `gunshot_layers`/`explosion_parameters` shape signature sounds by distance |

## How it's used

- `effects/weapons.py` attaches a `SoundEvent` to each fired round
  (`weapons.py:25` imports `SoundEvent`).
- `demos/game_server.py` imports the `audio.spatial` functions
  (`game_server.py:65`) and computes per-listener audio params into the frame.

## Palantir lens

- **Object:** `SoundEvent` — a positioned, typed sound (source, kind, loudness).
- **Typed actions:** each function is a pure map from geometry to an audio
  parameter — `distance_attenuation(source, listener) -> gain`,
  `stereo_pan(source, listener, heading) -> [-1..1]`,
  `doppler_factor(source_vel, listener_vel, …) -> pitch ratio`. No state.
- **Links:** a `SoundEvent` links a sound to the world position that emitted it;
  the listener geometry links it to what a given unit/camera hears.

## Related

- [`../effects/README.md`](../effects/README.md) — the firing model that emits `SoundEvent`s
- SC acoustic plugin: `tritium-sc/plugins/acoustic/` (the production side that
  classifies real sounds — the same acoustic domain this validates)

## Dependencies

None — pure Python / stdlib.
