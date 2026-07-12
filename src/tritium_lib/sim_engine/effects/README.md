# sim_engine/effects/ — particles & weapon firing

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

The backend that computes *what the frontend draws when things happen*:
particle systems (explosions, smoke, tracers, debris, fire, blood) and the
weapon-firing state machine that emits per-round events. The backend computes;
the Three.js frontend renders. Both files serialize to JSON via `to_three_js()`.

## Files

| File | Key objects | What it does |
|------|-------------|--------------|
| `particles.py` | `EffectsManager` (`:566`), `ParticleEmitter` (`:112`), `Particle` (`:36`) + factories `explosion`/`muzzle_flash`/`tracer`/`smoke`/`debris`/`blood_splatter`/`fire`/`sparks` (`:317`–`:527`) | Particle emitters with lifetime, color-lerp, gravity; `EffectsManager` pools up to N emitters and exports `to_three_js(max_particles)` |
| `weapons.py` | `WeaponFirer` (`:288`), `WeaponProfile` (`:49`), `FiredRound` (`:255`), `FireMode` (`:33`), `WEAPONS` catalog (`:88`), `create_firer` (`:575`) | The **firing-state model** — cooldowns, burst patterns, ammo, spread bloom; produces per-round events the frontend turns into particles + spatial audio |

> `weapons.py` is not "effect specs" — it is a stateful firer. `WeaponFirer.tick(dt)`
> (`weapons.py:350`) advances bursts and returns the `FiredRound`s produced this
> tick. It pulls in `audio.SoundEvent` and a `debug.DebugStream` to attach sound
> and diagnostics to each shot.

## How it's wired

`demos/game_server.py` owns a single `EffectsManager(max_emitters=256)`
(`game_server.py:1485`), imports the factory functions (`game_server.py:60`),
and folds `to_three_js()` output into the frame under the `effects` key. Each
`WeaponFirer` is created per weapon via `create_firer(weapon_id)`.

## Palantir lens

- **Objects:** `EffectsManager` (the pool), `ParticleEmitter` (one effect
  instance), `WeaponFirer` (one weapon's live state), `WeaponProfile` (the
  weapon's static spec in `WEAPONS`).
- **Typed actions:** `EffectsManager.add(emitter)` / `.tick(dt)` /
  `.to_three_js()`; `WeaponFirer.tick(dt) -> list[FiredRound]`. State in →
  render/round data out.
- **Links:** a `FiredRound` links back to its `WeaponProfile`; a spawned
  `ParticleEmitter` (e.g. from `muzzle_flash(pos, heading)`) is the visual
  consequence of a round.

## Related

- [`../audio/README.md`](../audio/README.md) — the `SoundEvent` each shot carries
- [`../debug/README.md`](../debug/README.md) — the `DebugStream` firers attach
- SC frontend rendering: `tritium-sc/src/frontend/js/`

## Dependencies

None — pure Python / stdlib.
