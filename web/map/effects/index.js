// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Combat effects — each effect type in its own file for clarity.
 *
 * Effects:
 *   FlashEffect      — brief muzzle/hit flash
 *   FloatingText     — rising DOM text (damage numbers)
 *   CombatEffects    — base class managing effect lifecycle
 *
 * Note (W201 cleanup): ProjectileEffect, ExplosionEffect, and
 * ParticleBurst were removed because every consumer used the
 * inline implementations in tritium-sc map-maplibre.js
 * (_spawnProjectile / _spawnExplosion / _spawnParticleBurst)
 * rather than the lib classes.  If a future consumer needs the
 * lib pattern, restore from git history (commits before 1d7bc4a).
 */

export { CombatEffects, DEFAULT_WEAPON_VFX } from './base.js';
export { FlashEffect } from './flash.js';
export { FloatingText } from './floating-text.js';
