// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Combat effects — each effect type in its own file for clarity.
 *
 * Effects:
 *   ProjectileEffect — tracer/bullet/missile flight path
 *   ExplosionEffect  — expanding sphere at impact
 *   ParticleBurst    — debris/sparks radiating outward
 *   FlashEffect      — brief muzzle/hit flash
 *   FloatingText     — rising DOM text (damage numbers)
 *   CombatEffects    — base class managing effect lifecycle
 */

export { CombatEffects, DEFAULT_WEAPON_VFX } from './base.js';
export { ProjectileEffect } from './projectile.js';
export { ExplosionEffect } from './explosion.js';
export { ParticleBurst } from './particles.js';
export { FlashEffect } from './flash.js';
export { FloatingText } from './floating-text.js';
