// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * CombatEffects — base class for Three.js tactical map visual effects.
 *
 * Provides the interface and shared utilities for projectile trails,
 * explosions, hit flashes, floating text, and streak announcements.
 * Subclass this in your app and override the spawn methods.
 *
 * Handles:
 * - Effect lifecycle (spawn → animate → dispose)
 * - Repaint loop management (requestAnimationFrame)
 * - Toggle state (showTracers, showExplosions, etc.)
 * - Object pooling for geometry reuse
 *
 * Usage:
 *   import { CombatEffects } from '/lib/map/effects.js';
 *   const fx = new CombatEffects(threeScene, coordSystem);
 *   fx.onProjectile({ from, to, type, speed });
 *   fx.onHit({ position, damage, targetId });
 *   fx.animate(deltaTime); // call in render loop
 */

// Default weapon VFX presets — apps can override via setWeaponPresets()
const DEFAULT_WEAPON_VFX = {
    bullet:       { color: 0xffcc00, trailWidth: 0.15, trailLength: 8,  speed: 200, explosionSize: 0,   particleCount: 0  },
    shell:        { color: 0xff6600, trailWidth: 0.3,  trailLength: 5,  speed: 120, explosionSize: 3,   particleCount: 15 },
    missile:      { color: 0xff2200, trailWidth: 0.4,  trailLength: 12, speed: 80,  explosionSize: 8,   particleCount: 30 },
    laser:        { color: 0x00ffff, trailWidth: 0.1,  trailLength: 50, speed: 999, explosionSize: 0.5, particleCount: 5  },
    plasma:       { color: 0xff00ff, trailWidth: 0.5,  trailLength: 6,  speed: 60,  explosionSize: 5,   particleCount: 20 },
    dart:         { color: 0x88ccff, trailWidth: 0.08, trailLength: 4,  speed: 250, explosionSize: 0,   particleCount: 0  },
    grenade:      { color: 0xff4400, trailWidth: 0.2,  trailLength: 3,  speed: 30,  explosionSize: 6,   particleCount: 25 },
    flamethrower: { color: 0xff3300, trailWidth: 0.8,  trailLength: 4,  speed: 15,  explosionSize: 2,   particleCount: 10 },
};

export class CombatEffects {
    /**
     * @param {THREE.Scene} scene — Three.js scene to add effects to
     * @param {Object} [options]
     * @param {boolean} [options.showTracers=true]
     * @param {boolean} [options.showExplosions=true]
     * @param {boolean} [options.showParticles=true]
     * @param {boolean} [options.showHitFlashes=true]
     * @param {boolean} [options.showFloatingText=true]
     */
    constructor(scene, options = {}) {
        this.scene = scene;
        this.effectsRoot = null; // Set via setEffectsRoot() or auto-created
        this.effects = [];       // Active effect objects
        this._animating = false;
        this._animId = null;
        this._weaponVFX = { ...DEFAULT_WEAPON_VFX };

        // Toggle state
        this.showTracers = options.showTracers ?? true;
        this.showExplosions = options.showExplosions ?? true;
        this.showParticles = options.showParticles ?? true;
        this.showHitFlashes = options.showHitFlashes ?? true;
        this.showFloatingText = options.showFloatingText ?? true;
    }

    /** Set the Three.js group that effects are added to. */
    setEffectsRoot(group) {
        this.effectsRoot = group;
    }

    /** Override weapon VFX presets. */
    setWeaponPresets(presets) {
        Object.assign(this._weaponVFX, presets);
    }

    /** Get VFX preset for a weapon type. */
    getWeaponVFX(projectileType) {
        return this._weaponVFX[projectileType] || this._weaponVFX.bullet;
    }

    /**
     * Handle a projectile event.
     * Override in subclass for Three.js mesh creation.
     * @param {Object} data — { from: {x,y}, to: {x,y}, projectileType, speed, shooterId, targetId }
     */
    onProjectile(data) {
        // Base: no-op. Override in subclass.
    }

    /**
     * Handle a hit event.
     * @param {Object} data — { position: {x,y}, damage, targetId, weapon }
     */
    onHit(data) {
        // Base: no-op. Override in subclass.
    }

    /**
     * Handle an elimination event.
     * @param {Object} data — { position: {x,y}, targetId, killerId, targetName }
     */
    onElimination(data) {
        // Base: no-op. Override in subclass.
    }

    /**
     * Handle a streak event.
     * @param {Object} data — { unitId, streakCount, streakType, position: {x,y} }
     */
    onStreak(data) {
        // Base: no-op. Override in subclass.
    }

    /**
     * Animate all active effects. Call this every frame.
     * @param {number} now — performance.now() timestamp
     */
    animate(now) {
        for (let i = this.effects.length - 1; i >= 0; i--) {
            const fx = this.effects[i];
            if (!fx.alive) {
                this._disposeEffect(fx);
                this.effects.splice(i, 1);
                continue;
            }
            if (fx.update) fx.update(now);
            if (fx.endTime && now >= fx.endTime) {
                fx.alive = false;
            }
        }
    }

    /** Add an effect to the active list. */
    addEffect(effect) {
        this.effects.push(effect);
    }

    /** Remove and dispose a single effect. */
    _disposeEffect(fx) {
        if (fx.mesh) {
            if (fx.mesh.parent) fx.mesh.parent.remove(fx.mesh);
            if (fx.mesh.geometry) fx.mesh.geometry.dispose();
            if (fx.mesh.material) {
                if (Array.isArray(fx.mesh.material)) {
                    fx.mesh.material.forEach(m => m.dispose());
                } else {
                    fx.mesh.material.dispose();
                }
            }
        }
        if (fx.domEl) fx.domEl.remove();
    }

    /** Remove all active effects. */
    clearAll() {
        for (const fx of this.effects) this._disposeEffect(fx);
        this.effects = [];
    }

    /** Clean up everything. */
    destroy() {
        this.clearAll();
        if (this._animId) cancelAnimationFrame(this._animId);
        this._animating = false;
    }

    /** Number of active effects. */
    get activeCount() {
        return this.effects.length;
    }
}

export { DEFAULT_WEAPON_VFX };
