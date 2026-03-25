// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * ExplosionEffect — expanding sphere + particle burst at impact point.
 *
 * A sphere mesh expands from 0 to radiusMeters over durationMs,
 * fading out as it grows. Optionally spawns particle debris.
 */

export class ExplosionEffect {
    /**
     * @param {Object} opts
     * @param {number} opts.x — game X position
     * @param {number} opts.y — game Y position
     * @param {number} opts.color — hex color
     * @param {number} opts.radius — max radius in meters
     * @param {number} opts.duration — ms
     * @param {THREE} THREE
     * @param {THREE.Group} parent
     */
    constructor(opts, THREE, parent) {
        this.alive = true;
        this.startTime = performance.now();
        this.endTime = this.startTime + (opts.duration || 600);
        this.duration = opts.duration || 600;

        const geom = new THREE.SphereGeometry(1, 12, 8);
        const mat = new THREE.MeshBasicMaterial({
            color: opts.color || 0xff6600,
            transparent: true,
            opacity: 0.7,
        });
        this.mesh = new THREE.Mesh(geom, mat);
        this.mesh.position.set(opts.x, opts.y, 0.5);
        this.mesh.scale.set(0.1, 0.1, 0.1);
        this.maxRadius = opts.radius || 3;
        if (parent) parent.add(this.mesh);
    }

    update(now) {
        const t = Math.min(1, (now - this.startTime) / this.duration);
        const r = this.maxRadius * t;
        this.mesh.scale.set(r, r, r * 0.6);
        if (this.mesh.material) {
            this.mesh.material.opacity = 0.7 * (1 - t);
        }
        if (t >= 1) this.alive = false;
    }
}
