// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * ProjectileEffect — a tracer/bullet/missile flying from shooter to target.
 *
 * Creates a line or tube mesh that interpolates from start to end position
 * over a duration based on weapon speed. Leaves a fading trail behind it.
 */

export class ProjectileEffect {
    /**
     * @param {Object} opts
     * @param {Object} opts.from — { x, y } start position (game meters)
     * @param {Object} opts.to — { x, y } end position
     * @param {number} opts.color — hex color (e.g., 0xffcc00)
     * @param {number} opts.speed — meters/second
     * @param {number} opts.trailWidth — trail thickness
     * @param {number} opts.trailLength — trail length in meters
     * @param {THREE} THREE — Three.js namespace
     * @param {THREE.Group} parent — group to add mesh to
     */
    constructor(opts, THREE, parent) {
        this.alive = true;
        this.startTime = performance.now();

        const dx = opts.to.x - opts.from.x;
        const dy = opts.to.y - opts.from.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        this.duration = (dist / Math.max(opts.speed, 1)) * 1000; // ms
        this.endTime = this.startTime + this.duration;

        this.from = opts.from;
        this.to = opts.to;
        this.dx = dx;
        this.dy = dy;

        // Create line mesh
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array([
            opts.from.x, opts.from.y, 1.5,
            opts.from.x, opts.from.y, 1.5,
        ]);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.LineBasicMaterial({
            color: opts.color,
            transparent: true,
            opacity: 0.9,
            linewidth: 2,
        });
        this.mesh = new THREE.Line(geom, mat);
        if (parent) parent.add(this.mesh);
        this._positions = positions;
    }

    update(now) {
        const t = Math.min(1, (now - this.startTime) / this.duration);
        const headX = this.from.x + this.dx * t;
        const headY = this.from.y + this.dy * t;
        const tailT = Math.max(0, t - 0.15);
        const tailX = this.from.x + this.dx * tailT;
        const tailY = this.from.y + this.dy * tailT;

        this._positions[0] = tailX;
        this._positions[1] = tailY;
        this._positions[3] = headX;
        this._positions[4] = headY;
        this.mesh.geometry.attributes.position.needsUpdate = true;

        if (this.mesh.material) {
            this.mesh.material.opacity = 0.9 * (1 - t * 0.3);
        }

        if (t >= 1) this.alive = false;
    }
}
