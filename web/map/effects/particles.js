// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * ParticleBurst — debris/sparks radiating from a point.
 *
 * Creates N small meshes that fly outward with gravity,
 * fading and shrinking over duration.
 */

export class ParticleBurst {
    /**
     * @param {Object} opts
     * @param {number} opts.x — game X
     * @param {number} opts.y — game Y
     * @param {number} opts.color — hex color
     * @param {number} opts.count — number of particles
     * @param {number} opts.duration — ms
     * @param {THREE} THREE
     * @param {THREE.Group} parent
     */
    constructor(opts, THREE, parent) {
        this.alive = true;
        this.startTime = performance.now();
        this.endTime = this.startTime + (opts.duration || 1000);
        this.duration = opts.duration || 1000;

        this.particles = [];
        const geom = new THREE.BoxGeometry(0.15, 0.15, 0.15);
        const mat = new THREE.MeshBasicMaterial({
            color: opts.color || 0xffaa00,
            transparent: true,
        });

        this.mesh = new THREE.Group();
        const count = opts.count || 15;
        for (let i = 0; i < count; i++) {
            const p = new THREE.Mesh(geom, mat.clone());
            const angle = Math.random() * Math.PI * 2;
            const speed = 2 + Math.random() * 6;
            p.position.set(opts.x, opts.y, 0.5 + Math.random() * 2);
            this.particles.push({
                mesh: p,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                vz: 2 + Math.random() * 4,
            });
            this.mesh.add(p);
        }
        if (parent) parent.add(this.mesh);
    }

    update(now) {
        const t = Math.min(1, (now - this.startTime) / this.duration);
        const dt = 0.016; // ~60fps step

        for (const p of this.particles) {
            p.mesh.position.x += p.vx * dt;
            p.mesh.position.y += p.vy * dt;
            p.mesh.position.z += p.vz * dt;
            p.vz -= 9.8 * dt; // gravity
            const s = 1 - t;
            p.mesh.scale.set(s, s, s);
            if (p.mesh.material) p.mesh.material.opacity = s;
        }

        if (t >= 1) this.alive = false;
    }
}
