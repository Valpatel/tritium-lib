// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * FlashEffect — brief bright sphere at impact point (muzzle flash or hit).
 */

export class FlashEffect {
    constructor(opts, THREE, parent) {
        this.alive = true;
        this.startTime = performance.now();
        this.endTime = this.startTime + (opts.duration || 150);
        this.duration = opts.duration || 150;

        const geom = new THREE.SphereGeometry(opts.radius || 0.5, 8, 6);
        const mat = new THREE.MeshBasicMaterial({
            color: opts.color || 0xffffff,
            transparent: true,
            opacity: 1.0,
        });
        this.mesh = new THREE.Mesh(geom, mat);
        this.mesh.position.set(opts.x, opts.y, opts.z || 1.5);
        if (parent) parent.add(this.mesh);
    }

    update(now) {
        const t = Math.min(1, (now - this.startTime) / this.duration);
        if (this.mesh.material) this.mesh.material.opacity = 1 - t;
        const s = 1 + t * 0.5;
        this.mesh.scale.set(s, s, s);
        if (t >= 1) this.alive = false;
    }
}
