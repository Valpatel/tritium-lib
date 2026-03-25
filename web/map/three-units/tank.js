// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { Base3DUnit } from './base.js';

export class TankModel extends Base3DUnit {
    static typeId = 'tank';

    build(THREE, opts = {}) {
        const color = opts.color || 0x556655;
        const g = new THREE.Group();

        // Hull
        g.add(this.box(THREE, 1.2, 0.3, 0.7, color));
        // Turret
        const turret = this.cylinder(THREE, 0.25, 0.3, 0.2, color);
        turret.position.y = 0.25;
        g.add(turret);
        // Barrel
        const barrel = this.cylinder(THREE, 0.05, 0.05, 1.0, 0x333333);
        barrel.rotation.z = Math.PI / 2;
        barrel.position.set(0.6, 0.3, 0);
        g.add(barrel);
        // Tracks
        for (const z of [0.35, -0.35]) {
            const track = this.box(THREE, 1.3, 0.15, 0.12, 0x222222);
            track.position.set(0, -0.1, z);
            g.add(track);
        }

        this.group = g;
        return g;
    }
}
