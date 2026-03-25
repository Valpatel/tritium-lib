// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { Base3DUnit } from './base.js';

export class TurretModel extends Base3DUnit {
    static typeId = 'turret';

    build(THREE, opts = {}) {
        const color = opts.color || 0x00f0ff;
        const dark = 0x003344;
        const g = new THREE.Group();

        // Base platform
        g.add(this.cylinder(THREE, 0.6, 0.7, 0.3, dark));
        // Pedestal
        const ped = this.cylinder(THREE, 0.25, 0.35, 0.6, color);
        ped.position.y = 0.45;
        g.add(ped);
        // Head
        const head = this.sphere(THREE, 0.3, color);
        head.position.y = 0.9;
        g.add(head);
        // Barrel
        const barrel = this.cylinder(THREE, 0.06, 0.06, 0.8, dark);
        barrel.rotation.z = Math.PI / 2;
        barrel.position.set(0.5, 0.9, 0);
        g.add(barrel);

        this.group = g;
        return g;
    }
}
