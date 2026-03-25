// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { Base3DUnit } from './base.js';

export class RoverModel extends Base3DUnit {
    static typeId = 'rover';

    build(THREE, opts = {}) {
        const color = opts.color || 0x05ffa1;
        const g = new THREE.Group();

        // Chassis
        g.add(this.box(THREE, 0.8, 0.25, 0.5, color));
        // Wheels (4)
        for (const [wx, wz] of [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]]) {
            const wheel = this.cylinder(THREE, 0.12, 0.12, 0.08, 0x222222);
            wheel.rotation.x = Math.PI / 2;
            wheel.position.set(wx, -0.1, wz);
            g.add(wheel);
        }
        // Sensor mast
        const mast = this.cylinder(THREE, 0.03, 0.03, 0.4, 0x444444);
        mast.position.set(0, 0.32, 0);
        g.add(mast);
        const sensor = this.sphere(THREE, 0.06, 0x00f0ff);
        sensor.position.set(0, 0.55, 0);
        g.add(sensor);

        this.group = g;
        return g;
    }
}
