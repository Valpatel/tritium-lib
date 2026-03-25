// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { Base3DUnit } from './base.js';

export class DroneModel extends Base3DUnit {
    static typeId = 'drone';
    _rotors = [];

    build(THREE, opts = {}) {
        const color = opts.color || 0x00f0ff;
        const g = new THREE.Group();

        // Body
        g.add(this.box(THREE, 0.4, 0.15, 0.4, color));
        // Camera pod
        const cam = this.sphere(THREE, 0.08, 0x222222);
        cam.position.set(0, -0.1, 0.15);
        g.add(cam);

        // 4 rotor arms + discs
        const armPositions = [[0.3, 0, 0.3], [-0.3, 0, 0.3], [0.3, 0, -0.3], [-0.3, 0, -0.3]];
        this._rotors = [];
        for (const [ax, ay, az] of armPositions) {
            const arm = this.box(THREE, 0.05, 0.05, 0.3, 0x333333);
            arm.position.set(ax * 0.5, 0.05, az * 0.5);
            arm.lookAt(ax, 0.05, az);
            g.add(arm);

            const rotor = this.cylinder(THREE, 0.18, 0.18, 0.02, color, 16);
            rotor.position.set(ax, 0.12, az);
            g.add(rotor);
            this._rotors.push(rotor);
        }

        // Default hover height
        g.position.y = 3;
        this.group = g;
        return g;
    }

    animate(dt) {
        for (const r of this._rotors) {
            r.rotation.y += dt * 30;
        }
    }
}
