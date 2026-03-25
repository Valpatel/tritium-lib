// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { Base3DUnit } from './base.js';

export class PersonModel extends Base3DUnit {
    static typeId = 'person';

    build(THREE, opts = {}) {
        const color = opts.color || 0x888888;
        const g = new THREE.Group();

        // Body
        g.add(this.cylinder(THREE, 0.12, 0.15, 0.5, color));
        // Head
        const head = this.sphere(THREE, 0.1, 0xddbbaa);
        head.position.y = 0.4;
        g.add(head);

        this.group = g;
        return g;
    }
}
