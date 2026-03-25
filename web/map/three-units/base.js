// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Base3DUnit — base class for Three.js tactical unit models.
 *
 * Each unit type (turret, drone, rover, person, etc.) extends this and
 * overrides build() to construct its mesh hierarchy. The base handles
 * common patterns: ground ring, selection glow, animation tick.
 *
 * Usage:
 *   class TurretModel extends Base3DUnit {
 *       static typeId = 'turret';
 *       build(THREE) { return this._buildTurret(THREE); }
 *   }
 */

export class Base3DUnit {
    static typeId = 'generic';

    constructor() {
        this.group = null;   // THREE.Group root
        this.disposed = false;
    }

    /**
     * Build the 3D model. Override in subclass.
     * @param {Object} THREE — Three.js namespace
     * @param {Object} [opts] — { color, alliance, scale }
     * @returns {THREE.Group}
     */
    build(THREE, opts = {}) {
        const g = new THREE.Group();
        const geo = new THREE.BoxGeometry(1, 1, 1);
        const mat = new THREE.MeshLambertMaterial({ color: opts.color || 0x888888 });
        g.add(new THREE.Mesh(geo, mat));
        this.group = g;
        return g;
    }

    /**
     * Animate per-frame. Override for spinning barrels, rotor blades, etc.
     * @param {number} dt — seconds since last frame
     */
    animate(dt) {}

    /** Add a ground ring (selection/range indicator). */
    addGroundRing(THREE, group, radius, color, opacity = 0.3) {
        const geo = new THREE.RingGeometry(radius * 0.9, radius, 32);
        const mat = new THREE.MeshBasicMaterial({
            color, transparent: true, opacity, side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(geo, mat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.02;
        group.add(ring);
        return ring;
    }

    /** Add a vertical beam (sensor/scanner indicator). */
    addBeam(THREE, group, height, color, opacity = 0.15) {
        const geo = new THREE.CylinderGeometry(0.05, 0.3, height, 8);
        const mat = new THREE.MeshBasicMaterial({
            color, transparent: true, opacity,
        });
        const beam = new THREE.Mesh(geo, mat);
        beam.position.y = height / 2;
        group.add(beam);
        return beam;
    }

    /** Create a box mesh helper. */
    box(THREE, w, h, d, color) {
        return new THREE.Mesh(
            new THREE.BoxGeometry(w, h, d),
            new THREE.MeshLambertMaterial({ color })
        );
    }

    /** Create a cylinder mesh helper. */
    cylinder(THREE, rTop, rBot, h, color, segments = 12) {
        return new THREE.Mesh(
            new THREE.CylinderGeometry(rTop, rBot, h, segments),
            new THREE.MeshLambertMaterial({ color })
        );
    }

    /** Create a sphere mesh helper. */
    sphere(THREE, r, color, segments = 10) {
        return new THREE.Mesh(
            new THREE.SphereGeometry(r, segments, segments),
            new THREE.MeshLambertMaterial({ color })
        );
    }

    /** Dispose all geometries and materials in the group. */
    dispose() {
        if (this.disposed || !this.group) return;
        this.disposed = true;
        this.group.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                else child.material.dispose();
            }
        });
        if (this.group.parent) this.group.parent.remove(this.group);
    }
}
