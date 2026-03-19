// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * GroundUnit — base class for all entities that move on the ground.
 *
 * Cars, pedestrians, motorcycles, tanks, and any future ground unit
 * all inherit from this. Each has:
 * - A Path (continuous parameterized curve)
 * - A position (x, z) derived from the path at distance d
 * - A heading derived from the path tangent
 * - A speed and acceleration
 * - A collision radius
 * - An instance handle for rendering
 *
 * Subclasses provide:
 * - computeAcceleration(world) — how this unit decides to speed up/slow down
 * - onPathEnd(world) — what to do when the path runs out (extend, loop, stop)
 * - getCollisionResponse(other) — what happens on collision
 */

import { Path } from './path.js';

let _nextId = 0;

export class GroundUnit {
    /**
     * @param {Object} config
     * @param {string} config.type - 'car', 'pedestrian', 'motorcycle', 'tank', etc.
     * @param {string} [config.id] - Unique ID (auto-generated if omitted)
     * @param {number} [config.speed] - Initial speed (m/s)
     * @param {number} [config.length] - Length (meters)
     * @param {number} [config.width] - Width (meters)
     * @param {number} [config.height] - Height (meters)
     * @param {number} [config.collisionRadius] - Collision circle radius
     */
    constructor(config = {}) {
        this.id = config.id || `unit_${_nextId++}`;
        this.type = config.type || 'generic';

        // Path state
        this.path = config.path || new Path();
        this.d = config.d || 0;           // distance along path (meters)

        // Motion state
        this.speed = config.speed || 0;
        this.acc = 0;

        // Derived position (updated each tick from path)
        this.x = 0;
        this.z = 0;
        this.y = 0;
        this.heading = 0;
        this.pitch = 0;

        // Physical dimensions
        this.length = config.length || 1;
        this.width = config.width || 1;
        this.height = config.height || 1;
        this.collisionRadius = config.collisionRadius || Math.max(this.length, this.width) / 2;

        // Rendering
        this.instanceHandle = -1;  // set by renderer
        this.visible = true;
        this.color = config.color || 0xffffff;

        // State flags
        this.alive = true;         // set false to remove
        this.inCurve = false;      // currently in a Bezier segment
    }

    /**
     * Main tick — advance along path, update position.
     * Subclasses should override computeAcceleration() not tick().
     *
     * @param {number} dt - Time step (seconds)
     * @param {Object} world - World context (spatial hash, other units, traffic, etc.)
     */
    tick(dt, world) {
        if (!this.alive) return;

        // 1. Subclass computes acceleration
        this.acc = this.computeAcceleration(dt, world);

        // 2. Update speed
        this.speed = Math.max(0, this.speed + this.acc * dt);

        // 3. Advance along path
        this.d = Math.max(0, this.d + this.speed * dt + 0.5 * this.acc * dt * dt);

        // 4. Extend path if running low
        if (this.path.remaining(this.d) < this.getPathExtendThreshold()) {
            this.onPathEnd(world);
        }

        // 5. Derive position from path
        const pos = this.path.getPosition(this.d);
        this.x = pos.x;
        this.y = pos.y || 0;
        this.z = pos.z;
        this.heading = this.path.getHeading(this.d);
        this.pitch = this.path.getPitch(this.d);
        this.inCurve = this.path.isInCurve(this.d);

        // 6. Trim old path behind
        this.path.trimBefore(this.d - 20);
    }

    /**
     * Compute acceleration for this unit. Override in subclasses.
     * @param {number} dt
     * @param {Object} world
     * @returns {number} acceleration in m/s²
     */
    computeAcceleration(dt, world) {
        return 0; // base: no acceleration, constant speed
    }

    /**
     * Called when path is running low. Override to extend or stop.
     * @param {Object} world
     */
    onPathEnd(world) {
        this.speed = 0; // default: stop
    }

    /**
     * How far ahead must the path extend before onPathEnd is called.
     * Vehicles need more lookahead than pedestrians.
     */
    getPathExtendThreshold() {
        return 50; // meters
    }

    /**
     * Get debug info string.
     */
    getDebugInfo() {
        return `${this.id} [${this.type}] pos=(${this.x.toFixed(0)},${this.z.toFixed(0)}) v=${this.speed.toFixed(1)} acc=${this.acc.toFixed(2)}`;
    }
}
