// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Pedestrian — a person walking through the city.
 *
 * Unlike vehicles, pedestrians don't follow road lanes. They walk
 * point-to-point using the Social Force Model for collision avoidance.
 * They use the path system for smooth curves but can deviate to dodge.
 *
 * Features:
 * - Social force separation from other pedestrians
 * - Car avoidance (dodge sideways)
 * - Knocked down by car impact (recovery after delay)
 * - Role-based coloring and behavior
 */

import { GroundUnit } from '../core/ground-unit.js';

export class Pedestrian extends GroundUnit {
    constructor(config = {}) {
        super({
            type: 'pedestrian',
            length: 0.5,
            width: 0.5,
            height: 1.7,
            collisionRadius: 0.4,
            speed: 0.8 + Math.random() * 0.6, // 0.8-1.4 m/s
            ...config,
        });

        this.desiredSpeed = this.speed;
        this.role = config.role || 'resident';

        // Direct position control (peds don't strictly follow path — they walk point-to-point)
        this.goalX = this.x;
        this.goalZ = this.z;
        this.goalReached = true;

        // Knocked down state
        this.knockedDown = false;
        this.knockedTimer = 0;
        this.knockedAngle = 0;

        // Walk animation
        this.bobPhase = Math.random() * Math.PI * 2;

        // Velocity (direct, not path-based)
        this.vx = 0;
        this.vz = 0;
    }

    /**
     * Pedestrians use direct velocity control, not path-based movement.
     * Override tick entirely.
     */
    tick(dt, world) {
        if (!this.alive) return;

        if (this.knockedDown) {
            this.knockedTimer -= dt;
            this.speed = 0;
            this.vx = 0;
            this.vz = 0;
            if (this.knockedTimer <= 0) {
                this.knockedDown = false;
                this.goalReached = true;
            }
            return;
        }

        // 1. Desired velocity toward goal
        let dvx = 0, dvz = 0;
        if (!this.goalReached) {
            const dx = this.goalX - this.x;
            const dz = this.goalZ - this.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < 2.0) {
                this.goalReached = true;
            } else {
                dvx = (dx / dist) * this.desiredSpeed;
                dvz = (dz / dist) * this.desiredSpeed;
            }
        }

        // 2. Separation force from nearby pedestrians
        const nearby = world.spatialHash ? world.spatialHash.getNearby(this.x, 0, this.z) : [];
        for (const other of nearby) {
            if (other === this || other.type !== 'pedestrian' || other.knockedDown) continue;
            const dx = this.x - other.x;
            const dz = this.z - other.z;
            const distSq = dx * dx + dz * dz;
            if (distSq < 2.25 && distSq > 0.01) {
                const dist = Math.sqrt(distSq);
                const repulsion = (1.5 - dist) * 0.3;
                dvx += (dx / dist) * repulsion;
                dvz += (dz / dist) * repulsion;
            }
        }

        // 3. Car avoidance (only very close, heading at us)
        for (const other of nearby) {
            if (other.type === 'pedestrian' || !other.speed || other.speed < 2) continue;
            const dx = this.x - other.x;
            const dz = this.z - other.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist > 6 || dist < 0.1) continue;

            const fwdX = Math.sin(other.heading || 0);
            const fwdZ = Math.cos(other.heading || 0);
            const dot = (-dx * fwdX + -dz * fwdZ) / dist;
            if (dot > 0.6 && dist < 5) {
                const urgency = Math.min(2.0, (5 - dist) * 0.4);
                const side = Math.sign(dx * fwdZ - dz * fwdX) || 1;
                dvx += -fwdZ * urgency * side;
                dvz += fwdX * urgency * side;
            }
        }

        // 4. Velocity relaxation (tau = 0.5s)
        const tau = 0.5;
        this.vx += (dvx - this.vx) / tau * dt;
        this.vz += (dvz - this.vz) / tau * dt;

        // Clamp
        this.speed = Math.sqrt(this.vx * this.vx + this.vz * this.vz);
        if (this.speed > this.desiredSpeed * 1.5) {
            const s = (this.desiredSpeed * 1.5) / this.speed;
            this.vx *= s;
            this.vz *= s;
            this.speed = this.desiredSpeed * 1.5;
        }

        // 5. Update position
        this.x += this.vx * dt;
        this.z += this.vz * dt;

        // Heading
        if (this.speed > 0.1) {
            const target = Math.atan2(this.vx, this.vz);
            let dh = target - this.heading;
            if (dh > Math.PI) dh -= Math.PI * 2;
            if (dh < -Math.PI) dh += Math.PI * 2;
            this.heading += dh * Math.min(1, dt * 5);
        }

        // Walk bob
        this.bobPhase += dt * this.speed * 8;

        // 6. Car collision → knocked down
        for (const other of nearby) {
            if (other.type === 'pedestrian' || !other.speed || other.speed < 1) continue;
            const dx = this.x - other.x;
            const dz = this.z - other.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < 2.0) {
                this.knockedDown = true;
                this.knockedTimer = 3 + Math.random() * 4;
                this.knockedAngle = Math.atan2(dx, dz);
                this.vx = (dx / dist) * other.speed * 0.3;
                this.vz = (dz / dist) * other.speed * 0.3;
                this.speed = 0;
                break;
            }
        }
    }

    /** Assign a goal position. */
    setGoal(x, z) {
        this.goalX = x;
        this.goalZ = z;
        this.goalReached = false;
    }

    /** Assign a random goal within radius. */
    setRandomGoal(radius = 50) {
        const a = Math.random() * Math.PI * 2;
        const d = 5 + Math.random() * radius;
        this.setGoal(this.x + Math.cos(a) * d, this.z + Math.sin(a) * d);
    }
}
