// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Pedestrian Controller — goal-driven NPC movement with car collision.
 *
 * Each pedestrian has:
 * - A position (x, z) on the sidewalk/crosswalk network
 * - A goal (destination building, park, bus stop, etc.)
 * - A daily routine from schedule-executor.js
 * - Collision detection with cars (can get knocked over)
 *
 * Movement uses a simple sidewalk graph (not NavMesh) for performance.
 * Pedestrians walk along sidewalk edges, cross at crosswalks, and
 * avoid each other with simple separation forces.
 *
 * Based on: Social Force Model (Helbing 1995), simplified for performance.
 */

// ============================================================
// PEDESTRIAN STATE
// ============================================================

/**
 * Create a new pedestrian state object.
 *
 * @param {number} x - Starting world X
 * @param {number} z - Starting world Z
 * @param {string} id - Unique identifier
 * @param {string} role - 'resident', 'worker', 'student', etc.
 * @returns {Object} Pedestrian state
 */
export function createPedestrian(x, z, id, role = 'resident') {
    return {
        id,
        role,
        x, z,
        vx: 0, vz: 0,          // velocity
        speed: 0,
        desiredSpeed: 0.8 + Math.random() * 0.6, // 0.8-1.4 m/s
        heading: Math.random() * Math.PI * 2,

        // Goal system
        goalX: x, goalZ: z,     // current destination
        goalType: 'idle',       // 'idle', 'walk_to', 'crossing', 'knocked_down'
        goalReached: true,

        // Knocked down state
        knockedDown: false,
        knockedTimer: 0,        // seconds remaining on ground
        knockedAngle: 0,        // rotation when knocked over

        // Visual
        meshIndex: -1,          // index in instanced mesh
        bobPhase: Math.random() * Math.PI * 2,

        // Schedule
        scheduleActive: false,
    };
}

// ============================================================
// TICK — one frame of pedestrian simulation
// ============================================================

/**
 * Update a single pedestrian for one frame.
 *
 * @param {Object} ped - Pedestrian state from createPedestrian()
 * @param {number} dt - Time step (seconds)
 * @param {Array} allPeds - All pedestrians (for separation)
 * @param {Array} cars - All cars with { worldX, worldZ, speed, worldHeading, length }
 * @param {Object} spatialHash - Spatial hash for nearby lookups (optional)
 */
export function tickPedestrian(ped, dt, allPeds, cars, spatialHash = null) {
    // Knocked down: lie on ground, recover after timer
    if (ped.knockedDown) {
        ped.knockedTimer -= dt;
        ped.speed = 0;
        ped.vx = 0;
        ped.vz = 0;
        if (ped.knockedTimer <= 0) {
            ped.knockedDown = false;
            ped.goalType = 'idle';
        }
        return;
    }

    // 1. Compute desired velocity toward goal
    let dvx = 0, dvz = 0;
    if (!ped.goalReached && ped.goalType !== 'idle') {
        const dx = ped.goalX - ped.x;
        const dz = ped.goalZ - ped.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (dist < 2.0) {
            // Reached goal
            ped.goalReached = true;
            ped.goalType = 'idle';
            dvx = 0;
            dvz = 0;
        } else {
            // Move toward goal at desired speed
            dvx = (dx / dist) * ped.desiredSpeed;
            dvz = (dz / dist) * ped.desiredSpeed;
        }
    }

    // 2. Separation force from nearby pedestrians
    const neighbors = spatialHash
        ? spatialHash.getNearby(ped.x, ped.z)
        : allPeds;

    for (const other of neighbors) {
        if (other === ped || other.knockedDown) continue;
        const dx = ped.x - other.x;
        const dz = ped.z - other.z;
        const distSq = dx * dx + dz * dz;
        if (distSq < 4 && distSq > 0.01) { // 2m radius
            const dist = Math.sqrt(distSq);
            const repulsion = (2 - dist) * 0.5; // stronger when closer
            dvx += (dx / dist) * repulsion;
            dvz += (dz / dist) * repulsion;
        }
    }

    // 3. Car avoidance: run away from approaching cars
    for (const car of cars) {
        if (!car.worldX && car.worldX !== 0) continue;
        const dx = ped.x - car.worldX;
        const dz = ped.z - car.worldZ;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist > 15 || dist < 0.1) continue;

        // Is car heading toward us?
        const carFwdX = Math.sin(car.worldHeading || 0);
        const carFwdZ = Math.cos(car.worldHeading || 0);
        const dot = (-dx * carFwdX + -dz * carFwdZ) / dist;

        if (dot > 0.3 && dist < 8) {
            // Car approaching — dodge sideways
            const urgency = (8 - dist) * car.speed * 0.3;
            // Perpendicular to car's heading
            dvx += -carFwdZ * urgency * Math.sign(dx * carFwdZ - dz * carFwdX);
            dvz += carFwdX * urgency * Math.sign(dx * carFwdZ - dz * carFwdX);
        }
    }

    // 4. Update velocity with relaxation (Social Force Model tau = 0.5s)
    const tau = 0.5;
    ped.vx += (dvx - ped.vx) / tau * dt;
    ped.vz += (dvz - ped.vz) / tau * dt;

    // Clamp speed
    ped.speed = Math.sqrt(ped.vx * ped.vx + ped.vz * ped.vz);
    if (ped.speed > ped.desiredSpeed * 1.5) {
        const scale = (ped.desiredSpeed * 1.5) / ped.speed;
        ped.vx *= scale;
        ped.vz *= scale;
        ped.speed = ped.desiredSpeed * 1.5;
    }

    // 5. Update position
    ped.x += ped.vx * dt;
    ped.z += ped.vz * dt;

    // Update heading (smooth)
    if (ped.speed > 0.1) {
        const targetHeading = Math.atan2(ped.vx, ped.vz);
        let dh = targetHeading - ped.heading;
        if (dh > Math.PI) dh -= Math.PI * 2;
        if (dh < -Math.PI) dh += Math.PI * 2;
        ped.heading += dh * Math.min(1, dt * 5);
    }

    // Walk bob phase
    ped.bobPhase += dt * ped.speed * 8;

    // 6. Car collision detection — get knocked over
    for (const car of cars) {
        if (!car.worldX && car.worldX !== 0) continue;
        if (car.speed < 1) continue; // stationary cars don't knock people over
        const dx = ped.x - car.worldX;
        const dz = ped.z - car.worldZ;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 2.0) { // hit radius
            ped.knockedDown = true;
            ped.knockedTimer = 3 + Math.random() * 4; // 3-7 seconds on ground
            ped.knockedAngle = Math.atan2(dx, dz); // fall away from car
            ped.vx = (dx / dist) * car.speed * 0.3; // pushed by car
            ped.vz = (dz / dist) * car.speed * 0.3;
            ped.speed = 0;
            break;
        }
    }
}

// ============================================================
// GOAL ASSIGNMENT
// ============================================================

/**
 * Assign a new walking goal to a pedestrian.
 *
 * @param {Object} ped
 * @param {number} goalX
 * @param {number} goalZ
 * @param {string} goalType - 'walk_to', 'crossing', etc.
 */
export function assignGoal(ped, goalX, goalZ, goalType = 'walk_to') {
    ped.goalX = goalX;
    ped.goalZ = goalZ;
    ped.goalType = goalType;
    ped.goalReached = false;
}

/**
 * Assign a random nearby goal (for idle wandering).
 *
 * @param {Object} ped
 * @param {number} radius - Max distance for random goal
 */
export function assignRandomGoal(ped, radius = 30) {
    const angle = Math.random() * Math.PI * 2;
    const dist = 5 + Math.random() * radius;
    assignGoal(ped, ped.x + Math.cos(angle) * dist, ped.z + Math.sin(angle) * dist, 'walk_to');
}

// ============================================================
// INSTANCED PEDESTRIAN RENDERER
// ============================================================

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export class InstancedPedRenderer {
    /**
     * @param {THREE.Scene} scene
     * @param {number} maxPeds
     */
    constructor(scene, maxPeds = 1000) {
        this.scene = scene;
        this.maxPeds = maxPeds;
        this.count = 0;
        this.dummy = new THREE.Object3D();

        // Body: capsule-like shape (cylinder + sphere head)
        const bodyGeo = new THREE.CylinderGeometry(0.25, 0.25, 1.2, 6);
        bodyGeo.translate(0, 0.8, 0); // bottom at ground
        const headGeo = new THREE.SphereGeometry(0.2, 6, 4);
        headGeo.translate(0, 1.6, 0);
        const merged = mergeGeometries([bodyGeo, headGeo], false);

        const mat = new THREE.MeshStandardMaterial({ color: 0x44aa66, roughness: 0.7 });
        this.mesh = new THREE.InstancedMesh(merged, mat, maxPeds);
        this.mesh.count = 0;
        this.mesh.castShadow = true;

        // Per-instance color for role differentiation
        const colors = new Float32Array(maxPeds * 3);
        this.mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);

        const hideMatrix = new THREE.Matrix4().makeTranslation(0, -1000, 0);
        for (let i = 0; i < maxPeds; i++) this.mesh.setMatrixAt(i, hideMatrix);
        this.mesh.instanceMatrix.needsUpdate = true;

        scene.add(this.mesh);
    }

    /**
     * Add a pedestrian instance.
     * @param {number} color - Hex color based on role
     * @returns {number} Instance index
     */
    addPed(color = 0x44aa66) {
        const idx = this.count++;
        if (idx >= this.maxPeds) return -1;
        this.mesh.count = this.count;
        const c = new THREE.Color(color);
        this.mesh.instanceColor.setXYZ(idx, c.r, c.g, c.b);
        this.mesh.instanceColor.needsUpdate = true;
        return idx;
    }

    /**
     * Update a pedestrian's visual position.
     * @param {number} idx - Instance index
     * @param {Object} ped - Pedestrian state
     */
    updatePed(idx, ped) {
        if (idx < 0 || idx >= this.count) return;

        // Walk bob: sinusoidal vertical offset
        const bob = ped.knockedDown ? 0 : Math.sin(ped.bobPhase) * 0.03;

        if (ped.knockedDown) {
            // Lying on ground — rotate 90° around Z
            this.dummy.position.set(ped.x, 0.3, ped.z);
            this.dummy.rotation.set(0, ped.knockedAngle, Math.PI / 2);
        } else {
            this.dummy.position.set(ped.x, bob, ped.z);
            this.dummy.rotation.set(0, ped.heading, 0);
        }
        this.dummy.updateMatrix();
        this.mesh.setMatrixAt(idx, this.dummy.matrix);
    }

    flush() {
        this.mesh.instanceMatrix.needsUpdate = true;
    }
}
