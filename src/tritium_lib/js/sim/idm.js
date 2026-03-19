// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Intelligent Driver Model (IDM) — car-following acceleration model.
 *
 * The IDM computes a smooth, realistic acceleration for a vehicle based on:
 * - Current speed and desired speed
 * - Gap to the car ahead
 * - Speed difference with the car ahead
 *
 * Reference: Treiber, Hennecke, Helbing (2000)
 * https://en.wikipedia.org/wiki/Intelligent_driver_model
 *
 * This is a pure math module with zero rendering dependencies.
 */

/**
 * Default IDM parameters for a car.
 * @typedef {Object} IDMParams
 * @property {number} v0 - Desired velocity (m/s). Default 12 (≈27 mph)
 * @property {number} a - Maximum acceleration (m/s²). Default 1.4
 * @property {number} b - Comfortable deceleration (m/s²). Default 2.0
 * @property {number} s0 - Minimum gap when stopped (m). Default 2.0
 * @property {number} T - Safe time headway (s). Default 1.5
 * @property {number} delta - Acceleration exponent. Default 4
 */
export const IDM_DEFAULTS = {
    v0: 12,     // desired speed m/s (≈27 mph city street)
    a: 1.4,     // max acceleration m/s²
    b: 2.0,     // comfortable deceleration m/s²
    s0: 2.0,    // minimum gap (m) — bumper to bumper when stopped
    T: 1.5,     // safe time headway (s)
    delta: 4,   // acceleration exponent
};

/**
 * Calculate IDM acceleration.
 *
 * @param {number} v - Current speed (m/s)
 * @param {number} s - Gap to leader (m, bumper-to-bumper). Use Infinity if no leader.
 * @param {number} vLeader - Leader's speed (m/s). Use v if no leader.
 * @param {IDMParams} [params] - IDM parameters (defaults used if omitted)
 * @returns {number} acceleration (m/s², positive = speed up, negative = brake)
 */
export function idmAcceleration(v, s, vLeader, params = IDM_DEFAULTS) {
    const { v0, a, b, s0, T, delta } = params;

    // Speed difference (positive = approaching leader)
    const deltaV = v - vLeader;

    // Desired dynamic gap: s* = s0 + v*T + v*Δv / (2*sqrt(a*b))
    const interaction = (v * deltaV) / (2 * Math.sqrt(a * b));
    const sStar = s0 + Math.max(0, v * T + interaction);

    // IDM acceleration: a * [1 - (v/v0)^δ - (s*/s)²]
    const vRatio = (v0 > 0) ? Math.pow(v / v0, delta) : 0;
    const sRatio = (s > 0.01) ? Math.pow(sStar / s, 2) : 100; // large braking if gap ≈ 0

    const acc = a * (1 - vRatio - sRatio);

    // Clamp to reasonable bounds
    return Math.max(-9.0, Math.min(a, acc)); // max 1g braking, max a acceleration
}

/**
 * Update a vehicle's position using ballistic integration.
 *
 * @param {number} u - Current position along road (m)
 * @param {number} v - Current speed (m/s)
 * @param {number} acc - Current acceleration (m/s²)
 * @param {number} dt - Time step (s)
 * @returns {{ u: number, v: number }} updated position and speed
 */
export function ballisticUpdate(u, v, acc, dt) {
    const newU = u + Math.max(0, v * dt + 0.5 * acc * dt * dt);
    const newV = Math.max(0, v + acc * dt);
    return { u: newU, v: newV };
}

/**
 * Convert road coordinates (road, lane, u) to world position (x, z).
 *
 * For a straight road segment from point A to point B:
 * - Position along road: lerp(A, B, u/length)
 * - Lane offset: perpendicular to road direction
 *
 * @param {Object} road - { ax, az, bx, bz, length, angle, laneWidth, numLanesPerDir }
 * @param {number} lane - Lane index (0 = rightmost in direction A→B)
 * @param {number} u - Position along road (0 to road.length)
 * @returns {{ x: number, z: number, angle: number }}
 */
export function roadToWorld(road, lane, u) {
    const t = Math.max(0, Math.min(1, u / road.length));

    // Position along road centerline
    const cx = road.ax + t * (road.bx - road.ax);
    const cz = road.az + t * (road.bz - road.az);

    // Road direction vector (normalized)
    const dx = road.bx - road.ax;
    const dz = road.bz - road.az;
    const len = Math.sqrt(dx * dx + dz * dz) || 1;
    const dirX = dx / len;
    const dirZ = dz / len;

    // Perpendicular (right-hand rule: rotate 90° clockwise in XZ plane)
    const perpX = -dirZ;
    const perpZ = dirX;

    // Lane offset from centerline
    // Lanes: 0,1 go in A→B direction (right side), 2,3 go in B→A direction (left side)
    // For A→B lanes: offset to the right (+perp)
    // For B→A lanes: offset to the left (-perp)
    const nPerDir = road.numLanesPerDir || 2;
    const laneWidth = road.laneWidth || 3;
    let offset;
    if (lane < nPerDir) {
        // A→B direction: right side of road
        offset = (lane + 0.5) * laneWidth;
    } else {
        // B→A direction: left side of road
        offset = -((lane - nPerDir) + 0.5) * laneWidth;
    }

    const x = cx + perpX * offset;
    const z = cz + perpZ * offset;

    // Angle: A→B direction for lanes 0..nPerDir-1, B→A for lanes nPerDir..
    const angle = (lane < nPerDir)
        ? Math.atan2(dirX, dirZ)
        : Math.atan2(-dirX, -dirZ);

    return { x, z, angle };
}

/**
 * Find the leader vehicle (car ahead in same lane, same road).
 *
 * @param {Object} car - { road, lane, u }
 * @param {Object[]} allCars - array of all cars
 * @returns {{ leader: Object|null, gap: number }}
 */
export function findLeader(car, allCars) {
    let bestGap = Infinity;
    let leader = null;

    const isForward = car.lane < (car.road.numLanesPerDir || 2);

    for (const other of allCars) {
        if (other === car) continue;
        if (other.road !== car.road) continue;
        if (other.lane !== car.lane) continue;

        // Gap: depends on direction
        let gap;
        if (isForward) {
            gap = other.u - car.u; // leader is ahead (higher u)
        } else {
            gap = car.u - other.u; // leader is ahead (lower u in reverse direction)
        }

        if (gap > 0 && gap < bestGap) {
            bestGap = gap;
            leader = other;
        }
    }

    // Subtract car lengths for bumper-to-bumper gap
    if (leader) {
        bestGap = Math.max(0.1, bestGap - (car.length || 4) - (leader.length || 4));
    }

    return { leader, gap: bestGap };
}
