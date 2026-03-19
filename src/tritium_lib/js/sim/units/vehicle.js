// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Vehicle — base class for all road vehicles (cars, motorcycles, trucks, buses, emergency).
 *
 * Uses IDM for longitudinal control (speed/acceleration based on gap to leader).
 * Follows a continuous Path along road lanes.
 *
 * Subclasses can override:
 * - getIDMParams() for different acceleration/deceleration profiles
 * - findLeader() for different leader detection logic
 * - onPathEnd() for different route planning
 */

import { GroundUnit } from '../core/ground-unit.js';
import { idmAcceleration, IDM_DEFAULTS } from '../idm.js';
import { extendPath, buildPath } from '../core/path.js';

export class Vehicle extends GroundUnit {
    /**
     * @param {Object} config
     * @param {Object} [config.idm] - IDM parameters override
     * @param {number} [config.laneOffset] - Lateral offset from road center (meters)
     * @param {boolean} [config.isEmergency] - Emergency vehicle (ignores red lights)
     */
    constructor(config = {}) {
        super({ type: 'vehicle', length: 5, width: 2.5, height: 1.6, ...config });

        // IDM parameters (per-vehicle variation)
        this.idmParams = { ...IDM_DEFAULTS, ...(config.idm || {}) };
        if (!config.idm?.v0) {
            this.idmParams.v0 = 8 + Math.random() * 5; // 8-13 m/s
        }

        // Lane offset from road center
        this.laneOffset = config.laneOffset || ((Math.random() < 0.5 ? 0.5 : 1.5) * 3);

        // Emergency vehicle flags
        this.isEmergency = config.isEmergency || false;

        // Visual state
        this.brakeLightsOn = false;

        // Route (sequence of intersection node IDs for Dijkstra path planning)
        this.route = [];
        this.routeIndex = 0;
        this.destination = null;
    }

    computeAcceleration(dt, world) {
        // Find nearest obstacle ahead (other vehicles, pedestrians, red lights)
        const leader = this.findLeader(world);

        // IDM acceleration
        let acc = idmAcceleration(this.speed, leader.gap, leader.speed, this.idmParams);

        // In curves: don't brake unless leader is very close (prevents false positives)
        if (this.inCurve && acc < 0 && leader.gap > 4) {
            acc = Math.max(acc, 0);
        }

        // Brake lights
        this.brakeLightsOn = acc < -0.5;

        return acc;
    }

    /**
     * Find the nearest obstacle ahead using forward-cone detection.
     * Checks other vehicles, pedestrians, and red traffic lights.
     *
     * @param {Object} world - { spatialHash, units, trafficCtrl }
     * @returns {{ gap: number, speed: number }}
     */
    findLeader(world) {
        let bestGap = Infinity;
        let bestSpeed = this.speed;

        const fwdX = Math.sin(this.heading);
        const fwdZ = Math.cos(this.heading);

        // Check nearby units from spatial hash
        const nearby = world.spatialHash ? world.spatialHash.getNearby(this.x, this.z) : [];

        for (const other of nearby) {
            if (other === this || !other.alive) continue;

            const dx = other.x - this.x;
            const dz = other.z - this.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist > 50) continue;

            // Must be ahead (forward cone)
            const dot = dx * fwdX + dz * fwdZ;
            if (dot <= 0) continue;

            // Lateral check: within 2m for vehicles (same lane), wider for peds
            const lateral = Math.abs(-dx * fwdZ + dz * fwdX);
            const lateralThreshold = other.type === 'pedestrian' ? 1.5 : 2.0;
            if (lateral > lateralThreshold) continue;

            const gap = Math.max(0.1, dot - this.length / 2 - (other.length || 1) / 2);
            if (gap < bestGap) {
                bestGap = gap;
                bestSpeed = other.speed || 0;
            }
        }

        // Check traffic lights (if not emergency vehicle)
        if (!this.isEmergency && world.trafficCtrl) {
            const signalGap = this._findRedLightGap(world.trafficCtrl, fwdX, fwdZ);
            if (signalGap > 0 && signalGap < bestGap) {
                bestGap = signalGap;
                bestSpeed = 0;
            }
        }

        // During curves, enforce minimum speed if no close leader
        if (this.inCurve && bestGap > 4) {
            bestGap = Infinity; // don't brake
        }

        return { gap: bestGap, speed: bestSpeed };
    }

    _findRedLightGap(trafficCtrl, fwdX, fwdZ) {
        const waypoints = this.path.waypoints;
        if (!waypoints || waypoints.length < 2) return Infinity;

        // Find nearest intersection ahead
        let nearestDot = Infinity;
        let nearestNode = null;

        for (const node of waypoints) {
            if (!node || !node.id) continue;
            const dx = node.x - this.x;
            const dz = node.z - this.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < 5 || dist > 60) continue;
            const dot = dx * fwdX + dz * fwdZ;
            if (dot <= 0) continue;
            if (dot < nearestDot) {
                nearestDot = dot;
                nearestNode = node;
            }
        }

        if (!nearestNode) return Infinity;

        // Determine approach direction
        const dx = nearestNode.x - this.x;
        const dz = nearestNode.z - this.z;
        const approachDir = Math.abs(dx) > Math.abs(dz)
            ? (dx > 0 ? 'W' : 'E')
            : (dz > 0 ? 'N' : 'S');

        // Check signal
        let isGreen = true;
        if (trafficCtrl.isGreenForApproach) {
            isGreen = trafficCtrl.isGreenForApproach(nearestNode.id, approachDir);
        } else if (trafficCtrl.getController) {
            const ctrl = trafficCtrl.getController(nearestNode.id);
            if (ctrl) isGreen = ctrl.isGreen(approachDir);
        }

        return isGreen ? Infinity : Math.max(0.5, nearestDot - 5);
    }

    onPathEnd(world) {
        if (!world.roadNetwork) { this.speed = 0; return; }

        const lastNode = this.path.getLastWaypoint();
        const prevNode = this.path.getPrevWaypoint();
        if (!lastNode) { this.speed = 0; return; }

        // Pick next node from route or random
        let nextNode;
        if (this.route && this.routeIndex < this.route.length) {
            nextNode = world.roadNetwork.nodes[this.route[this.routeIndex].nodeId];
            this.routeIndex++;
        }

        if (!nextNode) {
            // Random: pick a neighbor that isn't a U-turn
            const edges = world.roadNetwork.getEdgesForNode(lastNode.id);
            const candidates = [];
            for (const edge of edges) {
                const otherId = world.roadNetwork.getOtherNode(edge, lastNode.id);
                if (prevNode && otherId === prevNode.id) continue;
                const node = world.roadNetwork.nodes[otherId];
                if (node) candidates.push(node);
            }
            if (candidates.length > 0) {
                nextNode = candidates[Math.floor(Math.random() * candidates.length)];
            } else if (prevNode) {
                nextNode = world.roadNetwork.nodes[prevNode.id]; // U-turn
            }
        }

        if (nextNode) {
            extendPath(this.path, nextNode, this.laneOffset, 8);
        }
    }

    getPathExtendThreshold() {
        return 200; // vehicles need more lookahead
    }
}

// ============================================================
// VEHICLE SUBTYPES
// ============================================================

export class Car extends Vehicle {
    constructor(config = {}) {
        super({
            type: 'car',
            length: 5.0, width: 2.5, height: 1.6,
            idm: { v0: 8 + Math.random() * 5, a: 1.4, b: 2.0, s0: 2, T: 1.5 },
            ...config,
        });
        this.subtype = config.subtype || 'sedan'; // sedan, suv, truck
    }
}

export class Motorcycle extends Vehicle {
    constructor(config = {}) {
        super({
            type: 'motorcycle',
            length: 2.2, width: 0.8, height: 1.2,
            idm: { v0: 12 + Math.random() * 4, a: 2.5, b: 3.0, s0: 1.5, T: 1.0 },
            ...config,
        });
    }
}

export class EmergencyVehicle extends Vehicle {
    constructor(config = {}) {
        super({
            type: 'emergency',
            length: 5.0, width: 2.5, height: 1.8,
            idm: { v0: 18, a: 2.5, b: 3.0, s0: 2, T: 1.2 },
            isEmergency: true,
            ...config,
        });
        this.sirenActive = true;
        this.emergencyType = config.emergencyType || 'police'; // police, ambulance, fire
    }
}

export class Tank extends Vehicle {
    constructor(config = {}) {
        super({
            type: 'tank',
            length: 7.0, width: 3.5, height: 2.5,
            idm: { v0: 6, a: 0.5, b: 1.5, s0: 4, T: 2.5 },
            ...config,
        });
        this.turretAngle = 0;
    }
}
