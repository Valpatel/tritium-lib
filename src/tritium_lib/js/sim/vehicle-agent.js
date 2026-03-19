// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Vehicle Agent — manages a single vehicle in the traffic simulation.
 *
 * Each vehicle has:
 * - Road coordinates: (road, lane, u) — position derived from road geometry
 * - IDM acceleration: smooth car-following behavior
 * - Route: sequence of road edges from origin to destination
 * - Intersection turning: Bezier curves for smooth transitions
 * - Turn signals: blink based on upcoming turn direction
 * - Stuck detection: re-route if stuck for too long
 *
 * This module is pure logic — rendering is handled by the caller.
 */

import { idmAcceleration, ballisticUpdate, roadToWorld, IDM_DEFAULTS } from './idm.js';
import { bezierPosition, bezierTangent, tangentToAngle } from './road-network.js';
import { decideLaneChange, createLaneChangeState, updateLaneChange, MOBIL_DEFAULTS } from './mobil.js';

// ============================================================
// VEHICLE TYPES
// ============================================================

export const VEHICLE_TYPES = {
    sedan: {
        idm: { ...IDM_DEFAULTS, v0: 12, a: 1.4, b: 2.0, s0: 2.0, T: 1.5 },
        length: 4.5, width: 2.0, height: 1.4,
        color: null, // assigned randomly
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.3 },
    },
    suv: {
        idm: { ...IDM_DEFAULTS, v0: 11, a: 1.2, b: 2.0, s0: 2.5, T: 1.8 },
        length: 5.0, width: 2.2, height: 1.8,
        color: null,
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.4 },
    },
    truck: {
        idm: { ...IDM_DEFAULTS, v0: 10, a: 0.8, b: 1.5, s0: 3.0, T: 2.0 },
        length: 8.0, width: 2.5, height: 3.0,
        color: null,
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.5 },
    },
    motorcycle: {
        idm: { ...IDM_DEFAULTS, v0: 14, a: 2.5, b: 3.0, s0: 1.5, T: 1.0 },
        length: 2.2, width: 0.8, height: 1.2,
        color: null,
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.1, threshold: 0.1 },
    },
    police: {
        idm: { ...IDM_DEFAULTS, v0: 18, a: 2.5, b: 3.0, s0: 2.0, T: 1.2 },
        length: 4.5, width: 2.0, height: 1.5,
        color: 0x1155ff,
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.1 },
        emergency: true,
    },
    ambulance: {
        idm: { ...IDM_DEFAULTS, v0: 16, a: 2.0, b: 3.0, s0: 2.5, T: 1.3 },
        length: 6.0, width: 2.2, height: 2.5,
        color: 0xffffff,
        mobil: { ...MOBIL_DEFAULTS, politeness: 0.1 },
        emergency: true,
    },
};

// ============================================================
// TURN SIGNAL STATES
// ============================================================

const SIGNAL_OFF = 'off';
const SIGNAL_LEFT = 'left';
const SIGNAL_RIGHT = 'right';
const SIGNAL_HAZARD = 'hazard';

// ============================================================
// VEHICLE AGENT CLASS
// ============================================================

export class VehicleAgent {
    /**
     * @param {string} id - Unique vehicle identifier
     * @param {string} vehicleType - Key into VEHICLE_TYPES
     */
    constructor(id, vehicleType = 'sedan') {
        this.id = id;
        this.vehicleType = vehicleType;
        const typeInfo = VEHICLE_TYPES[vehicleType] || VEHICLE_TYPES.sedan;

        // Road state
        this.road = null;       // Current road edge
        this.lane = 0;          // Current lane index (0-3)
        this.u = 0;             // Position along road in direction of travel (m)
        this.speed = 0;         // Current speed (m/s)
        this.acc = 0;           // Current acceleration (m/s²)

        // IDM parameters (with per-vehicle variation)
        this.idmParams = { ...typeInfo.idm };
        this.idmParams.v0 *= (0.9 + Math.random() * 0.2); // ±10% speed variation

        // MOBIL parameters
        this.mobilParams = { ...typeInfo.mobil };

        // Physical dimensions
        this.length = typeInfo.length;
        this.width = typeInfo.width;
        this.height = typeInfo.height;
        this.isEmergency = typeInfo.emergency || false;

        // Route
        this.route = [];        // Sequence of { edge, nodeId }
        this.routeIndex = 0;    // Current position in route
        this.destination = null; // Target node ID

        // Intersection turning state
        this.turningState = null; // { p0, p1, p2, t, duration, toEdge, toLane, turnSpeed, turnType }

        // Lane change state
        this.laneChangeState = null; // from mobil.js createLaneChangeState

        // Turn signals
        this.turnSignal = SIGNAL_OFF;
        this.turnSignalTimer = 0;

        // Brake lights
        this.brakeLightsOn = false;

        // Stuck detection
        this.stuckTimer = 0;

        // Leader tracking (for debug)
        this.leader = null;
        this.gap = Infinity;

        // World position (derived, updated each tick)
        this.worldX = 0;
        this.worldZ = 0;
        this.worldAngle = 0;
    }

    /**
     * Assign a route from the road network.
     *
     * @param {Object} road - Starting road edge
     * @param {number} lane - Starting lane
     * @param {number} u - Starting position along road
     * @param {Object} roadNetwork - RoadNetwork instance
     * @param {string} [destNodeId] - Optional destination node ID
     */
    assignRoute(road, lane, u, roadNetwork, destNodeId = null) {
        this.road = road;
        this.lane = lane;
        this.u = u;
        this.speed = 2 + Math.random() * 4; // Start with some speed

        if (destNodeId) {
            const isForward = lane < (road.numLanesPerDir || 2);
            const currentNodeId = isForward ? road.to : road.from;
            this.route = roadNetwork.findPath(currentNodeId, destNodeId);
            this.destination = destNodeId;
        }
        this.routeIndex = 0;
    }

    /**
     * Assign a random starting position and cruise route.
     *
     * @param {Object} roadNetwork - RoadNetwork instance
     */
    assignRandomStart(roadNetwork) {
        const edges = roadNetwork.edges;
        if (edges.length === 0) return;

        const edge = edges[Math.floor(Math.random() * edges.length)];
        const n = edge.numLanesPerDir || 2;
        const lane = Math.floor(Math.random() * n * 2);
        const u = 5 + Math.random() * (edge.length - 10);

        this.assignRoute(edge, lane, u, roadNetwork);
    }

    /**
     * Main tick: advance the vehicle by dt seconds.
     *
     * @param {number} dt - Time step (seconds)
     * @param {Array} allVehicles - All VehicleAgent instances
     * @param {Array} virtualObstacles - Virtual obstacles from traffic controllers
     * @param {Object} roadNetwork - RoadNetwork instance
     * @param {Object} [trafficManager] - TrafficControllerManager (for preemption)
     */
    tick(dt, allVehicles, virtualObstacles, roadNetwork, trafficManager = null) {
        if (!this.road) return;

        if (this.turningState) {
            this._tickTurning(dt);
        } else {
            this._tickDriving(dt, allVehicles, virtualObstacles, roadNetwork, trafficManager);
        }

        this._updateSignals(dt, roadNetwork);
        this._checkStuck(dt, roadNetwork);
    }

    /**
     * Tick while driving on a road segment.
     */
    _tickDriving(dt, allVehicles, virtualObstacles, roadNetwork, trafficManager) {
        // Combine real vehicles and virtual obstacles for leader finding
        const allObstacles = this._buildObstacleList(allVehicles, virtualObstacles);

        // Find leader (car ahead in same lane, same road)
        const { leader, gap } = this._findLeader(allObstacles);
        this.leader = leader;
        this.gap = gap;

        // Calculate IDM acceleration
        const leaderSpeed = leader ? leader.speed : this.speed;
        this.acc = idmAcceleration(this.speed, gap, leaderSpeed, this.idmParams);

        // Brake lights
        this.brakeLightsOn = this.acc < -0.5;

        // MOBIL lane change (only if not already changing and not too close to intersection)
        if (!this.laneChangeState && this.u < this.road.length - 30) {
            this._considerLaneChange(allObstacles);
        }

        // Update lane change animation
        if (this.laneChangeState) {
            const result = updateLaneChange(this.laneChangeState, dt);
            if (result.complete) {
                this.lane = this.laneChangeState.toLane;
                this.laneChangeState = null;
            }
        }

        // Ballistic position update
        const updated = ballisticUpdate(this.u, this.speed, this.acc, dt);
        this.u = updated.u;
        this.speed = updated.v;

        // Check if reached end of road → start intersection turn
        if (this.u >= this.road.length) {
            this.u = this.road.length; // clamp
            this._startIntersectionTurn(roadNetwork, trafficManager);
        }

        // Derive world position
        this._updateWorldPosition();
    }

    /**
     * Tick while turning through an intersection (following Bezier curve).
     */
    _tickTurning(dt) {
        const ts = this.turningState;
        ts.t += dt / ts.duration;

        // Reduce speed through turn
        this.speed = Math.max(2, ts.turnSpeed);
        this.acc = 0;
        this.brakeLightsOn = false;

        if (ts.t >= 1) {
            // Turn complete — enter new road
            this.road = ts.toEdge;
            this.lane = ts.toLane;
            this.u = 0;
            // Keep current speed, IDM will adjust
            this.turningState = null;
            this._updateWorldPosition();
            return;
        }

        // Position from Bezier
        const pos = bezierPosition(ts.p0, ts.p1, ts.p2, ts.t);
        const tang = bezierTangent(ts.p0, ts.p1, ts.p2, ts.t);

        this.worldX = pos.x;
        this.worldZ = pos.z;
        this.worldAngle = tangentToAngle(tang);
    }

    /**
     * Build list of obstacles for leader detection.
     * Emergency vehicles ignore virtual obstacles (run red lights).
     */
    _buildObstacleList(allVehicles, virtualObstacles) {
        const list = [];

        // Add other real vehicles
        for (const v of allVehicles) {
            if (v === this) continue;
            if (v.turningState) continue; // skip vehicles in intersection
            list.push({
                road: v.road,
                lane: v.lane,
                u: v.u,
                speed: v.speed,
                length: v.length,
                id: v.id,
            });
        }

        // Add virtual obstacles (unless we're an emergency vehicle)
        if (!this.isEmergency) {
            for (const obs of virtualObstacles) {
                list.push(obs);
            }
        }

        return list;
    }

    /**
     * Find the leader vehicle (car ahead in same lane, same road, higher u).
     */
    _findLeader(allObstacles) {
        let bestGap = Infinity;
        let leader = null;

        for (const other of allObstacles) {
            if (other.road !== this.road) continue;

            // Handle lane change: check both lanes during transition
            if (this.laneChangeState) {
                if (other.lane !== this.laneChangeState.fromLane &&
                    other.lane !== this.laneChangeState.toLane) continue;
            } else {
                if (other.lane !== this.lane) continue;
            }

            const gap = other.u - this.u;
            if (gap > 0 && gap < bestGap) {
                bestGap = gap;
                leader = other;
            }
        }

        // Subtract car lengths for bumper-to-bumper gap
        if (leader) {
            bestGap = Math.max(0.1, bestGap - this.length - (leader.length || 4));
        }

        return { leader, gap: bestGap };
    }

    /**
     * Consider a MOBIL lane change.
     */
    _considerLaneChange(allObstacles) {
        const result = decideLaneChange(
            { ...this, road: this.road, lane: this.lane, u: this.u, speed: this.speed, idmParams: this.idmParams, length: this.length },
            allObstacles,
            this.mobilParams
        );

        if (result.direction && result.targetLane !== null) {
            this.laneChangeState = createLaneChangeState(this.lane, result.targetLane, 2.0);
            this.turnSignal = result.direction === 'left' ? SIGNAL_LEFT : SIGNAL_RIGHT;
        }
    }

    /**
     * Start an intersection turn with Bezier path.
     */
    _startIntersectionTurn(roadNetwork, trafficManager) {
        const n = this.road.numLanesPerDir || 2;
        const isForward = this.lane < n;
        const arrivalNodeId = isForward ? this.road.to : this.road.from;

        // Get lane connections
        let connection;

        if (this.route.length > 0 && this.routeIndex < this.route.length) {
            // Follow route
            const nextStep = this.route[this.routeIndex];
            const connections = roadNetwork.getLaneConnections(this.road.id, this.lane, arrivalNodeId);
            connection = connections.find(c => c.toEdge === nextStep.edge);
            if (connection) this.routeIndex++;
        }

        if (!connection) {
            // Random turn
            connection = roadNetwork.pickRandomConnection(this.road.id, this.lane, arrivalNodeId);
        }

        if (!connection) {
            // Dead end or no connections — U-turn by re-routing
            this.u = this.road.length - 5;
            this.speed = 0;
            return;
        }

        // Emergency vehicle preemption
        if (this.isEmergency && trafficManager) {
            const ctrl = trafficManager.getController(arrivalNodeId);
            if (ctrl) {
                const arrivalDir = this.road.horizontal
                    ? (isForward ? 'W' : 'E')
                    : (isForward ? 'N' : 'S');
                ctrl.forceGreen(arrivalDir, 10);
            }
        }

        // Calculate Bezier path points
        const entryU = isForward ? this.road.length : 0;
        const entryWorld = this._roadToWorldPos(this.road, this.lane, entryU);

        const toIsForward = connection.toLane < (connection.toEdge.numLanesPerDir || 2);
        const exitU = toIsForward ? 0 : connection.toEdge.length;
        const exitWorld = this._roadToWorldPos(connection.toEdge, connection.toLane, exitU);

        const controlPoint = connection.bezierControl || { x: roadNetwork.nodes[arrivalNodeId].x, z: roadNetwork.nodes[arrivalNodeId].z };

        // Turn speed and duration
        const turnSpeed = Math.max(2, this.speed * 0.5);
        const arcLength = this._estimateArcLength(entryWorld, controlPoint, exitWorld);
        const duration = Math.max(0.5, Math.min(2.5, arcLength / Math.max(1, turnSpeed)));

        this.turningState = {
            p0: entryWorld,
            p1: controlPoint,
            p2: exitWorld,
            t: 0,
            duration,
            toEdge: connection.toEdge,
            toLane: connection.toLane,
            turnSpeed,
            turnType: connection.turnType,
        };

        // Set turn signal based on turn type
        if (connection.turnType === 'left') {
            this.turnSignal = SIGNAL_LEFT;
        } else if (connection.turnType === 'right') {
            this.turnSignal = SIGNAL_RIGHT;
        }
    }

    /**
     * Update turn signal state.
     */
    _updateSignals(dt, roadNetwork) {
        this.turnSignalTimer += dt;

        // Clear turn signal after completing turn
        if (!this.turningState && !this.laneChangeState) {
            if (this.turnSignal === SIGNAL_LEFT || this.turnSignal === SIGNAL_RIGHT) {
                this.turnSignal = SIGNAL_OFF;
            }
        }

        // Approaching intersection — set signal for upcoming turn
        if (!this.turningState && !this.laneChangeState && this.road) {
            const distToEnd = this.road.length - this.u;
            if (distToEnd < 30 && distToEnd > 5 && this.turnSignal === SIGNAL_OFF) {
                const n = this.road.numLanesPerDir || 2;
                const isForward = this.lane < n;
                const nodeId = isForward ? this.road.to : this.road.from;
                const connection = roadNetwork.pickRandomConnection(this.road.id, this.lane, nodeId);
                if (connection) {
                    if (connection.turnType === 'left') this.turnSignal = SIGNAL_LEFT;
                    else if (connection.turnType === 'right') this.turnSignal = SIGNAL_RIGHT;
                }
            }
        }

        // Hazard lights when stopped for > 10 seconds
        if (this.speed < 0.1 && !this.turningState) {
            this.stuckTimer += 0; // stuckTimer managed in _checkStuck
            if (this.stuckTimer > 10 && this.turnSignal !== SIGNAL_HAZARD) {
                this.turnSignal = SIGNAL_HAZARD;
            }
        }
    }

    /**
     * Detect and recover from being stuck.
     */
    _checkStuck(dt, roadNetwork) {
        if (this.turningState) {
            this.stuckTimer = 0;
            return;
        }

        if (this.speed < 0.1) {
            this.stuckTimer += dt;
            if (this.stuckTimer > 30) {
                // Re-route: pick new random position
                this.assignRandomStart(roadNetwork);
                this.stuckTimer = 0;
                this.turnSignal = SIGNAL_OFF;
            }
        } else {
            this.stuckTimer = Math.max(0, this.stuckTimer - dt); // decay
        }
    }

    /**
     * Derive world position from road coordinates.
     */
    _updateWorldPosition() {
        if (this.turningState) return; // handled in _tickTurning

        const isForward = this.lane < (this.road.numLanesPerDir || 2);

        // Handle lane change interpolation
        let effectiveLane = this.lane;
        if (this.laneChangeState && this.laneChangeState.active) {
            const smoothT = 0.5 - 0.5 * Math.cos(this.laneChangeState.t * Math.PI);
            effectiveLane = this.laneChangeState.fromLane +
                (this.laneChangeState.toLane - this.laneChangeState.fromLane) * smoothT;
        }

        // Convert u to roadToWorld u (backward lanes need flip)
        const roadU = isForward ? this.u : (this.road.length - this.u);
        const world = roadToWorld(this.road, effectiveLane, roadU);

        this.worldX = world.x;
        this.worldZ = world.z;
        this.worldAngle = world.angle;
    }

    /**
     * Helper: compute world position for a given road/lane/u.
     */
    _roadToWorldPos(road, lane, u) {
        const t = Math.max(0, Math.min(1, u / road.length));
        const cx = road.ax + t * (road.bx - road.ax);
        const cz = road.az + t * (road.bz - road.az);

        const dx = road.bx - road.ax;
        const dz = road.bz - road.az;
        const len = Math.sqrt(dx * dx + dz * dz) || 1;
        const perpX = -dz / len;
        const perpZ = dx / len;

        const nPerDir = road.numLanesPerDir || 2;
        const laneWidth = road.laneWidth || 3;
        let offset;
        if (lane < nPerDir) {
            offset = (lane + 0.5) * laneWidth;
        } else {
            offset = -((lane - nPerDir) + 0.5) * laneWidth;
        }

        return { x: cx + perpX * offset, z: cz + perpZ * offset };
    }

    /**
     * Estimate arc length of quadratic Bezier.
     */
    _estimateArcLength(p0, p1, p2, samples = 8) {
        let length = 0;
        let prev = p0;
        for (let i = 1; i <= samples; i++) {
            const t = i / samples;
            const u = 1 - t;
            const curr = {
                x: u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
                z: u * u * p0.z + 2 * u * t * p1.z + t * t * p2.z,
            };
            const dx = curr.x - prev.x;
            const dz = curr.z - prev.z;
            length += Math.sqrt(dx * dx + dz * dz);
            prev = curr;
        }
        return length;
    }

    /**
     * Get debug info for the HUD.
     */
    getDebugInfo() {
        return {
            id: this.id,
            type: this.vehicleType,
            road: this.road?.id || 'none',
            lane: this.lane,
            u: this.u.toFixed(1),
            speed: this.speed.toFixed(1),
            acc: this.acc.toFixed(2),
            gap: this.gap === Infinity ? '∞' : this.gap.toFixed(1),
            leader: this.leader?.id || 'none',
            turnSignal: this.turnSignal,
            turning: this.turningState ? `${this.turningState.turnType} ${(this.turningState.t * 100).toFixed(0)}%` : 'no',
            laneChange: this.laneChangeState ? `${(this.laneChangeState.t * 100).toFixed(0)}%` : 'no',
            stuck: this.stuckTimer.toFixed(0),
            brake: this.brakeLightsOn ? 'ON' : 'off',
        };
    }

    /**
     * Is the turn signal currently visible? (for blinking animation)
     */
    isTurnSignalVisible() {
        if (this.turnSignal === SIGNAL_OFF) return { left: false, right: false };
        const blinkOn = Math.sin(this.turnSignalTimer * Math.PI * 2) > 0; // 1Hz blink
        if (this.turnSignal === SIGNAL_HAZARD) return { left: blinkOn, right: blinkOn };
        if (this.turnSignal === SIGNAL_LEFT) return { left: blinkOn, right: false };
        if (this.turnSignal === SIGNAL_RIGHT) return { left: false, right: blinkOn };
        return { left: false, right: false };
    }
}

// ============================================================
// VEHICLE MANAGER
// ============================================================

/**
 * Manages all vehicles in the simulation.
 */
export class VehicleManager {
    constructor() {
        this.vehicles = [];
        this.nextId = 0;
    }

    /**
     * Spawn a new vehicle.
     *
     * @param {Object} roadNetwork - RoadNetwork instance
     * @param {string} [type] - Vehicle type (random if omitted)
     * @returns {VehicleAgent}
     */
    spawn(roadNetwork, type = null) {
        if (!type) {
            const types = ['sedan', 'sedan', 'sedan', 'suv', 'suv', 'truck', 'motorcycle'];
            type = types[Math.floor(Math.random() * types.length)];
        }

        const id = `car_${this.nextId++}`;
        const vehicle = new VehicleAgent(id, type);
        vehicle.assignRandomStart(roadNetwork);
        this.vehicles.push(vehicle);
        return vehicle;
    }

    /**
     * Tick all vehicles.
     */
    tick(dt, virtualObstacles, roadNetwork, trafficManager) {
        for (const v of this.vehicles) {
            v.tick(dt, this.vehicles, virtualObstacles, roadNetwork, trafficManager);
        }
    }

    /**
     * Get all vehicles (for rendering).
     */
    getAll() {
        return this.vehicles;
    }

    /**
     * Remove a vehicle by ID.
     */
    remove(id) {
        this.vehicles = this.vehicles.filter(v => v.id !== id);
    }

    /**
     * Get debug info for all vehicles.
     */
    getDebugInfo() {
        return this.vehicles.map(v => v.getDebugInfo());
    }
}
