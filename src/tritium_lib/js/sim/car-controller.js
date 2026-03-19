// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Car Controller — unified vehicle tick function using continuous CarPath.
 *
 * Replaces ALL previous car logic (updateCar, transitionToNextRoad,
 * findLeaderCustom, buildVirtualStopCars, etc.) with one clean loop:
 *
 *   1. Extend path if running low
 *   2. Find effective leader (cars, pedestrians, red lights)
 *   3. IDM acceleration
 *   4. Advance d along path
 *   5. Derive (x, z, heading) from path
 *
 * No turning state. No pause timer. No state machine.
 * The car just follows the curve.
 *
 * Based on: docs/plans/vehicle-framework-rewrite.md
 */

import { idmAcceleration, IDM_DEFAULTS } from './idm.js';
import { CarPath, buildPathFromRoute, extendPath } from './car-path.js';

// ============================================================
// SPATIAL HASH GRID — O(1) neighbor lookup instead of O(n)
// ============================================================

class SpatialHash {
    constructor(cellSize = 50) {
        this.cellSize = cellSize;
        this.grid = new Map();
    }

    _key(x, z) {
        const cx = Math.floor(x / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        return (cx & 0xFFFF) | ((cz & 0xFFFF) << 16);
    }

    clear() {
        this.grid.clear();
    }

    insert(car) {
        const key = this._key(car.worldX, car.worldZ);
        let cell = this.grid.get(key);
        if (!cell) { cell = []; this.grid.set(key, cell); }
        cell.push(car);
    }

    /** Get all cars in the same cell and 8 neighbors */
    getNearby(x, z) {
        const cx = Math.floor(x / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        const result = [];
        for (let dx = -1; dx <= 1; dx++) {
            for (let dz = -1; dz <= 1; dz++) {
                const key = ((cx + dx) & 0xFFFF) | (((cz + dz) & 0xFFFF) << 16);
                const cell = this.grid.get(key);
                if (cell) {
                    for (let i = 0; i < cell.length; i++) result.push(cell[i]);
                }
            }
        }
        return result;
    }
}

const spatialHash = new SpatialHash(50);

// ============================================================
// TICK — one frame of vehicle simulation
// ============================================================

/**
 * Update a single car for one frame.
 *
 * @param {Object} car - Vehicle state: { d, speed, acc, path, idmParams, ... }
 * @param {number} dt - Time step (seconds)
 * @param {Array} allCars - All vehicles (for leader detection)
 * @param {Array} pedestrians - NPCs with { x, z, speed }
 * @param {Object} trafficCtrl - Traffic controller manager
 * @param {Object} roadNetwork - Road network (for path extension)
 */
/**
 * Rebuild spatial hash grid for O(1) neighbor lookups.
 * Call ONCE per frame before ticking cars.
 */
export function rebuildSpatialHash(allCars) {
    spatialHash.clear();
    for (const car of allCars) {
        if (car.worldX !== undefined) spatialHash.insert(car);
    }
}

export function tickCar(car, dt, allCars, pedestrians, trafficCtrl, roadNetwork) {
    if (!car.path || car.path.totalLength < 1) return;

    // 1. Extend path if running low (< 200m ahead)
    if (car.path.remainingLength(car.d) < 200) {
        extendCarPath(car, roadNetwork);
    }

    // 2. Find effective leader (nearest obstacle ahead)
    const leader = findEffectiveLeader(car, allCars, pedestrians, trafficCtrl);

    // 3. IDM acceleration
    car.acc = idmAcceleration(car.speed, leader.gap, leader.speed, car.idmParams);

    // During turns: don't brake for distant false positives, maintain minimum turn speed
    if (car.inTurn) {
        if (car.acc < 0 && leader.gap > 4) {
            car.acc = Math.max(car.acc, 0);
        }
    }

    // 4. Update speed and advance along path
    let newSpeed = Math.max(0, car.speed + car.acc * dt);
    // Minimum speed during turns to prevent stopping inside intersections
    if (car.inTurn && leader.gap > 4) {
        newSpeed = Math.max(newSpeed, 3.0);
    }
    car.d = Math.max(0, car.d + newSpeed * dt + 0.5 * car.acc * dt * dt);
    car.speed = newSpeed;

    // 5. Derive world position and heading from path
    const pos = car.path.getPosition(car.d);
    const heading = car.path.getHeading(car.d);
    car.worldX = pos.x;
    car.worldZ = pos.z;
    car.worldHeading = heading;

    // 6. Trim old path segments (keep 20m behind for smooth heading)
    car.path.trimBefore(car.d - 20);

    // 7. Brake lights
    car.brakeLightsOn = car.acc < -0.5;

    // 8. Track if in a turn (for turn signals)
    car.inTurn = car.path.isInTurn(car.d);
}

// ============================================================
// LEADER DETECTION — find nearest obstacle ahead on path
// ============================================================

/**
 * Find the effective leader (nearest thing the car must not hit).
 * Checks: other cars, pedestrians crossing path, red traffic lights.
 *
 * @returns {{ gap: number, speed: number }}
 */
export function findEffectiveLeader(car, allCars, pedestrians, trafficCtrl) {
    let bestGap = Infinity;
    let bestSpeed = car.speed; // no leader = cruise at own speed

    const carPos = car.path.getPosition(car.d);
    const carHeading = car.path.getHeading(car.d);
    const fwdX = Math.sin(carHeading);
    const fwdZ = Math.cos(carHeading);

    // 1. Other cars: spatial hash for O(1) neighbor lookup
    const nearby = spatialHash.getNearby(carPos.x, carPos.z);
    for (const other of nearby) {
        if (other === car) continue;

        const dx = other.worldX - carPos.x;
        const dz = other.worldZ - carPos.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (dist > 50) continue;

        // Only check cars ahead (in forward cone)
        const dot = (dx * fwdX + dz * fwdZ);
        if (dot <= 0) continue; // behind us

        // Lateral check: within 2m of our path (same lane only, not adjacent lane)
        const lateral = Math.abs(-dx * fwdZ + dz * fwdX);
        if (lateral > 2) continue;

        // Gap = forward distance minus car lengths
        const gap = Math.max(0.1, dot - (car.length || 4) / 2 - (other.length || 4) / 2);
        if (gap < bestGap) {
            bestGap = gap;
            bestSpeed = other.speed;
        }
    }

    // 2. Pedestrians in path
    for (const ped of pedestrians) {
        const dx = ped.x - carPos.x;
        const dz = ped.z - carPos.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        if (dist > 20) continue;

        const dot = dx * fwdX + dz * fwdZ;
        if (dot <= 0) continue;

        const lateral = Math.abs(-dx * fwdZ + dz * fwdX);
        if (lateral > 1.5) continue; // only brake for peds actually ON the road, not on sidewalk

        const gap = Math.max(0.5, dot - (car.length || 4) / 2 - 0.5);
        if (gap < bestGap) {
            bestGap = gap;
            bestSpeed = ped.speed || 1.0;
        }
    }

    // 3. Red traffic lights
    if (trafficCtrl) {
        const signalGap = findNextRedSignalGap(car, trafficCtrl);
        if (signalGap > 0 && signalGap < bestGap) {
            bestGap = signalGap;
            bestSpeed = 0;
        }
    }

    return { gap: bestGap, speed: bestSpeed };
}

/**
 * Find distance to the next red traffic signal along the car's path.
 *
 * @returns {number} Gap in meters (Infinity if no red signal ahead)
 */
function findNextRedSignalGap(car, trafficCtrl) {
    const path = car.path;
    if (!path.intersections || path.intersections.length < 2) return Infinity;
    if (!trafficCtrl) return Infinity;

    const heading = car.path.getHeading(car.d);
    const fwdX = Math.sin(heading);
    const fwdZ = Math.cos(heading);

    // Only check the NEAREST intersection ahead (not all of them)
    let nearestDot = Infinity;
    let nearestNode = null;

    for (let i = 0; i < path.intersections.length; i++) {
        const node = path.intersections[i];
        if (!node || !node.id) continue;

        const dx = node.x - car.worldX;
        const dz = node.z - car.worldZ;
        const dist = Math.sqrt(dx * dx + dz * dz);

        // Skip too close (already in intersection) or too far
        if (dist < 5 || dist > 60) continue;

        // Must be ahead
        const dot = dx * fwdX + dz * fwdZ;
        if (dot <= 0) continue;

        if (dot < nearestDot) {
            nearestDot = dot;
            nearestNode = node;
        }
    }

    if (!nearestNode) return Infinity;

    // Check signal state for the nearest intersection only
    const approachDir = getApproachDirection(car.worldX, car.worldZ, nearestNode.x, nearestNode.z);
    let isGreen = true;

    if (trafficCtrl.getController) {
        const ctrl = trafficCtrl.getController(nearestNode.id);
        if (ctrl) isGreen = ctrl.isGreen(approachDir);
    } else if (trafficCtrl.isGreenForApproach) {
        isGreen = trafficCtrl.isGreenForApproach(nearestNode.id, approachDir);
    }

    if (!isGreen) {
        return Math.max(0.5, nearestDot - 5);
    }

    return Infinity;
}

/**
 * Determine approach direction (N/S/E/W) based on car position relative to intersection.
 */
function getApproachDirection(carX, carZ, nodeX, nodeZ) {
    const dx = nodeX - carX;
    const dz = nodeZ - carZ;
    if (Math.abs(dx) > Math.abs(dz)) {
        return dx > 0 ? 'W' : 'E'; // approaching from west or east
    } else {
        return dz > 0 ? 'N' : 'S'; // approaching from north or south
    }
}

// ============================================================
// PATH EXTENSION — dynamically grow the path ahead
// ============================================================

/**
 * Extend a car's path by appending the next road segment.
 */
export function extendCarPath(car, roadNetwork) {
    if (!roadNetwork) return;

    const lastNode = car.path.getLastIntersection();
    if (!lastNode) return;

    let nextNode;

    // If car has a planned route, follow it
    if (car.route && car.routeIndex < car.route.length) {
        const step = car.route[car.routeIndex];
        nextNode = roadNetwork.nodes[step.nodeId];
        car.routeIndex++;
    }

    // If route exhausted or no route, pick random next
    if (!nextNode) {
        const prevNode = car.path.getPrevIntersection();
        nextNode = pickRandomNextNode(roadNetwork, lastNode, prevNode);
        // Plan a new destination if we've arrived
        if (car.destination && car.routeIndex >= (car.route?.length || 0)) {
            planNewRoute(car, roadNetwork);
        }
    }

    if (!nextNode) return;

    extendPath(car.path, nextNode, car.laneOffset || 3, 8);
}

/**
 * Pick a random next intersection, avoiding U-turns.
 */
function pickRandomNextNode(roadNetwork, currentNode, prevNode) {
    const edges = roadNetwork.getEdgesForNode(currentNode.id);
    const candidates = [];

    for (const edge of edges) {
        const otherId = roadNetwork.getOtherNode(edge, currentNode.id);
        if (prevNode && otherId === prevNode.id) continue; // no U-turn
        const otherNode = roadNetwork.nodes[otherId];
        if (otherNode) candidates.push(otherNode);
    }

    if (candidates.length === 0) {
        // Dead end — allow U-turn
        if (prevNode) return roadNetwork.nodes[prevNode.id] || null;
        return null;
    }

    // Weight: straight-ish directions preferred
    return candidates[Math.floor(Math.random() * candidates.length)];
}

/**
 * Plan a new Dijkstra route to a random destination.
 */
function planNewRoute(car, roadNetwork) {
    const lastNode = car.path.getLastIntersection();
    if (!lastNode) return;

    // Pick random interior destination, 2+ blocks away
    const nodeIds = Object.keys(roadNetwork.nodes);
    let destId = null;
    for (let i = 0; i < 20; i++) {
        const cand = nodeIds[Math.floor(Math.random() * nodeIds.length)];
        if (cand === lastNode.id) continue;
        const node = roadNetwork.nodes[cand];
        if (!node) continue;
        const dist = Math.abs(node.x - lastNode.x) + Math.abs(node.z - lastNode.z);
        if (dist > 60) { destId = cand; break; }
    }
    if (!destId) destId = nodeIds[Math.floor(Math.random() * nodeIds.length)];

    car.destination = destId;
    car.route = roadNetwork.findPath(lastNode.id, destId);
    car.routeIndex = 0;
}

// ============================================================
// OVERLAP RESOLUTION — visual safety net
// ============================================================

/**
 * Prevent visual overlap between cars.
 * NOT physics — just pushes cars apart along their paths.
 *
 * @param {Array} cars - All vehicles
 */
export function resolveOverlaps(cars) {
    // Hard collision prevention: push overlapping cars apart
    // Uses spatial hash for O(n*k) instead of O(n²)
    for (const car of cars) {
        // Skip cars in turns — heading changes rapidly, false collisions
        if (car.inTurn) continue;
        const nearby = spatialHash.getNearby(car.worldX, car.worldZ);
        for (const other of nearby) {
            if (other === car) continue;
            const dx = other.worldX - car.worldX;
            const dz = other.worldZ - car.worldZ;
            const distSq = dx * dx + dz * dz;
            const minDist = 3.0; // slightly less than car length for visual clearance
            if (distSq < minDist * minDist && distSq > 0.01) {
                const dist = Math.sqrt(distSq);
                const overlap = minDist - dist;
                // Gently push the rear car back along its path
                const heading = car.path ? car.path.getHeading(car.d) : 0;
                const fwdX = Math.sin(heading), fwdZ = Math.cos(heading);
                const dot = dx * fwdX + dz * fwdZ;
                if (dot > 0) {
                    // Other is ahead — nudge this car back slightly
                    car.d = Math.max(0, car.d - overlap * 0.15);
                } else {
                    // Other is behind — nudge this car forward slightly
                    car.d += overlap * 0.05;
                }
            }
        }
    }
}

// ============================================================
// INITIALIZATION — create a car with a CarPath
// ============================================================

/**
 * Initialize a car with a path from a starting position.
 *
 * @param {Object} roadNetwork - RoadNetwork instance
 * @param {number} [laneOffset] - Distance from road center to lane center
 * @returns {Object} Car state object
 */
export function createCar(roadNetwork, laneWidth = 3) {
    // Lane center offsets from road center (right-hand traffic):
    // Inner lane: 0.5 * laneWidth = 1.5m from center
    // Outer lane: 1.5 * laneWidth = 4.5m from center
    const laneOffset = (Math.random() < 0.5 ? 0.5 : 1.5) * laneWidth;
    // Pick random starting edge and position
    const nodeIds = Object.keys(roadNetwork.nodes);
    const startId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
    const startNode = roadNetwork.nodes[startId];

    // Pick a neighbor to determine initial direction
    const edges = roadNetwork.getEdgesForNode(startId);
    if (edges.length === 0) return null;
    const firstEdge = edges[Math.floor(Math.random() * edges.length)];
    const nextId = roadNetwork.getOtherNode(firstEdge, startId);
    const nextNode = roadNetwork.nodes[nextId];

    // Build initial path with 3+ segments ahead
    const route = [startNode, nextNode];
    // Extend path 4 intersections ahead (more added dynamically as car drives)
    let currNode = nextNode, prevNode = startNode;
    for (let i = 0; i < 4; i++) {
        const next = pickRandomNextNode(roadNetwork, currNode, prevNode);
        if (!next) break;
        route.push(next);
        prevNode = currNode;
        currNode = next;
    }

    const path = buildPathFromRoute(route, laneOffset, 8);

    // Place car at a well-spaced position (use global counter to distribute)
    if (!createCar._counter) createCar._counter = 0;
    createCar._counter++;
    // Spread cars evenly across the path length using golden ratio for even distribution
    const golden = 0.618033988749895;
    const startD = 5 + ((createCar._counter * golden) % 1) * Math.min(path.totalLength * 0.6, 200);

    return {
        path,
        d: startD,
        speed: 5 + Math.random() * 5, // start with decent speed so IDM doesn't gridlock
        acc: 0,
        idmParams: {
            ...IDM_DEFAULTS,
            v0: 8 + Math.random() * 5, // 8-13 m/s desired speed
        },
        length: 4.5,
        width: 2.0,
        laneOffset,
        worldX: 0,
        worldZ: 0,
        worldHeading: 0,
        brakeLightsOn: false,
        inTurn: false,
        route: [],
        routeIndex: 0,
        destination: null,
        stuckTimer: 0,
    };
}
