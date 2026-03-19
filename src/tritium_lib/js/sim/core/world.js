// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * World — the central simulation manager.
 *
 * Creates and manages all units (vehicles, pedestrians), the road network,
 * traffic lights, spatial indexing, weather, and rendering. One World instance
 * per simulation.
 *
 * Architecture:
 *   Vehicles → CarPath + IDM (lane-constrained road driving)
 *   Pedestrians → yuka Vehicle + NavMesh (free-form sidewalk movement)
 *   Rendering → InstancedRenderer (generic, one draw call per unit type)
 *   Spatial → SpatialHash3D (supports overpasses, underground)
 *
 * Usage:
 *   const world = new World(scene, roadNetwork);
 *   world.spawnCars(200);
 *   world.spawnPedestrians(50, buildingEntries);
 *   // In animation loop:
 *   world.tick(dt);
 */

import { SpatialHash } from './spatial-hash.js';
import { Path, buildPath, extendPath } from './path.js';
import { Vehicle, Car, Motorcycle, EmergencyVehicle } from '../units/vehicle.js';
import { Pedestrian } from '../units/pedestrian.js';
import { InstancedRenderer } from '../rendering/instanced-renderer.js';
import { idmAcceleration, IDM_DEFAULTS } from '../idm.js';
import { computeWeather } from '../weather.js';
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// ============================================================
// CAR MESH DEFINITIONS
// ============================================================

function createCarGeometry() {
    const body = new THREE.BoxGeometry(1.8, 1.4, 4.5); // realistic car width
    body.translate(0, 0.7, 0);
    const cabin = new THREE.BoxGeometry(1.5, 0.9, 2.2);
    cabin.translate(0, 1.8, -0.1);
    return mergeGeometries([body, cabin], false) || body;
}

function createHeadlightGeometry() {
    const l = new THREE.BoxGeometry(0.35, 0.3, 0.2);
    l.translate(0.5, 0.45, 2.3);
    const r = new THREE.BoxGeometry(0.35, 0.3, 0.2);
    r.translate(-0.5, 0.45, 2.3);
    return mergeGeometries([l, r], false);
}

function createTaillightGeometry() {
    const l = new THREE.BoxGeometry(0.3, 0.25, 0.15);
    l.translate(0.5, 0.45, -2.3);
    const r = new THREE.BoxGeometry(0.3, 0.25, 0.15);
    r.translate(-0.5, 0.45, -2.3);
    return mergeGeometries([l, r], false);
}

function createPedGeometry() {
    const body = new THREE.CylinderGeometry(0.25, 0.25, 1.2, 6);
    body.translate(0, 0.8, 0);
    const head = new THREE.SphereGeometry(0.2, 6, 4);
    head.translate(0, 1.6, 0);
    return mergeGeometries([body, head], false);
}

function createBeamGeometry() {
    // Headlight beam with gradient: bright at car, dim at far end
    function makeBeamCone(offsetX) {
        const geo = new THREE.ConeGeometry(2.0, 10, 6);
        geo.rotateX(-Math.PI / 2);
        geo.translate(offsetX, 0.25, 7.5);

        // Vertex color gradient: bright near car, dim far (RGB, no alpha)
        const pos = geo.attributes.position;
        const colors = new Float32Array(pos.count * 3); // RGB
        for (let i = 0; i < pos.count; i++) {
            const z = pos.getZ(i);
            const t = Math.max(0, Math.min(1, (z - 2) / 12)); // 0=car, 1=far
            const brightness = 1.0 - t * 0.7; // 1.0 at car → 0.3 at far
            colors[i * 3] = brightness;        // R
            colors[i * 3 + 1] = brightness;    // G
            colors[i * 3 + 2] = brightness * 0.75; // B (warm tint)
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        return geo;
    }
    return mergeGeometries([makeBeamCone(0.5), makeBeamCone(-0.5)], false);
    return mergeGeometries([l, r], false);
}

// ============================================================
// WORLD CLASS
// ============================================================

export class World {
    /**
     * @param {THREE.Scene} scene
     * @param {Object} roadNetwork — RoadNetwork instance
     * @param {Object} [config]
     * @param {number} [config.maxVehicles] — max instanced vehicles
     * @param {number} [config.maxPedestrians] — max instanced peds
     */
    constructor(scene, roadNetwork, config = {}) {
        this.scene = scene;
        this.roadNetwork = roadNetwork;
        this.maxVehicles = config.maxVehicles || 5000;
        this.maxPeds = config.maxPedestrians || 1000;

        // All units
        this.vehicles = [];
        this.pedestrians = [];

        // Spatial indexing (3D)
        this.spatialHash = new SpatialHash(50);

        // Rendering
        this.renderer = new InstancedRenderer(scene);
        this._initMeshTypes();

        // Traffic
        this.trafficCtrl = config.trafficCtrl || null;

        // Sim clock
        this.simHour = config.startHour || 7;
        this.simDay = 0;
        this.timeScale = config.timeScale || 1; // sim minutes per real second

        // Weather
        this.isNight = false;

        // Stats
        this.frameCount = 0;
        this.fps = 0;
        this._fpsTimer = 0;
        this._fpsFrames = 0;

        // Spawn queue (gradual spawning to avoid frame spikes)
        this._vehicleSpawnQueue = 0;
        this._pedSpawnQueue = 0;
        this._buildingEntries = config.buildingEntries || [];

        // Colors for random car colors
        this._carColors = [0xfcee0a, 0xff4444, 0x4488ff, 0x44ff44, 0xffffff, 0xcccccc, 0xff8844, 0x44ffff, 0xee66aa, 0x88ff44];
        this._pedColors = {
            resident: 0x44aa66, worker: 0x4488cc, student: 0xccaa44,
            police: 0x2244ff, shopkeeper: 0xcc6644, jogger: 0xff6644,
        };
    }

    _initMeshTypes() {
        // Car body
        this.renderer.defineMeshType('car_body', createCarGeometry(),
            new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6 }), this.maxVehicles);

        // Headlights (always bright white)
        this.renderer.defineMeshType('car_headlights', createHeadlightGeometry(),
            new THREE.MeshBasicMaterial({ color: 0xffffff }), this.maxVehicles);

        // Taillights (red, per-instance color for brake state)
        this.renderer.defineMeshType('car_taillights', createTaillightGeometry(),
            new THREE.MeshBasicMaterial({ color: 0xff0000 }), this.maxVehicles);

        // Headlight beams (night only, gradient cones)
        this.renderer.defineMeshType('car_beams', createBeamGeometry(),
            new THREE.MeshBasicMaterial({
                color: 0xffffaa,
                transparent: true,
                opacity: 0.08,
                depthWrite: false,
                vertexColors: true, // use per-vertex color for gradient
            }),
            this.maxVehicles);
        this.renderer.setVisible('car_beams', false);

        // Pedestrian body
        this.renderer.defineMeshType('ped_body', createPedGeometry(),
            new THREE.MeshStandardMaterial({ color: 0x44aa66, roughness: 0.7 }), this.maxPeds);
    }

    // ============================================================
    // SPAWNING
    // ============================================================

    /**
     * Queue N vehicles to spawn gradually (avoids frame spikes).
     */
    spawnCars(count) {
        this._vehicleSpawnQueue += count;
    }

    /**
     * Queue N pedestrians to spawn gradually.
     */
    spawnPedestrians(count, buildingEntries) {
        this._pedSpawnQueue += count;
        if (buildingEntries) this._buildingEntries = buildingEntries;
    }

    _spawnOneCar() {
        const rn = this.roadNetwork;
        if (!rn) return;

        // Pick random start: 2 connected nodes
        const nodeIds = Object.keys(rn.nodes);
        const startId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
        const startNode = rn.nodes[startId];
        const edges = rn.getEdgesForNode(startId);
        if (edges.length === 0) return;

        const firstEdge = edges[Math.floor(Math.random() * edges.length)];
        const nextId = rn.getOtherNode(firstEdge, startId);
        const nextNode = rn.nodes[nextId];

        // Build initial path (4+ intersections ahead)
        const route = [startNode, nextNode];
        let curr = nextNode, prev = startNode;
        for (let i = 0; i < 5; i++) {
            const e = rn.getEdgesForNode(curr.id);
            const cands = [];
            for (const edge of e) {
                const oid = rn.getOtherNode(edge, curr.id);
                if (prev && oid === prev.id) continue;
                const n = rn.nodes[oid];
                if (n) cands.push(n);
            }
            if (cands.length === 0) break;
            const next = cands[Math.floor(Math.random() * cands.length)];
            route.push(next);
            prev = curr;
            curr = next;
        }

        // Create car
        const laneOffset = (Math.random() < 0.5 ? 0.5 : 1.5) * 3;
        const car = new Car({
            speed: 5 + Math.random() * 5,
            laneOffset,
            path: buildPath(route, laneOffset, 8),
            d: 10 + Math.random() * 40,
        });

        // Color
        const color = this._carColors[Math.floor(Math.random() * this._carColors.length)];
        car.color = color;

        // Register with renderer
        const bodyIdx = this.renderer.addInstance('car_body', color);
        // Randomize headlight temperature: cool blue-white to warm yellow-white
        const warmth = Math.random();
        const hlR = warmth < 0.3 ? 0.7 : (0.85 + warmth * 0.15);  // cool cars are bluer
        const hlG = warmth < 0.3 ? 0.8 : (0.88 + warmth * 0.05);
        const hlB = warmth < 0.3 ? 1.0 : (1.0 - warmth * 0.4);    // warm cars are more yellow
        const hlIdx = this.renderer.addInstance('car_headlights', null);
        this.renderer.setInstanceColor('car_headlights', hlIdx, hlR, hlG, hlB);
        const tlIdx = this.renderer.addInstance('car_taillights', null);
        const beamIdx = this.renderer.addInstance('car_beams', null);
        car._renderHandles = { body: bodyIdx, headlights: hlIdx, taillights: tlIdx, beams: beamIdx };
        car.instanceHandle = bodyIdx;

        this.vehicles.push(car);
    }

    _spawnOnePed() {
        if (this._buildingEntries.length === 0) return;

        // Start at a random building entry
        const entry = this._buildingEntries[Math.floor(Math.random() * this._buildingEntries.length)];
        const roles = ['resident', 'worker', 'worker', 'student', 'shopkeeper', 'jogger'];
        const role = roles[Math.floor(Math.random() * roles.length)];

        const ped = new Pedestrian({
            x: entry.x + (Math.random() - 0.5) * 4,
            z: entry.z + (Math.random() - 0.5) * 4,
            role,
            color: this._pedColors[role] || 0x44aa66,
        });
        // Set x/z directly since pedestrians don't use path
        ped.x = entry.x + (Math.random() - 0.5) * 4;
        ped.z = entry.z + (Math.random() - 0.5) * 4;

        // Initial goal: another building entry
        const goal = this._buildingEntries[Math.floor(Math.random() * this._buildingEntries.length)];
        ped.setGoal(goal.x, goal.z);

        // Register with renderer
        const idx = this.renderer.addInstance('ped_body', ped.color);
        ped.instanceHandle = idx;
        ped._renderHandles = { body: idx };

        this.pedestrians.push(ped);
    }

    // ============================================================
    // TICK
    // ============================================================

    /**
     * Main simulation tick. Call once per frame.
     *
     * @param {number} dt — real seconds since last frame
     */
    tick(dt) {
        this.frameCount++;

        // FPS counter
        this._fpsFrames++;
        this._fpsTimer += dt;
        if (this._fpsTimer >= 1) {
            this.fps = this._fpsFrames;
            this._fpsFrames = 0;
            this._fpsTimer = 0;
        }

        // Sim clock
        this.simHour += dt * this.timeScale / 60;
        if (this.simHour >= 24) { this.simHour -= 24; this.simDay++; }

        // Gradual spawning (budget: 3ms per frame)
        const spawnStart = performance.now();
        while (this._vehicleSpawnQueue > 0 && performance.now() - spawnStart < 3) {
            this._spawnOneCar();
            this._vehicleSpawnQueue--;
        }
        while (this._pedSpawnQueue > 0 && performance.now() - spawnStart < 3) {
            this._spawnOnePed();
            this._pedSpawnQueue--;
        }

        // Rebuild spatial hash with ALL units
        this.spatialHash.clear();
        for (const v of this.vehicles) this.spatialHash.insert(v);
        for (const p of this.pedestrians) this.spatialHash.insert(p);

        // Tick vehicles
        const worldCtx = {
            spatialHash: this.spatialHash,
            roadNetwork: this.roadNetwork,
            trafficCtrl: this.trafficCtrl,
        };
        for (const v of this.vehicles) {
            v.tick(dt, worldCtx);
        }

        // Tick pedestrians
        for (const p of this.pedestrians) {
            p.tick(dt, worldCtx);
            // Reassign goal when reached
            if (p.goalReached && !p.knockedDown && this._buildingEntries.length > 0) {
                const goal = this._buildingEntries[Math.floor(Math.random() * this._buildingEntries.length)];
                p.setGoal(goal.x, goal.z);
            }
        }

        // Resolve overlaps (vehicles only, spatial hash accelerated)
        this._resolveOverlaps();

        // Update rendering
        this._updateRendering();

        // Weather / day-night
        this.isNight = this.simHour >= 22 || this.simHour < 6;
    }

    _resolveOverlaps() {
        // Hard collision: push overlapping vehicles apart in world space
        // AND reduce speed of the car behind to prevent re-collision
        for (const car of this.vehicles) {
            if (car.inCurve || car._aggressive) continue; // skip during turns and aggressive mode
            const nearby = this.spatialHash.getNearby(car.x, 0, car.z);
            for (const other of nearby) {
                if (other === car || other.type === 'pedestrian') continue;
                const dx = other.x - car.x;
                const dz = other.z - car.z;
                const distSq = dx * dx + dz * dz;
                const minDist = (car.length + (other.length || 4)) / 2;
                if (distSq < minDist * minDist && distSq > 0.01) {
                    const dist = Math.sqrt(distSq);
                    const overlap = minDist - dist;

                    // Determine who is "behind" (should slow down)
                    const fwdX = Math.sin(car.heading);
                    const fwdZ = Math.cos(car.heading);
                    const dot = dx * fwdX + dz * fwdZ;

                    if (dot > 0) {
                        // Other is ahead — back up along path, gentle speed reduction
                        car.d = Math.max(0, car.d - overlap * 0.3);
                        car.speed = Math.max(0, car.speed * 0.95); // 5% speed loss per overlap frame
                    } else {
                        // Other is behind — nudge forward
                        car.d += overlap * 0.1;
                    }
                }
            }
        }
    }

    _updateRendering() {
        // Vehicles
        for (const v of this.vehicles) {
            const h = v._renderHandles;
            if (!h) continue;
            this.renderer.updateInstance('car_body', h.body, v.x, v.y, v.z, v.heading);
            this.renderer.updateInstance('car_headlights', h.headlights, v.x, v.y, v.z, v.heading);
            this.renderer.updateInstance('car_taillights', h.taillights, v.x, v.y, v.z, v.heading);
            // Adaptive beam tilt: slight dip when car ahead, like real low/high beam
            const gap = v._leaderGap !== undefined ? v._leaderGap : Infinity;
            // Gap 30m+ = level (highbeam), gap 5m = max tilt 8° (0.14 rad) = lowbeam
            // Real headlights only tilt a few degrees — beam always projects forward
            const beamPitch = gap < 30 ? Math.max(0, (30 - gap) / 25) * 0.14 : 0;
            this.renderer.updateInstance('car_beams', h.beams, v.x, v.y, v.z, v.heading, beamPitch);

            // Brake lights: bright red when braking
            if (v.brakeLightsOn) {
                this.renderer.setInstanceColor('car_taillights', h.taillights, 1.0, 0.1, 0.0);
            } else {
                this.renderer.setInstanceColor('car_taillights', h.taillights, 0.5, 0.0, 0.0);
            }
        }

        // Pedestrians
        for (const p of this.pedestrians) {
            const h = p._renderHandles;
            if (!h) continue;
            const bob = p.knockedDown ? 0 : Math.sin(p.bobPhase || 0) * 0.03;
            if (p.knockedDown) {
                this.renderer.updateInstance('ped_body', h.body, p.x, 0.3, p.z, p.knockedAngle, 0, Math.PI / 2);
            } else {
                this.renderer.updateInstance('ped_body', h.body, p.x, bob, p.z, p.heading);
            }
        }

        // Night mode
        this.renderer.setVisible('car_beams', this.isNight);

        // Headlight brightness
        if (this.isNight) {
            this.renderer.setMaterialProp('car_headlights', 'color', new THREE.Color(1, 1, 0.85));
        } else {
            this.renderer.setMaterialProp('car_headlights', 'color', new THREE.Color(0.9, 0.9, 0.8));
        }

        // Flush to GPU
        this.renderer.flush();
    }

    // ============================================================
    // QUERIES
    // ============================================================

    /** Get simulation time as "HH:MM" string. */
    getTimeString() {
        const h = Math.floor(this.simHour);
        const m = Math.floor((this.simHour - h) * 60);
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
    }

    /** Get weather state for rendering. */
    getWeather() {
        return computeWeather({
            simTime: this.simHour,
            isNight: this.isNight,
            weather: { rain: false, fog: false },
            phase: 'NORMAL',
        });
    }

    /** Get stats for HUD. */
    getStats() {
        const vStopped = this.vehicles.filter(v => v.speed < 0.5).length;
        const pDown = this.pedestrians.filter(p => p.knockedDown).length;
        return {
            fps: this.fps,
            vehicles: this.vehicles.length,
            vehiclesStopped: vStopped,
            pedestrians: this.pedestrians.length,
            pedsDown: pDown,
            simTime: this.getTimeString(),
            day: this.simDay,
            isNight: this.isNight,
            drawCalls: this.renderer.getDrawCallCount(),
            spawning: this._vehicleSpawnQueue + this._pedSpawnQueue,
        };
    }

    /** Add more cars at runtime. */
    addCars(n) { this._vehicleSpawnQueue += n; }

    /** Add more peds at runtime. */
    addPeds(n) { this._pedSpawnQueue += n; }

    /** Set sim speed (sim minutes per real second). */
    setTimeScale(scale) { this.timeScale = scale; }
}
