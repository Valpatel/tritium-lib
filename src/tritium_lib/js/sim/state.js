// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * SimState — Pure simulation state with zero rendering dependencies.
 *
 * This is THE single source of truth for any simulation. Renderers read from it,
 * sim systems write to it, and it can be serialized/streamed via WebSocket.
 *
 * Key design constraints:
 * - NO Three.js imports — this module is pure JavaScript
 * - All positions use plain {x, z} or {x, y, z} objects
 * - Entity arrays use plain objects with string IDs
 * - State shape is the wire format (JSON-serializable)
 */

// =========================================================================
// Entity factory helpers
// =========================================================================

let _nextEntityId = 0;

export function makeEntity(type, x, z, props = {}) {
  return {
    id: type + '_' + (_nextEntityId++),
    type,           // 'pedestrian', 'protestor', 'police', 'car', 'taxi', etc.
    x, z, y: 0,
    rotY: 0,
    speed: props.speed || 2,
    alive: true,
    ...props,
  };
}

export function makeVehicle(type, x, z, props = {}) {
  return {
    id: type + '_' + (_nextEntityId++),
    type,           // 'car', 'taxi', 'ambulance', 'fire_truck', 'police_van', 'robot'
    x, z,
    rotY: 0,
    speed: props.speed || 8,
    dir: props.dir || 1,
    horizontal: props.horizontal !== undefined ? props.horizontal : true,
    alive: true,
    pauseTimer: 0,
    ...props,
  };
}

export function makeProjectile(type, startX, startZ, targetX, targetZ) {
  const dx = targetX - startX;
  const dz = targetZ - startZ;
  const dist = Math.sqrt(dx * dx + dz * dz);
  const flightTime = Math.max(0.3, dist / 30);
  const vy = dist * 0.15 + 5;
  return {
    id: 'proj_' + (_nextEntityId++),
    type,           // 'molotov', 'rock'
    startX, startZ,
    vx: dx / flightTime,
    vz: dz / flightTime,
    vy,
    gravity: vy * 2 / flightTime,
    age: 0,
    flightTime,
    alive: true,
  };
}

// =========================================================================
// City grid constants (pure data, no rendering)
// =========================================================================

export const GRID_DEFAULTS = {
  BLOCK_W: 60,
  BLOCK_H: 50,
  GRID_COLS: 8,
  GRID_ROWS: 6,
  ROAD_W: 16,
  SIDEWALK_W: 3,
};

export function cityDimensions(grid = GRID_DEFAULTS) {
  return {
    CITY_W: grid.GRID_COLS * (grid.BLOCK_W + grid.ROAD_W) + grid.ROAD_W,
    CITY_H: grid.GRID_ROWS * (grid.BLOCK_H + grid.ROAD_W) + grid.ROAD_W,
  };
}

export function buildRoadGrid(grid = GRID_DEFAULTS) {
  const hRoads = [];
  const vRoads = [];
  const intersections = [];
  for (let row = 0; row <= grid.GRID_ROWS; row++) {
    hRoads.push(row * (grid.BLOCK_H + grid.ROAD_W) + grid.ROAD_W / 2);
  }
  for (let col = 0; col <= grid.GRID_COLS; col++) {
    vRoads.push(col * (grid.BLOCK_W + grid.ROAD_W) + grid.ROAD_W / 2);
  }
  for (const x of vRoads) {
    for (const z of hRoads) {
      intersections.push({ x, z });
    }
  }
  return { hRoads, vRoads, intersections };
}

// =========================================================================
// SimState class
// =========================================================================

export class SimState {
  constructor(gridConfig = GRID_DEFAULTS) {
    // Grid
    this.grid = gridConfig;
    const dims = cityDimensions(gridConfig);
    this.CITY_W = dims.CITY_W;
    this.CITY_H = dims.CITY_H;
    this.PLAZA_X = dims.CITY_W / 2;
    this.PLAZA_Z = dims.CITY_H / 2;

    const roads = buildRoadGrid(gridConfig);
    this.hRoads = roads.hRoads;
    this.vRoads = roads.vRoads;
    this.intersections = roads.intersections;

    // Time
    this.simTime = 8.0;    // 24h clock
    this.riotTimer = 0;

    // Phase
    this.phase = 'PEACEFUL';  // PEACEFUL, TENSION, RIOT, DISPERSAL
    this.riotMode = false;
    this.gatheringPhase = 'NONE'; // NONE, MARCHING, MILLING, TENSION_BUILDUP, RIOT_TRIGGERED

    // Campaign
    this.campaignPhase = 0;
    this.campaignTimer = 0;
    this.campaignComplete = false;

    // Weather
    this.weather = {
      rain: false,
      fog: false,
      nightMode: false,
      lightningTimer: 0,
      lightningCount: 0,
    };

    // Entity arrays (plain objects, no Three.js)
    this.pedestrians = [];
    this.cars = [];
    this.taxis = [];
    this.protestors = [];
    this.police = [];
    this.ambulances = [];
    this.fireTrucks = [];
    this.policeVans = [];
    this.carDrivers = [];
    this.robotCars = [];
    this.medics = [];

    // Combat
    this.projectiles = [];
    this.fires = [];
    this.barricades = [];
    this.ieds = [];
    this.tracers = [];

    // Buildings (generated during world init, pure data)
    this.buildings = [];
    this.houseBuildings = [];
    this.hospitalBuilding = null;

    // Counters
    this.molotovCount = 0;
    this.teargasCount = 0;
    this.rockCount = 0;
    this.arrestCount = 0;
    this.injuryCount = 0;

    // Supply
    this.supplyTearGas = 20;
    this.supplyRubberBullets = 100;
    this.supplyMolotovs = 15;

    // Economy
    this.policeBudget = 10000;
    this.propertyDamage = 0;

    // Status effects
    this.activeEffects = [];

    // Comms
    this.commsStatus = 'ACTIVE';
    this.commsDegradedTimer = 0;

    // EW
    this.ewJamActive = false;
    this.ewJamTimer = 0;
    this.ewJamX = 0;
    this.ewJamZ = 0;
    this.ewJamRadius = 40;

    // Fog of war
    this.fogOfWarEnabled = false;
    this.detectedCount = 0;

    // Territory
    this.territoryZones = [];

    // Objectives
    this.objectives = [];

    // Scores
    this.totalScore = 0;
    this.achievementsAwarded = {};

    // Events log (narration, kill feed)
    this.narrationMessages = [];
    this.killFeed = [];

    // Injured tracking
    this.injuredOnGround = [];
    this.injuredQueue = [];

    // Traffic
    this.trafficPhase = 0;
  }

  /**
   * Returns true if it's currently night in the simulation.
   */
  get isNight() {
    return this.weather.nightMode || this.simTime < 6 || this.simTime > 20;
  }

  /**
   * Advance simulation time. 1 real second = 2 sim minutes.
   */
  advanceTime(dt) {
    this.simTime += dt * (2 / 60);
    if (this.simTime >= 24) this.simTime -= 24;
    this.trafficPhase += dt;
  }

  /**
   * Add a narration message.
   */
  addNarration(html) {
    this.narrationMessages.push({ html, age: 0 });
    if (this.narrationMessages.length > 4) this.narrationMessages.shift();
  }

  /**
   * Add a kill feed entry.
   */
  addKillFeed(text) {
    this.killFeed.unshift({ text, age: 0 });
    if (this.killFeed.length > 10) this.killFeed.pop();
  }

  /**
   * Export a JSON-serializable snapshot of the full state.
   * Useful for WebSocket streaming or replay recording.
   */
  snapshot() {
    return {
      simTime: this.simTime,
      phase: this.phase,
      riotMode: this.riotMode,
      weather: { ...this.weather },
      pedestrians: this.pedestrians.filter(p => p.alive).map(p => ({
        id: p.id, x: p.x, z: p.z, rotY: p.rotY, type: p.type,
        bodyColor: p.bodyColor, headColor: p.headColor, scale: p.scale,
      })),
      cars: this.cars.filter(c => c.alive).map(c => ({
        id: c.id, x: c.x, z: c.z, rotY: c.rotY, type: c.type,
        bodyColor: c.bodyColor, cabinColor: c.cabinColor,
      })),
      protestors: this.protestors.filter(p => p.alive && !p.arrested).map(p => ({
        id: p.id, x: p.x, z: p.z, rotY: p.rotY, morale: p.morale, health: p.health,
      })),
      police: this.police.filter(p => p.alive).map(p => ({
        id: p.id, x: p.x, z: p.z, rotY: p.rotY, morale: p.morale, health: p.health,
      })),
      fires: this.fires.map(f => ({ x: f.x, z: f.z, age: f.age })),
      projectiles: this.projectiles.filter(p => p.alive).map(p => ({
        type: p.type, x: p.startX + p.vx * p.age, z: p.startZ + p.vz * p.age,
        y: 2 + p.vy * p.age - 0.5 * p.gravity * p.age * p.age,
      })),
    };
  }
}

// =========================================================================
// Spatial utilities (pure math, no rendering)
// =========================================================================

export function dist2d(a, b) {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}

export function formatGrid(x, z, cityW, cityH) {
  const col = String.fromCharCode(65 + Math.min(7, Math.max(0, Math.floor((x / cityW) * 8))));
  const row = Math.min(6, Math.max(1, Math.floor((z / cityH) * 6) + 1));
  return col + row;
}

export function isInsideBuilding(px, pz, buildings) {
  for (const b of buildings) {
    if (Math.abs(px - b.x) < b.w / 2 && Math.abs(pz - b.z) < b.d / 2) return b;
  }
  return null;
}

export function hasLineOfSight(fromX, fromZ, toX, toZ, buildings) {
  const dx = toX - fromX;
  const dz = toZ - fromZ;
  const dist = Math.sqrt(dx * dx + dz * dz);
  const steps = Math.ceil(dist / 3);
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    const px = fromX + dx * t;
    const pz = fromZ + dz * t;
    if (isInsideBuilding(px, pz, buildings)) return false;
  }
  return true;
}

export function resolveCollision(oldX, oldZ, newX, newZ, buildings) {
  for (const b of buildings) {
    const hw = b.w / 2;
    const hd = b.d / 2;
    if (Math.abs(newX - b.x) < hw && Math.abs(newZ - b.z) < hd) {
      const penX = hw - Math.abs(newX - b.x);
      const penZ = hd - Math.abs(newZ - b.z);
      if (penX < penZ) {
        newX = b.x + Math.sign(newX - b.x) * (hw + 0.3);
      } else {
        newZ = b.z + Math.sign(newZ - b.z) * (hd + 0.3);
      }
    }
  }
  return { x: newX, z: newZ };
}
