/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Pedestrian spawning, movement, pathfinding, collision resolution,
           person instance management, car-driving civilians.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  CITY_W, CITY_H, BLOCK_W, BLOCK_H, GRID_COLS, GRID_ROWS, ROAD_W, SIDEWALK_W,
  PLAZA_X, PLAZA_Z, MAX_PEOPLE, BODY_W, BODY_H, BODY_D, HEAD_SIZE, PERSON_SCALE,
  NUM_PEDESTRIANS, COLOR_CIVILIAN, SKIN_TONES,
} from './config.js';

// =========================================================================
// Person InstancedMesh Init
// =========================================================================
export function initPeople() {
  const scene = state.scene;

  const personBodyGeo = new THREE.BoxGeometry(1, 1, 1);
  const personBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.7 });
  const personBodyMesh = new THREE.InstancedMesh(personBodyGeo, personBodyMat, MAX_PEOPLE);
  personBodyMesh.castShadow = false;
  personBodyMesh.count = 0;
  scene.add(personBodyMesh);
  state.personBodyMesh = personBodyMesh;

  const personHeadGeo = new THREE.BoxGeometry(1, 1, 1);
  const personHeadMat = new THREE.MeshStandardMaterial({ roughness: 0.6 });
  const personHeadMesh = new THREE.InstancedMesh(personHeadGeo, personHeadMat, MAX_PEOPLE);
  personHeadMesh.castShadow = false;
  personHeadMesh.count = 0;
  scene.add(personHeadMesh);
  state.personHeadMesh = personHeadMesh;
}

export function allocPersonSlot() {
  if (state.personSlotCount >= MAX_PEOPLE) return -1;
  return state.personSlotCount++;
}

export function updatePersonInstance(slot, x, y, z, rotY, bodyColor, headColor, scale) {
  if (slot < 0) return;
  const s = scale * PERSON_SCALE;
  _euler.set(0, rotY, 0);
  _quat.setFromEuler(_euler);
  const bodyY = y + (BODY_H * s) / 2;
  _pos.set(x, bodyY, z);
  _scale.set(BODY_W * s, BODY_H * s, BODY_D * s);
  _mat4.compose(_pos, _quat, _scale);
  state.personBodyMesh.setMatrixAt(slot, _mat4);
  state.personBodyMesh.setColorAt(slot, _color.set(bodyColor));
  const headY = y + BODY_H * s + (HEAD_SIZE * s) / 2;
  _pos.set(x, headY, z);
  _scale.set(HEAD_SIZE * s, HEAD_SIZE * s, HEAD_SIZE * s);
  _mat4.compose(_pos, _quat, _scale);
  state.personHeadMesh.setMatrixAt(slot, _mat4);
  state.personHeadMesh.setColorAt(slot, _color.set(headColor));
}

export function hidePersonInstance(slot) {
  if (slot < 0) return;
  _pos.set(0, -100, 0);
  _quat.identity();
  _scale.set(0, 0, 0);
  _mat4.compose(_pos, _quat, _scale);
  state.personBodyMesh.setMatrixAt(slot, _mat4);
  state.personHeadMesh.setMatrixAt(slot, _mat4);
}

// =========================================================================
// Spatial Helpers
// =========================================================================
export function randomSidewalkPos() {
  if (rng() < 0.5) {
    const row = Math.floor(rng() * (GRID_ROWS + 1));
    const z = row * (BLOCK_H + ROAD_W) + ROAD_W / 2 + (rng() < 0.5 ? 1 : -1) * (ROAD_W / 2 + SIDEWALK_W / 2);
    const x = rng() * CITY_W;
    return { x, z };
  } else {
    const col = Math.floor(rng() * (GRID_COLS + 1));
    const x = col * (BLOCK_W + ROAD_W) + ROAD_W / 2 + (rng() < 0.5 ? 1 : -1) * (ROAD_W / 2 + SIDEWALK_W / 2);
    const z = rng() * CITY_H;
    return { x, z };
  }
}

export function dist2d(a, b) {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}

export function edgeSpawnPos() {
  const side = Math.floor(rng() * 4);
  switch (side) {
    case 0: return { x: -5, z: rng() * CITY_H };
    case 1: return { x: CITY_W + 5, z: rng() * CITY_H };
    case 2: return { x: rng() * CITY_W, z: -5 };
    default: return { x: rng() * CITY_W, z: CITY_H + 5 };
  }
}

export function isInsideBuilding(px, pz) {
  for (const b of state.buildings) {
    if (Math.abs(px - b.x) < b.w / 2 && Math.abs(pz - b.z) < b.d / 2) return b;
  }
  return null;
}

export function hasLineOfSight(fromX, fromZ, toX, toZ) {
  const dx = toX - fromX;
  const dz = toZ - fromZ;
  const dist = Math.sqrt(dx * dx + dz * dz);
  const steps = Math.ceil(dist / 3);
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    const px = fromX + dx * t;
    const pz = fromZ + dz * t;
    if (isInsideBuilding(px, pz)) return false;
  }
  return true;
}

export function nearbyBuildings(px, pz, radius) {
  const result = [];
  for (const b of state.buildings) {
    if (Math.abs(px - b.x) < (b.w / 2 + radius) && Math.abs(pz - b.z) < (b.d / 2 + radius)) {
      result.push(b);
    }
  }
  return result;
}

export function snapOutOfBuilding(px, pz, bldg) {
  const hw = bldg.w / 2;
  const hd = bldg.d / 2;
  const penX = hw - Math.abs(px - bldg.x);
  const penZ = hd - Math.abs(pz - bldg.z);
  if (penX < penZ) {
    return { x: bldg.x + Math.sign(px - bldg.x) * (hw + 0.5), z: pz };
  } else {
    return { x: px, z: bldg.z + Math.sign(pz - bldg.z) * (hd + 0.5) };
  }
}

export function resolveCollision(oldX, oldZ, newX, newZ) {
  const nearby = nearbyBuildings(newX, newZ, 30);
  for (const b of nearby) {
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

export function planPedPath(from, to) {
  let bestS = null, bestE = null, dS = Infinity, dE = Infinity;
  for (const i of state.intersections) {
    const ds = dist2d(from, i), de = dist2d(to, i);
    if (ds < dS) { dS = ds; bestS = i; }
    if (de < dE) { dE = de; bestE = i; }
  }
  if (!bestS || !bestE) return [to];
  const hw = ROAD_W / 2 + 1.5;
  const ssx = Math.sign(from.x - bestS.x) || 1, ssz = Math.sign(from.z - bestS.z) || 1;
  const esx = Math.sign(to.x - bestE.x) || 1, esz = Math.sign(to.z - bestE.z) || 1;
  const cS = { x: bestS.x + ssx * hw, z: bestS.z + ssz * hw };
  const cE = { x: bestE.x + esx * hw, z: bestE.z + esz * hw };
  if (bestS === bestE) return [cS, cE, to];
  const cM = { x: bestE.x + esx * hw, z: bestS.z + ssz * hw };
  return [cS, cM, cE, to];
}

export function snapToNearestRoad(px, pz) {
  let best = null, bestDist = Infinity;
  for (const z of state.hRoads) {
    const d = Math.abs(pz - z);
    if (d < bestDist) { bestDist = d; best = { x: px, z, horizontal: true }; }
  }
  for (const x of state.vRoads) {
    const d = Math.abs(px - x);
    if (d < bestDist) { bestDist = d; best = { x, z: pz, horizontal: false }; }
  }
  return best;
}

export function planRoadPath(fromX, fromZ, toX, toZ) {
  const startRoad = snapToNearestRoad(fromX, fromZ);
  const endRoad = snapToNearestRoad(toX, toZ);
  const waypoints = [];
  waypoints.push({ x: startRoad.x, z: startRoad.z });
  let bestIntStart = null, bestDistStart = Infinity;
  for (const inter of state.intersections) {
    const d = dist2d(startRoad, inter);
    if (d < bestDistStart) { bestDistStart = d; bestIntStart = inter; }
  }
  if (bestIntStart) waypoints.push({ ...bestIntStart });
  let bestIntEnd = null, bestDistEnd = Infinity;
  for (const inter of state.intersections) {
    const d = dist2d(endRoad, inter);
    if (d < bestDistEnd) { bestDistEnd = d; bestIntEnd = inter; }
  }
  if (bestIntEnd && bestIntEnd !== bestIntStart) {
    waypoints.push({ x: bestIntEnd.x, z: bestIntStart.z });
    waypoints.push({ ...bestIntEnd });
  }
  waypoints.push({ x: endRoad.x, z: endRoad.z });
  waypoints.push({ x: toX, z: toZ });
  return waypoints;
}

export function curbNearBuilding(bldg) {
  let bestDist = Infinity, bestPos = null;
  for (const vx of state.vRoads) {
    const d = Math.abs(bldg.x - vx);
    if (d < bestDist) {
      bestDist = d;
      const side = bldg.x > vx ? 1 : -1;
      bestPos = { x: vx + side * (ROAD_W / 2 - 1.5), z: bldg.z };
    }
  }
  for (const hz of state.hRoads) {
    const d = Math.abs(bldg.z - hz);
    if (d < bestDist) {
      bestDist = d;
      const side = bldg.z > hz ? 1 : -1;
      bestPos = { x: bldg.x, z: hz + side * (ROAD_W / 2 - 1.5) };
    }
  }
  return bestPos || { x: bldg.x, z: bldg.z + (bldg.d || 10) / 2 + 3 };
}

export function doorPos(bldg) {
  const hw = (bldg.w || 10) / 2;
  const hd = (bldg.d || 10) / 2;
  let bestDist = Infinity, bestPos = { x: bldg.x, z: bldg.z + hd + 0.5 };
  for (const hz of state.hRoads) {
    const dPlus = Math.abs((bldg.z + hd) - hz);
    if (dPlus < bestDist) { bestDist = dPlus; bestPos = { x: bldg.x, z: bldg.z + hd + 0.5 }; }
    const dMinus = Math.abs((bldg.z - hd) - hz);
    if (dMinus < bestDist) { bestDist = dMinus; bestPos = { x: bldg.x, z: bldg.z - hd - 0.5 }; }
  }
  for (const vx of state.vRoads) {
    const dPlus = Math.abs((bldg.x + hw) - vx);
    if (dPlus < bestDist) { bestDist = dPlus; bestPos = { x: bldg.x + hw + 0.5, z: bldg.z }; }
    const dMinus = Math.abs((bldg.x - hw) - vx);
    if (dMinus < bestDist) { bestDist = dMinus; bestPos = { x: bldg.x - hw - 0.5, z: bldg.z }; }
  }
  return bestPos;
}

export function addKillFeedEntry(text) {
  state.killFeed.unshift({ text, age: 0 });
  if (state.killFeed.length > 10) state.killFeed.pop();
}

export function formatGrid(x, z) {
  const col = String.fromCharCode(65 + Math.min(7, Math.max(0, Math.floor((x / CITY_W) * 8))));
  const row = Math.min(6, Math.max(1, Math.floor((z / CITY_H) * 6) + 1));
  return col + row;
}

// =========================================================================
// Narration System
// =========================================================================
export const narrationTemplates = {
  riot_start: [
    '<span class="callsign">DISPATCH:</span> All units, Code 10-10 at Central Plaza. Riot in progress.',
    '<span class="callsign">DISPATCH:</span> 10-10 Central Plaza. Civil disturbance, all available units respond.',
    '<span class="callsign">COMMAND:</span> Declaring Code 10-10. Central Plaza. All units converge.',
  ],
  tear_gas: [
    '<span class="callsign">ALPHA LEAD:</span> Gas deployed at grid {grid}. Wind from northwest.',
    '<span class="callsign">BRAVO-2:</span> Tear gas away at {grid}. Masks on.',
    '<span class="callsign">ALPHA-1:</span> CS deployed grid {grid}. Crowd beginning to disperse.',
  ],
  molotov: [
    '<span class="callsign">ALPHA-3:</span> <span class="alert">CONTACT</span> Incendiary device thrown at our position!',
    '<span class="callsign">BRAVO-1:</span> <span class="alert">MOLOTOV</span> at police line! Taking fire!',
    '<span class="callsign">CHARLIE-2:</span> <span class="alert">FIREBOMB</span> inbound! Officers take cover!',
  ],
  officer_down: [
    '<span class="callsign">MEDIC:</span> <span class="alert">Officer down</span> at grid {grid}! Requesting immediate evac!',
    '<span class="callsign">ALPHA-4:</span> <span class="alert">Man down!</span> Need medic at grid {grid}!',
    '<span class="callsign">DISPATCH:</span> <span class="alert">10-999</span> officer needs assistance, grid {grid}.',
  ],
  fire_started: [
    '<span class="callsign">FIRE CONTROL:</span> Structure fire at grid {grid}. Dispatch engine.',
    '<span class="callsign">DISPATCH:</span> Fire reported grid {grid}. Engine company responding.',
    '<span class="callsign">BRAVO LEAD:</span> Fire at {grid}. Request fire suppression.',
  ],
  arrest: [
    '<span class="callsign">DISPATCH:</span> <span class="info">Suspect in custody.</span> {n} total arrests.',
    '<span class="callsign">ALPHA-2:</span> <span class="info">One in custody.</span> {n} total.',
    '<span class="callsign">BRAVO-3:</span> <span class="info">Apprehended subject.</span> Running total: {n}.',
  ],
  all_clear: [
    '<span class="callsign">COMMAND:</span> <span class="info">Situation stabilized.</span> Return to normal operations.',
    '<span class="callsign">DISPATCH:</span> <span class="info">All clear.</span> Stand down from Code 10-10.',
    '<span class="callsign">COMMAND:</span> <span class="info">Riot contained.</span> Resume patrol duties.',
  ],
  ambulance_dispatch: [
    '<span class="callsign">DISPATCH:</span> Ambulance en route to grid {grid}. ETA 2 minutes.',
    '<span class="callsign">MEDIC-1:</span> Responding to casualty at {grid}. Lights and sirens.',
  ],
  helicopter: [
    '<span class="callsign">AIR-1:</span> Overhead Central Plaza. Searchlight active. Eyes on crowd.',
    '<span class="callsign">AIR-1:</span> Orbiting plaza. Estimating {n} hostiles remaining.',
  ],
  van_deploy: [
    '<span class="callsign">DISPATCH:</span> Tactical unit deploying at grid {grid}. Reinforcements on scene.',
    '<span class="callsign">TAC-1:</span> Van on station. Deploying officers at {grid}.',
  ],
  fire_truck: [
    '<span class="callsign">ENGINE-1:</span> Responding to fire at grid {grid}. ETA 90 seconds.',
    '<span class="callsign">FIRE CONTROL:</span> Pumper en route to {grid}.',
  ],
  fire_extinguished: [
    '<span class="callsign">ENGINE-1:</span> <span class="info">Fire knocked down</span> at grid {grid}. Overhaul in progress.',
    '<span class="callsign">FIRE CONTROL:</span> <span class="info">Fire contained</span> at {grid}. All clear.',
  ],
  artillery_warning: [
    '<span class="callsign">COMMAND:</span> <span class="alert">FIRE MISSION INBOUND</span> grid {grid}. All units clear the area!',
    '<span class="callsign">FDC:</span> <span class="alert">SHOT OUT</span> — 3 rounds HE, grid {grid}. Splash in 3 seconds.',
  ],
  artillery_complete: [
    '<span class="callsign">COMMAND:</span> Fire mission complete. Grid {grid}. Assess damage.',
    '<span class="callsign">FDC:</span> Rounds complete grid {grid}. BDA requested.',
  ],
  ied_detonate: [
    '<span class="callsign">ALPHA-1:</span> <span class="alert">CONTACT IED</span> Grid {grid}! Casualties reported!',
    '<span class="callsign">DISPATCH:</span> <span class="alert">IED DETONATION</span> grid {grid}. All units hold position!',
  ],
  ied_detected: [
    '<span class="callsign">ROS2-{n}:</span> <span class="info">Suspicious device detected</span> grid {grid}. EOD requested.',
    '<span class="callsign">DISPATCH:</span> <span class="info">IED located</span> by robot at grid {grid}. Marking area.',
  ],
  ew_jam_start: [
    '<span class="callsign">COMMS:</span> <span class="alert">Signal interference detected!</span> Grid {grid} is being jammed!',
  ],
  ew_counter: [
    '<span class="callsign">EW TEAM:</span> <span class="info">Deploying counter-measures...</span>',
  ],
  building_breach: [
    '<span class="callsign">ALPHA TEAM:</span> <span class="alert">BREACH</span> Building at grid {grid}. Clearing in progress.',
    '<span class="callsign">BRAVO-1:</span> <span class="alert">FLASHBANG</span> deployed grid {grid}. Entry team moving.',
  ],
};

export function pickTemplate(type, vars) {
  const templates = narrationTemplates[type];
  if (!templates || templates.length === 0) return '';
  let msg = templates[Math.floor(rng() * templates.length)];
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      msg = msg.replace(new RegExp('\\{' + k + '\\}', 'g'), v);
    }
  }
  return msg;
}

export function addNarration(type, vars) {
  const html = pickTemplate(type, vars);
  if (!html) return;
  state.narrationMessages.push({ html, age: 0 });
  if (state.narrationMessages.length > 4) {
    state.narrationMessages.shift();
  }
}

// =========================================================================
// Spawn Pedestrians
// =========================================================================
export function spawnPedestrians() {
  for (let i = 0; i < NUM_PEDESTRIANS; i++) {
    const pos = randomSidewalkPos();
    const headColor = SKIN_TONES[Math.floor(rng() * SKIN_TONES.length)];
    const slot = allocPersonSlot();
    const dest = randomSidewalkPos();
    const wps = planPedPath(pos, dest);
    state.pedestrians.push({
      slot,
      x: pos.x, z: pos.z, y: 0, rotY: 0,
      bodyColor: COLOR_CIVILIAN,
      headColor,
      scale: 1.0,
      speed: 1.5 + rng() * 1.5,
      target: wps[0],
      waypoints: wps.slice(1),
      alive: true,
      fleeing: false,
    });
  }
}

// =========================================================================
// Injured Person Logic
// =========================================================================
// Note: showInjuredMarker is imported at top-level by the main HTML module
// and passed into knockDownPerson via the effects module reference.
// To avoid circular imports, knockDownPerson accesses it through the global registry.
export function knockDownPerson(person, showInjuredMarkerFn) {
  if (!person || !person.alive || person.onGround) return;
  person.onGround = true;
  person.injured = true;
  let marker = null;
  if (showInjuredMarkerFn) {
    marker = showInjuredMarkerFn(person.x, 0, person.z);
  }
  state.injuredOnGround.push({ person, marker, timer: 15 + rng() * 10, ambulanceDispatched: false });
  state.injuryCount++;
}
