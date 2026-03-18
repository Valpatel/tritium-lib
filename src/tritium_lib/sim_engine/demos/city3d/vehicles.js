/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Cars, taxis, ambulances, fire trucks, police vans, helicopter,
           car instance management, collision detection, road pathfinding.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  CITY_W, CITY_H, BLOCK_W, BLOCK_H, GRID_COLS, GRID_ROWS,
  ROAD_W, PLAZA_X, PLAZA_Z,
  MAX_CARS_INST, LANE_OFFSET, TRAFFIC_CYCLE,
  NUM_CARS, NUM_TAXIS, NUM_ROBOT_CARS,
  LIDAR_RAYS, LIDAR_RANGE, ROBOT_TRAIL_LENGTH,
  COLOR_CIVILIAN, COLOR_POLICE, SKIN_TONES, CYAN,
  MAX_RINGS, MAX_SHIELDS,
} from './config.js';
import { dist2d, snapToNearestRoad, planRoadPath, edgeSpawnPos, addKillFeedEntry, addNarration, formatGrid, allocPersonSlot } from './people.js';
import { isBlockedByBarricade } from './effects.js';

// =========================================================================
// Car InstancedMesh Init
// =========================================================================
export function initCars() {
  const scene = state.scene;

  const carBodyGeo = new THREE.BoxGeometry(2, 1.5, 4);
  const carBodyMat = new THREE.MeshStandardMaterial({ metalness: 0.3, roughness: 0.6 });
  const carBodyMesh = new THREE.InstancedMesh(carBodyGeo, carBodyMat, MAX_CARS_INST);
  carBodyMesh.castShadow = false;
  carBodyMesh.count = 0;
  scene.add(carBodyMesh);
  state.carBodyMesh = carBodyMesh;

  const carCabinGeo = new THREE.BoxGeometry(1.8, 1.0, 2.2);
  const carCabinMat = new THREE.MeshStandardMaterial({ metalness: 0.2, roughness: 0.7 });
  const carCabinMesh = new THREE.InstancedMesh(carCabinGeo, carCabinMat, MAX_CARS_INST);
  carCabinMesh.castShadow = false;
  carCabinMesh.count = 0;
  scene.add(carCabinMesh);
  state.carCabinMesh = carCabinMesh;

  const carLightGeo = new THREE.BoxGeometry(0.3, 0.3, 0.5);
  const carHLMat = new THREE.MeshBasicMaterial({ color: 0xffffee });
  const carTLMat = new THREE.MeshBasicMaterial({ color: 0xff2200, transparent: true, opacity: 0.7 });
  const carHLMesh = new THREE.InstancedMesh(carLightGeo, carHLMat, MAX_CARS_INST * 2);
  carHLMesh.count = 0;
  scene.add(carHLMesh);
  const carTLMesh = new THREE.InstancedMesh(carLightGeo, carTLMat, MAX_CARS_INST * 2);
  carTLMesh.count = 0;
  scene.add(carTLMesh);
  state.carHLMesh = carHLMesh;
  state.carTLMesh = carTLMesh;

  // Signal colors
  state._hlDefault = new THREE.Color(0xffffee);
  state._tlDefault = new THREE.Color(0xff2200);
  state._tlBrake  = new THREE.Color(0xff0000);
  state._hlOrange = new THREE.Color(0xff8800);
  state._tlColor = new THREE.Color();

  // Car headlight PointLights
  for (let i = 0; i < MAX_CARS_INST; i++) {
    const l1 = new THREE.PointLight(0xffffee, 0, 20);
    const l2 = new THREE.PointLight(0xffffee, 0, 20);
    scene.add(l1);
    scene.add(l2);
    state.carHeadlights.push({ light1: l1, light2: l2 });
  }
}

export function allocCarSlot() {
  if (state.nextCarSlot >= MAX_CARS_INST) return -1;
  const s = state.nextCarSlot;
  state.nextCarSlot++;
  state.carBodyMesh.count = state.nextCarSlot;
  state.carCabinMesh.count = state.nextCarSlot;
  state.carHLMesh.count = state.nextCarSlot * 2;
  state.carTLMesh.count = state.nextCarSlot * 2;
  state.carHLMesh.setColorAt(s * 2, state._hlDefault);
  state.carHLMesh.setColorAt(s * 2 + 1, state._hlDefault);
  state.carTLMesh.setColorAt(s * 2, state._tlDefault);
  state.carTLMesh.setColorAt(s * 2 + 1, state._tlDefault);
  return s;
}

export function vehicleRotY(dx, dz) {
  return -Math.atan2(dx, dz);
}

export function vehicleRotYFromDir(horizontal, dir) {
  if (horizontal) {
    return dir > 0 ? -Math.PI / 2 : Math.PI / 2;
  }
  return dir > 0 ? 0 : Math.PI;
}

export function updateCarInstance(idx, x, y, z, rotY, bodyColor, cabinColor, vehScale) {
  const sc = vehScale || 1;
  const cosR = Math.cos(rotY);
  const sinR = Math.sin(rotY);
  _euler.set(0, rotY, 0);
  _quat.setFromEuler(_euler);
  _pos.set(x, 0.75 * sc, z);
  _scale.set(sc, sc, sc);
  _mat4.compose(_pos, _quat, _scale);
  state.carBodyMesh.setMatrixAt(idx, _mat4);
  state.carBodyMesh.setColorAt(idx, _color.set(bodyColor));
  const cLocalZ = -0.3 * sc;
  const cWorldX = x + cLocalZ * sinR;
  const cWorldZ = z + cLocalZ * cosR;
  _pos.set(cWorldX, 1.7 * sc, cWorldZ);
  _mat4.compose(_pos, _quat, _scale);
  state.carCabinMesh.setMatrixAt(idx, _mat4);
  state.carCabinMesh.setColorAt(idx, _color.set(cabinColor));
  const hlLocalZ = 2 * sc;
  _scale.set(sc, sc, sc);
  for (let s = 0; s < 2; s++) {
    const hlLocalX = (s === 0 ? 0.7 : -0.7) * sc;
    const hlX = x + hlLocalX * cosR + hlLocalZ * sinR;
    const hlZ = z + hlLocalX * (-sinR) + hlLocalZ * cosR;
    _pos.set(hlX, 0.6 * sc, hlZ);
    _mat4.compose(_pos, _quat, _scale);
    state.carHLMesh.setMatrixAt(idx * 2 + s, _mat4);
  }
  const tlLocalZ = -2 * sc;
  for (let s = 0; s < 2; s++) {
    const tlLocalX = (s === 0 ? 0.7 : -0.7) * sc;
    const tlX = x + tlLocalX * cosR + tlLocalZ * sinR;
    const tlZ = z + tlLocalX * (-sinR) + tlLocalZ * cosR;
    _pos.set(tlX, 0.6 * sc, tlZ);
    _mat4.compose(_pos, _quat, _scale);
    state.carTLMesh.setMatrixAt(idx * 2 + s, _mat4);
  }
}

export function hideCarInstance(idx) {
  _pos.set(0, -100, 0);
  _quat.identity();
  _scale.set(0, 0, 0);
  _mat4.compose(_pos, _quat, _scale);
  state.carBodyMesh.setMatrixAt(idx, _mat4);
  state.carCabinMesh.setMatrixAt(idx, _mat4);
  for (let s = 0; s < 2; s++) {
    state.carHLMesh.setMatrixAt(idx * 2 + s, _mat4);
    state.carTLMesh.setMatrixAt(idx * 2 + s, _mat4);
  }
}

export function updateCarHeadlights(idx, x, z, rotY, isNight) {
  const hl = state.carHeadlights[idx];
  if (!hl) return;
  const intensity = isNight ? 6.0 : 0;
  hl.light1.intensity = intensity;
  hl.light2.intensity = intensity;
  if (isNight) {
    const cosR = Math.cos(rotY);
    const sinR = Math.sin(rotY);
    const hlLocalX = 2.5;
    const hlX1 = x + hlLocalX * sinR + 0.7 * cosR;
    const hlZ1 = z + hlLocalX * cosR - 0.7 * sinR;
    const hlX2 = x + hlLocalX * sinR - 0.7 * cosR;
    const hlZ2 = z + hlLocalX * cosR + 0.7 * sinR;
    hl.light1.position.set(hlX1, 0.8, hlZ1);
    hl.light2.position.set(hlX2, 0.8, hlZ2);
  }
}

// =========================================================================
// Traffic Lights
// =========================================================================
export function isGreen(intersection, horizontal) {
  return (Math.floor(state.trafficPhase / TRAFFIC_CYCLE) % 2 === 0) === horizontal;
}

export function initTrafficLights() {
  const scene = state.scene;
  const tlGeo = new THREE.BoxGeometry(0.5, 2.5, 0.5);
  const tlMat = new THREE.MeshStandardMaterial({ emissive: 0x00ff00, emissiveIntensity: 0.8, roughness: 0.5 });
  const trafficLightMesh = new THREE.InstancedMesh(tlGeo, tlMat, state.intersections.length);
  trafficLightMesh.count = state.intersections.length;
  scene.add(trafficLightMesh);
  state.trafficLightMesh = trafficLightMesh;
  state._tlColor = new THREE.Color();
  for (let i = 0; i < state.intersections.length; i++) {
    _pos.set(state.intersections[i].x + ROAD_W / 2, 2.5, state.intersections[i].z + ROAD_W / 2);
    _quat.identity(); _scale.set(1, 1, 1);
    _mat4.compose(_pos, _quat, _scale);
    trafficLightMesh.setMatrixAt(i, _mat4);
    trafficLightMesh.setColorAt(i, state._tlColor.set(0x00ff00));
  }
  trafficLightMesh.instanceMatrix.needsUpdate = true;
  if (trafficLightMesh.instanceColor) trafficLightMesh.instanceColor.needsUpdate = true;
  for (let i = 0; i < 4; i++) {
    const pl = new THREE.PointLight(0x00ff00, 0.6, 15);
    pl.position.set(0, -200, 0);
    scene.add(pl);
    state.tlLights.push(pl);
  }
}

// =========================================================================
// Vehicle Collision
// =========================================================================
export function collectAllRoadVehicles() {
  const list = [];
  for (const c of state.cars) {
    if (c.alive) list.push({ x: c.x, z: c.z, dir: c.dir, horizontal: c.horizontal, speed: c.speed, paused: c.pauseTimer > 0, ref: c });
  }
  for (const t of state.taxis) {
    if (t.alive) list.push({ x: t.x, z: t.z, dir: t.dir, horizontal: t.horizontal, speed: t.speed, paused: t.pauseTimer > 0, ref: t });
  }
  for (const cd of state.carDrivers) {
    if (cd.alive && cd.carMoving) list.push({ x: cd.x, z: cd.z, dir: 0, horizontal: true, speed: 8, paused: false, ref: cd });
  }
  for (const a of state.ambulances) {
    if (a.alive) list.push({ x: a.x, z: a.z, dir: 0, horizontal: true, speed: a.speed, paused: false, ref: a });
  }
  for (const ft of state.fireTrucks) {
    if (ft.alive) list.push({ x: ft.x, z: ft.z, dir: 0, horizontal: true, speed: ft.speed, paused: false, ref: ft });
  }
  for (const pv of state.policeVans) {
    if (pv.alive) list.push({ x: pv.x, z: pv.z, dir: 0, horizontal: true, speed: pv.speed, paused: false, ref: pv });
  }
  state._allRoadVehicles = list;
}

export function checkVehicleCollision(vehRef, vehX, vehZ, horizontal, dir) {
  let minDist = Infinity;
  let minAhead = Infinity;
  for (const other of state._allRoadVehicles) {
    if (other.ref === vehRef) continue;
    const odx = other.x - vehX;
    const odz = other.z - vehZ;
    const dist = Math.sqrt(odx * odx + odz * odz);
    if (dist < minDist) minDist = dist;
    if (horizontal) {
      if (Math.abs(odz) > 2.5) continue;
      const ahead = odx * dir;
      if (ahead > 0 && ahead < minAhead) minAhead = ahead;
    } else {
      if (Math.abs(odx) > 2.5) continue;
      const ahead = odz * dir;
      if (ahead > 0 && ahead < minAhead) minAhead = ahead;
    }
  }
  if (minDist < 4) return 0;
  if (minAhead < 5) return 0.05;
  if (minAhead < 8) return 0.3;
  return 1.0;
}

export function findNextIntersection(x, z, horizontal, dir) {
  if (horizontal) {
    if (dir > 0) {
      for (const vx of state.vRoads) {
        if (vx > x + 1) return { x: vx, z };
      }
      return null;
    } else {
      for (let i = state.vRoads.length - 1; i >= 0; i--) {
        if (state.vRoads[i] < x - 1) return { x: state.vRoads[i], z };
      }
      return null;
    }
  } else {
    if (dir > 0) {
      for (const hz of state.hRoads) {
        if (hz > z + 1) return { x, z: hz };
      }
      return null;
    } else {
      for (let i = state.hRoads.length - 1; i >= 0; i--) {
        if (state.hRoads[i] < z - 1) return { x, z: state.hRoads[i] };
      }
      return null;
    }
  }
}

export function pickRandomRoadStart() {
  if (rng() < 0.5) {
    const row = Math.floor(rng() * (GRID_ROWS + 1));
    const roadZ = state.hRoads[row];
    const col = Math.floor(rng() * GRID_COLS);
    const x = state.vRoads[col] + rng() * (BLOCK_W + ROAD_W);
    const dir = rng() < 0.5 ? 1 : -1;
    const z = roadZ + dir * LANE_OFFSET;
    return { x, z, horizontal: true, dir };
  } else {
    const col = Math.floor(rng() * (GRID_COLS + 1));
    const roadX = state.vRoads[col];
    const row = Math.floor(rng() * GRID_ROWS);
    const z = state.hRoads[row] + rng() * (BLOCK_H + ROAD_W);
    const dir = rng() < 0.5 ? 1 : -1;
    const x = roadX - dir * LANE_OFFSET;
    return { x, z, horizontal: false, dir };
  }
}
