/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Protestors, police, riot phases, escalation, molotovs, tear gas,
           helicopter, ambulances, fire trucks, police vans, medics.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  CITY_W, CITY_H, PLAZA_X, PLAZA_Z, MAGENTA, CYAN,
  NUM_PROTESTORS, NUM_POLICE, MAX_RINGS, MAX_SHIELDS,
  COLOR_PROTESTOR, COLOR_POLICE, SKIN_TONES, PERSON_SCALE,
  COST_TEAR_GAS, COST_RUBBER_BULLET, REWARD_ARREST, COST_FIRE, COST_INJURY,
  SUPPLY_TEAR_GAS_MAX, SUPPLY_RUBBER_BULLETS_MAX, SUPPLY_MOLOTOVS_MAX,
} from './config.js';
import {
  dist2d, hasLineOfSight, addKillFeedEntry, addNarration, formatGrid,
  allocPersonSlot, updatePersonInstance, hidePersonInstance,
  edgeSpawnPos, snapToNearestRoad, planRoadPath, resolveCollision,
  knockDownPerson,
} from './people.js';
import {
  allocCarSlot, vehicleRotY, updateCarInstance, hideCarInstance,
  updateCarHeadlights, checkVehicleCollision,
} from './vehicles.js';
import {
  spawnParticle, spawnBlood, createTracer, createMuzzleFlash, createExplosion,
  createHitEffect, launchProjectile, showInjuredMarker, hideInjuredMarker,
  moraleColor, healthTint, healthSpeedMul, updateBarricadeInstances,
} from './effects.js';
import { playGunshot, playHiss, playExplosion, playSiren, ensureAudio } from './audio.js';
import { awardAchievement, addStatusEffect } from './systems.js';

// =========================================================================
// Spawn Units
// =========================================================================
export function spawnProtestors() {
  for (let i = 0; i < NUM_PROTESTORS; i++) {
    const angle = rng() * Math.PI * 2;
    const radius = rng() * 20;
    const x = PLAZA_X + Math.cos(angle) * radius;
    const z = PLAZA_Z + Math.sin(angle) * radius;
    const headColor = SKIN_TONES[Math.floor(rng() * SKIN_TONES.length)];
    const slot = allocPersonSlot();
    let myRingIdx = -1;
    if (state.ringIdx < MAX_RINGS) {
      myRingIdx = state.ringIdx++;
      state.ringMesh.count = state.ringIdx;
    }
    state.protestors.push({
      slot, myRingIdx,
      x, z, y: 0, rotY: 0,
      bodyColor: COLOR_PROTESTOR, headColor, ringColor: MAGENTA,
      scale: 1.1, speed: 1.5 + rng() * 2,
      alive: true, health: 1.0, morale: 0.8,
      fleeing: false, throwTimer: 3 + rng() * 8,
      arrested: false, injured: false, stunnedTimer: 0,
      aggression: 0.0, panicFlash: 0,
      target: { x: PLAZA_X + (rng() - 0.5) * 40, z: PLAZA_Z + (rng() - 0.5) * 40 },
    });
  }
}

export function spawnPolice() {
  const lineZ = PLAZA_Z - 35;
  const lineStartX = PLAZA_X - 25;
  for (let i = 0; i < NUM_POLICE; i++) {
    const x = lineStartX + i * 5;
    const z = lineZ + (rng() - 0.5) * 3;
    const headColor = SKIN_TONES[Math.floor(rng() * SKIN_TONES.length)];
    const slot = allocPersonSlot();
    let myRingIdx = -1;
    if (state.ringIdx < MAX_RINGS) {
      myRingIdx = state.ringIdx++;
      state.ringMesh.count = state.ringIdx;
    }
    let myShieldIdx = -1;
    if (state.shieldMesh.count < MAX_SHIELDS) {
      myShieldIdx = state.shieldMesh.count++;
    }
    state.police.push({
      slot, myRingIdx, myShieldIdx,
      x, z, y: 0, rotY: 0,
      bodyColor: COLOR_POLICE, headColor, ringColor: CYAN,
      scale: 1.15, speed: 2 + rng() * 1.5,
      alive: true, health: 1.0, morale: 1.0,
      fireTimer: 2 + rng() * 4,
      target: null, holdLine: true,
      lineX: x, lineZ: lineZ,
      injured: false, gasAffected: false,
    });
  }
}

export function spawnBarricades() {
  const lineZ = PLAZA_Z - 35;
  const positions = [
    { x: PLAZA_X - 15, z: lineZ + 5, rotY: 0 },
    { x: PLAZA_X + 15, z: lineZ + 5, rotY: 0 },
    { x: PLAZA_X - 30, z: PLAZA_Z - 15, rotY: Math.PI / 2 },
    { x: PLAZA_X + 30, z: PLAZA_Z - 15, rotY: Math.PI / 2 },
  ];
  for (const pos of positions) {
    if (state.barricades.length >= 8) break;
    state.barricades.push({
      x: pos.x, z: pos.z, rotY: pos.rotY,
      health: 1.0, maxHealth: 1.0, active: true,
    });
  }
  updateBarricadeInstances();
}

// =========================================================================
// Helicopter
// =========================================================================
export function createHelicopter() {
  const scene = state.scene;
  const heliGroup = new THREE.Group();
  const bodyGeo = new THREE.BoxGeometry(3, 1.5, 2);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.6, metalness: 0.4 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  heliGroup.add(body);
  const tailGeo = new THREE.BoxGeometry(4, 0.5, 0.5);
  const tail = new THREE.Mesh(tailGeo, bodyMat);
  tail.position.set(-3, 0.2, 0);
  heliGroup.add(tail);
  const finGeo = new THREE.BoxGeometry(0.3, 1.2, 0.8);
  const fin = new THREE.Mesh(finGeo, bodyMat);
  fin.position.set(-4.8, 0.8, 0);
  heliGroup.add(fin);
  const rotorGeo = new THREE.BoxGeometry(8, 0.08, 0.4);
  const rotorMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.6 });
  const rotor = new THREE.Mesh(rotorGeo, rotorMat);
  rotor.position.y = 1.0;
  heliGroup.add(rotor);
  const rotor2 = new THREE.Mesh(rotorGeo, rotorMat);
  rotor2.position.y = 1.0;
  rotor2.rotation.y = Math.PI / 2;
  heliGroup.add(rotor2);
  const tailRotorGeo = new THREE.BoxGeometry(0.08, 1.5, 0.3);
  const tailRotor = new THREE.Mesh(tailRotorGeo, rotorMat);
  tailRotor.position.set(-4.8, 0.8, 0.5);
  heliGroup.add(tailRotor);
  const searchlight = new THREE.SpotLight(0xffffff, 15, 300, 0.06, 0.5, 0.8);
  searchlight.position.set(0, -0.3, 0);
  searchlight.castShadow = false;
  heliGroup.add(searchlight);
  const slTarget = new THREE.Object3D();
  slTarget.position.set(0, -80, 0);
  heliGroup.add(slTarget);
  searchlight.target = slTarget;
  const beamHeight = 80;
  const beamGeo = new THREE.CylinderGeometry(0.3, 4, beamHeight, 8, 1, true);
  const beamMat = new THREE.MeshBasicMaterial({
    color: 0xffffcc, transparent: true, opacity: 0.08,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const beam = new THREE.Mesh(beamGeo, beamMat);
  beam.position.y = -beamHeight / 2;
  heliGroup.add(beam);
  const splashGeo = new THREE.CircleGeometry(12, 24);
  const splashMat = new THREE.MeshBasicMaterial({
    color: 0xffffcc, transparent: true, opacity: 0.15,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const splash = new THREE.Mesh(splashGeo, splashMat);
  splash.rotation.x = -Math.PI / 2;
  scene.add(splash);
  const glowGeo = new THREE.SphereGeometry(0.3, 8, 6);
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xffffcc, transparent: true, opacity: 0.9 });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.set(0, -0.5, 0);
  heliGroup.add(glow);
  heliGroup.position.set(PLAZA_X, 80, PLAZA_Z);
  scene.add(heliGroup);
  state.helicopter = {
    group: heliGroup,
    rotor, rotor2, tailRotor,
    searchlight, slTarget, beam, splash,
    angle: 0, radius: 60,
    centerX: PLAZA_X, centerZ: PLAZA_Z,
    rotorAngle: 0,
    searchTarget: null, searchTimer: 0,
  };
}

// =========================================================================
// Tear Gas, Fire, Smoke helpers
// =========================================================================
export function createTearGasCloud(pos) {
  for (let i = 0; i < 4; i++) {
    const size = 0.8 + rng() * 1.0;
    spawnParticle(
      pos.x + (rng() - 0.5) * 6, 0.5 + rng() * 2, pos.z + (rng() - 0.5) * 6,
      (rng() - 0.5) * 0.8, 0.2 + rng() * 0.5, (rng() - 0.5) * 0.8,
      size, 0xcccccc, 3 + rng() * 2, true
    );
  }
  playHiss(pos.x, pos.z);
}

export function createFire(pos) {
  if (state.fires.length >= 6) return;
  state.policeBudget -= COST_FIRE;
  state.propertyDamage += COST_FIRE;
  const light = new THREE.PointLight(0xff6600, 5, 30);
  light.position.set(pos.x, 3, pos.z);
  state.scene.add(light);
  const totalLife = 20 + rng() * 10;
  state.fires.push({ light, life: totalLife, age: 0, x: pos.x, z: pos.z, maxIntensity: 5 });
  addNarration('fire_started', { grid: formatGrid(pos.x, pos.z) });
  for (let i = 0; i < 12; i++) {
    const fireColors = [0xff4400, 0xff8800, 0xffaa00, 0xff2200];
    spawnParticle(
      pos.x + (rng() - 0.5) * 3, 0.5 + rng() * 2, pos.z + (rng() - 0.5) * 3,
      (rng() - 0.5) * 1, 0.5 + rng() * 2, (rng() - 0.5) * 1,
      0.6 + rng() * 0.6,
      fireColors[Math.floor(rng() * fireColors.length)],
      2 + rng() * 3
    );
  }
}
