/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Particle pool, tracers, fire/smoke, artillery, morale colors, injured markers.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  MAX_PARTICLES, MAX_TRACERS, MAX_FLASHES, MAX_PROJECTILES,
  MAX_INJURED_MARKERS, MAX_BARRICADES,
  BODY_W, BODY_H, BODY_D, HEAD_SIZE, PERSON_SCALE,
  CITY_W, CITY_H, PLAZA_X, PLAZA_Z,
  COST_FIRE, MAX_IEDS,
} from './config.js';

// =========================================================================
// Morale / Health Color Helpers
// =========================================================================
const _moraleColorA = new THREE.Color();
const _moraleColorB = new THREE.Color();

export function moraleColor(baseColor, morale) {
  _moraleColorA.set(baseColor);
  if (morale >= 0.7) return baseColor;
  if (morale >= 0.3) {
    _moraleColorB.set(0xcccc00);
    _moraleColorA.lerp(_moraleColorB, 1.0 - (morale - 0.3) / 0.4);
    return _moraleColorA.getHex();
  }
  _moraleColorA.set(0xcccc00);
  _moraleColorB.set(0xcc2222);
  _moraleColorA.lerp(_moraleColorB, 1.0 - morale / 0.3);
  return _moraleColorA.getHex();
}

export function lerpMoraleColor(baseColor, morale) { return moraleColor(baseColor, morale); }

const _healthColorA = new THREE.Color();
const _healthColorB = new THREE.Color();

export function healthTint(displayColor, health) {
  if (health > 0.7) return displayColor;
  _healthColorA.set(displayColor);
  if (health > 0.5) { _healthColorA.multiplyScalar(0.7 + health * 0.3); return _healthColorA.getHex(); }
  if (health > 0.3) { _healthColorB.set(0xff8800); _healthColorA.lerp(_healthColorB, (0.5 - health) / 0.2 * 0.4); return _healthColorA.getHex(); }
  _healthColorB.set(0xcc0000); _healthColorA.lerp(_healthColorB, (0.3 - health) / 0.3 * 0.6); return _healthColorA.getHex();
}

export function healthSpeedMul(health) { return health > 0.5 ? 1.0 : health > 0.3 ? 0.6 : 0.4; }

// =========================================================================
// Particle System
// =========================================================================
export function initParticles() {
  const scene = state.scene;

  const particleGeo = new THREE.BoxGeometry(1, 1, 1);
  const particleMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.3, depthWrite: false,
    blending: THREE.NormalBlending,
  });
  const particleMesh = new THREE.InstancedMesh(particleGeo, particleMat, MAX_PARTICLES);
  particleMesh.count = 0;
  particleMesh.renderOrder = 10;
  scene.add(particleMesh);
  state.particleMesh = particleMesh;

  // Pre-allocate particle pool
  for (let i = 0; i < MAX_PARTICLES; i++) {
    state.particlePool.push(i);
    _pos.set(0, -200, 0);
    _quat.identity();
    _scale.set(0, 0, 0);
    _mat4.compose(_pos, _quat, _scale);
    particleMesh.setMatrixAt(i, _mat4);
    particleMesh.setColorAt(i, _color.set(0x000000));
  }
  particleMesh.instanceMatrix.needsUpdate = true;
  if (particleMesh.instanceColor) particleMesh.instanceColor.needsUpdate = true;
}

export function spawnParticle(x, y, z, vx, vy, vz, size, color, life, isTearGas = false, isSmoke = false) {
  if (state.particlePool.length === 0) return null;
  const slot = state.particlePool.pop();
  const p = {
    slot, x, y, z, vx, vy, vz, size, color, life, maxLife: life,
    isTearGas, isSmoke, rotX: rng() * Math.PI, rotZ: rng() * Math.PI,
    growRate: isTearGas ? 0.15 : (isSmoke ? 0.08 : 0.3),
  };
  state.activeParticles.push(p);
  return p;
}

export function spawnBlood(x, z) {
  for (let i = 0; i < 2; i++) spawnParticle(x, 1.2, z, (rng()-0.5)*2, 1+rng()*2, (rng()-0.5)*2, 0.2, 0xaa0000, 0.5);
}

export function updateParticles(dt) {
  const particleMesh = state.particleMesh;
  let needsUpdate = false;
  for (let i = state.activeParticles.length - 1; i >= 0; i--) {
    const p = state.activeParticles[i];
    p.life -= dt;
    if (p.life <= 0) {
      _pos.set(0, -200, 0);
      _quat.identity();
      _scale.set(0, 0, 0);
      _mat4.compose(_pos, _quat, _scale);
      particleMesh.setMatrixAt(p.slot, _mat4);
      state.particlePool.push(p.slot);
      state.activeParticles.splice(i, 1);
      needsUpdate = true;
      continue;
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
    p.rotX += dt * 3;
    p.rotZ += dt * 2;
    if (p.isTearGas) {
      p.size *= (1 + p.growRate * dt);
      p.vx += (rng() - 0.5) * dt * 2;
      p.vz += (rng() - 0.5) * dt * 2;
    } else {
      p.size *= (1 + p.growRate * dt);
    }
    const alpha = Math.max(0, p.life / p.maxLife);
    const fadeScale = p.isTearGas ? Math.sqrt(alpha) : (p.isSmoke ? alpha * alpha : alpha);
    _euler.set(p.rotX, 0, p.rotZ);
    _quat.setFromEuler(_euler);
    _pos.set(p.x, p.y, p.z);
    const sz = p.size * fadeScale;
    _scale.set(sz, sz, sz);
    _mat4.compose(_pos, _quat, _scale);
    particleMesh.setMatrixAt(p.slot, _mat4);
    particleMesh.setColorAt(p.slot, _color.set(p.color));
    needsUpdate = true;
  }
  let maxSlot = 0;
  for (const p of state.activeParticles) {
    if (p.slot >= maxSlot) maxSlot = p.slot + 1;
  }
  particleMesh.count = maxSlot || 0;
  if (needsUpdate) {
    particleMesh.instanceMatrix.needsUpdate = true;
    if (particleMesh.instanceColor) particleMesh.instanceColor.needsUpdate = true;
  }
}

// =========================================================================
// Tracers, Flashes, Projectiles
// =========================================================================
export function initTracers() {
  const scene = state.scene;
  const tracerMat = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2, transparent: true, opacity: 1, depthWrite: false });
  for (let i = 0; i < MAX_TRACERS; i++) {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(6);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const line = new THREE.Line(geo, tracerMat.clone());
    line.visible = false;
    line.frustumCulled = false;
    scene.add(line);
    state.tracerPool.push(line);
  }

  for (let i = 0; i < MAX_FLASHES; i++) {
    const light = new THREE.PointLight(0xffaa00, 0, 20);
    light.visible = false;
    scene.add(light);
    state.flashPool.push(light);
  }

  const projGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
  state.projMats = {
    molotov: new THREE.MeshBasicMaterial({ color: 0xff6600 }),
    rock: new THREE.MeshBasicMaterial({ color: 0x888888 }),
  };
  for (let i = 0; i < MAX_PROJECTILES; i++) {
    const mesh = new THREE.Mesh(projGeo, state.projMats.rock);
    mesh.visible = false;
    mesh.frustumCulled = false;
    scene.add(mesh);
    state.projPool.push(mesh);
  }
}

export function createTracer(from, to, color = 0xffff00, duration = 0.2) {
  if (state.tracers.length >= MAX_TRACERS) return;
  const line = state.tracerPool.find(l => !l.visible);
  if (!line) return;
  const pos = line.geometry.attributes.position.array;
  pos[0] = from.x; pos[1] = 2; pos[2] = from.z;
  pos[3] = to.x; pos[4] = 2; pos[5] = to.z;
  line.geometry.attributes.position.needsUpdate = true;
  line.material.color.setHex(color);
  line.material.opacity = 1;
  line.visible = true;
  state.tracers.push({ line, life: duration, maxLife: duration });
}

export function createMuzzleFlash(pos) {
  if (state.muzzleFlashes.length >= MAX_FLASHES) return;
  const light = state.flashPool.find(l => !l.visible);
  if (!light) return;
  light.position.set(pos.x, 2.5, pos.z);
  light.intensity = 5;
  light.visible = true;
  state.muzzleFlashes.push({ light, life: 0.05 });
}

export function createExplosion(pos) {
  const fireColors = [0xff4400, 0xff8800, 0xffaa00, 0x666666];
  for (let i = 0; i < 18; i++) {
    const size = 0.3 + rng() * 0.6;
    const color = fireColors[Math.floor(rng() * fireColors.length)];
    spawnParticle(
      pos.x, 2, pos.z,
      (rng() - 0.5) * 25, 5 + rng() * 18, (rng() - 0.5) * 25,
      size, color, 1.0 + rng() * 0.5
    );
  }
  const exLight = state.flashPool.find(l => !l.visible);
  if (exLight) {
    exLight.color.setHex(0xff6600);
    exLight.intensity = 8;
    exLight.distance = 50;
    exLight.position.set(pos.x, 4, pos.z);
    exLight.visible = true;
    state.muzzleFlashes.push({ light: exLight, life: 0.3 });
  }
}

export function createHitEffect(pos) {
  const light = new THREE.PointLight(0xff0000, 5, 12);
  light.position.set(pos.x, 2, pos.z);
  state.scene.add(light);
  state.muzzleFlashes.push({ light, life: 0.1 });
}

export function launchProjectile(from, to, type) {
  if (state.projectiles.length >= MAX_PROJECTILES) return;
  const mesh = state.projPool.find(m => !m.visible);
  if (!mesh) return;
  mesh.material = state.projMats[type] || state.projMats.rock;
  mesh.scale.set(type === 'molotov' ? 1.2 : 0.8, type === 'molotov' ? 1.6 : 0.8, type === 'molotov' ? 1.2 : 0.8);
  mesh.position.set(from.x, 2, from.z);
  mesh.visible = true;
  const dx = to.x - from.x;
  const dz = to.z - from.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  const flightTime = Math.max(0.3, dist / 30);
  const vy = dist * 0.15 + 5;
  state.projectiles.push({
    mesh, type,
    startX: from.x, startZ: from.z,
    vx: dx / flightTime, vz: dz / flightTime,
    vy, gravity: vy * 2 / flightTime,
    age: 0, flightTime,
  });
}

// =========================================================================
// Injured Cross Markers
// =========================================================================
export function initInjuredMarkers() {
  const scene = state.scene;
  const crossHGeo = new THREE.PlaneGeometry(1.5, 0.5);
  const crossVGeo = new THREE.PlaneGeometry(0.5, 1.5);
  for (let i = 0; i < MAX_INJURED_MARKERS; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide, depthTest: false });
    const h = new THREE.Mesh(crossHGeo, mat);
    const v = new THREE.Mesh(crossVGeo, mat);
    h.visible = false;
    v.visible = false;
    h.renderOrder = 999;
    v.renderOrder = 999;
    scene.add(h);
    scene.add(v);
    state.injuredMarkers.push({ crossH: h, crossV: v, active: false });
  }
}

export function showInjuredMarker(x, y, z) {
  for (const m of state.injuredMarkers) {
    if (!m.active) {
      m.active = true;
      m.crossH.visible = true;
      m.crossV.visible = true;
      m.crossH.position.set(x, y + 5, z);
      m.crossV.position.set(x, y + 5, z);
      return m;
    }
  }
  return null;
}

export function hideInjuredMarker(marker) {
  if (!marker) return;
  marker.active = false;
  marker.crossH.visible = false;
  marker.crossV.visible = false;
}

// =========================================================================
// Barricades
// =========================================================================
export function initBarricades() {
  const scene = state.scene;
  const barricadeGeo = new THREE.BoxGeometry(6, 1.5, 1);
  const barricadeMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.1 });
  const barricadeMesh = new THREE.InstancedMesh(barricadeGeo, barricadeMat, MAX_BARRICADES);
  barricadeMesh.castShadow = true;
  barricadeMesh.receiveShadow = true;
  barricadeMesh.count = 0;
  scene.add(barricadeMesh);
  state.barricadeMesh = barricadeMesh;
}

export function updateBarricadeInstances() {
  let count = 0;
  for (let i = 0; i < state.barricades.length; i++) {
    const b = state.barricades[i];
    if (!b.active) {
      _pos.set(0, -100, 0);
      _quat.identity();
      _scale.set(0, 0, 0);
      _mat4.compose(_pos, _quat, _scale);
      state.barricadeMesh.setMatrixAt(i, _mat4);
      state.barricadeMesh.setColorAt(i, _color.set(0x000000));
    } else {
      _pos.set(b.x, 0.75, b.z);
      _euler.set(0, b.rotY, 0);
      _quat.setFromEuler(_euler);
      _scale.set(1, 1, 1);
      _mat4.compose(_pos, _quat, _scale);
      state.barricadeMesh.setMatrixAt(i, _mat4);
      const r = 0.1 + (1.0 - b.health) * 0.8;
      const g = 0.1 * b.health;
      const bl = 0.3 * b.health;
      _color.setRGB(r, g, bl);
      state.barricadeMesh.setColorAt(i, _color);
    }
    count = i + 1;
  }
  state.barricadeMesh.count = count;
  state.barricadeMesh.instanceMatrix.needsUpdate = true;
  if (state.barricadeMesh.instanceColor) state.barricadeMesh.instanceColor.needsUpdate = true;
}

export function isBlockedByBarricade(px, pz, threshold) {
  for (const b of state.barricades) {
    if (!b.active) continue;
    const dx = px - b.x;
    const dz = pz - b.z;
    if (Math.sqrt(dx * dx + dz * dz) < threshold) return true;
  }
  return false;
}
