// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * EffectsRenderer — Reusable particle pool and visual effects.
 *
 * Manages instanced particle system (fire, smoke, tear gas, debris),
 * pooled tracer lines, muzzle flash lights, and projectile meshes.
 */

import * as THREE from 'three';

const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _color = new THREE.Color();
const _euler = new THREE.Euler();

export class ParticlePool {
  /**
   * @param {THREE.Scene} scene
   * @param {number} maxParticles
   */
  constructor(scene, maxParticles = 500) {
    this.maxParticles = maxParticles;
    this.pool = [];       // available slot indices
    this.active = [];     // active particle objects

    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshBasicMaterial({
      transparent: true, opacity: 0.3, depthWrite: false,
      blending: THREE.NormalBlending,
    });
    this.mesh = new THREE.InstancedMesh(geo, mat, maxParticles);
    this.mesh.count = 0;
    this.mesh.renderOrder = 10;
    scene.add(this.mesh);

    // Pre-allocate all slots
    for (let i = 0; i < maxParticles; i++) {
      this.pool.push(i);
      _pos.set(0, -200, 0);
      _quat.identity();
      _scale.set(0, 0, 0);
      _mat4.compose(_pos, _quat, _scale);
      this.mesh.setMatrixAt(i, _mat4);
      this.mesh.setColorAt(i, _color.set(0x000000));
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  /**
   * Spawn a particle.
   */
  spawn(x, y, z, vx, vy, vz, size, color, life, isTearGas = false, isSmoke = false) {
    if (this.pool.length === 0) return null;
    const slot = this.pool.pop();
    const p = {
      slot, x, y, z, vx, vy, vz, size, color, life, maxLife: life,
      isTearGas, isSmoke,
      rotX: Math.random() * Math.PI,
      rotZ: Math.random() * Math.PI,
      growRate: isTearGas ? 0.15 : (isSmoke ? 0.08 : 0.3),
    };
    this.active.push(p);
    return p;
  }

  /**
   * Update all active particles. Call once per sim tick.
   */
  update(dt) {
    let needsUpdate = false;
    for (let i = this.active.length - 1; i >= 0; i--) {
      const p = this.active[i];
      p.life -= dt;
      if (p.life <= 0) {
        _pos.set(0, -200, 0);
        _quat.identity();
        _scale.set(0, 0, 0);
        _mat4.compose(_pos, _quat, _scale);
        this.mesh.setMatrixAt(p.slot, _mat4);
        this.pool.push(p.slot);
        this.active.splice(i, 1);
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
        p.vx += (Math.random() - 0.5) * dt * 2;
        p.vz += (Math.random() - 0.5) * dt * 2;
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
      this.mesh.setMatrixAt(p.slot, _mat4);
      this.mesh.setColorAt(p.slot, _color.set(p.color));
      needsUpdate = true;
    }
    // Update count to highest active slot
    let maxSlot = 0;
    for (const p of this.active) {
      if (p.slot >= maxSlot) maxSlot = p.slot + 1;
    }
    this.mesh.count = maxSlot || 0;
    if (needsUpdate) {
      this.mesh.instanceMatrix.needsUpdate = true;
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    }
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh.parent?.remove(this.mesh);
  }
}

/**
 * Morale color helper — tints body color based on morale level.
 * Pure function, no Three.js dependency needed in sim layer.
 */
export function moraleColorHex(baseColor, morale) {
  if (morale >= 0.7) return baseColor;
  // Lerp toward yellow then red as morale drops
  const br = (baseColor >> 16) & 0xff, bg = (baseColor >> 8) & 0xff, bb = baseColor & 0xff;
  if (morale >= 0.3) {
    const t = 1.0 - (morale - 0.3) / 0.4;
    return lerpRGB(br, bg, bb, 0xcc, 0xcc, 0x00, t);
  }
  const t = 1.0 - morale / 0.3;
  return lerpRGB(0xcc, 0xcc, 0x00, 0xcc, 0x22, 0x22, t);
}

function lerpRGB(r1, g1, b1, r2, g2, b2, t) {
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return (r << 16) | (g << 8) | b;
}
