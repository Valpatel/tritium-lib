// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * VehicleRenderer — Reusable InstancedMesh renderer for vehicles.
 *
 * Renders box-style vehicles (body + cabin + headlights + taillights)
 * using InstancedMesh. Supports brake lights, turn signals, headlight glow.
 *
 * Usage:
 *   const renderer = new VehicleRenderer(scene, 45);
 *   const slot = renderer.alloc();
 *   renderer.set(slot, x, 0, z, rotY, bodyColor, cabinColor);
 *   renderer.end();
 */

import * as THREE from 'three';

const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _color = new THREE.Color();
const _euler = new THREE.Euler();

export class VehicleRenderer {
  /**
   * @param {THREE.Scene} scene
   * @param {number} maxCount - Max vehicle instances
   */
  constructor(scene, maxCount = 45) {
    this.maxCount = maxCount;
    this._nextSlot = 0;
    this._dirty = false;

    // Body
    const bodyGeo = new THREE.BoxGeometry(2, 1.5, 4);
    const bodyMat = new THREE.MeshStandardMaterial({ metalness: 0.3, roughness: 0.6 });
    this.bodyMesh = new THREE.InstancedMesh(bodyGeo, bodyMat, maxCount);
    this.bodyMesh.castShadow = false;
    this.bodyMesh.count = 0;
    scene.add(this.bodyMesh);

    // Cabin
    const cabinGeo = new THREE.BoxGeometry(1.8, 1.0, 2.2);
    const cabinMat = new THREE.MeshStandardMaterial({ metalness: 0.2, roughness: 0.7 });
    this.cabinMesh = new THREE.InstancedMesh(cabinGeo, cabinMat, maxCount);
    this.cabinMesh.castShadow = false;
    this.cabinMesh.count = 0;
    scene.add(this.cabinMesh);

    // Lights
    const lightGeo = new THREE.BoxGeometry(0.3, 0.3, 0.5);
    const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffee });
    const tlMat = new THREE.MeshBasicMaterial({ color: 0xff2200, transparent: true, opacity: 0.7 });
    this.hlMesh = new THREE.InstancedMesh(lightGeo, hlMat, maxCount * 2);
    this.hlMesh.count = 0;
    scene.add(this.hlMesh);
    this.tlMesh = new THREE.InstancedMesh(lightGeo, tlMat, maxCount * 2);
    this.tlMesh.count = 0;
    scene.add(this.tlMesh);

    // Signal colors
    this._hlDefault = new THREE.Color(0xffffee);
    this._tlDefault = new THREE.Color(0xff2200);
    this._tlBrake = new THREE.Color(0xff0000);
    this._hlOrange = new THREE.Color(0xff8800);

    // Headlight point lights
    this.headlights = [];
    for (let i = 0; i < maxCount; i++) {
      const l1 = new THREE.PointLight(0xffffee, 0, 20);
      const l2 = new THREE.PointLight(0xffffee, 0, 20);
      scene.add(l1);
      scene.add(l2);
      this.headlights.push({ light1: l1, light2: l2 });
    }
  }

  /**
   * Allocate the next available vehicle slot.
   * @returns {number} Slot index, or -1 if at capacity
   */
  alloc() {
    if (this._nextSlot >= this.maxCount) return -1;
    const s = this._nextSlot++;
    this.bodyMesh.count = this._nextSlot;
    this.cabinMesh.count = this._nextSlot;
    this.hlMesh.count = this._nextSlot * 2;
    this.tlMesh.count = this._nextSlot * 2;
    this.hlMesh.setColorAt(s * 2, this._hlDefault);
    this.hlMesh.setColorAt(s * 2 + 1, this._hlDefault);
    this.tlMesh.setColorAt(s * 2, this._tlDefault);
    this.tlMesh.setColorAt(s * 2 + 1, this._tlDefault);
    return s;
  }

  /**
   * Set a vehicle instance's position and appearance.
   */
  set(idx, x, y, z, rotY, bodyColor, cabinColor, vehScale = 1) {
    const sc = vehScale;
    const cosR = Math.cos(rotY);
    const sinR = Math.sin(rotY);

    _euler.set(0, rotY, 0);
    _quat.setFromEuler(_euler);
    _scale.set(sc, sc, sc);

    // Body
    _pos.set(x, 0.75 * sc, z);
    _mat4.compose(_pos, _quat, _scale);
    this.bodyMesh.setMatrixAt(idx, _mat4);
    this.bodyMesh.setColorAt(idx, _color.set(bodyColor));

    // Cabin
    const cLocalZ = -0.3 * sc;
    _pos.set(x + cLocalZ * sinR, 1.7 * sc, z + cLocalZ * cosR);
    _mat4.compose(_pos, _quat, _scale);
    this.cabinMesh.setMatrixAt(idx, _mat4);
    this.cabinMesh.setColorAt(idx, _color.set(cabinColor));

    // Headlights
    const hlLocalZ = 2 * sc;
    for (let s = 0; s < 2; s++) {
      const hlLocalX = (s === 0 ? 0.7 : -0.7) * sc;
      const hlX = x + hlLocalX * cosR + hlLocalZ * sinR;
      const hlZ = z + hlLocalX * (-sinR) + hlLocalZ * cosR;
      _pos.set(hlX, 0.6 * sc, hlZ);
      _mat4.compose(_pos, _quat, _scale);
      this.hlMesh.setMatrixAt(idx * 2 + s, _mat4);
    }

    // Taillights
    const tlLocalZ = -2 * sc;
    for (let s = 0; s < 2; s++) {
      const tlLocalX = (s === 0 ? 0.7 : -0.7) * sc;
      const tlX = x + tlLocalX * cosR + tlLocalZ * sinR;
      const tlZ = z + tlLocalX * (-sinR) + tlLocalZ * cosR;
      _pos.set(tlX, 0.6 * sc, tlZ);
      _mat4.compose(_pos, _quat, _scale);
      this.tlMesh.setMatrixAt(idx * 2 + s, _mat4);
    }

    this._dirty = true;
  }

  /**
   * Hide a vehicle slot.
   */
  hide(idx) {
    _pos.set(0, -100, 0);
    _quat.identity();
    _scale.set(0, 0, 0);
    _mat4.compose(_pos, _quat, _scale);
    this.bodyMesh.setMatrixAt(idx, _mat4);
    this.cabinMesh.setMatrixAt(idx, _mat4);
    for (let s = 0; s < 2; s++) {
      this.hlMesh.setMatrixAt(idx * 2 + s, _mat4);
      this.tlMesh.setMatrixAt(idx * 2 + s, _mat4);
    }
    this._dirty = true;
  }

  /**
   * Set brake light state.
   */
  setBrake(idx, braking) {
    const c = braking ? this._tlBrake : this._tlDefault;
    this.tlMesh.setColorAt(idx * 2, c);
    this.tlMesh.setColorAt(idx * 2 + 1, c);
  }

  /**
   * Update headlight point lights for night driving.
   */
  updateHeadlights(idx, x, z, rotY, isNight) {
    const hl = this.headlights[idx];
    if (!hl) return;
    const intensity = isNight ? 6.0 : 0;
    hl.light1.intensity = intensity;
    hl.light2.intensity = intensity;
    if (isNight) {
      const cosR = Math.cos(rotY);
      const sinR = Math.sin(rotY);
      hl.light1.position.set(x + 2.5 * sinR + 0.7 * cosR, 0.8, z + 2.5 * cosR - 0.7 * sinR);
      hl.light2.position.set(x + 2.5 * sinR - 0.7 * cosR, 0.8, z + 2.5 * cosR + 0.7 * sinR);
    }
  }

  /**
   * Flush GPU buffers after all updates.
   */
  end() {
    if (!this._dirty) return;
    this.bodyMesh.instanceMatrix.needsUpdate = true;
    if (this.bodyMesh.instanceColor) this.bodyMesh.instanceColor.needsUpdate = true;
    this.cabinMesh.instanceMatrix.needsUpdate = true;
    if (this.cabinMesh.instanceColor) this.cabinMesh.instanceColor.needsUpdate = true;
    this.hlMesh.instanceMatrix.needsUpdate = true;
    this.tlMesh.instanceMatrix.needsUpdate = true;
    if (this.hlMesh.instanceColor) this.hlMesh.instanceColor.needsUpdate = true;
    if (this.tlMesh.instanceColor) this.tlMesh.instanceColor.needsUpdate = true;
    this._dirty = false;
  }

  dispose() {
    for (const mesh of [this.bodyMesh, this.cabinMesh, this.hlMesh, this.tlMesh]) {
      mesh.geometry.dispose();
      mesh.material.dispose();
      mesh.parent?.remove(mesh);
    }
    for (const hl of this.headlights) {
      hl.light1.parent?.remove(hl.light1);
      hl.light2.parent?.remove(hl.light2);
    }
  }
}
