// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * PeopleRenderer — Reusable InstancedMesh renderer for humanoid characters.
 *
 * Renders Minecraft-style characters (body box + head box) using InstancedMesh
 * for performance. Can be used by city3d, war3d, or any 3D view that needs people.
 *
 * Usage:
 *   const renderer = new PeopleRenderer(scene, 180);
 *   // Each frame:
 *   renderer.begin();
 *   renderer.set(0, x, y, z, rotY, bodyColor, headColor, scale);
 *   renderer.set(1, ...);
 *   renderer.end(slotCount);
 */

import * as THREE from 'three';

// Person dimensions (Minecraft-style, visible from 200m orbit camera)
const BODY_W = 1.0, BODY_H = 1.5, BODY_D = 0.6;
const HEAD_SIZE = 0.7;
const PERSON_SCALE = 0.9;

// Reusable temporaries (module-scoped, not global)
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _color = new THREE.Color();
const _euler = new THREE.Euler();

export class PeopleRenderer {
  /**
   * @param {THREE.Scene} scene - Scene to add meshes to
   * @param {number} maxCount - Maximum number of people to render
   */
  constructor(scene, maxCount = 180) {
    this.maxCount = maxCount;
    this._dirty = false;

    // Body InstancedMesh
    const bodyGeo = new THREE.BoxGeometry(1, 1, 1);
    const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.7 });
    this.bodyMesh = new THREE.InstancedMesh(bodyGeo, bodyMat, maxCount);
    this.bodyMesh.castShadow = false;
    this.bodyMesh.count = 0;
    scene.add(this.bodyMesh);

    // Head InstancedMesh
    const headGeo = new THREE.BoxGeometry(1, 1, 1);
    const headMat = new THREE.MeshStandardMaterial({ roughness: 0.6 });
    this.headMesh = new THREE.InstancedMesh(headGeo, headMat, maxCount);
    this.headMesh.castShadow = false;
    this.headMesh.count = 0;
    scene.add(this.headMesh);
  }

  /**
   * Call before setting instances for this frame.
   */
  begin() {
    this._dirty = false;
  }

  /**
   * Set a person instance at the given slot.
   * @param {number} slot - Instance index (0..maxCount-1)
   * @param {number} x - World X position
   * @param {number} y - World Y offset (usually 0, can bob)
   * @param {number} z - World Z position
   * @param {number} rotY - Rotation around Y axis (radians)
   * @param {number} bodyColor - Hex color for body (e.g., 0x33AA55)
   * @param {number} headColor - Hex color for head/skin
   * @param {number} scale - Scale multiplier (1.0 = normal)
   */
  set(slot, x, y, z, rotY, bodyColor, headColor, scale = 1.0) {
    if (slot < 0 || slot >= this.maxCount) return;
    const s = scale * PERSON_SCALE;

    _euler.set(0, rotY, 0);
    _quat.setFromEuler(_euler);

    // Body
    const bodyY = y + (BODY_H * s) / 2;
    _pos.set(x, bodyY, z);
    _scale.set(BODY_W * s, BODY_H * s, BODY_D * s);
    _mat4.compose(_pos, _quat, _scale);
    this.bodyMesh.setMatrixAt(slot, _mat4);
    this.bodyMesh.setColorAt(slot, _color.set(bodyColor));

    // Head
    const headY = y + BODY_H * s + (HEAD_SIZE * s) / 2;
    _pos.set(x, headY, z);
    _scale.set(HEAD_SIZE * s, HEAD_SIZE * s, HEAD_SIZE * s);
    _mat4.compose(_pos, _quat, _scale);
    this.headMesh.setMatrixAt(slot, _mat4);
    this.headMesh.setColorAt(slot, _color.set(headColor));

    this._dirty = true;
  }

  /**
   * Hide a person instance (move offscreen).
   * @param {number} slot - Instance index to hide
   */
  hide(slot) {
    if (slot < 0 || slot >= this.maxCount) return;
    _pos.set(0, -100, 0);
    _quat.identity();
    _scale.set(0, 0, 0);
    _mat4.compose(_pos, _quat, _scale);
    this.bodyMesh.setMatrixAt(slot, _mat4);
    this.headMesh.setMatrixAt(slot, _mat4);
    this._dirty = true;
  }

  /**
   * Call after all set() calls. Updates GPU buffers.
   * @param {number} count - Number of active slots
   */
  end(count) {
    this.bodyMesh.count = count;
    this.headMesh.count = count;
    if (this._dirty && count > 0) {
      this.bodyMesh.instanceMatrix.needsUpdate = true;
      if (this.bodyMesh.instanceColor) this.bodyMesh.instanceColor.needsUpdate = true;
      this.headMesh.instanceMatrix.needsUpdate = true;
      if (this.headMesh.instanceColor) this.headMesh.instanceColor.needsUpdate = true;
    }
  }

  /**
   * Remove meshes from scene and dispose geometry/materials.
   */
  dispose() {
    this.bodyMesh.geometry.dispose();
    this.bodyMesh.material.dispose();
    this.headMesh.geometry.dispose();
    this.headMesh.material.dispose();
    this.bodyMesh.parent?.remove(this.bodyMesh);
    this.headMesh.parent?.remove(this.headMesh);
  }
}

// Export constants for external use
export { BODY_W, BODY_H, BODY_D, HEAD_SIZE, PERSON_SCALE };
