// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * BuildingRenderer — Reusable InstancedMesh renderer for block-style buildings.
 *
 * Renders Minecraft-style buildings (body + roof + windows + doors) using
 * InstancedMesh for performance. Produces building data objects that the
 * sim layer can use for collision detection.
 *
 * Usage:
 *   const renderer = new BuildingRenderer(scene, { maxBuildings: 300, maxWindows: 3000, maxDoors: 300 });
 *   const buildingData = renderer.addBuilding(x, z, w, d, h, 'office', roadGrid);
 *   renderer.finalize();
 */

import * as THREE from 'three';

// Building type colors
const BLDG_COLORS = {
  house:    0x8B6914,
  office:   0x888888,
  shop:     0xD2B48C,
  police:   0x4466AA,
  hospital: 0xEEEEEE,
};
const BLDG_ROOF = 0x333333;

// Reusable temporaries
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _color = new THREE.Color();
const _euler = new THREE.Euler();

export class BuildingRenderer {
  /**
   * @param {THREE.Scene} scene
   * @param {Object} opts
   * @param {number} opts.maxBuildings - Max building instances (default 300)
   * @param {number} opts.maxWindows - Max window instances (default 3000)
   * @param {number} opts.maxDoors - Max door instances (default 300)
   */
  constructor(scene, opts = {}) {
    const maxB = opts.maxBuildings || 300;
    const maxW = opts.maxWindows || 3000;
    const maxD = opts.maxDoors || 300;

    this.scene = scene;
    this.maxBuildings = maxB;
    this.maxWindows = maxW;
    this.maxDoors = maxD;

    // Building bodies
    const bodyGeo = new THREE.BoxGeometry(1, 1, 1);
    const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.0 });
    this.bodyMesh = new THREE.InstancedMesh(bodyGeo, bodyMat, maxB);
    this.bodyMesh.castShadow = true;
    this.bodyMesh.receiveShadow = true;
    this.bodyMesh.count = 0;
    scene.add(this.bodyMesh);

    // Roofs
    const roofGeo = new THREE.BoxGeometry(1, 1, 1);
    const roofMat = new THREE.MeshStandardMaterial({ color: BLDG_ROOF, roughness: 0.9 });
    this.roofMesh = new THREE.InstancedMesh(roofGeo, roofMat, maxB);
    this.roofMesh.castShadow = true;
    this.roofMesh.count = 0;
    scene.add(this.roofMesh);

    // Windows
    const winGeo = new THREE.PlaneGeometry(1.5, 1.2);
    this.windowMat = new THREE.MeshStandardMaterial({
      color: 0xffdd66, emissive: 0xffdd44, emissiveIntensity: 0.15,
      transparent: true, opacity: 0.7,
    });
    this.windowMesh = new THREE.InstancedMesh(winGeo, this.windowMat, maxW);
    this.windowMesh.count = 0;
    scene.add(this.windowMesh);

    // Doors
    const doorGeo = new THREE.PlaneGeometry(1, 1);
    const doorMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    this.doorMesh = new THREE.InstancedMesh(doorGeo, doorMat, maxD);
    this.doorMesh.count = 0;
    scene.add(this.doorMesh);

    // Counters
    this._buildingIdx = 0;
    this._windowIdx = 0;
    this._doorIdx = 0;

    // Output: building data array for sim collision
    this.buildings = [];

    // RNG
    this._rng = Math.random;
  }

  /**
   * Set the RNG function (for deterministic generation).
   */
  setRNG(rng) {
    this._rng = rng;
  }

  /**
   * Add a building at position (x, z) with dimensions w, d, h.
   *
   * @param {number} x - Center X
   * @param {number} z - Center Z
   * @param {number} w - Width
   * @param {number} d - Depth
   * @param {number} h - Height
   * @param {string} type - Building type ('house', 'office', 'shop', 'police', 'hospital')
   * @param {Object} roadGrid - { hRoads: number[], vRoads: number[] } for door placement
   * @returns {Object|null} Building data object, or null if at capacity
   */
  addBuilding(x, z, w, d, h, type, roadGrid) {
    if (this._buildingIdx >= this.maxBuildings) return null;

    const color = BLDG_COLORS[type] || BLDG_COLORS.office;
    const rng = this._rng;
    const idx = this._buildingIdx;

    // Body
    _pos.set(x, h / 2, z);
    _quat.identity();
    _scale.set(w, h, d);
    _mat4.compose(_pos, _quat, _scale);
    this.bodyMesh.setMatrixAt(idx, _mat4);
    this.bodyMesh.setColorAt(idx, _color.set(color));

    // Roof
    _pos.set(x, h + 0.25, z);
    _scale.set(w + 0.2, 0.5, d + 0.2);
    _mat4.compose(_pos, _quat, _scale);
    this.roofMesh.setMatrixAt(idx, _mat4);

    this._buildingIdx++;
    this.bodyMesh.count = this._buildingIdx;
    this.roofMesh.count = this._buildingIdx;

    // Door — face nearest road
    if (this._doorIdx < this.maxDoors && roadGrid) {
      const doorW = Math.min(2, w * 0.3);
      const doorH = Math.min(3, h * 0.5);
      let bestFaceDist = Infinity;
      let doorX = x, doorZ = z + d / 2 + 0.05, doorRotY = 0;

      for (const hz of roadGrid.hRoads) {
        const fd = Math.abs((z + d / 2) - hz);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x; doorZ = z + d / 2 + 0.05; doorRotY = 0; }
      }
      for (const hz of roadGrid.hRoads) {
        const fd = Math.abs((z - d / 2) - hz);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x; doorZ = z - d / 2 - 0.05; doorRotY = Math.PI; }
      }
      for (const vx of roadGrid.vRoads) {
        const fd = Math.abs((x + w / 2) - vx);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x + w / 2 + 0.05; doorZ = z; doorRotY = Math.PI / 2; }
      }
      for (const vx of roadGrid.vRoads) {
        const fd = Math.abs((x - w / 2) - vx);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x - w / 2 - 0.05; doorZ = z; doorRotY = -Math.PI / 2; }
      }

      _pos.set(doorX, doorH / 2, doorZ);
      _euler.set(0, doorRotY, 0);
      _quat.setFromEuler(_euler);
      _scale.set(doorW, doorH, 1);
      _mat4.compose(_pos, _quat, _scale);
      this.doorMesh.setMatrixAt(this._doorIdx, _mat4);
      this._doorIdx++;
      this.doorMesh.count = this._doorIdx;
    }

    // Windows
    const winStart = this._windowIdx;
    const floors = Math.floor(h / 4);
    for (let f = 1; f <= floors; f++) {
      const wy = f * 3.5;
      if (wy > h - 1) break;
      // Front/back windows
      for (const side of [1, -1]) {
        const numWins = Math.floor(w / 4);
        for (let wi = 0; wi < numWins; wi++) {
          if (rng() < 0.3 || this._windowIdx >= this.maxWindows) continue;
          const wx = x - w / 2 + 2 + wi * 4;
          _pos.set(wx, wy, z + side * (d / 2 + 0.05));
          _euler.set(0, side < 0 ? Math.PI : 0, 0);
          _quat.setFromEuler(_euler);
          _scale.set(1, 1, 1);
          _mat4.compose(_pos, _quat, _scale);
          this.windowMesh.setMatrixAt(this._windowIdx, _mat4);
          const winColor = rng() < 0.7 ? 0xffdd66 : 0x66ddff;
          this.windowMesh.setColorAt(this._windowIdx, _color.set(winColor));
          this._windowIdx++;
        }
      }
      // Side windows
      for (const side of [1, -1]) {
        const numWins = Math.floor(d / 4);
        for (let wi = 0; wi < numWins; wi++) {
          if (rng() < 0.4 || this._windowIdx >= this.maxWindows) continue;
          const wz = z - d / 2 + 2 + wi * 4;
          _pos.set(x + side * (w / 2 + 0.05), wy, wz);
          _euler.set(0, side * Math.PI / 2, 0);
          _quat.setFromEuler(_euler);
          _scale.set(1, 1, 1);
          _mat4.compose(_pos, _quat, _scale);
          this.windowMesh.setMatrixAt(this._windowIdx, _mat4);
          const winColor = rng() < 0.7 ? 0xffdd66 : 0x66ddff;
          this.windowMesh.setColorAt(this._windowIdx, _color.set(winColor));
          this._windowIdx++;
        }
      }
    }
    this.windowMesh.count = this._windowIdx;

    // Build data object for sim layer
    const bldgData = {
      h, type, x, z, w, d,
      idx,
      winStart,
      winEnd: this._windowIdx,
      cleared: false,
      damage: 0,
      origColor: color,
    };
    this.buildings.push(bldgData);
    return bldgData;
  }

  /**
   * Call after all addBuilding() calls to upload GPU buffers.
   */
  finalize() {
    this.bodyMesh.instanceMatrix.needsUpdate = true;
    if (this.bodyMesh.instanceColor) this.bodyMesh.instanceColor.needsUpdate = true;
    this.roofMesh.instanceMatrix.needsUpdate = true;
    this.windowMesh.instanceMatrix.needsUpdate = true;
    if (this.windowMesh.instanceColor) this.windowMesh.instanceColor.needsUpdate = true;
    this.doorMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Update window emissive intensity (for day/night transition).
   * @param {number} intensity - Emissive intensity (0.15 day, 0.8 night)
   */
  setWindowEmissive(intensity) {
    this.windowMat.emissiveIntensity = intensity;
  }

  /**
   * Flash a building's windows (e.g., during breach).
   * @param {Object} bldg - Building data object from addBuilding()
   * @param {number} color - Hex color to flash
   * @param {number} restoreColor - Hex color to restore after duration
   * @param {number} durationMs - Flash duration in milliseconds
   */
  flashWindows(bldg, color, restoreColor = 0xffdd66, durationMs = 300) {
    for (let wi = bldg.winStart; wi < bldg.winEnd; wi++) {
      this.windowMesh.setColorAt(wi, _color.set(color));
    }
    if (this.windowMesh.instanceColor) this.windowMesh.instanceColor.needsUpdate = true;
    setTimeout(() => {
      for (let wi = bldg.winStart; wi < bldg.winEnd; wi++) {
        this.windowMesh.setColorAt(wi, _color.set(restoreColor));
      }
      if (this.windowMesh.instanceColor) this.windowMesh.instanceColor.needsUpdate = true;
    }, durationMs);
  }

  /**
   * Tint a building body color (e.g., cleared = green tint).
   * @param {Object} bldg - Building data object
   * @param {number} color - New hex color
   */
  setBodyColor(bldg, color) {
    this.bodyMesh.setColorAt(bldg.idx, _color.set(color));
    if (this.bodyMesh.instanceColor) this.bodyMesh.instanceColor.needsUpdate = true;
  }

  /**
   * Dispose all GPU resources.
   */
  dispose() {
    for (const mesh of [this.bodyMesh, this.roofMesh, this.windowMesh, this.doorMesh]) {
      mesh.geometry.dispose();
      mesh.material.dispose();
      mesh.parent?.remove(mesh);
    }
  }
}

// Export color constants for external use
export { BLDG_COLORS, BLDG_ROOF };
