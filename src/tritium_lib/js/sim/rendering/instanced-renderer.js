// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * InstancedRenderer — renders any unit type using Three.js InstancedMesh.
 *
 * Generic: works for cars, pedestrians, motorcycles, tanks, etc.
 * Each "mesh type" (body, headlights, taillights, etc.) gets one InstancedMesh.
 * One draw call per mesh type regardless of unit count.
 *
 * Usage:
 *   const renderer = new InstancedRenderer(scene);
 *   renderer.defineMeshType('car_body', geometry, material, maxInstances);
 *   renderer.defineMeshType('car_headlights', hlGeometry, hlMaterial, maxInstances);
 *   const handle = renderer.addInstance('car_body', color);
 *   renderer.updateInstance(handle, x, y, z, heading);
 *   renderer.flush();
 */

import * as THREE from 'three';

export class InstancedRenderer {
    /**
     * @param {THREE.Scene} scene
     */
    constructor(scene) {
        this.scene = scene;
        this.meshTypes = {};  // name → { mesh, count, maxInstances }
        this.dummy = new THREE.Object3D();
    }

    /**
     * Define a new mesh type that can be instanced.
     *
     * @param {string} name - Unique name (e.g., 'car_body', 'ped_capsule')
     * @param {THREE.BufferGeometry} geometry
     * @param {THREE.Material} material
     * @param {number} maxInstances
     */
    defineMeshType(name, geometry, material, maxInstances = 5000) {
        const hideMatrix = new THREE.Matrix4().makeTranslation(0, -1000, 0);
        const mesh = new THREE.InstancedMesh(geometry, material, maxInstances);
        // WebGPU bug: count must be set to max at creation, not incremented later
        // (three.js issue #32099: WebGPU doesn't re-allocate instance buffer on count change)
        mesh.count = maxInstances;
        mesh.frustumCulled = false;
        mesh.castShadow = true;

        // Initialize all instance colors to white
        const colors = new Float32Array(maxInstances * 3);
        for (let i = 0; i < maxInstances * 3; i++) colors[i] = 1.0;
        mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);

        // Hide all
        for (let i = 0; i < maxInstances; i++) mesh.setMatrixAt(i, hideMatrix);
        mesh.instanceMatrix.needsUpdate = true;

        this.scene.add(mesh);
        this.meshTypes[name] = { mesh, count: 0, maxInstances, _colorDirty: false };
    }

    /**
     * Add an instance and return its index.
     *
     * @param {string} meshType - Name of the mesh type
     * @param {number} [color] - Hex color (null = white)
     * @returns {number} Instance index (-1 if full)
     */
    addInstance(meshType, color = null) {
        const mt = this.meshTypes[meshType];
        if (!mt || mt.count >= mt.maxInstances) return -1;

        const idx = mt.count++;
        // Don't change mesh.count — it stays at maxInstances for WebGPU compatibility
        // Unused instances stay at the hide position (y=-1000)

        if (color !== null) {
            const c = new THREE.Color(color);
            mt.mesh.instanceColor.setXYZ(idx, c.r, c.g, c.b);
            mt._colorDirty = true;
        }

        return idx;
    }

    /**
     * Update an instance's transform.
     *
     * @param {string} meshType
     * @param {number} index
     * @param {number} x
     * @param {number} y - Altitude (0 for ground)
     * @param {number} z
     * @param {number} heading - Rotation Y
     * @param {number} [pitch] - Rotation X (for aircraft)
     * @param {number} [roll] - Rotation Z (for aircraft banking)
     */
    updateInstance(meshType, index, x, y, z, heading, pitch = 0, roll = 0, scaleZ = 1) {
        const mt = this.meshTypes[meshType];
        if (!mt || index < 0 || index >= mt.count) return;

        this.dummy.position.set(x, y, z);
        this.dummy.rotation.set(pitch, heading, roll);
        this.dummy.scale.set(1, 1, scaleZ);
        this.dummy.updateMatrix();
        mt.mesh.setMatrixAt(index, this.dummy.matrix);
        // Reset scale for next call
        this.dummy.scale.set(1, 1, 1);
    }

    /**
     * Set per-instance color.
     *
     * @param {string} meshType
     * @param {number} index
     * @param {number} r - Red [0,1]
     * @param {number} g - Green [0,1]
     * @param {number} b - Blue [0,1]
     */
    setInstanceColor(meshType, index, r, g, b) {
        const mt = this.meshTypes[meshType];
        if (!mt || index < 0) return;
        mt.mesh.instanceColor.setXYZ(index, r, g, b);
        mt._colorDirty = true;
    }

    /**
     * Set material-level property (affects ALL instances of this type).
     *
     * @param {string} meshType
     * @param {string} prop - Material property name
     * @param {*} value
     */
    setMaterialProp(meshType, prop, value) {
        const mt = this.meshTypes[meshType];
        if (!mt) return;
        mt.mesh.material[prop] = value;
    }

    /**
     * Set visibility of a mesh type.
     */
    setVisible(meshType, visible) {
        const mt = this.meshTypes[meshType];
        if (mt) mt.mesh.visible = visible;
    }

    /**
     * Flush all matrix and color updates to GPU. Call once per frame.
     */
    flush() {
        for (const name in this.meshTypes) {
            const mt = this.meshTypes[name];
            mt.mesh.instanceMatrix.needsUpdate = true;
            if (mt._colorDirty) {
                mt.mesh.instanceColor.needsUpdate = true;
                mt._colorDirty = false;
            }
        }
    }

    /**
     * Get total draw calls used.
     */
    getDrawCallCount() {
        let n = 0;
        for (const name in this.meshTypes) {
            if (this.meshTypes[name].count > 0) n++;
        }
        return n;
    }
}
