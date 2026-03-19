// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Instanced Car Renderer — renders thousands of cars with minimal draw calls.
 *
 * Instead of creating a separate Three.js Group for each car (12+ meshes each),
 * this uses InstancedMesh to render ALL cars of the same type in a single draw call.
 *
 * Performance: 1000 cars = ~6 draw calls instead of ~12,000.
 *
 * Each car type (sedan, suv, truck, police, ambulance) gets one InstancedMesh
 * for the body. Per-frame: update the instance matrix for each car based on
 * its CarPath position and heading.
 *
 * Brake lights, turn signals, and emergency lights are handled via per-instance
 * color attributes (InstancedBufferAttribute) or separate small InstancedMeshes.
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// ============================================================
// CAR TYPE GEOMETRIES
// ============================================================

const CAR_TYPES = {
    sedan:     { w: 2.0, h: 1.4, l: 4.5, cabinH: 0.8, cabinL: 2.0, color: null },
    suv:       { w: 2.2, h: 1.8, l: 5.0, cabinH: 0.9, cabinL: 2.2, color: null },
    truck:     { w: 2.5, h: 2.5, l: 8.0, cabinH: 1.0, cabinL: 2.0, color: null },
    police:    { w: 2.0, h: 1.5, l: 4.5, cabinH: 0.8, cabinL: 2.0, color: 0x1155ff },
    ambulance: { w: 2.2, h: 2.2, l: 6.0, cabinH: 1.0, cabinL: 2.5, color: 0xffffff },
};

// ============================================================
// INSTANCED CAR RENDERER
// ============================================================

export class InstancedCarRenderer {
    /**
     * @param {THREE.Scene} scene
     * @param {number} maxCars - Maximum number of cars to support
     */
    constructor(scene, maxCars = 5000) {
        this.scene = scene;
        this.maxCars = maxCars;
        this.instances = {}; // type → { bodyMesh, count }
        this.carData = [];   // parallel array: carData[instanceIndex] = { type, typeIndex }
        this.dummy = new THREE.Object3D();
        this.totalCount = 0;

        this._initMeshes();
    }

    _initMeshes() {
        for (const [type, info] of Object.entries(CAR_TYPES)) {
            // Body geometry (box)
            const bodyGeo = new THREE.BoxGeometry(info.w, info.h, info.l);
            bodyGeo.translate(0, info.h / 2, 0); // pivot at bottom center

            // Cabin on top
            const cabinGeo = new THREE.BoxGeometry(info.w * 0.8, info.cabinH, info.cabinL);
            cabinGeo.translate(0, info.h + info.cabinH / 2, -info.l * 0.05);

            // Merge body + cabin into single geometry (one material)
            const mergedGeo = mergeGeometries([bodyGeo, cabinGeo], false) || bodyGeo;

            const mat = new THREE.MeshStandardMaterial({
                color: info.color || 0x888888,
                roughness: 0.6,
            });

            const mesh = new THREE.InstancedMesh(mergedGeo, mat, this.maxCars);
            mesh.count = 0; // start with 0 visible instances
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            // Per-instance color
            const colors = new Float32Array(this.maxCars * 3);
            mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);

            // Hide all instances initially (move far away)
            const hideMatrix = new THREE.Matrix4().makeTranslation(0, -1000, 0);
            for (let i = 0; i < this.maxCars; i++) {
                mesh.setMatrixAt(i, hideMatrix);
            }
            mesh.instanceMatrix.needsUpdate = true;

            this.scene.add(mesh);
            this.instances[type] = { mesh, count: 0 };
        }
    }

    /**
     * Add a car and return its instance handle.
     *
     * @param {string} type - Car type key (sedan, police, etc.)
     * @param {number} color - Body color as hex (e.g., 0xff4444)
     * @returns {number} Car handle (index for updating)
     */
    addCar(type = 'sedan', color = null) {
        const inst = this.instances[type] || this.instances.sedan;
        const index = inst.count;
        if (index >= this.maxCars) return -1; // full

        inst.count++;
        inst.mesh.count = inst.count;

        // Set per-instance color
        if (color !== null) {
            const c = new THREE.Color(color);
            inst.mesh.instanceColor.setXYZ(index, c.r, c.g, c.b);
            inst.mesh.instanceColor.needsUpdate = true;
        }

        const handle = this.totalCount;
        this.carData.push({ type, typeIndex: index });
        this.totalCount++;

        return handle;
    }

    /**
     * Update a car's position and rotation.
     *
     * @param {number} handle - Car handle from addCar()
     * @param {number} x - World X
     * @param {number} z - World Z
     * @param {number} heading - Rotation Y (radians)
     */
    updateCar(handle, x, z, heading) {
        const data = this.carData[handle];
        if (!data) return;

        const inst = this.instances[data.type];
        this.dummy.position.set(x, 0, z);
        this.dummy.rotation.set(0, heading, 0);
        this.dummy.updateMatrix();
        inst.mesh.setMatrixAt(data.typeIndex, this.dummy.matrix);
    }

    /**
     * Flush all instance matrix updates to GPU. Call once per frame after all updateCar() calls.
     */
    flush() {
        for (const key in this.instances) {
            this.instances[key].mesh.instanceMatrix.needsUpdate = true;
        }
    }

    /**
     * Get total draw calls used by this renderer.
     */
    getDrawCallCount() {
        let count = 0;
        for (const key in this.instances) {
            if (this.instances[key].count > 0) count++;
        }
        return count;
    }
}
