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

        this.headlightMesh = null;  // shared across all types
        this.taillightMesh = null;
        this.brakeLightMesh = null;  // same as taillight but brighter when braking
        this._initMeshes();
    }

    _initMeshes() {
        const hideMatrix = new THREE.Matrix4().makeTranslation(0, -1000, 0);

        for (const [type, info] of Object.entries(CAR_TYPES)) {
            // Body geometry (box)
            const bodyGeo = new THREE.BoxGeometry(info.w, info.h, info.l);
            bodyGeo.translate(0, info.h / 2, 0);

            // Cabin on top
            const cabinGeo = new THREE.BoxGeometry(info.w * 0.8, info.cabinH, info.cabinL);
            cabinGeo.translate(0, info.h + info.cabinH / 2, -info.l * 0.05);

            // Merge body + cabin
            const mergedGeo = mergeGeometries([bodyGeo, cabinGeo], false) || bodyGeo;

            const mat = new THREE.MeshStandardMaterial({
                color: info.color || 0x888888,
                roughness: 0.6,
            });

            const mesh = new THREE.InstancedMesh(mergedGeo, mat, this.maxCars);
            mesh.count = 0;
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            const colors = new Float32Array(this.maxCars * 3);
            mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);

            for (let i = 0; i < this.maxCars; i++) mesh.setMatrixAt(i, hideMatrix);
            mesh.instanceMatrix.needsUpdate = true;

            this.scene.add(mesh);
            this.instances[type] = { mesh, count: 0 };
        }

        // Shared headlights (2 bright white boxes, oversized for visibility at distance)
        const hlGeoL = new THREE.BoxGeometry(0.5, 0.4, 0.3);
        hlGeoL.translate(0.6, 0.5, 2.3);
        const hlGeoR = new THREE.BoxGeometry(0.5, 0.4, 0.3);
        hlGeoR.translate(-0.6, 0.5, 2.3);
        const hlMerged = mergeGeometries([hlGeoL, hlGeoR], false);
        const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffff }); // pure white, always visible
        this.headlightMesh = new THREE.InstancedMesh(hlMerged, hlMat, this.maxCars);
        this.headlightMesh.count = 0;
        for (let i = 0; i < this.maxCars; i++) this.headlightMesh.setMatrixAt(i, hideMatrix);
        this.headlightMesh.instanceMatrix.needsUpdate = true;
        this.scene.add(this.headlightMesh);

        // Shared taillights (2 red boxes at rear, oversized for visibility)
        const tlGeoL = new THREE.BoxGeometry(0.45, 0.35, 0.25);
        tlGeoL.translate(0.6, 0.5, -2.3);
        const tlGeoR = new THREE.BoxGeometry(0.45, 0.35, 0.25);
        tlGeoR.translate(-0.6, 0.5, -2.3);
        const tlMerged = mergeGeometries([tlGeoL, tlGeoR], false);
        const tlMat = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // bright red
        this.taillightMesh = new THREE.InstancedMesh(tlMerged, tlMat, this.maxCars);
        this.taillightMesh.count = 0;
        for (let i = 0; i < this.maxCars; i++) this.taillightMesh.setMatrixAt(i, hideMatrix);
        this.taillightMesh.instanceMatrix.needsUpdate = true;
        this.scene.add(this.taillightMesh);

        // Per-instance taillight color (for brake light toggling)
        const tlColors = new Float32Array(this.maxCars * 3);
        // Default: dim red
        for (let i = 0; i < this.maxCars; i++) {
            tlColors[i * 3] = 0.3;     // R
            tlColors[i * 3 + 1] = 0;   // G
            tlColors[i * 3 + 2] = 0;   // B
        }
        this.taillightMesh.instanceColor = new THREE.InstancedBufferAttribute(tlColors, 3);

        // Headlight beams (cone mesh projecting forward from each car)
        const beamGeoL = new THREE.ConeGeometry(1.5, 8, 4); // wide cone, 8m long
        beamGeoL.rotateX(Math.PI / 2); // point along +Z (forward)
        beamGeoL.translate(0.5, 0.4, 6); // offset forward
        const beamGeoR = new THREE.ConeGeometry(1.5, 8, 4);
        beamGeoR.rotateX(Math.PI / 2);
        beamGeoR.translate(-0.5, 0.4, 6);
        const beamMerged = mergeGeometries([beamGeoL, beamGeoR], false);
        const beamMat = new THREE.MeshBasicMaterial({
            color: 0xffffaa, transparent: true, opacity: 0.06,
            depthWrite: false, // don't block other geometry
        });
        this.beamMesh = new THREE.InstancedMesh(beamMerged, beamMat, this.maxCars);
        this.beamMesh.count = 0;
        this.beamMesh.renderOrder = 999; // render last (transparent)
        for (let i = 0; i < this.maxCars; i++) this.beamMesh.setMatrixAt(i, hideMatrix);
        this.beamMesh.instanceMatrix.needsUpdate = true;
        this.scene.add(this.beamMesh);
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

        // Also allocate headlight + taillight instances
        const globalIdx = this.totalCount;
        if (this.headlightMesh) {
            this.headlightMesh.count = globalIdx + 1;
        }
        if (this.taillightMesh) {
            this.taillightMesh.count = globalIdx + 1;
        }
        if (this.beamMesh) {
            this.beamMesh.count = globalIdx + 1;
        }

        this.carData.push({ type, typeIndex: index, globalIndex: globalIdx });
        this.totalCount++;

        return globalIdx;
    }

    /**
     * Update a car's position and rotation.
     *
     * @param {number} handle - Car handle from addCar()
     * @param {number} x - World X
     * @param {number} z - World Z
     * @param {number} heading - Rotation Y (radians)
     */
    updateCar(handle, x, z, heading, braking = false) {
        const data = this.carData[handle];
        if (!data) return;

        this.dummy.position.set(x, 0, z);
        this.dummy.rotation.set(0, heading, 0);
        this.dummy.updateMatrix();

        const gi = data.globalIndex;

        // Body
        data._inst = data._inst || this.instances[data.type];
        data._inst.mesh.setMatrixAt(data.typeIndex, this.dummy.matrix);

        // Headlights, taillights, beams all follow same transform
        if (this.headlightMesh) this.headlightMesh.setMatrixAt(gi, this.dummy.matrix);
        if (this.taillightMesh) this.taillightMesh.setMatrixAt(gi, this.dummy.matrix);
        if (this.beamMesh) this.beamMesh.setMatrixAt(gi, this.dummy.matrix);

        // Brake light: bright red when braking, dim when not
        if (this.taillightMesh?.instanceColor) {
            if (braking) {
                this.taillightMesh.instanceColor.setXYZ(gi, 1.0, 0.1, 0.0);
            } else {
                this.taillightMesh.instanceColor.setXYZ(gi, 0.4, 0.0, 0.0);
            }
            this._tlColorDirty = true;
        }
    }

    /**
     * Flush all instance matrix updates to GPU. Call once per frame after all updateCar() calls.
     */
    flush() {
        for (const key in this.instances) {
            this.instances[key].mesh.instanceMatrix.needsUpdate = true;
        }
        if (this.headlightMesh) this.headlightMesh.instanceMatrix.needsUpdate = true;
        if (this.taillightMesh) {
            this.taillightMesh.instanceMatrix.needsUpdate = true;
            if (this._tlColorDirty) {
                this.taillightMesh.instanceColor.needsUpdate = true;
                this._tlColorDirty = false;
            }
        }
        if (this.beamMesh) this.beamMesh.instanceMatrix.needsUpdate = true;
    }

    /**
     * Set headlight brightness (day/night toggle).
     * @param {number} intensity - 0 = off (day), 0.8 = on (night)
     */
    setHeadlightIntensity(intensity) {
        if (this.headlightMesh) {
            // MeshBasicMaterial: adjust color brightness
            const b = Math.floor(intensity * 255);
            this.headlightMesh.material.color.setRGB(b/255, b/255, (b*0.85)/255);
        }
    }

    setTaillightIntensity(intensity) {
        if (this.taillightMesh) {
            const b = Math.floor(intensity * 255);
            this.taillightMesh.material.color.setRGB(b/255, 0, 0);
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
