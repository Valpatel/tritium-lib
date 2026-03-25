// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * OSMCityBuilder — builds a city from real OpenStreetMap data.
 *
 * Takes city-data JSON from /api/geo/city-data and produces a CityData
 * object compatible with World, including RoadNetwork, TrafficControllerManager,
 * building entries, and 3D geometry.
 *
 * This is the real-world counterpart to CityBuilder.buildGrid().
 * Vehicles, IDM, MOBIL, and traffic controllers all work unchanged —
 * they consume the graph, not the grid layout.
 *
 * Usage:
 *   const cityData = await fetch('/api/geo/city-data?lat=...&lng=...&radius=300').then(r => r.json());
 *   const city = OSMCityBuilder.build(scene, cityData);
 *   const world = new World(scene, city.roadNetwork, { trafficCtrl: city.trafficMgr, buildingEntries: city.entries });
 */

import * as THREE from 'three';
import { RoadNetwork } from '../road-network.js';
import { TrafficControllerManager } from '../traffic-controller.js';

/**
 * @typedef {Object} OSMCityData
 * @property {RoadNetwork} roadNetwork
 * @property {TrafficControllerManager} trafficMgr
 * @property {Array} entries — building entries for pedestrian destinations
 * @property {number} width — city extent width (meters)
 * @property {number} height — city extent height (meters)
 * @property {Object} raw — original city-data JSON
 */

export class OSMCityBuilder {
    /**
     * Build city from OSM data and optionally add geometry to the scene.
     *
     * @param {THREE.Scene|null} scene — Three.js scene (null for headless/test)
     * @param {Object} cityData — city-data JSON from /api/geo/city-data
     * @param {Object} [options]
     * @param {boolean} [options.renderGeometry=true] — add 3D geometry to scene
     * @param {number} [options.mergeRadius=5] — intersection merge radius (meters)
     * @returns {OSMCityData}
     */
    static build(scene, cityData, options = {}) {
        const renderGeometry = options.renderGeometry !== false;
        const mergeRadius = options.mergeRadius || 5;
        const radius = cityData.radius || 300;

        // Build road network from OSM roads
        const roadNetwork = new RoadNetwork();
        roadNetwork.buildFromOSM(cityData.roads || [], mergeRadius);

        // Initialize traffic controllers at intersections with 3+ connections
        const trafficMgr = new TrafficControllerManager();
        trafficMgr.initFromNetwork(roadNetwork);

        // Generate building entries for pedestrian destinations
        const entries = [];
        for (const bldg of (cityData.buildings || [])) {
            if (!bldg.polygon || bldg.polygon.length < 3) continue;
            // Entry at centroid of polygon
            let cx = 0, cz = 0;
            for (const [x, z] of bldg.polygon) {
                cx += x;
                cz += z;
            }
            cx /= bldg.polygon.length;
            cz /= bldg.polygon.length;
            entries.push({
                x: cx,
                z: cz,
                buildingId: bldg.id,
                buildingType: bldg.type || 'yes',
                category: bldg.category || 'residential',
                name: bldg.name || '',
            });
        }

        // Use entrance data if available (more accurate than centroids)
        for (const ent of (cityData.entrances || [])) {
            entries.push({
                x: ent.pos[0],
                z: ent.pos[1],
                buildingId: null,
                buildingType: 'entrance',
                category: 'entrance',
                name: ent.name || '',
            });
        }

        // Add POIs as destinations too
        for (const poi of (cityData.pois || [])) {
            entries.push({
                x: poi.pos[0],
                z: poi.pos[1],
                buildingId: null,
                buildingType: poi.type || 'amenity',
                category: 'amenity',
                name: poi.name || '',
            });
        }

        // Render 3D geometry if scene provided
        if (scene && renderGeometry) {
            OSMCityBuilder._renderGround(scene, radius);
            OSMCityBuilder._renderRoads(scene, cityData.roads || []);
            OSMCityBuilder._renderTrees(scene, cityData.trees || []);
        }

        return {
            roadNetwork,
            trafficMgr,
            entries,
            width: radius * 2,
            height: radius * 2,
            raw: cityData,
        };
    }

    /**
     * Render ground plane.
     */
    static _renderGround(scene, radius) {
        const size = radius * 2 + 40;
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(size, size),
            new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(0, -0.05, 0);
        ground.receiveShadow = true;
        scene.add(ground);
    }

    /**
     * Render road surfaces from OSM data.
     */
    static _renderRoads(scene, roads) {
        if (!roads.length) return;
        const roadMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.8 });

        for (const road of roads) {
            const pts = road.points;
            if (!pts || pts.length < 2) continue;
            const width = road.width || 6;
            const half = width / 2;

            // Create road ribbon
            const positions = [];
            for (let i = 0; i < pts.length; i++) {
                const [x, z] = pts[i];
                let dx, dz;
                if (i < pts.length - 1) {
                    dx = pts[i + 1][0] - x;
                    dz = pts[i + 1][1] - z;
                } else {
                    dx = x - pts[i - 1][0];
                    dz = z - pts[i - 1][1];
                }
                const len = Math.sqrt(dx * dx + dz * dz) || 1;
                const px = -dz / len, pz = dx / len;
                positions.push(
                    x + px * half, 0.01, -(z + pz * half),
                    x - px * half, 0.01, -(z - pz * half),
                );
            }

            const indices = [];
            for (let i = 0; i < pts.length - 1; i++) {
                const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
                indices.push(a, b, c, b, d, c);
            }

            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geo.setIndex(indices);
            geo.computeVertexNormals();
            scene.add(new THREE.Mesh(geo, roadMat));
        }
    }

    /**
     * Render instanced trees.
     */
    static _renderTrees(scene, trees) {
        if (!trees.length) return;
        const max = Math.min(trees.length, 2000);

        const trunkGeo = new THREE.CylinderGeometry(0.15, 0.2, 3, 5);
        const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2820, roughness: 0.9 });
        const crownGeo = new THREE.SphereGeometry(1, 6, 5);
        const crownMat = new THREE.MeshStandardMaterial({ color: 0x1a4a1a, roughness: 0.8 });

        const trunkMesh = new THREE.InstancedMesh(trunkGeo, trunkMat, max);
        const crownMesh = new THREE.InstancedMesh(crownGeo, crownMat, max);
        trunkMesh.castShadow = true;
        crownMesh.castShadow = true;

        const dummy = new THREE.Object3D();
        for (let i = 0; i < max; i++) {
            const t = trees[i];
            const h = t.height || 6;
            const [x, z] = t.pos;

            dummy.position.set(x, h * 0.22, -z);
            dummy.scale.set(1, h * 0.15, 1);
            dummy.updateMatrix();
            trunkMesh.setMatrixAt(i, dummy.matrix);

            const cr = h * 0.3;
            dummy.position.set(x, h * 0.55, -z);
            dummy.scale.set(cr, cr, cr);
            dummy.updateMatrix();
            crownMesh.setMatrixAt(i, dummy.matrix);
        }

        trunkMesh.count = max;
        crownMesh.count = max;
        trunkMesh.instanceMatrix.needsUpdate = true;
        crownMesh.instanceMatrix.needsUpdate = true;
        scene.add(trunkMesh);
        scene.add(crownMesh);
    }
}
