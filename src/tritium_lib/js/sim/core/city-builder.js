// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * CityBuilder — generates a city grid with roads, buildings, traffic lights.
 *
 * Reusable across applications. Creates:
 * - Road network (graph + geometry)
 * - Lane markings (yellow center, white lanes, crosswalks)
 * - Buildings (zone-based: residential, commercial, office, civic, park)
 * - Sidewalks
 * - Traffic lights (instanced poles + colored bulbs)
 * - Trees (instanced)
 * - Building entries (for pedestrian destinations)
 *
 * All geometry is batched/instanced for performance.
 * Returns a CityData object that World uses.
 *
 * Usage:
 *   const city = CityBuilder.buildGrid(scene, { cols: 8, rows: 6 });
 *   const world = new World(scene, city.roadNetwork, { trafficCtrl: city.trafficMgr, buildingEntries: city.entries });
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoadNetwork } from '../road-network.js';
import { TrafficControllerManager } from '../traffic-controller.js';

/**
 * @typedef {Object} CityConfig
 * @property {number} cols — grid columns (default 8)
 * @property {number} rows — grid rows (default 6)
 * @property {number} blockW — block width in meters (default 60)
 * @property {number} blockH — block height in meters (default 50)
 * @property {number} roadW — road width including sidewalks (default 16)
 * @property {number} laneW — lane width (default 3)
 * @property {number} lanesPerDir — lanes per direction (default 2)
 */

/**
 * @typedef {Object} CityData
 * @property {RoadNetwork} roadNetwork
 * @property {TrafficControllerManager} trafficMgr
 * @property {Array} entries — building entries for ped destinations
 * @property {number} width — total city width
 * @property {number} height — total city height
 */

export class CityBuilder {
    /**
     * Build a grid city and add geometry to the scene.
     *
     * @param {THREE.Scene} scene
     * @param {CityConfig} config
     * @returns {CityData}
     */
    static buildGrid(scene, config = {}) {
        const cols = config.cols || 8;
        const rows = config.rows || 6;
        const blockW = config.blockW || 60;
        const blockH = config.blockH || 50;
        const roadW = config.roadW || 16;
        const laneW = config.laneW || 3;
        const lanesPerDir = config.lanesPerDir || 2;

        const cityW = cols * (blockW + roadW) + roadW;
        const cityH = rows * (blockH + roadW) + roadW;
        const hRoadZ = (row) => row * (blockH + roadW) + roadW / 2;
        const vRoadX = (col) => col * (blockW + roadW) + roadW / 2;

        const seeded = ((s) => () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; })(42);

        // ---- Ground ----
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(cityW + 40, cityH + 40),
            new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(cityW / 2, -0.05, cityH / 2);
        scene.add(ground);

        // ---- Roads ----
        const roadMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.8 });
        for (let row = 0; row <= rows; row++) {
            const g = new THREE.PlaneGeometry(cityW, roadW - 4);
            g.rotateX(-Math.PI / 2); g.translate(cityW / 2, 0.01, hRoadZ(row));
            scene.add(new THREE.Mesh(g, roadMat));
        }
        for (let col = 0; col <= cols; col++) {
            const g = new THREE.PlaneGeometry(roadW - 4, cityH);
            g.rotateX(-Math.PI / 2); g.translate(vRoadX(col), 0.01, cityH / 2);
            scene.add(new THREE.Mesh(g, roadMat));
        }
        // Intersections
        for (let row = 0; row <= rows; row++) {
            for (let col = 0; col <= cols; col++) {
                const g = new THREE.PlaneGeometry(roadW, roadW);
                g.rotateX(-Math.PI / 2); g.translate(vRoadX(col), 0.01, hRoadZ(row));
                scene.add(new THREE.Mesh(g, roadMat));
            }
        }

        // ---- Lane Markings (batched) ----
        const yellowGeos = [], whiteGeos = [], crosswalkGeos = [];
        const DASH = 2, GAP = 3, MW = 0.15;

        // Horizontal roads
        for (let row = 0; row <= rows; row++) {
            const z = hRoadZ(row);
            for (let col = 0; col < cols; col++) {
                const x1 = vRoadX(col) + roadW / 2, x2 = vRoadX(col + 1) - roadW / 2;
                const segLen = x2 - x1;
                // SOLID yellow center line (no dashes)
                const yg = new THREE.PlaneGeometry(segLen, MW); yg.rotateX(-Math.PI / 2); yg.translate((x1 + x2) / 2, 0.015, z);
                yellowGeos.push(yg);
                // DASHED white lane lines
                const n = Math.floor(segLen / (DASH + GAP));
                for (let d = 0; d < n; d++) {
                    const cx = x1 + (d + 0.5) * (DASH + GAP);
                    const w1 = new THREE.PlaneGeometry(DASH, MW); w1.rotateX(-Math.PI / 2); w1.translate(cx, 0.015, z - laneW);
                    const w2 = new THREE.PlaneGeometry(DASH, MW); w2.rotateX(-Math.PI / 2); w2.translate(cx, 0.015, z + laneW);
                    whiteGeos.push(w1, w2);
                }
            }
        }
        // Vertical roads
        for (let col = 0; col <= cols; col++) {
            const x = vRoadX(col);
            for (let row = 0; row < rows; row++) {
                const z1 = hRoadZ(row) + roadW / 2, z2 = hRoadZ(row + 1) - roadW / 2;
                const segLen = z2 - z1;
                // SOLID yellow center line
                const yg = new THREE.PlaneGeometry(MW, segLen); yg.rotateX(-Math.PI / 2); yg.translate(x, 0.015, (z1 + z2) / 2);
                yellowGeos.push(yg);
                // DASHED white lane lines
                const n = Math.floor(segLen / (DASH + GAP));
                for (let d = 0; d < n; d++) {
                    const cz = z1 + (d + 0.5) * (DASH + GAP);
                    const w1 = new THREE.PlaneGeometry(MW, DASH); w1.rotateX(-Math.PI / 2); w1.translate(x - laneW, 0.015, cz);
                    const w2 = new THREE.PlaneGeometry(MW, DASH); w2.rotateX(-Math.PI / 2); w2.translate(x + laneW, 0.015, cz);
                    whiteGeos.push(w1, w2);
                }
            }
        }

        // Crosswalks
        for (let row = 0; row <= rows; row++) {
            for (let col = 0; col <= cols; col++) {
                const ix = vRoadX(col), iz = hRoadZ(row);
                for (const [dx, dz, isH] of [[0, -(roadW / 2 - 1), true], [0, roadW / 2 - 1, true], [-(roadW / 2 - 1), 0, false], [roadW / 2 - 1, 0, false]]) {
                    for (let s = -3; s <= 3; s++) {
                        const sg = isH ? new THREE.PlaneGeometry(1.2, 0.4) : new THREE.PlaneGeometry(0.4, 1.2);
                        sg.rotateX(-Math.PI / 2);
                        sg.translate(ix + dx + (isH ? s * 1.5 : 0), 0.02, iz + dz + (isH ? 0 : s * 1.5));
                        crosswalkGeos.push(sg);
                    }
                }
            }
        }

        if (yellowGeos.length) scene.add(new THREE.Mesh(mergeGeometries(yellowGeos, false), new THREE.MeshBasicMaterial({ color: 0xccaa00 })));
        if (whiteGeos.length) scene.add(new THREE.Mesh(mergeGeometries(whiteGeos, false), new THREE.MeshBasicMaterial({ color: 0xdddddd })));
        if (crosswalkGeos.length) scene.add(new THREE.Mesh(mergeGeometries(crosswalkGeos, false), new THREE.MeshBasicMaterial({ color: 0xffffff })));

        // ---- Sidewalks ----
        const swGeos = [];
        const SW = 2;
        for (let row = 0; row <= rows; row++) {
            const z = hRoadZ(row);
            for (const side of [-1, 1]) {
                const g = new THREE.PlaneGeometry(cityW, SW); g.rotateX(-Math.PI / 2);
                g.translate(cityW / 2, 0.05, z + side * (roadW / 2 - SW / 2 + 0.5));
                swGeos.push(g);
            }
        }
        for (let col = 0; col <= cols; col++) {
            const x = vRoadX(col);
            for (const side of [-1, 1]) {
                const g = new THREE.PlaneGeometry(SW, cityH); g.rotateX(-Math.PI / 2);
                g.translate(x + side * (roadW / 2 - SW / 2 + 0.5), 0.05, cityH / 2);
                swGeos.push(g);
            }
        }
        if (swGeos.length) scene.add(new THREE.Mesh(mergeGeometries(swGeos, false), new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.9 })));

        // ---- Buildings + Parks ----
        const ZONES = ['residential', 'commercial', 'office', 'residential', 'civic', 'residential'];
        const ZONE_COLORS = { residential: 0x8B7355, commercial: 0x556677, office: 0x667788, civic: 0x557788 };
        const buildGeos = {};
        const treePositions = [];
        const entries = [];

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const bx = col * (blockW + roadW) + roadW + blockW / 2;
                const bz = row * (blockH + roadW) + roadW + blockH / 2;
                const zone = (col === Math.floor(cols / 2) && row === Math.floor(rows / 2)) ? 'park' : ZONES[Math.floor(seeded() * ZONES.length)];

                if (zone === 'park') {
                    const pg = new THREE.PlaneGeometry(blockW - 2, blockH - 2);
                    pg.rotateX(-Math.PI / 2); pg.translate(bx, 0.04, bz);
                    scene.add(new THREE.Mesh(pg, new THREE.MeshStandardMaterial({ color: 0x2a5a2a })));
                    for (let t = 0; t < 6; t++) {
                        treePositions.push({ x: bx + (seeded() - 0.5) * (blockW - 8), z: bz + (seeded() - 0.5) * (blockH - 8), r: 1.5 + seeded() * 1.5, h: 3.5 + seeded() });
                    }
                } else {
                    const zoneColor = ZONE_COLORS[zone] || 0x667788;
                    const count = 1 + Math.floor(seeded() * 2);
                    for (let b = 0; b < count; b++) {
                        const h = zone === 'office' ? 12 + seeded() * 30 : 5 + seeded() * 20;
                        const w = 10 + seeded() * 25, d = 10 + seeded() * 18;
                        const ox = (seeded() - 0.5) * (blockW - w - 4), oz = (seeded() - 0.5) * (blockH - d - 4);
                        if (!buildGeos[zoneColor]) buildGeos[zoneColor] = [];
                        const bg = new THREE.BoxGeometry(w, h, d); bg.translate(bx + ox, h / 2, bz + oz);
                        buildGeos[zoneColor].push(bg);
                    }
                }
                entries.push({ x: bx, z: bz + blockH / 2 - 2, blockRow: row, blockCol: col, zone });
                entries.push({ x: bx, z: bz - blockH / 2 + 2, blockRow: row, blockCol: col, zone });
            }
        }

        for (const [c, geos] of Object.entries(buildGeos)) {
            if (geos.length === 0) continue;
            const m = new THREE.Mesh(mergeGeometries(geos, false), new THREE.MeshStandardMaterial({ color: parseInt(c), roughness: 0.7 }));
            m.castShadow = true;
            scene.add(m);
        }

        // ---- Instanced Trees ----
        if (treePositions.length > 0) {
            const tGeo = new THREE.CylinderGeometry(0.3, 0.4, 3, 6);
            const tMesh = new THREE.InstancedMesh(tGeo, new THREE.MeshStandardMaterial({ color: 0x4a3520 }), treePositions.length);
            const cGeo = new THREE.SphereGeometry(1, 8, 6);
            const cMesh = new THREE.InstancedMesh(cGeo, new THREE.MeshStandardMaterial({ color: 0x228833 }), treePositions.length);
            const dm = new THREE.Object3D();
            for (let i = 0; i < treePositions.length; i++) {
                const t = treePositions[i];
                dm.position.set(t.x, 1.5, t.z); dm.scale.set(1, 1, 1); dm.updateMatrix();
                tMesh.setMatrixAt(i, dm.matrix);
                dm.position.set(t.x, t.h, t.z); dm.scale.set(t.r, t.r, t.r); dm.updateMatrix();
                cMesh.setMatrixAt(i, dm.matrix);
            }
            tMesh.instanceMatrix.needsUpdate = true; cMesh.instanceMatrix.needsUpdate = true;
            tMesh.castShadow = true; cMesh.castShadow = true;
            scene.add(tMesh); scene.add(cMesh);
        }

        // ---- Road Network ----
        const roadNetwork = new RoadNetwork();
        roadNetwork.buildFromGrid(cols, rows, blockW, blockH, roadW, laneW, lanesPerDir);

        // ---- Traffic Lights ----
        const trafficMgr = new TrafficControllerManager();
        trafficMgr.initFromNetwork(roadNetwork);

        // Instanced traffic light visuals
        const tlPositions = [];
        for (const nodeId in roadNetwork.nodes) {
            const node = roadNetwork.nodes[nodeId];
            const ctrl = trafficMgr.getController(nodeId);
            if (!ctrl) continue;
            for (const approach of node.approaches) {
                const offset = roadW / 2 - 1;
                let px = node.x, pz = node.z;
                if (approach === 'N') pz -= offset;
                else if (approach === 'S') pz += offset;
                else if (approach === 'W') px -= offset;
                else if (approach === 'E') px += offset;
                tlPositions.push({ x: px, z: pz, nodeId, approach });
            }
        }

        const poleGeo = new THREE.CylinderGeometry(0.1, 0.1, 5, 4); poleGeo.translate(0, 2.5, 0);
        const poleMesh = new THREE.InstancedMesh(poleGeo, new THREE.MeshStandardMaterial({ color: 0x333333 }), Math.max(1, tlPositions.length));
        poleMesh.count = tlPositions.length; poleMesh.frustumCulled = false;

        const bulbGeo = new THREE.SphereGeometry(0.3, 6, 4); bulbGeo.translate(0, 5.5, 0);
        const bulbMesh = new THREE.InstancedMesh(bulbGeo, new THREE.MeshBasicMaterial({ color: 0xff0000 }), Math.max(1, tlPositions.length));
        bulbMesh.count = tlPositions.length; bulbMesh.frustumCulled = false;
        const bulbColors = new Float32Array(Math.max(1, tlPositions.length) * 3);
        bulbMesh.instanceColor = new THREE.InstancedBufferAttribute(bulbColors, 3);

        const tlDummy = new THREE.Object3D();
        for (let i = 0; i < tlPositions.length; i++) {
            tlDummy.position.set(tlPositions[i].x, 0, tlPositions[i].z);
            tlDummy.updateMatrix();
            poleMesh.setMatrixAt(i, tlDummy.matrix);
            bulbMesh.setMatrixAt(i, tlDummy.matrix);
        }
        poleMesh.instanceMatrix.needsUpdate = true;
        bulbMesh.instanceMatrix.needsUpdate = true;
        scene.add(poleMesh); scene.add(bulbMesh);

        return {
            roadNetwork,
            trafficMgr,
            entries,
            width: cityW,
            height: cityH,
            tlPositions,
            bulbMesh,
        };
    }
}
