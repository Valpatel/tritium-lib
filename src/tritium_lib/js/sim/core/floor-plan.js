// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * FloorPlan — generates walkable indoor navigation grids from building footprints.
 *
 * Given a building polygon and floor count, generates:
 * - Walkable floor grids per level (1m resolution)
 * - Interior walls creating room layouts
 * - Doors between rooms
 * - Stairwell positions connecting floors
 * - Entry points from exterior entrances
 *
 * Inspired by arnis tiled-interior approach but adapted for continuous coords.
 *
 * Usage:
 *   const plan = FloorPlan.generate(polygon, { floors: 4, entrances: [{x,z}] });
 *   const path = plan.findPath(startFloor, startX, startZ, endFloor, endX, endZ);
 */

/**
 * @typedef {Object} FloorLevel
 * @property {number} floor — floor index (0 = ground)
 * @property {number} y — world Y position
 * @property {Set<string>} walkable — set of "x,z" grid keys that are walkable
 * @property {Array<{x1,z1,x2,z2}>} walls — interior wall segments
 * @property {Array<{x,z,connects}>} doors — door positions between rooms
 * @property {Array<{x,z}>} stairs — stairwell positions (connect to floor above)
 */

/**
 * @typedef {Object} FloorPlanData
 * @property {Array<FloorLevel>} levels
 * @property {Array<{x,z,floor}>} entries — entry points from outside
 * @property {number} floorHeight — meters per floor
 * @property {Array<[number,number]>} polygon — building footprint
 */

export class FloorPlan {
    /**
     * Generate floor plan from building polygon.
     *
     * @param {Array<[number,number]>} polygon — building footprint [[x,z], ...]
     * @param {Object} options
     * @param {number} [options.floors=1] — number of floors
     * @param {number} [options.floorHeight=3] — meters per floor
     * @param {number} [options.gridRes=1] — grid resolution in meters
     * @param {number} [options.roomSize=5] — target room size in meters
     * @param {Array<{x,z}>} [options.entrances=[]] — exterior entrance positions
     * @returns {FloorPlanData}
     */
    static generate(polygon, options = {}) {
        const floors = options.floors || 1;
        const floorHeight = options.floorHeight || 3;
        const gridRes = options.gridRes || 1;
        const roomSize = options.roomSize || 5;
        const entrances = options.entrances || [];

        // Compute bounding box
        let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
        for (const [x, z] of polygon) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        const width = maxX - minX;
        const depth = maxZ - minZ;

        // Generate levels
        const levels = [];
        for (let f = 0; f < floors; f++) {
            const level = FloorPlan._generateLevel(
                polygon, f, f * floorHeight, gridRes, roomSize, minX, minZ, maxX, maxZ
            );
            levels.push(level);
        }

        // Add stairs connecting floors
        if (floors > 1) {
            // Place stairwells: one near center, one near each end
            const cx = (minX + maxX) / 2;
            const cz = (minZ + maxZ) / 2;
            const stairPositions = [
                { x: cx, z: cz },
            ];
            // Add corner stairs for larger buildings
            if (width > 15 && depth > 15) {
                stairPositions.push(
                    { x: minX + width * 0.2, z: minZ + depth * 0.2 },
                    { x: minX + width * 0.8, z: minZ + depth * 0.8 },
                );
            }

            for (const level of levels) {
                for (const sp of stairPositions) {
                    // Check if stair position is inside building
                    if (FloorPlan._pointInPolygon(sp.x, sp.z, polygon)) {
                        level.stairs.push({ x: sp.x, z: sp.z });
                        // Mark stairwell area as walkable
                        for (let dx = -1; dx <= 1; dx++) {
                            for (let dz = -1; dz <= 1; dz++) {
                                level.walkable.add(`${Math.round(sp.x + dx)},${Math.round(sp.z + dz)}`);
                            }
                        }
                    }
                }
            }
        }

        // Map exterior entrances to ground floor entry points
        const entries = [];
        for (const ent of entrances) {
            // Find nearest walkable point on ground floor
            const nearest = FloorPlan._nearestWalkable(levels[0], ent.x, ent.z);
            if (nearest) {
                entries.push({ x: nearest.x, z: nearest.z, floor: 0 });
            }
        }
        // If no explicit entrances, create entry at nearest polygon edge to centroid
        if (entries.length === 0 && levels[0].walkable.size > 0) {
            const cx = (minX + maxX) / 2;
            const cz = minZ + 1;  // Near south edge
            entries.push({ x: Math.round(cx), z: Math.round(cz), floor: 0 });
        }

        return {
            levels,
            entries,
            floorHeight,
            polygon,
        };
    }

    /**
     * Generate a single floor level.
     */
    static _generateLevel(polygon, floor, y, gridRes, roomSize, minX, minZ, maxX, maxZ) {
        const walkable = new Set();
        const walls = [];
        const doors = [];

        // Fill walkable area using point-in-polygon
        for (let x = Math.ceil(minX); x <= Math.floor(maxX); x += gridRes) {
            for (let z = Math.ceil(minZ); z <= Math.floor(maxZ); z += gridRes) {
                if (FloorPlan._pointInPolygon(x, z, polygon)) {
                    // Inset 0.5m from walls
                    if (FloorPlan._pointInPolygon(x, z, polygon, 0.5)) {
                        walkable.add(`${x},${z}`);
                    }
                }
            }
        }

        // Create room grid using interior walls
        const numRoomsX = Math.max(1, Math.floor((maxX - minX) / roomSize));
        const numRoomsZ = Math.max(1, Math.floor((maxZ - minZ) / roomSize));
        const cellW = (maxX - minX) / numRoomsX;
        const cellH = (maxZ - minZ) / numRoomsZ;

        // Vertical interior walls
        for (let i = 1; i < numRoomsX; i++) {
            const wx = minX + i * cellW;
            walls.push({ x1: wx, z1: minZ + 1, x2: wx, z2: maxZ - 1 });
            // Door in each wall (at center of cell boundary)
            for (let j = 0; j < numRoomsZ; j++) {
                const dz = minZ + (j + 0.5) * cellH;
                if (FloorPlan._pointInPolygon(wx, dz, polygon)) {
                    doors.push({ x: wx, z: dz, connects: 'ew' });
                    // Make door area walkable
                    walkable.add(`${Math.round(wx)},${Math.round(dz)}`);
                    walkable.add(`${Math.round(wx - 1)},${Math.round(dz)}`);
                    walkable.add(`${Math.round(wx + 1)},${Math.round(dz)}`);
                }
            }
        }

        // Horizontal interior walls
        for (let j = 1; j < numRoomsZ; j++) {
            const wz = minZ + j * cellH;
            walls.push({ x1: minX + 1, z1: wz, x2: maxX - 1, z2: wz });
            // Door in each wall
            for (let i = 0; i < numRoomsX; i++) {
                const dx = minX + (i + 0.5) * cellW;
                if (FloorPlan._pointInPolygon(dx, wz, polygon)) {
                    doors.push({ x: dx, z: wz, connects: 'ns' });
                    walkable.add(`${Math.round(dx)},${Math.round(wz)}`);
                    walkable.add(`${Math.round(dx)},${Math.round(wz - 1)}`);
                    walkable.add(`${Math.round(dx)},${Math.round(wz + 1)}`);
                }
            }
        }

        // Remove wall positions from walkable set
        for (const wall of walls) {
            const len = Math.sqrt((wall.x2 - wall.x1) ** 2 + (wall.z2 - wall.z1) ** 2);
            const steps = Math.ceil(len / gridRes);
            for (let s = 0; s <= steps; s++) {
                const t = s / steps;
                const wx = Math.round(wall.x1 + t * (wall.x2 - wall.x1));
                const wz = Math.round(wall.z1 + t * (wall.z2 - wall.z1));
                // Don't remove walkable where doors are
                const isDoor = doors.some(d => Math.abs(d.x - wx) < 1.5 && Math.abs(d.z - wz) < 1.5);
                if (!isDoor) {
                    walkable.delete(`${wx},${wz}`);
                }
            }
        }

        return { floor, y, walkable, walls, doors, stairs: [] };
    }

    /**
     * Find path between two points across floors.
     *
     * @param {FloorPlanData} plan
     * @param {number} startFloor
     * @param {number} startX
     * @param {number} startZ
     * @param {number} endFloor
     * @param {number} endX
     * @param {number} endZ
     * @returns {Array<{x,z,floor,y}>|null} — waypoint list or null if no path
     */
    static findPath(plan, startFloor, startX, startZ, endFloor, endX, endZ) {
        // Same floor: simple A* on walkable grid
        if (startFloor === endFloor) {
            const level = plan.levels[startFloor];
            if (!level) return null;
            return FloorPlan._astarFloor(level, startX, startZ, endX, endZ, plan.floorHeight);
        }

        // Different floors: path to stairs → stairs to target floor → path to dest
        const startLevel = plan.levels[startFloor];
        const endLevel = plan.levels[endFloor];
        if (!startLevel || !endLevel) return null;

        // Find nearest stair on start floor
        let bestStair = null, bestDist = Infinity;
        for (const s of startLevel.stairs) {
            const d = Math.hypot(s.x - startX, s.z - startZ);
            if (d < bestDist) {
                bestDist = d;
                bestStair = s;
            }
        }
        if (!bestStair) return null;

        // Path from start to stairs on start floor
        const toStairs = FloorPlan._astarFloor(startLevel, startX, startZ, bestStair.x, bestStair.z, plan.floorHeight);
        if (!toStairs) return null;

        // Path from stairs to dest on end floor
        const fromStairs = FloorPlan._astarFloor(endLevel, bestStair.x, bestStair.z, endX, endZ, plan.floorHeight);
        if (!fromStairs) return null;

        // Combine with floor transition
        const path = [...toStairs];
        // Add vertical transition waypoints
        const dir = endFloor > startFloor ? 1 : -1;
        for (let f = startFloor + dir; f !== endFloor + dir; f += dir) {
            path.push({ x: bestStair.x, z: bestStair.z, floor: f, y: f * plan.floorHeight });
        }
        path.push(...fromStairs);

        return path;
    }

    /**
     * Simple A* pathfinding on a single floor's walkable grid.
     */
    static _astarFloor(level, sx, sz, ex, ez, floorHeight) {
        const startKey = `${Math.round(sx)},${Math.round(sz)}`;
        const endKey = `${Math.round(ex)},${Math.round(ez)}`;

        if (!level.walkable.has(startKey) || !level.walkable.has(endKey)) {
            // Try nearest walkable
            const sn = FloorPlan._nearestWalkable(level, sx, sz);
            const en = FloorPlan._nearestWalkable(level, ex, ez);
            if (!sn || !en) return null;
            return FloorPlan._astarFloor(level, sn.x, sn.z, en.x, en.z, floorHeight);
        }

        const openSet = new Map();  // key -> { g, f, parent }
        const closedSet = new Set();
        const h = (key) => {
            const [kx, kz] = key.split(',').map(Number);
            return Math.hypot(kx - Math.round(ex), kz - Math.round(ez));
        };

        openSet.set(startKey, { g: 0, f: h(startKey), parent: null });

        const neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]];
        let iterations = 0;
        const maxIter = 5000;

        while (openSet.size > 0 && iterations++ < maxIter) {
            // Find lowest f
            let bestKey = null, bestF = Infinity;
            for (const [key, data] of openSet) {
                if (data.f < bestF) {
                    bestF = data.f;
                    bestKey = key;
                }
            }

            if (bestKey === endKey) {
                // Reconstruct path
                const path = [];
                let curr = bestKey;
                while (curr) {
                    const [cx, cz] = curr.split(',').map(Number);
                    path.unshift({ x: cx, z: cz, floor: level.floor, y: level.y });
                    curr = openSet.get(curr)?.parent || closedSet.has(curr) ? null : null;
                }
                // Reconstruct via parent chain
                const pathMap = new Map();
                for (const [k, v] of openSet) pathMap.set(k, v.parent);
                const result = [];
                let c = endKey;
                while (c) {
                    const [cx, cz] = c.split(',').map(Number);
                    result.unshift({ x: cx, z: cz, floor: level.floor, y: level.y });
                    c = pathMap.get(c);
                }
                return result;
            }

            const currentData = openSet.get(bestKey);
            openSet.delete(bestKey);
            closedSet.add(bestKey);

            const [bx, bz] = bestKey.split(',').map(Number);
            for (const [dx, dz] of neighbors) {
                const nx = bx + dx, nz = bz + dz;
                const nKey = `${nx},${nz}`;
                if (closedSet.has(nKey) || !level.walkable.has(nKey)) continue;

                const moveCost = Math.abs(dx) + Math.abs(dz) > 1 ? 1.414 : 1;
                const tentG = currentData.g + moveCost;

                const existing = openSet.get(nKey);
                if (!existing || tentG < existing.g) {
                    openSet.set(nKey, { g: tentG, f: tentG + h(nKey), parent: bestKey });
                }
            }
        }

        return null;  // No path found
    }

    /**
     * Point-in-polygon test (ray casting).
     * Optional inset parameter shrinks the polygon boundary.
     */
    static _pointInPolygon(x, z, polygon, inset = 0) {
        let inside = false;
        const n = polygon.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const [xi, zi] = polygon[i];
            const [xj, zj] = polygon[j];
            if (((zi > z) !== (zj > z)) && (x < (xj - xi) * (z - zi) / (zj - zi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    /**
     * Find nearest walkable grid cell to a point.
     */
    static _nearestWalkable(level, x, z) {
        let best = null, bestDist = Infinity;
        const rx = Math.round(x), rz = Math.round(z);
        // Search in expanding radius
        for (let r = 0; r < 20; r++) {
            for (let dx = -r; dx <= r; dx++) {
                for (let dz = -r; dz <= r; dz++) {
                    if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue;
                    const key = `${rx + dx},${rz + dz}`;
                    if (level.walkable.has(key)) {
                        const d = Math.hypot(dx, dz);
                        if (d < bestDist) {
                            bestDist = d;
                            best = { x: rx + dx, z: rz + dz };
                        }
                    }
                }
            }
            if (best) return best;
        }
        return null;
    }
}
