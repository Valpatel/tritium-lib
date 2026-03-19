// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * SpatialHash3D — O(1) neighbor lookup for units in 3D space.
 *
 * Divides the world into cubic cells. Insert all units each frame,
 * then query getNearby(x, y, z) to get all units in the same cell + 26 neighbors.
 *
 * Works for ground units (y≈0), underground parking (y<0), overpasses (y>0),
 * and aircraft (y>>0). Units at different altitudes in the same XZ cell
 * are in different Y layers and won't collide.
 */

export class SpatialHash {
    /**
     * @param {number} cellSize - Cell size in meters (default 50)
     */
    constructor(cellSize = 50) {
        this.cellSize = cellSize;
        this.grid = new Map();
    }

    _key(x, y, z) {
        const cx = Math.floor(x / this.cellSize);
        const cy = Math.floor(y / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        // Pack three 10-bit signed coords into one integer
        // Range: ±512 cells = ±25,600m at 50m cells
        return ((cx & 0x3FF) | ((cy & 0x3FF) << 10) | ((cz & 0x3FF) << 20));
    }

    clear() {
        this.grid.clear();
    }

    /**
     * Insert a unit. Unit must have x, y (optional, defaults 0), z properties.
     */
    insert(item) {
        const key = this._key(item.x, item.y || 0, item.z);
        let cell = this.grid.get(key);
        if (!cell) { cell = []; this.grid.set(key, cell); }
        cell.push(item);
    }

    /**
     * Get all units in the same cell and 26 neighbors (3×3×3 cube).
     * For ground-only queries, most Y layers will be empty → fast.
     *
     * @param {number} x
     * @param {number} y - Altitude (0 for ground)
     * @param {number} z
     * @returns {Array}
     */
    getNearby(x, y = 0, z) {
        // Handle 2-arg call: getNearby(x, z)
        if (z === undefined) { z = y; y = 0; }

        const cx = Math.floor(x / this.cellSize);
        const cy = Math.floor(y / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        const result = [];

        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const key = ((cx + dx) & 0x3FF) | (((cy + dy) & 0x3FF) << 10) | (((cz + dz) & 0x3FF) << 20);
                    const cell = this.grid.get(key);
                    if (cell) {
                        for (let i = 0; i < cell.length; i++) result.push(cell[i]);
                    }
                }
            }
        }
        return result;
    }

    /**
     * 2D convenience: get nearby at ground level only (y=0).
     * Equivalent to getNearby(x, 0, z) but checks only 9 cells instead of 27.
     */
    getNearby2D(x, z) {
        const cx = Math.floor(x / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        const result = [];
        const cy = 0; // ground level

        for (let dx = -1; dx <= 1; dx++) {
            for (let dz = -1; dz <= 1; dz++) {
                const key = ((cx + dx) & 0x3FF) | ((cy & 0x3FF) << 10) | (((cz + dz) & 0x3FF) << 20);
                const cell = this.grid.get(key);
                if (cell) {
                    for (let i = 0; i < cell.length; i++) result.push(cell[i]);
                }
            }
        }
        return result;
    }
}
