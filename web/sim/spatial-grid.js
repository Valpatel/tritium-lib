// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * SpatialGrid — O(1) proximity queries for city sim entities.
 *
 * Divides the world into cells. Each entity is assigned to a cell.
 * Queries return entities in the same cell + 8 neighbors = 9 cells total.
 *
 * Replaces O(n²) loops in vehicle.tick() and pedestrian.tick() with
 * O(1) cell lookup + O(k) iteration where k = entities in 9 cells.
 */

const DEFAULT_CELL_SIZE = 20;  // meters — covers ~2 car lengths

export class SpatialGrid {
    /**
     * @param {number} cellSize — grid cell size in meters
     */
    constructor(cellSize = DEFAULT_CELL_SIZE) {
        this.cellSize = cellSize;
        this._cells = new Map();  // "cx,cz" → Set<entity>
        this._entityCell = new Map();  // entityId → "cx,cz"
    }

    /**
     * Clear the grid (call at start of each tick).
     */
    clear() {
        this._cells.clear();
        this._entityCell.clear();
    }

    /**
     * Insert an entity into the grid.
     * @param {Object} entity — must have { id, x, z }
     */
    insert(entity) {
        const key = this._cellKey(entity.x, entity.z);

        if (!this._cells.has(key)) {
            this._cells.set(key, new Set());
        }
        this._cells.get(key).add(entity);
        this._entityCell.set(entity.id, key);
    }

    /**
     * Get all entities in the same cell and 8 neighbors.
     * @param {number} x
     * @param {number} z
     * @returns {Array} entities in 3×3 neighborhood
     */
    getNearby(x, z) {
        const cx = Math.floor(x / this.cellSize);
        const cz = Math.floor(z / this.cellSize);
        const result = [];

        for (let dx = -1; dx <= 1; dx++) {
            for (let dz = -1; dz <= 1; dz++) {
                const key = `${cx + dx},${cz + dz}`;
                const cell = this._cells.get(key);
                if (cell) {
                    for (const entity of cell) {
                        result.push(entity);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Get entities on a specific road edge.
     * @param {string} edgeId
     * @param {Array} vehicles — all vehicles (filters by edge)
     * @returns {Array} vehicles on this edge
     */
    getOnEdge(edgeId, vehicles) {
        // Simple filter — for edge-specific queries, spatial grid isn't optimal
        // but still faster than iterating all vehicles when combined with getNearby
        return vehicles.filter(v => v.edge?.id === edgeId);
    }

    _cellKey(x, z) {
        return `${Math.floor(x / this.cellSize)},${Math.floor(z / this.cellSize)}`;
    }

    /**
     * Stats for debugging.
     */
    stats() {
        let maxCellSize = 0;
        let totalEntities = 0;
        for (const [, cell] of this._cells) {
            maxCellSize = Math.max(maxCellSize, cell.size);
            totalEntities += cell.size;
        }
        return {
            cells: this._cells.size,
            entities: totalEntities,
            maxCellSize,
            avgCellSize: this._cells.size > 0 ? Math.round(totalEntities / this._cells.size * 10) / 10 : 0,
        };
    }
}
