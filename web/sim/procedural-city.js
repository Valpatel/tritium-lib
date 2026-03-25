// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * ProceduralCity — generates a fictional city without OSM data.
 *
 * For demo mode, offline operation, or testing. Produces the same
 * data format as /api/geo/city-data so it plugs into the existing
 * rendering and simulation pipeline.
 *
 * Generation approach:
 * - Grid road network with organic perturbation
 * - Zone-based building placement (residential, commercial, industrial)
 * - Tree/park placement in parks
 * - Procedural building heights based on zone + distance from center
 */

/**
 * Generate a procedural city.
 *
 * @param {Object} options
 * @param {number} [options.radius=300] — city radius in meters
 * @param {number} [options.blockSize=60] — block size in meters
 * @param {number} [options.roadWidth=12] — road width in meters
 * @param {number} [options.seed=42] — random seed
 * @param {string} [options.style='grid'] — 'grid' | 'organic' | 'radial'
 * @returns {Object} city-data format JSON
 */
export function generateProceduralCity(options = {}) {
    const radius = options.radius || 300;
    const blockSize = options.blockSize || 60;
    const roadWidth = options.roadWidth || 12;
    const seed = options.seed || 42;
    const style = options.style || 'grid';

    // Seeded RNG
    let _s = seed;
    const rng = () => { _s = (_s * 16807) % 2147483647; return (_s - 1) / 2147483646; };

    const buildings = [];
    const roads = [];
    const trees = [];
    const landuse = [];
    const barriers = [];

    // Generate road grid
    const gridSpacing = blockSize + roadWidth;
    const halfRadius = radius * 0.8;
    const cols = Math.floor(halfRadius * 2 / gridSpacing);
    const rows = Math.floor(halfRadius * 2 / gridSpacing);
    const startX = -halfRadius;
    const startZ = -halfRadius;

    let roadId = 1;
    let bldgId = 1;

    // Horizontal roads
    for (let r = 0; r <= rows; r++) {
        const z = startZ + r * gridSpacing;
        const perturbZ = style === 'organic' ? (rng() - 0.5) * 10 : 0;
        roads.push({
            id: roadId++,
            points: [[-halfRadius, z + perturbZ], [halfRadius, z + perturbZ]],
            class: r === Math.floor(rows / 2) ? 'primary' : 'residential',
            name: `${r + 1}th Street`,
            width: r === Math.floor(rows / 2) ? 14 : roadWidth,
            lanes: r === Math.floor(rows / 2) ? 4 : 2,
            surface: 'asphalt',
            oneway: false,
            bridge: false,
            tunnel: false,
            maxspeed: '',
        });
    }

    // Vertical roads
    for (let c = 0; c <= cols; c++) {
        const x = startX + c * gridSpacing;
        const perturbX = style === 'organic' ? (rng() - 0.5) * 10 : 0;
        roads.push({
            id: roadId++,
            points: [[x + perturbX, -halfRadius], [x + perturbX, halfRadius]],
            class: c === Math.floor(cols / 2) ? 'secondary' : 'residential',
            name: `${String.fromCharCode(65 + c % 26)} Avenue`,
            width: c === Math.floor(cols / 2) ? 10 : roadWidth,
            lanes: c === Math.floor(cols / 2) ? 3 : 2,
            surface: 'asphalt',
            oneway: false,
            bridge: false,
            tunnel: false,
            maxspeed: '',
        });
    }

    // Zone classification based on distance from center
    const getZone = (x, z) => {
        const dist = Math.sqrt(x * x + z * z);
        if (dist < radius * 0.2) return 'commercial';  // downtown
        if (dist < radius * 0.5) return 'mixed';
        if (dist > radius * 0.7 && rng() < 0.3) return 'industrial';
        return 'residential';
    };

    // Generate buildings in each block
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const bx = startX + c * gridSpacing + roadWidth / 2;
            const bz = startZ + r * gridSpacing + roadWidth / 2;
            const blockW = blockSize;
            const blockH = blockSize;

            const zone = getZone(bx + blockW / 2, bz + blockH / 2);

            // Park block (10% chance, more likely far from center)
            const dist = Math.sqrt((bx + blockW / 2) ** 2 + (bz + blockH / 2) ** 2);
            if (rng() < 0.1 + dist / radius * 0.1) {
                // Park
                landuse.push({
                    id: bldgId++,
                    polygon: [[bx + 2, bz + 2], [bx + blockW - 2, bz + 2],
                              [bx + blockW - 2, bz + blockH - 2], [bx + 2, bz + blockH - 2]],
                    type: 'park',
                    name: `${String.fromCharCode(65 + c % 26)}${r + 1} Park`,
                });
                // Trees in park
                for (let t = 0; t < 4 + Math.floor(rng() * 4); t++) {
                    trees.push({
                        pos: [bx + 5 + rng() * (blockW - 10), bz + 5 + rng() * (blockH - 10)],
                        species: rng() < 0.5 ? 'oak' : 'maple',
                        height: 5 + rng() * 6,
                        leaf_type: 'broadleaved',
                    });
                }
                continue;
            }

            // Buildings in block
            const numBldgs = zone === 'commercial' ? 1 + Math.floor(rng() * 2)
                           : zone === 'industrial' ? 1
                           : 2 + Math.floor(rng() * 3);

            for (let b = 0; b < numBldgs; b++) {
                const bw = zone === 'industrial' ? blockW * 0.7 + rng() * blockW * 0.2
                         : zone === 'commercial' ? blockW * 0.6 + rng() * blockW * 0.3
                         : 8 + rng() * 20;
                const bd = zone === 'industrial' ? blockH * 0.6 + rng() * blockH * 0.2
                         : zone === 'commercial' ? blockH * 0.6 + rng() * blockH * 0.3
                         : 8 + rng() * 15;
                const ox = (rng() - 0.5) * (blockW - bw - 4);
                const oz = (rng() - 0.5) * (blockH - bd - 4);
                const cx = bx + blockW / 2 + ox;
                const cz = bz + blockH / 2 + oz;

                // Height based on zone and distance
                let height;
                if (zone === 'commercial') {
                    height = 12 + rng() * 40;  // 12-52m downtown
                } else if (zone === 'industrial') {
                    height = 6 + rng() * 8;     // 6-14m
                } else if (zone === 'mixed') {
                    height = 8 + rng() * 20;    // 8-28m
                } else {
                    height = 5 + rng() * 12;    // 5-17m residential
                }

                const category = zone === 'commercial' ? 'commercial'
                               : zone === 'industrial' ? 'industrial'
                               : 'residential';

                buildings.push({
                    id: bldgId++,
                    polygon: [
                        [cx - bw / 2, cz - bd / 2],
                        [cx + bw / 2, cz - bd / 2],
                        [cx + bw / 2, cz + bd / 2],
                        [cx - bw / 2, cz + bd / 2],
                    ],
                    height: Math.round(height * 10) / 10,
                    type: zone === 'commercial' ? 'office' : zone === 'industrial' ? 'warehouse' : 'apartments',
                    category,
                    name: '',
                    levels: Math.max(1, Math.floor(height / 3)),
                    roof_shape: category === 'residential' && height < 12 ? 'gabled' : 'flat',
                    colour: '',
                    material: '',
                    address: `${Math.floor(rng() * 900 + 100)}`,
                    street: roads[c]?.name || '',
                });

                // Street trees along block edges
                if (b === 0 && rng() < 0.5) {
                    trees.push({
                        pos: [bx + 3, bz + 3],
                        species: 'plane',
                        height: 8 + rng() * 4,
                        leaf_type: 'broadleaved',
                    });
                }
            }
        }
    }

    return {
        center: { lat: 0, lng: 0 },
        radius,
        schema_version: 2,
        buildings,
        roads,
        trees,
        landuse,
        barriers: [],
        water: [],
        entrances: [],
        pois: [],
        stats: {
            buildings: buildings.length,
            roads: roads.length,
            trees: trees.length,
            landuse: landuse.length,
            barriers: 0,
            water: 0,
            entrances: 0,
            pois: 0,
        },
        _procedural: true,  // flag indicating this is generated, not from OSM
    };
}
