// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Road Network — directed graph for vehicle simulation.
 *
 * The road network is a graph where:
 * - Nodes = intersections (with approach directions)
 * - Edges = road segments (with lanes, length, geometry)
 * - Lane connections define how lanes connect across intersections
 *
 * Vehicles navigate this graph using:
 * - Dijkstra for route planning (sequence of edges)
 * - Lane connections for intersection transitions
 * - IDM for speed/acceleration on each segment
 *
 * This is a pure data module — no rendering, no Three.js dependencies.
 */

// ============================================================
// ROAD NETWORK CLASS
// ============================================================

export class RoadNetwork {
    constructor() {
        this.nodes = {};       // { id: { id, x, z, col, row, approaches[], type } }
        this.edges = [];       // [{ id, from, to, horizontal, lanes[], length, ax,az,bx,bz, numLanesPerDir, laneWidth }]
        this.adjList = {};     // nodeId → [edgeIndex]
        this.edgeById = {};    // edgeId → edge
        this.laneConnections = {}; // "edgeId:lane:nodeId" → [{ toEdge, toLane, turnType, bezierControl }]
    }

    /**
     * Build a grid city road network.
     *
     * @param {number} cols - Number of block columns
     * @param {number} rows - Number of block rows
     * @param {number} blockW - Block width (meters)
     * @param {number} blockH - Block height (meters)
     * @param {number} roadW - Total road width including sidewalks (meters)
     * @param {number} laneW - Lane width (meters)
     * @param {number} lanesPerDir - Lanes per direction (default 2)
     */
    buildFromGrid(cols, rows, blockW, blockH, roadW, laneW = 3, lanesPerDir = 2) {
        this.nodes = {};
        this.edges = [];
        this.adjList = {};
        this.edgeById = {};
        this.laneConnections = {};

        const hRoadZ = (row) => row * (blockH + roadW) + roadW / 2;
        const vRoadX = (col) => col * (blockW + roadW) + roadW / 2;

        // Create intersection nodes
        for (let row = 0; row <= rows; row++) {
            for (let col = 0; col <= cols; col++) {
                const id = `${col}_${row}`;
                const isTop = row === 0;
                const isBot = row === rows;
                const isLeft = col === 0;
                const isRight = col === cols;
                const approaches = [];
                if (!isTop) approaches.push('N');
                if (!isBot) approaches.push('S');
                if (!isLeft) approaches.push('W');
                if (!isRight) approaches.push('E');

                this.nodes[id] = {
                    id, col, row,
                    x: vRoadX(col),
                    z: hRoadZ(row),
                    approaches,
                    type: approaches.length === 4 ? '4-way' :
                          approaches.length === 3 ? '3-way' :
                          approaches.length === 2 ? '2-way' : '1-way',
                };
                this.adjList[id] = [];
            }
        }

        // Create horizontal edges (between (col,row) and (col+1,row))
        for (let row = 0; row <= rows; row++) {
            for (let col = 0; col < cols; col++) {
                const fromId = `${col}_${row}`;
                const toId = `${col + 1}_${row}`;
                const z = hRoadZ(row);
                const x1 = vRoadX(col);
                const x2 = vRoadX(col + 1);
                const edgeId = `h_${col}_${row}`;

                const edge = {
                    id: edgeId,
                    from: fromId,
                    to: toId,
                    horizontal: true,
                    ax: x1, az: z, bx: x2, bz: z,
                    length: Math.abs(x2 - x1),
                    numLanesPerDir: lanesPerDir,
                    laneWidth: laneW,
                    lanes: this._buildLanes(lanesPerDir, laneW),
                };

                const idx = this.edges.length;
                this.edges.push(edge);
                this.edgeById[edgeId] = edge;
                this.adjList[fromId].push(idx);
                this.adjList[toId].push(idx);
            }
        }

        // Create vertical edges (between (col,row) and (col,row+1))
        for (let col = 0; col <= cols; col++) {
            for (let row = 0; row < rows; row++) {
                const fromId = `${col}_${row}`;
                const toId = `${col}_${row + 1}`;
                const x = vRoadX(col);
                const z1 = hRoadZ(row);
                const z2 = hRoadZ(row + 1);
                const edgeId = `v_${col}_${row}`;

                const edge = {
                    id: edgeId,
                    from: fromId,
                    to: toId,
                    horizontal: false,
                    ax: x, az: z1, bx: x, bz: z2,
                    length: Math.abs(z2 - z1),
                    numLanesPerDir: lanesPerDir,
                    laneWidth: laneW,
                    lanes: this._buildLanes(lanesPerDir, laneW),
                };

                const idx = this.edges.length;
                this.edges.push(edge);
                this.edgeById[edgeId] = edge;
                this.adjList[fromId].push(idx);
                this.adjList[toId].push(idx);
            }
        }

        // Build lane connections for every intersection
        this._buildAllLaneConnections();

        return this;
    }

    _buildLanes(lanesPerDir, laneW) {
        const lanes = [];
        for (let i = 0; i < lanesPerDir; i++) {
            // Forward (A→B): offset to the right of center
            lanes.push({ offset: -(lanesPerDir - 0.5 - i) * laneW, dir: 1 });
        }
        for (let i = 0; i < lanesPerDir; i++) {
            // Backward (B→A): offset to the left of center
            lanes.push({ offset: (i + 0.5) * laneW, dir: -1 });
        }
        return lanes;
    }

    /**
     * Build lane connections for all intersections.
     * For each edge arriving at a node, determine which lanes on which outgoing edges
     * the vehicle can transition to, and what type of turn it is.
     */
    _buildAllLaneConnections() {
        for (const nodeId in this.nodes) {
            const node = this.nodes[nodeId];
            const edgeIndices = this.adjList[nodeId];

            for (const edgeIdx of edgeIndices) {
                const edge = this.edges[edgeIdx];

                // Determine which lanes arrive at this node
                // Forward lanes (0..nPerDir-1): travel A→B, arrive at node 'to'
                // Backward lanes (nPerDir..2*nPerDir-1): travel B→A, arrive at node 'from'
                const n = edge.numLanesPerDir;

                // Forward lanes arriving at 'to' node
                if (edge.to === nodeId) {
                    for (let lane = 0; lane < n; lane++) {
                        this._buildLaneConnectionsForArrival(edge, lane, nodeId, 'forward');
                    }
                }

                // Backward lanes arriving at 'from' node
                if (edge.from === nodeId) {
                    for (let lane = n; lane < 2 * n; lane++) {
                        this._buildLaneConnectionsForArrival(edge, lane, nodeId, 'backward');
                    }
                }
            }
        }
    }

    _buildLaneConnectionsForArrival(arrivalEdge, arrivalLane, nodeId, direction) {
        const key = `${arrivalEdge.id}:${arrivalLane}:${nodeId}`;
        const connections = [];
        const node = this.nodes[nodeId];
        const n = arrivalEdge.numLanesPerDir;

        // Determine arrival direction (NSEW)
        const arrivalDir = this._getArrivalDirection(arrivalEdge, nodeId, direction);

        // Find all outgoing edges from this node (excluding U-turn on same edge)
        const edgeIndices = this.adjList[nodeId];

        for (const edgeIdx of edgeIndices) {
            const outEdge = this.edges[edgeIdx];
            if (outEdge === arrivalEdge) continue; // no U-turns

            // Determine departure direction and which lanes depart from this node
            const departures = this._getDepartureLanes(outEdge, nodeId);

            for (const dep of departures) {
                const turnType = this._getTurnType(arrivalDir, dep.direction);
                const bezierControl = this._getBezierControl(
                    arrivalEdge, arrivalLane, outEdge, dep.lane, nodeId, turnType
                );

                connections.push({
                    toEdge: outEdge,
                    toLane: dep.lane,
                    turnType,
                    bezierControl,
                    departureDir: dep.direction,
                });
            }
        }

        this.laneConnections[key] = connections;
    }

    /**
     * Get the approach direction when arriving at a node.
     */
    _getArrivalDirection(edge, nodeId, travelDir) {
        if (edge.horizontal) {
            if (travelDir === 'forward') {
                // A→B = going +X = arriving from West
                return 'W';
            } else {
                // B→A = going -X = arriving from East
                return 'E';
            }
        } else {
            if (travelDir === 'forward') {
                // A→B = going +Z = arriving from North
                return 'N';
            } else {
                // B→A = going -Z = arriving from South
                return 'S';
            }
        }
    }

    /**
     * Get lanes that depart from a node on a given edge.
     */
    _getDepartureLanes(edge, nodeId) {
        const n = edge.numLanesPerDir;
        const departures = [];

        // Forward lanes depart from 'from' node
        if (edge.from === nodeId) {
            const dir = edge.horizontal ? 'E' : 'S'; // A→B = +X or +Z
            for (let lane = 0; lane < n; lane++) {
                departures.push({ lane, direction: dir });
            }
        }

        // Backward lanes depart from 'to' node
        if (edge.to === nodeId) {
            const dir = edge.horizontal ? 'W' : 'N'; // B→A = -X or -Z
            for (let lane = n; lane < 2 * n; lane++) {
                departures.push({ lane, direction: dir });
            }
        }

        return departures;
    }

    /**
     * Determine turn type from arrival and departure directions.
     */
    _getTurnType(arrivalDir, departureDir) {
        const order = ['N', 'E', 'S', 'W'];
        const ai = order.indexOf(arrivalDir);
        const di = order.indexOf(departureDir);

        // Arrival direction is where you came FROM.
        // If arriving from W and departing E → straight
        // If arriving from W and departing N → right turn
        // If arriving from W and departing S → left turn

        // The "forward" direction when arriving from a given direction:
        const forwardDir = { 'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E' }[arrivalDir];
        const fi = order.indexOf(forwardDir);

        if (departureDir === forwardDir) return 'straight';

        // Clockwise from forward = right, counterclockwise = left
        const rightDir = order[(fi + 1) % 4];
        const leftDir = order[(fi + 3) % 4];

        if (departureDir === rightDir) return 'right';
        if (departureDir === leftDir) return 'left';
        return 'uturn'; // shouldn't happen (we filter out same-edge)
    }

    /**
     * Calculate Bezier control point for intersection turn.
     * For straight: center point (degenerate straight Bezier).
     * For turns: offset toward the corner the car should arc through.
     */
    _getBezierControl(fromEdge, fromLane, toEdge, toLane, nodeId, turnType) {
        const node = this.nodes[nodeId];

        if (turnType === 'straight') {
            return { x: node.x, z: node.z };
        }

        // Compute entry and exit world positions to find the corner to arc through
        const fromIsForward = fromLane < (fromEdge.numLanesPerDir || 2);
        const entryU = fromIsForward ? fromEdge.length : 0;
        const entry = this._roadToWorldSimple(fromEdge, fromLane, entryU);

        const toIsForward = toLane < (toEdge.numLanesPerDir || 2);
        const exitU = toIsForward ? 0 : toEdge.length;
        const exit = this._roadToWorldSimple(toEdge, toLane, exitU);

        // For turns: the control point is the corner where the entry tangent
        // and exit tangent intersect. For perpendicular grid roads this is simply:
        // (entry.x extended along entry direction, entry.z extended along exit direction)
        // i.e., the L-corner that makes entry→corner→exit a right angle.
        if (turnType === 'right' || turnType === 'left') {
            // The corner point: take one coordinate from entry, the other from exit.
            // For perpendicular roads, one of these combinations creates the correct
            // 90° arc. The right one is: extend entry straight, then turn to meet exit.
            // That means: if entry is moving horizontally, keep entry.z but use exit.x
            //             if entry is moving vertically, keep entry.x but use exit.z
            if (fromEdge.horizontal) {
                // Entry moves in X, so extend X from entry, use Z from exit
                return { x: exit.x, z: entry.z };
            } else {
                // Entry moves in Z, so extend Z from entry, use X from exit
                return { x: entry.x, z: exit.z };
            }
        }

        return { x: node.x, z: node.z };
    }

    /**
     * Simple world position from road coordinates (no imports needed).
     */
    _roadToWorldSimple(road, lane, u) {
        const t = Math.max(0, Math.min(1, u / road.length));
        const cx = road.ax + t * (road.bx - road.ax);
        const cz = road.az + t * (road.bz - road.az);
        const dx = road.bx - road.ax;
        const dz = road.bz - road.az;
        const len = Math.sqrt(dx * dx + dz * dz) || 1;
        const perpX = -dz / len;
        const perpZ = dx / len;
        const nPerDir = road.numLanesPerDir || 2;
        const laneWidth = road.laneWidth || 3;
        let offset;
        if (lane < nPerDir) {
            offset = (lane + 0.5) * laneWidth;
        } else {
            offset = -((lane - nPerDir) + 0.5) * laneWidth;
        }
        return { x: cx + perpX * offset, z: cz + perpZ * offset };
    }

    /**
     * Get lane connections for a vehicle arriving at a node.
     *
     * @param {string} edgeId - Current edge ID
     * @param {number} lane - Current lane
     * @param {string} nodeId - Intersection node ID
     * @returns {Array} Array of { toEdge, toLane, turnType, bezierControl, departureDir }
     */
    getLaneConnections(edgeId, lane, nodeId) {
        const key = `${edgeId}:${lane}:${nodeId}`;
        return this.laneConnections[key] || [];
    }

    /**
     * Pick a random outgoing connection for a vehicle.
     * Prefers straight-through, then right, then left.
     *
     * @param {string} edgeId - Current edge ID
     * @param {number} lane - Current lane
     * @param {string} nodeId - Intersection node ID
     * @returns {{ toEdge, toLane, turnType, bezierControl } | null}
     */
    pickRandomConnection(edgeId, lane, nodeId) {
        const connections = this.getLaneConnections(edgeId, lane, nodeId);
        if (connections.length === 0) return null;

        // Weight: straight 60%, right 20%, left 20%
        const straight = connections.filter(c => c.turnType === 'straight');
        const right = connections.filter(c => c.turnType === 'right');
        const left = connections.filter(c => c.turnType === 'left');

        const r = Math.random();
        if (straight.length > 0 && r < 0.6) {
            return straight[Math.floor(Math.random() * straight.length)];
        }
        if (right.length > 0 && r < 0.8) {
            return right[Math.floor(Math.random() * right.length)];
        }
        if (left.length > 0) {
            return left[Math.floor(Math.random() * left.length)];
        }
        // Fallback: any connection
        return connections[Math.floor(Math.random() * connections.length)];
    }

    /**
     * Find shortest path between two nodes using Dijkstra.
     *
     * @param {string} fromNodeId - Start intersection
     * @param {string} toNodeId - End intersection
     * @returns {Array<{ edge, nodeId }>} Sequence of edges and nodes to traverse
     */
    findPath(fromNodeId, toNodeId) {
        if (fromNodeId === toNodeId) return [];

        const dist = {};
        const prev = {};
        const visited = new Set();
        const queue = []; // Simple priority queue (array sorted by distance)

        for (const id in this.nodes) {
            dist[id] = Infinity;
        }
        dist[fromNodeId] = 0;
        queue.push({ id: fromNodeId, dist: 0 });

        while (queue.length > 0) {
            // Sort and pop minimum (simple but O(n log n) — fine for our grid size)
            queue.sort((a, b) => a.dist - b.dist);
            const { id: current } = queue.shift();

            if (current === toNodeId) break;
            if (visited.has(current)) continue;
            visited.add(current);

            for (const edgeIdx of this.adjList[current]) {
                const edge = this.edges[edgeIdx];
                const neighbor = edge.from === current ? edge.to : edge.from;
                const newDist = dist[current] + edge.length;

                if (newDist < dist[neighbor]) {
                    dist[neighbor] = newDist;
                    prev[neighbor] = { nodeId: current, edgeIdx };
                    queue.push({ id: neighbor, dist: newDist });
                }
            }
        }

        // Reconstruct path
        if (dist[toNodeId] === Infinity) return []; // no path

        const path = [];
        let current = toNodeId;
        while (current !== fromNodeId) {
            const { nodeId: prevNode, edgeIdx } = prev[current];
            path.unshift({ edge: this.edges[edgeIdx], nodeId: current });
            current = prevNode;
        }

        return path;
    }

    /**
     * Get all edges connected to a node.
     */
    getEdgesForNode(nodeId) {
        return (this.adjList[nodeId] || []).map(idx => this.edges[idx]);
    }

    /**
     * Get the node at the other end of an edge from a given node.
     */
    getOtherNode(edge, nodeId) {
        return edge.from === nodeId ? edge.to : edge.from;
    }
}

// ============================================================
// BEZIER TURN PATH
// ============================================================

/**
 * Compute position on a quadratic Bezier curve.
 *
 * @param {{ x: number, z: number }} p0 - Entry point
 * @param {{ x: number, z: number }} p1 - Control point (intersection center)
 * @param {{ x: number, z: number }} p2 - Exit point
 * @param {number} t - Parameter [0, 1]
 * @returns {{ x: number, z: number }}
 */
export function bezierPosition(p0, p1, p2, t) {
    const u = 1 - t;
    return {
        x: u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
        z: u * u * p0.z + 2 * u * t * p1.z + t * t * p2.z,
    };
}

/**
 * Compute tangent (direction) on a quadratic Bezier curve.
 *
 * @param {{ x: number, z: number }} p0 - Entry point
 * @param {{ x: number, z: number }} p1 - Control point
 * @param {{ x: number, z: number }} p2 - Exit point
 * @param {number} t - Parameter [0, 1]
 * @returns {{ x: number, z: number }} Unnormalized tangent vector
 */
export function bezierTangent(p0, p1, p2, t) {
    const u = 1 - t;
    return {
        x: 2 * u * (p1.x - p0.x) + 2 * t * (p2.x - p1.x),
        z: 2 * u * (p1.z - p0.z) + 2 * t * (p2.z - p1.z),
    };
}

/**
 * Compute angle from a tangent vector (for mesh rotation.y).
 *
 * @param {{ x: number, z: number }} tangent
 * @returns {number} Angle in radians (compatible with Three.js rotation.y)
 */
export function tangentToAngle(tangent) {
    return Math.atan2(tangent.x, tangent.z);
}

// ============================================================
// INTERSECTION TURN STATE
// ============================================================

/**
 * Create an intersection turn state for a vehicle transitioning between edges.
 *
 * @param {Object} fromEdge - Current road edge
 * @param {number} fromLane - Current lane
 * @param {Object} toEdge - Next road edge
 * @param {number} toLane - Next lane
 * @param {string} nodeId - Intersection node ID
 * @param {{ x: number, z: number }} controlPoint - Bezier control point
 * @param {number} entrySpeed - Vehicle speed at entry (m/s)
 * @returns {Object} Turn state to attach to vehicle
 */
export function createTurnState(fromEdge, fromLane, toEdge, toLane, nodeId, controlPoint, entrySpeed) {
    // Entry point: end of current road at current lane
    const isForwardFrom = fromLane < fromEdge.numLanesPerDir;
    const entryU = isForwardFrom ? fromEdge.length : 0;
    const entryWorld = roadToWorldStatic(fromEdge, fromLane, entryU);

    // Exit point: start of next road at next lane
    const isForwardTo = toLane < toEdge.numLanesPerDir;
    const exitU = isForwardTo ? 0 : toEdge.length;
    const exitWorld = roadToWorldStatic(toEdge, toLane, exitU);

    // Turn duration based on speed (slower = longer turn, min 0.5s, max 2.5s)
    const turnSpeed = Math.max(2, entrySpeed * 0.5); // reduce speed through turn
    const arcLength = estimateBezierLength(entryWorld, controlPoint, exitWorld);
    const duration = Math.max(0.5, Math.min(2.5, arcLength / turnSpeed));

    return {
        p0: entryWorld,
        p1: controlPoint,
        p2: exitWorld,
        t: 0,
        duration,
        toEdge,
        toLane,
        turnSpeed,
    };
}

/**
 * Estimate Bezier curve length by sampling.
 */
function estimateBezierLength(p0, p1, p2, samples = 10) {
    let length = 0;
    let prev = p0;
    for (let i = 1; i <= samples; i++) {
        const t = i / samples;
        const curr = bezierPosition(p0, p1, p2, t);
        const dx = curr.x - prev.x;
        const dz = curr.z - prev.z;
        length += Math.sqrt(dx * dx + dz * dz);
        prev = curr;
    }
    return length;
}

/**
 * Simplified roadToWorld that doesn't need the full idm.js import.
 * Computes world position from road coordinates for straight roads.
 */
function roadToWorldStatic(road, lane, u) {
    const t = Math.max(0, Math.min(1, u / road.length));
    const cx = road.ax + t * (road.bx - road.ax);
    const cz = road.az + t * (road.bz - road.az);

    const dx = road.bx - road.ax;
    const dz = road.bz - road.az;
    const len = Math.sqrt(dx * dx + dz * dz) || 1;
    const dirX = dx / len;
    const dirZ = dz / len;

    // Perpendicular (right-hand rule)
    const perpX = -dirZ;
    const perpZ = dirX;

    const nPerDir = road.numLanesPerDir || 2;
    const laneWidth = road.laneWidth || 3;
    let offset;
    if (lane < nPerDir) {
        offset = (lane + 0.5) * laneWidth;
    } else {
        offset = -((lane - nPerDir) + 0.5) * laneWidth;
    }

    return {
        x: cx + perpX * offset,
        z: cz + perpZ * offset,
    };
}
