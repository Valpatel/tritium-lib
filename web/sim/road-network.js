// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * RoadNetwork — directed graph for vehicle simulation on real OSM roads.
 *
 * Builds a navigable road graph from city-data JSON. Nodes are intersections
 * (where road endpoints cluster within mergeRadius). Edges are road segments
 * with proper lane configuration from OSM tags.
 *
 * Used by CitySimManager for vehicle pathfinding and route assignment.
 * Pure data module — no Three.js dependency.
 */

export class RoadNetwork {
    constructor() {
        this.nodes = {};        // { id: { id, x, z, type, degree } }
        this.edges = [];        // [{ id, from, to, ax, az, bx, bz, length, ... }]
        this.adjList = {};      // nodeId → [edgeIndex]
        this.edgeById = {};     // edgeId → edge
    }

    /**
     * Build road network from OSM city-data roads.
     *
     * @param {Array} osmRoads — [{points, class, width, lanes, oneway, bridge}]
     * @param {number} mergeRadius — meters to merge nearby endpoints (default 5)
     * @returns {RoadNetwork} this
     */
    buildFromOSM(osmRoads, mergeRadius = 5) {
        this.nodes = {};
        this.edges = [];
        this.adjList = {};
        this.edgeById = {};

        if (!osmRoads?.length) return this;

        // Filter to vehicle-drivable roads
        const vehicleTypes = new Set([
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'unclassified', 'living_street',
            'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        ]);
        const roads = osmRoads.filter(r => vehicleTypes.has(r.class));
        if (!roads.length) return this;

        // Collect all road endpoints
        const endpoints = [];
        for (let ri = 0; ri < roads.length; ri++) {
            const pts = roads[ri].points;
            if (!pts || pts.length < 2) continue;
            endpoints.push({ x: pts[0][0], z: pts[0][1], ri, pi: 0 });
            endpoints.push({ x: pts[pts.length - 1][0], z: pts[pts.length - 1][1], ri, pi: pts.length - 1 });
        }

        // Merge nearby endpoints into intersection nodes
        const assigned = new Map();  // "ri:pi" → nodeId
        let nextId = 0;

        for (let i = 0; i < endpoints.length; i++) {
            const ki = `${endpoints[i].ri}:${endpoints[i].pi}`;
            if (assigned.has(ki)) continue;

            const cluster = [i];
            for (let j = i + 1; j < endpoints.length; j++) {
                const kj = `${endpoints[j].ri}:${endpoints[j].pi}`;
                if (assigned.has(kj)) continue;
                const dx = endpoints[i].x - endpoints[j].x;
                const dz = endpoints[i].z - endpoints[j].z;
                if (Math.sqrt(dx * dx + dz * dz) <= mergeRadius) {
                    cluster.push(j);
                }
            }

            // Centroid
            let cx = 0, cz = 0;
            for (const ci of cluster) { cx += endpoints[ci].x; cz += endpoints[ci].z; }
            cx /= cluster.length;
            cz /= cluster.length;

            const nodeId = `n${nextId++}`;
            this.nodes[nodeId] = { id: nodeId, x: cx, z: cz, type: 'osm', degree: 0 };
            this.adjList[nodeId] = [];

            for (const ci of cluster) {
                assigned.set(`${endpoints[ci].ri}:${endpoints[ci].pi}`, nodeId);
            }
        }

        // Create edges
        for (let ri = 0; ri < roads.length; ri++) {
            const road = roads[ri];
            const pts = road.points;
            if (!pts || pts.length < 2) continue;

            const fromId = assigned.get(`${ri}:0`);
            const toId = assigned.get(`${ri}:${pts.length - 1}`);
            if (!fromId || !toId) continue;

            if (fromId === toId) {
                console.warn(`[RoadNetwork] Skipping road ${ri}: both endpoints merged to node ${fromId}`);
                continue;
            }

            let length = 0;
            for (let i = 1; i < pts.length; i++) {
                const dx = pts[i][0] - pts[i - 1][0];
                const dz = pts[i][1] - pts[i - 1][1];
                length += Math.sqrt(dx * dx + dz * dz);
            }

            const lanesPerDir = road.oneway
                ? Math.max(1, road.lanes || 1)
                : Math.max(1, Math.floor((road.lanes || 2) / 2));

            const edgeId = `e${ri}`;
            const edge = {
                id: edgeId,
                from: fromId,
                to: toId,
                ax: pts[0][0], az: pts[0][1],
                bx: pts[pts.length - 1][0], bz: pts[pts.length - 1][1],
                length,
                lanesPerDir,
                laneWidth: Math.max(2, (road.width || 6) / (lanesPerDir * 2)),
                roadClass: road.class || 'residential',
                oneway: !!road.oneway,
                bridge: !!road.bridge,
                waypoints: pts,
                speedLimit: road.maxspeed ? parseFloat(road.maxspeed) / 3.6 : null, // km/h → m/s
            };

            const idx = this.edges.length;
            this.edges.push(edge);
            this.edgeById[edgeId] = edge;
            this.adjList[fromId].push(idx);
            this.adjList[toId].push(idx);
        }

        // Update node degrees
        for (const nodeId in this.nodes) {
            this.nodes[nodeId].degree = (this.adjList[nodeId] || []).length;
            const d = this.nodes[nodeId].degree;
            this.nodes[nodeId].type = d >= 4 ? '4+-way' : d === 3 ? '3-way' : d === 2 ? '2-way' : d === 1 ? 'dead-end' : 'isolated';
        }

        return this;
    }

    /**
     * Dijkstra shortest path.
     * @returns {Array<{edge, nodeId}>} or empty array if no path
     */
    findPath(fromNodeId, toNodeId) {
        if (fromNodeId === toNodeId) return [];
        if (!this.nodes[fromNodeId] || !this.nodes[toNodeId]) return [];

        const dist = {};
        const prev = {};
        const visited = new Set();
        const queue = [];

        for (const id in this.nodes) dist[id] = Infinity;
        dist[fromNodeId] = 0;
        queue.push({ id: fromNodeId, dist: 0 });

        while (queue.length > 0) {
            queue.sort((a, b) => a.dist - b.dist);
            const { id: current } = queue.shift();
            if (current === toNodeId) break;
            if (visited.has(current)) continue;
            visited.add(current);

            for (const edgeIdx of (this.adjList[current] || [])) {
                const edge = this.edges[edgeIdx];
                const neighbor = edge.from === current ? edge.to : edge.from;
                if (edge.oneway && edge.to === current) continue; // Can't go backwards on one-way

                const newDist = dist[current] + edge.length;
                if (newDist < dist[neighbor]) {
                    dist[neighbor] = newDist;
                    prev[neighbor] = { nodeId: current, edgeIdx };
                    queue.push({ id: neighbor, dist: newDist });
                }
            }
        }

        if (dist[toNodeId] === Infinity) return [];

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
     * Find nearest node to a point.
     * @returns {{ nodeId, dist }} or null
     */
    nearestNode(x, z) {
        let best = null, bestDist = Infinity;
        for (const id in this.nodes) {
            const n = this.nodes[id];
            const d = Math.hypot(n.x - x, n.z - z);
            if (d < bestDist) {
                bestDist = d;
                best = id;
            }
        }
        return best ? { nodeId: best, dist: bestDist } : null;
    }

    /**
     * Pick a random connected edge for spawning vehicles.
     * Prefers longer edges on higher-class roads.
     */
    randomEdge() {
        if (!this.edges.length) return null;
        // Weight by length (longer roads get more vehicles)
        const totalLen = this.edges.reduce((s, e) => s + e.length, 0);
        let r = Math.random() * totalLen;
        for (const edge of this.edges) {
            r -= edge.length;
            if (r <= 0) return edge;
        }
        return this.edges[this.edges.length - 1];
    }

    /**
     * Get stats for logging/debug.
     */
    stats() {
        const classCount = {};
        for (const e of this.edges) {
            classCount[e.roadClass] = (classCount[e.roadClass] || 0) + 1;
        }
        return {
            nodes: Object.keys(this.nodes).length,
            edges: this.edges.length,
            roadClasses: classCount,
            totalLengthM: Math.round(this.edges.reduce((s, e) => s + e.length, 0)),
        };
    }
}
