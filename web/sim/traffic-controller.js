// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * TrafficController — manages traffic signal phases at intersections.
 *
 * Uses the virtual-obstacle mechanism: red lights are modeled as phantom
 * stopped cars at the stop line. IDM naturally brakes behind them.
 * When the light turns green, the virtual obstacle is removed.
 *
 * Based on SUMO's virtual-obstacle approach (proven in production).
 */

/**
 * Signal phase definition.
 * @typedef {Object} SignalPhase
 * @property {Set<string>} greenEdges — edge IDs that have green in this phase
 * @property {number} duration — phase duration in seconds
 * @property {string} type — 'green', 'yellow', 'allred'
 */

export class TrafficController {
    /**
     * @param {string} nodeId — intersection node ID
     * @param {Array} edges — edges connected to this intersection
     * @param {Object} node — intersection node {x, z, degree}
     */
    constructor(nodeId, edges, node) {
        this.nodeId = nodeId;
        this.node = node;
        this.edges = edges;
        this.phases = [];
        this.currentPhase = 0;
        this.phaseTimer = 0;
        this.staggerOffset = 0;

        // Adaptive (actuated) signal parameters
        this.mode = 'fixed';       // 'fixed' | 'adaptive' | 'rl'
        this.minGreen = 5;         // minimum green to prevent flickering
        this.maxGreen = 45;        // maximum green extension
        this.extensionPerVehicle = 3; // seconds per queued vehicle
        this._queueCounts = {};    // edgeId → queued vehicle count
        this._effectiveDuration = 0; // computed duration for current green phase

        this._buildPhases();
    }

    /**
     * Build signal phases based on intersection degree.
     */
    _buildPhases() {
        if (this.edges.length < 2) {
            // Dead end or single road — always green
            this.phases = [{ greenEdges: new Set(this.edges.map(e => e.id)), duration: 999, type: 'green' }];
            return;
        }

        // Ensure we have at least 2 edges for phase splitting
        console.assert(this.edges.length >= 2, `[TrafficController] Edge count < 2: ${this.edges.length}`);

        // Split edges into two groups by angle
        // Group A: first half of edges, Group B: second half
        const halfIdx = Math.ceil(this.edges.length / 2);
        const groupA = this.edges.slice(0, halfIdx).map(e => e.id);
        const groupB = this.edges.slice(halfIdx).map(e => e.id);

        const greenDuration = this.edges.length >= 4 ? 20 : 15;
        const yellowDuration = 2;
        const allRedDuration = 1;

        // Phase A green
        this.phases.push({
            greenEdges: new Set(groupA),
            duration: greenDuration,
            type: 'green',
        });
        // Yellow
        this.phases.push({
            greenEdges: new Set(groupA),
            duration: yellowDuration,
            type: 'yellow',
        });
        // All red
        this.phases.push({
            greenEdges: new Set(),
            duration: allRedDuration,
            type: 'allred',
        });
        // Phase B green
        this.phases.push({
            greenEdges: new Set(groupB),
            duration: greenDuration,
            type: 'green',
        });
        // Yellow
        this.phases.push({
            greenEdges: new Set(groupB),
            duration: yellowDuration,
            type: 'yellow',
        });
        // All red
        this.phases.push({
            greenEdges: new Set(),
            duration: allRedDuration,
            type: 'allred',
        });
    }

    /**
     * Update queue counts for adaptive mode.
     * @param {Object} counts — { edgeId: numberOfQueuedVehicles }
     */
    updateQueueCounts(counts) {
        this._queueCounts = counts || {};
    }

    /**
     * Get effective duration for current phase in adaptive mode.
     * Extends green phases based on queue length; skips empty phases.
     * @returns {number} duration in seconds
     */
    _getAdaptiveDuration() {
        const phase = this.phases[this.currentPhase];
        if (phase.type !== 'green') return phase.duration;

        // Count queued vehicles on edges that are green in this phase
        let totalQueued = 0;
        for (const edgeId of phase.greenEdges) {
            totalQueued += this._queueCounts[edgeId] || 0;
        }

        if (totalQueued === 0) {
            // Check if next green phase has demand — skip this one with minimum green
            const nextGreenIdx = this._findNextGreenPhase();
            if (nextGreenIdx !== -1) {
                let nextQueued = 0;
                const nextPhase = this.phases[nextGreenIdx];
                for (const edgeId of nextPhase.greenEdges) {
                    nextQueued += this._queueCounts[edgeId] || 0;
                }
                if (nextQueued > 0) return this.minGreen;
            }
            return phase.duration; // no demand anywhere, use fixed timing
        }

        // Extend green: base duration + extension per queued vehicle, capped at maxGreen
        const extended = phase.duration + totalQueued * this.extensionPerVehicle;
        return Math.min(extended, this.maxGreen);
    }

    /**
     * Find the next green phase index after current.
     * @returns {number} phase index or -1
     */
    _findNextGreenPhase() {
        for (let i = 1; i < this.phases.length; i++) {
            const idx = (this.currentPhase + i) % this.phases.length;
            if (this.phases[idx].type === 'green') return idx;
        }
        return -1;
    }

    /**
     * Advance phase timer.
     * @param {number} dt
     */
    tick(dt) {
        if (this.mode === 'rl') return; // RL mode: external control only

        this.phaseTimer += dt;

        const phase = this.phases[this.currentPhase];
        const duration = this.mode === 'adaptive'
            ? this._getAdaptiveDuration()
            : phase.duration;

        if (this.phaseTimer >= duration) {
            this.phaseTimer -= duration;
            this.currentPhase = (this.currentPhase + 1) % this.phases.length;
        }
    }

    /**
     * Check if an edge has green light at this intersection.
     * @param {string} edgeId
     * @returns {boolean}
     */
    isGreen(edgeId) {
        const phase = this.phases[this.currentPhase];
        return phase.type === 'green' && phase.greenEdges.has(edgeId);
    }

    /**
     * Check if intersection is in yellow phase.
     */
    isYellow() {
        return this.phases[this.currentPhase].type === 'yellow';
    }

    /**
     * Get signal color for rendering.
     * @param {string} edgeId
     * @returns {'green'|'yellow'|'red'}
     */
    getSignalColor(edgeId) {
        const phase = this.phases[this.currentPhase];
        if (phase.type === 'allred') return 'red';
        if (phase.greenEdges.has(edgeId)) return phase.type === 'yellow' ? 'yellow' : 'green';
        return 'red';
    }
}

/**
 * TrafficControllerManager — manages all traffic controllers.
 */
export class TrafficControllerManager {
    constructor() {
        this.controllers = {};  // nodeId → TrafficController
    }

    /**
     * Initialize controllers for all 3+ way intersections.
     * @param {RoadNetwork} roadNetwork
     */
    initFromNetwork(roadNetwork) {
        this.controllers = {};

        for (const nodeId in roadNetwork.nodes) {
            const node = roadNetwork.nodes[nodeId];
            if (node.degree < 3) continue;  // Only signal-controlled intersections

            const edgeIndices = roadNetwork.adjList[nodeId] || [];
            const edges = edgeIndices.map(idx => roadNetwork.edges[idx]);

            const ctrl = new TrafficController(nodeId, edges, node);
            // Stagger start times to avoid synchronized lights
            const hash = nodeId.split('').reduce((h, c) => ((h << 5) - h + c.charCodeAt(0)) | 0, 0);
            ctrl.staggerOffset = Math.abs(hash % 30);
            ctrl.phaseTimer = ctrl.staggerOffset;

            this.controllers[nodeId] = ctrl;
        }

        console.log(`[TrafficCtrl] ${Object.keys(this.controllers).length} controllers initialized`);
    }

    /**
     * Set mode for all controllers.
     * @param {'fixed'|'adaptive'|'rl'} mode
     */
    setMode(mode) {
        this.mode = mode;
        for (const nodeId in this.controllers) {
            this.controllers[nodeId].mode = mode;
        }
        console.log(`[TrafficCtrl] mode → ${mode}`);
    }

    /**
     * Update queue counts for all controllers from vehicle data.
     * Counts vehicles within 30m of stop line per approach edge.
     * @param {Array} vehicles — current vehicle list
     */
    updateQueues(vehicles) {
        // Build per-intersection, per-edge queue counts
        const queues = {}; // nodeId → { edgeId → count }
        for (const car of vehicles) {
            if (!car.edge || car.parked) continue;
            const approachNode = car.direction > 0 ? car.edge.to : car.edge.from;
            const remaining = car.direction > 0 ? car.edge.length - car.u : car.u;
            if (remaining < 30 && remaining > 0) {
                if (!queues[approachNode]) queues[approachNode] = {};
                const q = queues[approachNode];
                q[car.edge.id] = (q[car.edge.id] || 0) + 1;
            }
        }
        // Push counts into each controller
        for (const nodeId in this.controllers) {
            this.controllers[nodeId].updateQueueCounts(queues[nodeId] || {});
        }
    }

    /**
     * Tick all controllers.
     */
    tick(dt) {
        for (const nodeId in this.controllers) {
            this.controllers[nodeId].tick(dt);
        }
    }

    /**
     * Check if an edge has green at a given node.
     */
    isGreen(nodeId, edgeId) {
        const ctrl = this.controllers[nodeId];
        if (!ctrl) return true;  // No controller = always green
        return ctrl.isGreen(edgeId);
    }

    /**
     * Get signal color for rendering.
     */
    getSignalColor(nodeId, edgeId) {
        const ctrl = this.controllers[nodeId];
        if (!ctrl) return 'green';
        return ctrl.getSignalColor(edgeId);
    }

    /**
     * Get all controller positions + current states for rendering.
     * @returns {Array<{nodeId, x, z, edgeId, color}>}
     */
    getSignalStates(roadNetwork) {
        const states = [];
        for (const nodeId in this.controllers) {
            const ctrl = this.controllers[nodeId];
            const node = ctrl.node;
            for (const edge of ctrl.edges) {
                states.push({
                    nodeId,
                    x: node.x,
                    z: node.z,
                    edgeId: edge.id,
                    color: ctrl.getSignalColor(edge.id),
                });
            }
        }
        return states;
    }
}
