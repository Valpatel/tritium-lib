// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Traffic Light Controller — per-intersection signal management.
 *
 * Each intersection has a controller that:
 * - Cycles through signal phases (green/yellow/all-red)
 * - Manages virtual obstacles at stop lines for red approaches
 * - Supports emergency vehicle preemption
 *
 * Virtual obstacles are the key mechanism: when a light is red,
 * a virtual stopped car is placed at the stop line. The IDM
 * naturally makes real cars brake and stop behind it. When the
 * light turns green, the virtual car is removed.
 *
 * This is a pure logic module — no rendering dependencies.
 */

// ============================================================
// SIGNAL PHASES
// ============================================================

/**
 * Build signal phases for an intersection based on its type.
 *
 * @param {string[]} approaches - Array of approach directions (e.g., ['N','S','E','W'])
 * @returns {Array<{ greenApproaches: string[], duration: number, type: string }>}
 */
export function buildSignalPhases(approaches) {
    const phases = [];

    if (approaches.length === 4) {
        // 4-way intersection: NS green, then EW green
        phases.push({ greenApproaches: ['N', 'S'], duration: 30, type: 'green' });
        phases.push({ greenApproaches: ['N', 'S'], duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
        phases.push({ greenApproaches: ['E', 'W'], duration: 30, type: 'green' });
        phases.push({ greenApproaches: ['E', 'W'], duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
    } else if (approaches.length === 3) {
        // 3-way (T-junction): two phases
        // Find the pair and the single approach
        const hasN = approaches.includes('N');
        const hasS = approaches.includes('S');
        const hasE = approaches.includes('E');
        const hasW = approaches.includes('W');

        let pair, single;
        if (hasN && hasS) {
            pair = ['N', 'S'];
            single = hasE ? ['E'] : ['W'];
        } else if (hasE && hasW) {
            pair = ['E', 'W'];
            single = hasN ? ['N'] : ['S'];
        } else {
            // Two non-opposing (corner-like T)
            pair = [approaches[0], approaches[1]];
            single = [approaches[2]];
        }

        phases.push({ greenApproaches: pair, duration: 30, type: 'green' });
        phases.push({ greenApproaches: pair, duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
        phases.push({ greenApproaches: single, duration: 20, type: 'green' });
        phases.push({ greenApproaches: single, duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
    } else if (approaches.length === 2) {
        // 2-way (corner): alternate each approach
        phases.push({ greenApproaches: [approaches[0]], duration: 25, type: 'green' });
        phases.push({ greenApproaches: [approaches[0]], duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
        phases.push({ greenApproaches: [approaches[1]], duration: 25, type: 'green' });
        phases.push({ greenApproaches: [approaches[1]], duration: 3, type: 'yellow' });
        phases.push({ greenApproaches: [], duration: 2, type: 'allRed' });
    } else {
        // 1-way: always green
        phases.push({ greenApproaches: approaches, duration: 60, type: 'green' });
    }

    return phases;
}

// ============================================================
// TRAFFIC CONTROLLER CLASS
// ============================================================

export class TrafficController {
    /**
     * @param {Object} node - Intersection node from RoadNetwork
     * @param {Object} roadNetwork - RoadNetwork instance
     */
    constructor(node, roadNetwork) {
        this.nodeId = node.id;
        this.node = node;
        this.roadNetwork = roadNetwork;
        this.phases = buildSignalPhases(node.approaches);
        this.currentPhase = 0;
        this.phaseTimer = 0;
        this.preemptionActive = false;
        this.preemptionDir = null;
        this.preemptionTimer = 0;

        // Stagger start time based on position to avoid synchronized lights
        const stagger = ((node.col * 7 + node.row * 13) % 17) * 2;
        this.phaseTimer = stagger % (this.phases[0]?.duration || 30);
    }

    /**
     * Get the current phase object.
     */
    getCurrentPhase() {
        return this.phases[this.currentPhase];
    }

    /**
     * Check if a given approach direction currently has green.
     *
     * @param {string} approachDir - 'N', 'S', 'E', or 'W'
     * @returns {boolean}
     */
    isGreen(approachDir) {
        if (this.preemptionActive) {
            return approachDir === this.preemptionDir;
        }
        const phase = this.getCurrentPhase();
        return phase.greenApproaches.includes(approachDir);
    }

    /**
     * Check if a given approach is in yellow phase.
     */
    isYellow(approachDir) {
        if (this.preemptionActive) return false;
        const phase = this.getCurrentPhase();
        return phase.type === 'yellow' && phase.greenApproaches.includes(approachDir);
    }

    /**
     * Get the light state for an approach: 'green', 'yellow', or 'red'.
     *
     * @param {string} approachDir
     * @returns {string}
     */
    getLightState(approachDir) {
        if (this.preemptionActive) {
            return approachDir === this.preemptionDir ? 'green' : 'red';
        }
        const phase = this.getCurrentPhase();
        if (phase.type === 'allRed') return 'red';
        if (phase.greenApproaches.includes(approachDir)) {
            return phase.type === 'yellow' ? 'yellow' : 'green';
        }
        return 'red';
    }

    /**
     * Advance the traffic light by dt seconds.
     *
     * @param {number} dt - Time step in seconds
     */
    tick(dt) {
        if (this.preemptionActive) {
            this.preemptionTimer -= dt;
            if (this.preemptionTimer <= 0) {
                this.preemptionActive = false;
                this.preemptionDir = null;
            }
            return;
        }

        this.phaseTimer += dt;
        const phase = this.getCurrentPhase();
        if (this.phaseTimer >= phase.duration) {
            this.phaseTimer -= phase.duration;
            this.currentPhase = (this.currentPhase + 1) % this.phases.length;
        }
    }

    /**
     * Force green for an approach direction (emergency vehicle preemption).
     *
     * @param {string} approachDir - Direction to force green
     * @param {number} duration - How long to hold preemption (seconds)
     */
    forceGreen(approachDir, duration = 15) {
        this.preemptionActive = true;
        this.preemptionDir = approachDir;
        this.preemptionTimer = duration;
    }

    /**
     * Get time remaining in current phase.
     */
    getTimeRemaining() {
        const phase = this.getCurrentPhase();
        return Math.max(0, phase.duration - this.phaseTimer);
    }

    /**
     * Build virtual obstacles for all red approaches.
     * Returns array of virtual stopped cars at stop lines.
     *
     * @param {Object} roadNetwork - RoadNetwork instance
     * @returns {Array<{ id, road, lane, u, speed, length, isVirtual }>}
     */
    buildVirtualObstacles() {
        const obstacles = [];
        const edges = this.roadNetwork.getEdgesForNode(this.nodeId);

        for (const edge of edges) {
            const n = edge.numLanesPerDir || 2;

            // Check each approach direction for this edge
            // Forward lanes (0..n-1) arrive at 'to' node
            if (edge.to === this.nodeId) {
                const arrivalDir = edge.horizontal ? 'W' : 'N';
                if (!this.isGreen(arrivalDir)) {
                    for (let lane = 0; lane < n; lane++) {
                        obstacles.push({
                            id: `virt_${this.nodeId}_${arrivalDir}_${lane}`,
                            road: edge,
                            lane,
                            u: edge.length - 2, // 2m before intersection
                            speed: 0,
                            length: 1,
                            isVirtual: true,
                        });
                    }
                }
            }

            // Backward lanes (n..2n-1) arrive at 'from' node
            if (edge.from === this.nodeId) {
                const arrivalDir = edge.horizontal ? 'E' : 'S';
                if (!this.isGreen(arrivalDir)) {
                    for (let lane = n; lane < 2 * n; lane++) {
                        obstacles.push({
                            id: `virt_${this.nodeId}_${arrivalDir}_${lane}`,
                            road: edge,
                            lane,
                            u: edge.length - 2,
                            speed: 0,
                            length: 1,
                            isVirtual: true,
                        });
                    }
                }
            }
        }

        return obstacles;
    }

    /**
     * Get debug info for HUD display.
     */
    getDebugInfo() {
        const phase = this.getCurrentPhase();
        return {
            nodeId: this.nodeId,
            phase: phase.type,
            greenApproaches: phase.greenApproaches,
            timeRemaining: this.getTimeRemaining().toFixed(1),
            preemption: this.preemptionActive ? this.preemptionDir : null,
        };
    }
}

// ============================================================
// TRAFFIC CONTROLLER MANAGER
// ============================================================

/**
 * Manages all traffic controllers in the city.
 */
export class TrafficControllerManager {
    constructor() {
        this.controllers = {}; // nodeId → TrafficController
    }

    /**
     * Initialize controllers for all intersections in a road network.
     *
     * @param {Object} roadNetwork - RoadNetwork instance
     */
    initFromNetwork(roadNetwork) {
        this.controllers = {};
        for (const nodeId in roadNetwork.nodes) {
            const node = roadNetwork.nodes[nodeId];
            if (node.approaches.length >= 2) {
                this.controllers[nodeId] = new TrafficController(node, roadNetwork);
            }
        }
    }

    /**
     * Tick all controllers.
     */
    tick(dt) {
        for (const id in this.controllers) {
            this.controllers[id].tick(dt);
        }
    }

    /**
     * Get all virtual obstacles from all controllers.
     */
    getAllVirtualObstacles() {
        const all = [];
        for (const id in this.controllers) {
            all.push(...this.controllers[id].buildVirtualObstacles());
        }
        return all;
    }

    /**
     * Check if an approach at an intersection is green.
     */
    isGreen(nodeId, approachDir) {
        const ctrl = this.controllers[nodeId];
        return ctrl ? ctrl.isGreen(approachDir) : true; // no controller = always green
    }

    /**
     * Get light state for rendering.
     */
    getLightState(nodeId, approachDir) {
        const ctrl = this.controllers[nodeId];
        return ctrl ? ctrl.getLightState(approachDir) : 'green';
    }

    /**
     * Get controller for a specific intersection.
     */
    getController(nodeId) {
        return this.controllers[nodeId] || null;
    }
}
