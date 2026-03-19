// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Protest Engine — Epstein-based protest/riot simulation.
 *
 * Uses the Epstein civil violence model (2002) to drive individual
 * agent activation decisions. Grievance vs risk determines whether
 * each NPC joins the protest.
 *
 * Grievance = hardship × (1 - legitimacy)
 * Risk = arrest_probability × risk_aversion
 * Activate if: grievance - risk > threshold
 *
 * Phase transitions are driven by the percentage of active agents.
 *
 * Reference: Epstein, J.M. (2002). "Modeling civil violence."
 * Proceedings of the National Academy of Sciences.
 */

import { PHASES } from './protest-scenario.js';

// ============================================================
// PROTEST ENGINE
// ============================================================

export class ProtestEngine {
    /**
     * @param {Object} config
     * @param {number} config.legitimacy - Government legitimacy [0, 1] (default 0.7)
     * @param {number} config.threshold - Activation threshold (default 0.1)
     * @param {number} config.maxJailTerm - Max jail time in sim-seconds (default 300)
     * @param {{ x: number, z: number }} config.plazaCenter - Gathering point
     * @param {number} config.plazaRadius - Gathering area radius (default 25)
     * @param {{ x: number, z: number }} config.policeStation - Police deployment origin
     */
    constructor(config = {}) {
        this.legitimacy = config.legitimacy ?? 0.7;
        this.threshold = config.threshold ?? 0.1;
        this.maxJailTerm = config.maxJailTerm ?? 300;
        this.plazaCenter = config.plazaCenter ?? { x: 0, z: 0 };
        this.plazaRadius = config.plazaRadius ?? 25;
        this.policeStation = config.policeStation ?? { x: 0, z: 0 };

        this.currentPhase = PHASES.NORMAL;
        this.phaseTimer = 0;
        this.active = false;

        // Per-agent state
        this.agents = new Map(); // npcId → { hardship, riskAversion, state, jailTimer }
    }

    /**
     * Register an NPC as a potential protest participant.
     */
    registerAgent(npcId, hardship = null, riskAversion = null) {
        this.agents.set(npcId, {
            hardship: hardship ?? Math.random(),
            riskAversion: riskAversion ?? Math.random(),
            state: 'passive', // passive | active | arrested | fleeing
            jailTimer: 0,
        });
    }

    /**
     * Start the protest event.
     */
    start() {
        this.active = true;
        this.currentPhase = PHASES.CALL_TO_ACTION;
        this.phaseTimer = 0;
        // Lower legitimacy to trigger more activation
        this.legitimacy = Math.max(0.3, this.legitimacy - 0.2);
    }

    /**
     * Stop/reset the protest.
     */
    stop() {
        this.active = false;
        this.currentPhase = PHASES.NORMAL;
        this.phaseTimer = 0;
        this.legitimacy = 0.7;
        for (const [, agent] of this.agents) {
            agent.state = 'passive';
            agent.jailTimer = 0;
        }
    }

    /**
     * Tick the protest engine.
     *
     * @param {number} dt - Time step (seconds)
     * @param {Array} npcPositions - Array of { id, x, z, type } for all NPCs
     * @returns {{ phase: string, activeCount: number, arrestedCount: number, tensionLevel: number }}
     */
    tick(dt, npcPositions) {
        if (!this.active) {
            return { phase: PHASES.NORMAL, activeCount: 0, arrestedCount: 0, tensionLevel: 0 };
        }

        this.phaseTimer += dt;

        // Count nearby agents by type
        let copsNearPlaza = 0;
        let activesNearPlaza = 0;
        let totalAgents = 0;

        for (const npc of npcPositions) {
            const agent = this.agents.get(npc.id);
            if (!agent) continue;
            totalAgents++;

            const dx = npc.x - this.plazaCenter.x;
            const dz = npc.z - this.plazaCenter.z;
            const dist = Math.sqrt(dx * dx + dz * dz);

            if (npc.type === 'police' && dist < this.plazaRadius * 2) {
                copsNearPlaza++;
            }

            // Update jail timer
            if (agent.state === 'arrested') {
                agent.jailTimer -= dt;
                if (agent.jailTimer <= 0) {
                    agent.state = 'passive';
                }
                continue;
            }

            // Epstein activation check
            if (agent.state === 'passive' || agent.state === 'active') {
                const grievance = agent.hardship * (1 - this.legitimacy);
                const arrestProb = 1 - Math.exp(-2.3 * (copsNearPlaza / (activesNearPlaza + 1)));
                const risk = arrestProb * agent.riskAversion;

                if (grievance - risk > this.threshold) {
                    agent.state = 'active';
                } else {
                    // Stay active if already active and in crowd (inertia)
                    if (agent.state === 'active' && dist < this.plazaRadius) {
                        // Keep active (crowd inertia)
                    } else {
                        agent.state = 'passive';
                    }
                }
            }

            if (agent.state === 'active' && dist < this.plazaRadius * 2) {
                activesNearPlaza++;
            }
        }

        // Police arrest active agents nearby
        if (this.currentPhase === PHASES.RIOT || this.currentPhase === PHASES.DISPERSAL) {
            for (const npc of npcPositions) {
                if (npc.type !== 'police') continue;
                const agent = this.agents.get(npc.id);
                if (agent) continue; // skip police themselves

                // Find nearest active agent
                for (const other of npcPositions) {
                    const otherAgent = this.agents.get(other.id);
                    if (!otherAgent || otherAgent.state !== 'active') continue;
                    const dx = npc.x - other.x;
                    const dz = npc.z - other.z;
                    if (dx * dx + dz * dz < 4) { // within 2m
                        otherAgent.state = 'arrested';
                        otherAgent.jailTimer = Math.random() * this.maxJailTerm;
                        break; // one arrest per cop per tick
                    }
                }
            }
        }

        // Phase transitions based on active percentage
        const activePercent = totalAgents > 0 ? activesNearPlaza / totalAgents : 0;
        const tensionLevel = activePercent;

        this._updatePhase(activePercent, activesNearPlaza, copsNearPlaza);

        const arrestedCount = [...this.agents.values()].filter(a => a.state === 'arrested').length;

        return {
            phase: this.currentPhase,
            activeCount: activesNearPlaza,
            arrestedCount,
            tensionLevel,
        };
    }

    _updatePhase(activePercent, actives, cops) {
        switch (this.currentPhase) {
            case PHASES.CALL_TO_ACTION:
                if (this.phaseTimer > 10) { // 10s after trigger
                    this.currentPhase = PHASES.MARCHING;
                    this.phaseTimer = 0;
                }
                break;
            case PHASES.MARCHING:
                if (actives >= 3 || this.phaseTimer > 30) {
                    this.currentPhase = PHASES.ASSEMBLED;
                    this.phaseTimer = 0;
                }
                break;
            case PHASES.ASSEMBLED:
                if (activePercent > 0.15 || this.phaseTimer > 45) {
                    this.currentPhase = PHASES.TENSION;
                    this.phaseTimer = 0;
                }
                break;
            case PHASES.TENSION:
                if (activePercent > 0.3 || this.phaseTimer > 30) {
                    this.currentPhase = PHASES.FIRST_INCIDENT;
                    this.phaseTimer = 0;
                    // Legitimacy drops further after first incident
                    this.legitimacy = Math.max(0.2, this.legitimacy - 0.15);
                }
                break;
            case PHASES.FIRST_INCIDENT:
                if (this.phaseTimer > 10) {
                    this.currentPhase = PHASES.RIOT;
                    this.phaseTimer = 0;
                }
                break;
            case PHASES.RIOT:
                if (cops > actives * 0.5 || this.phaseTimer > 60) {
                    this.currentPhase = PHASES.DISPERSAL;
                    this.phaseTimer = 0;
                    // Risk increases dramatically
                    this.legitimacy = Math.min(0.9, this.legitimacy + 0.3);
                }
                break;
            case PHASES.DISPERSAL:
                if (actives < 3 || this.phaseTimer > 45) {
                    this.currentPhase = PHASES.AFTERMATH;
                    this.phaseTimer = 0;
                }
                break;
            case PHASES.AFTERMATH:
                if (this.phaseTimer > 30) {
                    this.stop();
                }
                break;
        }
    }

    /**
     * Get the state of a specific agent.
     */
    getAgentState(npcId) {
        const agent = this.agents.get(npcId);
        return agent ? agent.state : 'passive';
    }

    /**
     * Get goal for an agent based on current phase and state.
     */
    getAgentGoal(npcId) {
        const agent = this.agents.get(npcId);
        if (!agent) return null;

        switch (agent.state) {
            case 'active':
                if (this.currentPhase === PHASES.MARCHING || this.currentPhase === PHASES.CALL_TO_ACTION) {
                    return { action: 'go_to', target: this.plazaCenter, speed: 2.5 };
                }
                if (this.currentPhase === PHASES.ASSEMBLED || this.currentPhase === PHASES.TENSION) {
                    return { action: 'mill', target: this.plazaCenter, radius: this.plazaRadius, speed: 1.0 };
                }
                if (this.currentPhase === PHASES.RIOT || this.currentPhase === PHASES.FIRST_INCIDENT) {
                    return { action: 'mill', target: this.plazaCenter, radius: this.plazaRadius * 0.5, speed: 2.0 };
                }
                if (this.currentPhase === PHASES.DISPERSAL) {
                    return { action: 'flee', speed: 3.5 };
                }
                return { action: 'go_to', target: this.plazaCenter, speed: 2.0 };

            case 'arrested':
                return { action: 'stay', speed: 0 };

            case 'fleeing':
                return { action: 'flee', speed: 3.5 };

            default: // passive
                return null; // follow normal daily routine
        }
    }

    /**
     * Get debug summary.
     */
    getDebugInfo() {
        const states = { passive: 0, active: 0, arrested: 0, fleeing: 0 };
        for (const [, agent] of this.agents) {
            states[agent.state] = (states[agent.state] || 0) + 1;
        }
        return {
            phase: this.currentPhase,
            timer: this.phaseTimer.toFixed(1),
            legitimacy: this.legitimacy.toFixed(2),
            ...states,
        };
    }
}
