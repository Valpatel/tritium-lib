// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Protest → Riot scenario goal system.
 *
 * Defines the escalation sequence from peaceful protest to full riot,
 * with goals for protestors, police, and bystanders at each phase.
 *
 * Phases:
 *   1. CALL_TO_ACTION — word spreads, some NPCs switch from daily routine to protest
 *   2. MARCHING — protestors walk from homes/workplaces to the plaza
 *   3. ASSEMBLED — protestors arrive, mill around, hold signs, chant
 *   4. TENSION — crowd tightens, chanting intensifies, police alerted
 *   5. FIRST_INCIDENT — someone throws a rock, police form line
 *   6. RIOT — full escalation: molotovs, tear gas, running battles
 *   7. DISPERSAL — police advance, crowd breaks, people flee
 *   8. AFTERMATH — injuries treated, arrests made, cleanup
 */

/**
 * @typedef {Object} ScenarioPhase
 * @property {string} name
 * @property {number} duration - seconds (sim time)
 * @property {Object} protestorGoals - what protestors do in this phase
 * @property {Object} policeGoals - what police do
 * @property {Object} civilianGoals - what bystanders do
 * @property {function} [trigger] - condition to advance to next phase
 */

export const PHASES = {
    NORMAL: 'NORMAL',
    CALL_TO_ACTION: 'CALL_TO_ACTION',
    MARCHING: 'MARCHING',
    ASSEMBLED: 'ASSEMBLED',
    TENSION: 'TENSION',
    FIRST_INCIDENT: 'FIRST_INCIDENT',
    RIOT: 'RIOT',
    DISPERSAL: 'DISPERSAL',
    AFTERMATH: 'AFTERMATH',
};

/**
 * @typedef {Object} ProtestConfig
 * @property {number} protestorCount - how many NPCs become protestors
 * @property {number} policeCount - how many police respond
 * @property {{x: number, z: number}} plazaCenter - gathering point
 * @property {number} plazaRadius - radius of the gathering area
 * @property {{x: number, z: number}} policeStation - where police deploy from
 * @property {number} escalationSpeed - 0.5 (slow) to 2.0 (fast)
 */

/**
 * Generate the scenario phase sequence with goals for each actor type.
 * @param {ProtestConfig} config
 * @returns {ScenarioPhase[]}
 */
export function generateProtestScenario(config) {
    const { plazaCenter, plazaRadius, policeStation, escalationSpeed = 1.0 } = config;
    const px = plazaCenter.x, pz = plazaCenter.z;

    return [
        {
            name: PHASES.CALL_TO_ACTION,
            duration: 30 / escalationSpeed,
            protestorGoals: {
                action: 'continue_routine', // keep doing what they were doing
                prepare: true, // but start heading toward plaza soon
            },
            policeGoals: { action: 'idle', destination: policeStation },
            civilianGoals: { action: 'continue_routine' },
            description: 'Word spreads on social media. Some residents start heading to the plaza.',
        },
        {
            name: PHASES.MARCHING,
            duration: 60 / escalationSpeed,
            protestorGoals: {
                action: 'go_to',
                destination: 'plaza',
                targetX: px, targetZ: pz,
                transport: 'walk',
                useRoad: true, // march in the street, not on sidewalks
                speed: 2.0,
                mood: 'determined',
            },
            policeGoals: {
                action: 'alert', // monitoring, not yet deployed
                destination: policeStation,
            },
            civilianGoals: {
                action: 'continue_routine',
                avoid: { x: px, z: pz, radius: plazaRadius * 2 }, // avoid the plaza area
            },
            description: 'Protestors march through streets toward the central plaza.',
        },
        {
            name: PHASES.ASSEMBLED,
            duration: 45 / escalationSpeed,
            protestorGoals: {
                action: 'wander',
                destination: 'plaza',
                center: { x: px, z: pz },
                radius: plazaRadius,
                speed: 0.5,
                holdSign: true,
                chant: true,
            },
            policeGoals: {
                action: 'deploy',
                destination: 'near_plaza',
                targetX: px, targetZ: pz - plazaRadius - 30, // form up south of plaza
                formation: 'line',
                transport: 'car',
            },
            civilianGoals: {
                action: 'flee_area',
                avoid: { x: px, z: pz, radius: plazaRadius * 1.5 },
            },
            description: 'Crowd assembled. Signs raised. Chanting. Police mobilizing.',
        },
        {
            name: PHASES.TENSION,
            duration: 30 / escalationSpeed,
            protestorGoals: {
                action: 'cluster',
                center: { x: px, z: pz },
                radius: plazaRadius * 0.6, // tighter clustering
                speed: 0.3,
                aggression: 0.3, // rising
                chant: true,
                chantIntensity: 'loud',
            },
            policeGoals: {
                action: 'form_line',
                lineCenter: { x: px, z: pz - plazaRadius - 10 },
                lineWidth: plazaRadius * 2,
                facing: 'north', // toward the crowd
                shields: true,
            },
            civilianGoals: {
                action: 'flee_area',
                avoid: { x: px, z: pz, radius: plazaRadius * 2 },
                urgency: 'high',
            },
            description: 'Tension rising. Crowd tightening. Police line formed with shields.',
        },
        {
            name: PHASES.FIRST_INCIDENT,
            duration: 5 / escalationSpeed,
            protestorGoals: {
                action: 'first_rock', // ONE protestor throws the first rock
                throwerCount: 1,
                target: 'police_line',
            },
            policeGoals: {
                action: 'hold_line',
                alert: 'high',
            },
            civilianGoals: {
                action: 'flee',
                urgency: 'panic',
            },
            description: 'FIRST ROCK THROWN. The crowd erupts.',
        },
        {
            name: PHASES.RIOT,
            duration: 120 / escalationSpeed,
            protestorGoals: {
                action: 'riot',
                behaviors: ['throw_rocks', 'throw_molotov', 'charge', 'flee_teargas'],
                aggressionRange: [0.3, 0.9], // varies per NPC
                moraleEffect: true, // low morale → flee, high → charge
            },
            policeGoals: {
                action: 'riot_response',
                behaviors: ['teargas', 'rubber_bullets', 'advance', 'arrest'],
                formation: 'advancing_line',
                advanceSpeed: 0.3,
            },
            civilianGoals: {
                action: 'flee',
                urgency: 'panic',
                seekShelter: true, // run to nearest building entry
            },
            description: 'Full riot. Molotovs, tear gas, running battles.',
        },
        {
            name: PHASES.DISPERSAL,
            duration: 60 / escalationSpeed,
            protestorGoals: {
                action: 'disperse',
                fleeDirection: 'away_from_police',
                moraleThreshold: 0.2, // below this → surrender
                surrenderChance: 0.3,
            },
            policeGoals: {
                action: 'advance_and_arrest',
                advanceSpeed: 0.5,
                arrestRadius: 5,
            },
            civilianGoals: {
                action: 'stay_indoors',
            },
            description: 'Police advancing. Crowd breaking. Arrests being made.',
        },
        {
            name: PHASES.AFTERMATH,
            duration: 120 / escalationSpeed,
            protestorGoals: {
                action: 'scatter',
                returnHome: true,
            },
            policeGoals: {
                action: 'patrol_and_cleanup',
                zone: { x: px, z: pz, radius: plazaRadius * 3 },
            },
            civilianGoals: {
                action: 'cautious_return',
                delay: 30, // wait 30s before venturing out
            },
            description: 'Aftermath. Injuries treated. Area being secured.',
        },
    ];
}

/**
 * Get the current phase based on elapsed time since scenario start.
 * @param {ScenarioPhase[]} phases
 * @param {number} elapsed - seconds since scenario started
 * @returns {{ phase: ScenarioPhase, phaseIndex: number, phaseElapsed: number }}
 */
export function getCurrentPhase(phases, elapsed) {
    let accumulated = 0;
    for (let i = 0; i < phases.length; i++) {
        if (elapsed < accumulated + phases[i].duration) {
            return {
                phase: phases[i],
                phaseIndex: i,
                phaseElapsed: elapsed - accumulated,
            };
        }
        accumulated += phases[i].duration;
    }
    // Past all phases — stay in last one
    return {
        phase: phases[phases.length - 1],
        phaseIndex: phases.length - 1,
        phaseElapsed: elapsed - accumulated + phases[phases.length - 1].duration,
    };
}
