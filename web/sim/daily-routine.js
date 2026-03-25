// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Daily routine goal system for NPCs.
 *
 * Each NPC has a schedule of activities based on their role and the time of day.
 * The routine drives goal-driven behavior: wake up → commute → work → lunch →
 * work → commute → leisure → home.
 *
 * This is a pure data module (no Three.js) — it returns goal sequences that
 * the NPC AI system (yuka StateMachine or GoalDrivenAgent) executes.
 */

/**
 * @typedef {Object} RoutineGoal
 * @property {string} action - What to do: 'go_to', 'stay_at', 'wander', 'idle'
 * @property {string} destination - POI id or zone type: 'home', 'work', 'park', 'commercial', 'plaza'
 * @property {number} startHour - When this goal begins (24h format, e.g., 7.5 = 7:30am)
 * @property {number} duration - How long to stay (hours)
 * @property {string} [transport] - 'walk', 'car', 'bus'
 * @property {string} [mood] - 'calm', 'hurried', 'relaxed'
 */

/**
 * Generate a daily routine for an NPC based on their role.
 * @param {string} role - 'resident', 'worker', 'student', 'police', 'shopkeeper', 'jogger', 'dogwalker'
 * @param {Object} pois - Available POIs: { home: {x,z}, work: {x,z}, school: {x,z}, park: {x,z}, ... }
 * @param {function} rng - Seeded random function (for deterministic routines)
 * @returns {RoutineGoal[]}
 */
export function generateDailyRoutine(role, pois, rng = Math.random) {
    const routines = {
        resident: () => residentRoutine(pois, rng),
        worker: () => workerRoutine(pois, rng),
        student: () => studentRoutine(pois, rng),
        police: () => policeRoutine(pois, rng),
        shopkeeper: () => shopkeeperRoutine(pois, rng),
        jogger: () => joggerRoutine(pois, rng),
        dogwalker: () => dogwalkerRoutine(pois, rng),
    };
    return (routines[role] || routines.resident)();
}

function residentRoutine(pois, rng) {
    const wakeHour = 6.5 + rng() * 2; // 6:30-8:30
    const workStart = wakeHour + 0.5 + rng() * 0.5;
    const lunchHour = 12 + rng() * 0.5;
    const workEnd = 17 + rng();
    const leisureChance = rng();

    const goals = [
        { action: 'stay_at', destination: 'home', startHour: 0, duration: wakeHour },
        { action: 'go_to', destination: 'work', startHour: wakeHour, transport: rng() < 0.6 ? 'car' : 'walk', mood: 'hurried' },
        { action: 'stay_at', destination: 'work', startHour: workStart, duration: lunchHour - workStart },
        { action: 'go_to', destination: 'commercial', startHour: lunchHour, transport: 'walk', mood: 'relaxed' },
        { action: 'stay_at', destination: 'commercial', startHour: lunchHour + 0.1, duration: 0.7 },
        { action: 'go_to', destination: 'work', startHour: lunchHour + 0.8, transport: 'walk' },
        { action: 'stay_at', destination: 'work', startHour: lunchHour + 1, duration: workEnd - lunchHour - 1 },
        { action: 'go_to', destination: 'home', startHour: workEnd, transport: rng() < 0.6 ? 'car' : 'walk' },
    ];

    // 40% chance of evening leisure
    if (leisureChance < 0.4) {
        const leisureDest = rng() < 0.5 ? 'park' : 'commercial';
        goals.push({ action: 'go_to', destination: leisureDest, startHour: workEnd + 0.5, transport: 'walk', mood: 'relaxed' });
        goals.push({ action: 'wander', destination: leisureDest, startHour: workEnd + 0.7, duration: 0.5 + rng() });
        goals.push({ action: 'go_to', destination: 'home', startHour: workEnd + 1.5 + rng(), transport: 'walk' });
    }

    goals.push({ action: 'stay_at', destination: 'home', startHour: 21, duration: 10 });
    return goals;
}

function workerRoutine(pois, rng) {
    // Same as resident but always drives
    const goals = residentRoutine(pois, rng);
    goals.forEach(g => { if (g.transport) g.transport = 'car'; });
    return goals;
}

function studentRoutine(pois, rng) {
    const wakeHour = 7 + rng() * 0.5;
    return [
        { action: 'stay_at', destination: 'home', startHour: 0, duration: wakeHour },
        { action: 'go_to', destination: 'school', startHour: wakeHour, transport: rng() < 0.3 ? 'bus' : 'walk', mood: 'calm' },
        { action: 'stay_at', destination: 'school', startHour: wakeHour + 0.3, duration: 7 },
        { action: 'go_to', destination: rng() < 0.5 ? 'park' : 'home', startHour: 15, transport: 'walk', mood: 'relaxed' },
        { action: 'wander', destination: 'park', startHour: 15.3, duration: 1 + rng() },
        { action: 'go_to', destination: 'home', startHour: 17, transport: 'walk' },
        { action: 'stay_at', destination: 'home', startHour: 17.5, duration: 13 },
    ];
}

function policeRoutine(pois, rng) {
    return [
        { action: 'stay_at', destination: 'police_station', startHour: 0, duration: 6 + rng() * 2 },
        { action: 'go_to', destination: 'patrol_route', startHour: 7, transport: 'car' },
        { action: 'wander', destination: 'patrol_zone', startHour: 7.5, duration: 4 },
        { action: 'go_to', destination: 'commercial', startHour: 12, transport: 'car', mood: 'calm' },
        { action: 'stay_at', destination: 'commercial', startHour: 12.2, duration: 0.5 },
        { action: 'go_to', destination: 'patrol_route', startHour: 13, transport: 'car' },
        { action: 'wander', destination: 'patrol_zone', startHour: 13.5, duration: 4 },
        { action: 'go_to', destination: 'police_station', startHour: 17, transport: 'car' },
        { action: 'stay_at', destination: 'police_station', startHour: 17.5, duration: 13.5 },
    ];
}

function shopkeeperRoutine(pois, rng) {
    const openHour = 9 + rng() * 0.5;
    const closeHour = 18 + rng() * 2;
    return [
        { action: 'stay_at', destination: 'home', startHour: 0, duration: openHour - 0.5 },
        { action: 'go_to', destination: 'commercial', startHour: openHour - 0.5, transport: 'walk' },
        { action: 'stay_at', destination: 'commercial', startHour: openHour, duration: closeHour - openHour },
        { action: 'go_to', destination: 'home', startHour: closeHour, transport: 'walk', mood: 'relaxed' },
        { action: 'stay_at', destination: 'home', startHour: closeHour + 0.3, duration: 24 - closeHour },
    ];
}

function joggerRoutine(pois, rng) {
    const jogHour = 6 + rng() * 2;
    return [
        { action: 'stay_at', destination: 'home', startHour: 0, duration: jogHour },
        { action: 'go_to', destination: 'park', startHour: jogHour, transport: 'walk', mood: 'hurried' },
        { action: 'wander', destination: 'park', startHour: jogHour + 0.2, duration: 0.5 + rng() * 0.5 },
        { action: 'go_to', destination: 'home', startHour: jogHour + 1, transport: 'walk', mood: 'relaxed' },
        // Rest of day: regular resident routine
        ...residentRoutine(pois, rng).filter(g => g.startHour > jogHour + 2),
    ];
}

function dogwalkerRoutine(pois, rng) {
    const walkHour = 7 + rng();
    const eveningWalk = 18 + rng();
    const base = residentRoutine(pois, rng);
    return [
        { action: 'stay_at', destination: 'home', startHour: 0, duration: walkHour },
        { action: 'go_to', destination: 'park', startHour: walkHour, transport: 'walk', mood: 'calm' },
        { action: 'wander', destination: 'park', startHour: walkHour + 0.2, duration: 0.4 + rng() * 0.3 },
        { action: 'go_to', destination: 'home', startHour: walkHour + 0.7, transport: 'walk' },
        ...base.filter(g => g.startHour > walkHour + 1 && g.startHour < eveningWalk),
        { action: 'go_to', destination: 'park', startHour: eveningWalk, transport: 'walk', mood: 'relaxed' },
        { action: 'wander', destination: 'park', startHour: eveningWalk + 0.2, duration: 0.3 },
        { action: 'go_to', destination: 'home', startHour: eveningWalk + 0.6, transport: 'walk' },
        { action: 'stay_at', destination: 'home', startHour: 21, duration: 10 },
    ];
}

/**
 * Get the current goal for an NPC based on time of day.
 * @param {RoutineGoal[]} routine
 * @param {number} hour - Current hour (0-24, fractional)
 * @returns {RoutineGoal|null}
 */
export function getCurrentGoal(routine, hour) {
    for (let i = routine.length - 1; i >= 0; i--) {
        if (hour >= routine[i].startHour) return routine[i];
    }
    return routine[0] || null;
}

/**
 * Get the NEXT goal after the current one.
 * @param {RoutineGoal[]} routine
 * @param {number} hour
 * @returns {RoutineGoal|null}
 */
export function getNextGoal(routine, hour) {
    for (const goal of routine) {
        if (goal.startHour > hour) return goal;
    }
    return null; // No more goals today
}

/**
 * NPC role distribution for a residential neighborhood.
 * @param {function} rng
 * @returns {string} role
 */
export function randomRole(rng = Math.random) {
    const r = rng();
    if (r < 0.35) return 'resident';
    if (r < 0.55) return 'worker';
    if (r < 0.65) return 'student';
    if (r < 0.70) return 'police';
    if (r < 0.78) return 'shopkeeper';
    if (r < 0.88) return 'jogger';
    return 'dogwalker';
}
