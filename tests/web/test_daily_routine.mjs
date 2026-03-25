// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/daily-routine.js — NPC daily routines
// Run: node tests/web/test_daily_routine.mjs

import { generateDailyRoutine, getCurrentGoal, getNextGoal, randomRole } from '../../web/sim/daily-routine.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

const POIS = {
    home: { x: 0, z: 0 },
    work: { x: 100, z: 0 },
    school: { x: 50, z: 50 },
    park: { x: -50, z: 30 },
    commercial: { x: 80, z: 20 },
    police_station: { x: 20, z: 80 },
    patrol_route: { x: 60, z: 60 },
    patrol_zone: { x: 70, z: 70 },
};

// Deterministic RNG for reproducible tests
function seededRng(seed) {
    let s = seed;
    return function() {
        s = (s * 16807 + 0) % 2147483647;
        return (s - 1) / 2147483646;
    };
}

console.log('=== DailyRoutine Tests ===');

// ── generateDailyRoutine ──────────────────────────────────────────

test('resident routine returns non-empty goal list', () => {
    const goals = generateDailyRoutine('resident', POIS, seededRng(42));
    assert.ok(goals.length >= 5, `expected >= 5 goals, got ${goals.length}`);
});

test('worker routine returns goals with car transport', () => {
    const goals = generateDailyRoutine('worker', POIS, seededRng(42));
    const carGoals = goals.filter(g => g.transport === 'car');
    assert.ok(carGoals.length > 0, 'workers should have at least one car trip');
});

test('student routine includes school destination', () => {
    const goals = generateDailyRoutine('student', POIS, seededRng(42));
    const schoolGoals = goals.filter(g => g.destination === 'school');
    assert.ok(schoolGoals.length > 0, 'students should go to school');
});

test('police routine includes patrol_route destination', () => {
    const goals = generateDailyRoutine('police', POIS, seededRng(42));
    const patrolGoals = goals.filter(g => g.destination === 'patrol_route' || g.destination === 'patrol_zone');
    assert.ok(patrolGoals.length > 0, 'police should patrol');
});

test('shopkeeper routine includes commercial destination', () => {
    const goals = generateDailyRoutine('shopkeeper', POIS, seededRng(42));
    const commercialGoals = goals.filter(g => g.destination === 'commercial');
    assert.ok(commercialGoals.length > 0, 'shopkeepers should go to commercial area');
});

test('jogger routine includes park destination', () => {
    const goals = generateDailyRoutine('jogger', POIS, seededRng(42));
    const parkGoals = goals.filter(g => g.destination === 'park');
    assert.ok(parkGoals.length > 0, 'joggers should go to park');
});

test('dogwalker routine includes park destination', () => {
    const goals = generateDailyRoutine('dogwalker', POIS, seededRng(42));
    const parkGoals = goals.filter(g => g.destination === 'park');
    assert.ok(parkGoals.length >= 2, 'dogwalkers should go to park at least twice');
});

test('unknown role falls back to resident routine', () => {
    const goals = generateDailyRoutine('alien_visitor', POIS, seededRng(42));
    assert.ok(goals.length >= 5, 'unknown role should produce a valid routine');
});

test('all goals have startHour property', () => {
    const roles = ['resident', 'worker', 'student', 'police', 'shopkeeper', 'jogger', 'dogwalker'];
    for (const role of roles) {
        const goals = generateDailyRoutine(role, POIS, seededRng(42));
        for (const goal of goals) {
            assert.strictEqual(typeof goal.startHour, 'number', `${role}: goal missing startHour`);
        }
    }
});

test('all goals have valid action', () => {
    const validActions = new Set(['go_to', 'stay_at', 'wander', 'idle']);
    const goals = generateDailyRoutine('resident', POIS, seededRng(42));
    for (const goal of goals) {
        assert.ok(validActions.has(goal.action), `invalid action: ${goal.action}`);
    }
});

test('routine is deterministic with same seed', () => {
    const goals1 = generateDailyRoutine('resident', POIS, seededRng(42));
    const goals2 = generateDailyRoutine('resident', POIS, seededRng(42));
    assert.strictEqual(goals1.length, goals2.length);
    for (let i = 0; i < goals1.length; i++) {
        assert.strictEqual(goals1[i].action, goals2[i].action);
        assert.strictEqual(goals1[i].destination, goals2[i].destination);
    }
});

// ── getCurrentGoal ────────────────────────────────────────────────

test('getCurrentGoal returns last goal with startHour <= hour', () => {
    const routine = [
        { startHour: 0, action: 'stay_at', destination: 'home' },
        { startHour: 7, action: 'go_to', destination: 'work' },
        { startHour: 17, action: 'go_to', destination: 'home' },
    ];
    const goal = getCurrentGoal(routine, 12);
    assert.strictEqual(goal.destination, 'work');
});

test('getCurrentGoal returns first goal for very early hour', () => {
    const routine = [
        { startHour: 0, action: 'stay_at', destination: 'home' },
        { startHour: 7, action: 'go_to', destination: 'work' },
    ];
    const goal = getCurrentGoal(routine, 3);
    assert.strictEqual(goal.destination, 'home');
});

test('getCurrentGoal returns last goal for late hour', () => {
    const routine = [
        { startHour: 0, action: 'stay_at', destination: 'home' },
        { startHour: 21, action: 'stay_at', destination: 'home' },
    ];
    const goal = getCurrentGoal(routine, 23);
    assert.strictEqual(goal.startHour, 21);
});

test('getCurrentGoal returns null for empty routine', () => {
    const goal = getCurrentGoal([], 12);
    assert.strictEqual(goal, null);
});

// ── getNextGoal ───────────────────────────────────────────────────

test('getNextGoal returns first goal after current hour', () => {
    const routine = [
        { startHour: 0, action: 'stay_at', destination: 'home' },
        { startHour: 7, action: 'go_to', destination: 'work' },
        { startHour: 17, action: 'go_to', destination: 'home' },
    ];
    const goal = getNextGoal(routine, 10);
    assert.strictEqual(goal.startHour, 17);
});

test('getNextGoal returns null when no more goals', () => {
    const routine = [
        { startHour: 0, action: 'stay_at', destination: 'home' },
        { startHour: 7, action: 'go_to', destination: 'work' },
    ];
    const goal = getNextGoal(routine, 20);
    assert.strictEqual(goal, null);
});

// ── randomRole ────────────────────────────────────────────────────

test('randomRole returns a valid role string', () => {
    const validRoles = new Set(['resident', 'worker', 'student', 'police', 'shopkeeper', 'jogger', 'dogwalker']);
    for (let i = 0; i < 20; i++) {
        const role = randomRole(() => i / 20);
        assert.ok(validRoles.has(role), `invalid role: ${role}`);
    }
});

test('randomRole: low rng returns resident', () => {
    assert.strictEqual(randomRole(() => 0.1), 'resident');
});

test('randomRole: mid rng returns worker or student', () => {
    const role = randomRole(() => 0.5);
    assert.ok(role === 'worker' || role === 'student', `expected worker/student, got ${role}`);
});

test('randomRole: high rng returns dogwalker', () => {
    assert.strictEqual(randomRole(() => 0.99), 'dogwalker');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
