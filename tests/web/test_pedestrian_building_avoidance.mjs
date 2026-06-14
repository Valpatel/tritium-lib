// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/pedestrian.js — building avoidance.
// Proves a pedestrian routes A->B around a building and never enters a
// footprint (the operator-facing crowd must not walk through buildings).
// Run: node tests/web/test_pedestrian_building_avoidance.mjs

import { SimPedestrian } from '../../web/sim/pedestrian.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Axis-aligned building obstacles, open-SDK shape: point_in_building +
// path_crosses_building. rects are [x0, x1, z0, z1].
function makeObstacles(rects) {
    const inRect = (x, z, [x0, x1, z0, z1]) =>
        x >= x0 && x <= x1 && z >= z0 && z <= z1;
    const segHitsRect = (ax, az, bx, bz, r) => {
        if (inRect(ax, az, r) || inRect(bx, bz, r)) return true;
        if (inRect((ax + bx) / 2, (az + bz) / 2, r)) return true;
        const [x0, x1, z0, z1] = r;
        const edges = [[x0, z0, x1, z0], [x1, z0, x1, z1],
                       [x1, z1, x0, z1], [x0, z1, x0, z0]];
        const ccw = (px, py, qx, qy, rx, ry) =>
            (ry - py) * (qx - px) - (qy - py) * (rx - px);
        for (const [cx, cz, dx, dz] of edges) {
            const d1 = ccw(ax, az, bx, bz, cx, cz);
            const d2 = ccw(ax, az, bx, bz, dx, dz);
            const d3 = ccw(cx, cz, dx, dz, ax, az);
            const d4 = ccw(cx, cz, dx, dz, bx, bz);
            if (((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0))) return true;
        }
        return false;
    };
    return {
        rects,
        point_in_building(x, z) { return rects.some(r => inRect(x, z, r)); },
        path_crosses_building(pts) {
            for (let i = 0; i < pts.length - 1; i++)
                for (const r of rects)
                    if (segHitsRect(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], r))
                        return true;
            return false;
        },
    };
}

function makePed(x, z) {
    const ped = new SimPedestrian(x, z, { x, z }, { x: 0, z: 0 });
    return ped;
}

function drive(ped, goal, obstacles, maxTicks = 3000, dt = 0.1) {
    ped.setObstacles(obstacles);
    const pts = [[ped.x, ped.z]];
    for (let i = 0; i < maxTicks; i++) {
        // Force a fixed A->B goal each tick (bypass schedule/building-entry).
        ped.overrideGoal = { action: 'go_to', target: { x: goal.x, z: goal.z }, speed: 2.0 };
        ped.tick(dt, 12, [], []);
        pts.push([ped.x, ped.z]);
        if (Math.hypot(ped.x - goal.x, ped.z - goal.z) < 2.5) break;
    }
    return pts;
}

console.log('=== Pedestrian building-avoidance Tests ===');

test('baseline (no obstacles): ped reaches the goal', () => {
    const ped = makePed(-30, 0);
    const pts = drive(ped, { x: 30, z: 0 }, makeObstacles([]));
    const end = pts[pts.length - 1];
    assert.ok(Math.hypot(end[0] - 30, end[1] - 0) < 3,
        `ped did not reach goal: ${end}`);
});

test('ped NEVER enters a building footprint while travelling', () => {
    const obs = makeObstacles([[-3, 3, -20, 20]]); // wall between A and B
    const ped = makePed(-30, 0);
    const pts = drive(ped, { x: 30, z: 0 }, obs);
    for (const [x, z] of pts)
        assert.ok(!obs.point_in_building(x, z),
            `ped entered building at (${x.toFixed(1)},${z.toFixed(1)})`);
});

test('traversed polyline never crosses a footprint (swept)', () => {
    const obs = makeObstacles([[-3, 3, -20, 20]]);
    const ped = makePed(-30, 0);
    const pts = drive(ped, { x: 30, z: 0 }, obs);
    for (let i = 0; i < pts.length - 1; i++)
        assert.ok(!obs.path_crosses_building([pts[i], pts[i + 1]]),
            `segment ${i} crossed a building: ${pts[i]}->${pts[i + 1]}`);
});

test('ped routes AROUND the wall (gets to the far side, no deadlock)', () => {
    const obs = makeObstacles([[-3, 3, -20, 20]]);
    const ped = makePed(-30, 0);
    const pts = drive(ped, { x: 30, z: 0 }, obs);
    const end = pts[pts.length - 1];
    assert.ok(end[0] > 5,
        `ped stalled before clearing the wall (x=${end[0].toFixed(1)})`);
});

test('ped spawned INSIDE a building can still walk out', () => {
    const obs = makeObstacles([[-10, 10, -10, 10]]);
    const ped = makePed(0, 0); // dead center of the footprint
    const pts = drive(ped, { x: 40, z: 0 }, obs, 1500);
    const end = pts[pts.length - 1];
    assert.ok(end[0] > 12 && !obs.point_in_building(end[0], end[1]),
        `ped failed to exit the building it spawned in (end ${end})`);
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);
