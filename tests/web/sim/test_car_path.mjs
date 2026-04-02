// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/car-path.js — CarPath vehicle path system
// Run: node tests/web/sim/test_car_path.mjs

import {
    StraightSegment,
    TurnSegment,
    CarPath,
    computeTurnControlFromHeadings,
} from '../../../web/sim/car-path.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// ── StraightSegment ───────────────────────────────────────────────

console.log('=== StraightSegment Tests ===');

test('StraightSegment: length is euclidean distance', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 30, z: 40 });
    assert.ok(Math.abs(seg.length - 50) < 0.001, `expected 50, got ${seg.length}`);
});

test('StraightSegment: horizontal length', () => {
    const seg = new StraightSegment({ x: 10, z: 5 }, { x: 110, z: 5 });
    assert.ok(Math.abs(seg.length - 100) < 0.001, `expected 100, got ${seg.length}`);
});

test('StraightSegment: getPositionAt(0) returns start', () => {
    const seg = new StraightSegment({ x: 10, z: 20 }, { x: 60, z: 20 });
    const pos = seg.getPositionAt(0);
    assert.ok(Math.abs(pos.x - 10) < 0.001, `expected x=10, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - 20) < 0.001, `expected z=20, got ${pos.z}`);
});

test('StraightSegment: getPositionAt(1) returns end', () => {
    const seg = new StraightSegment({ x: 10, z: 20 }, { x: 60, z: 20 });
    const pos = seg.getPositionAt(1);
    assert.ok(Math.abs(pos.x - 60) < 0.001, `expected x=60, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - 20) < 0.001, `expected z=20, got ${pos.z}`);
});

test('StraightSegment: getPositionAt(0.5) returns midpoint', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 100, z: 0 });
    const pos = seg.getPositionAt(0.5);
    assert.ok(Math.abs(pos.x - 50) < 0.001, `expected x=50, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - 0) < 0.001, `expected z=0, got ${pos.z}`);
});

test('StraightSegment: getPositionAt(0.25) for diagonal', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 40, z: 30 });
    const pos = seg.getPositionAt(0.25);
    assert.ok(Math.abs(pos.x - 10) < 0.001, `expected x=10, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - 7.5) < 0.001, `expected z=7.5, got ${pos.z}`);
});

test('StraightSegment: heading East (0 radians)', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 100, z: 0 });
    const h = seg.getHeadingAt(0.5);
    assert.ok(Math.abs(h) < 0.001, `expected 0, got ${h}`);
});

test('StraightSegment: heading South (PI/2 radians)', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 0, z: 100 });
    const h = seg.getHeadingAt(0.5);
    assert.ok(Math.abs(h - Math.PI / 2) < 0.001, `expected PI/2, got ${h}`);
});

test('StraightSegment: heading West (PI radians)', () => {
    const seg = new StraightSegment({ x: 100, z: 0 }, { x: 0, z: 0 });
    const h = seg.getHeadingAt(0);
    assert.ok(Math.abs(Math.abs(h) - Math.PI) < 0.001, `expected +/-PI, got ${h}`);
});

test('StraightSegment: heading North (-PI/2 radians)', () => {
    const seg = new StraightSegment({ x: 0, z: 100 }, { x: 0, z: 0 });
    const h = seg.getHeadingAt(0);
    assert.ok(Math.abs(h + Math.PI / 2) < 0.001, `expected -PI/2, got ${h}`);
});

test('StraightSegment: heading constant along segment', () => {
    const seg = new StraightSegment({ x: 0, z: 0 }, { x: 100, z: 50 });
    const h0 = seg.getHeadingAt(0);
    const h1 = seg.getHeadingAt(0.5);
    const h2 = seg.getHeadingAt(1);
    assert.ok(Math.abs(h0 - h1) < 0.001, 'heading should be constant');
    assert.ok(Math.abs(h1 - h2) < 0.001, 'heading should be constant');
});

test('StraightSegment: getEndPoint and getStartPoint', () => {
    const seg = new StraightSegment({ x: 5, z: 10 }, { x: 50, z: 60 });
    const start = seg.getStartPoint();
    const end = seg.getEndPoint();
    assert.ok(Math.abs(start.x - 5) < 0.001);
    assert.ok(Math.abs(start.z - 10) < 0.001);
    assert.ok(Math.abs(end.x - 50) < 0.001);
    assert.ok(Math.abs(end.z - 60) < 0.001);
});

// ── TurnSegment ───────────────────────────────────────────────────

console.log('\n=== TurnSegment Tests ===');

test('TurnSegment: getPositionAt(0) returns entry', () => {
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const pos = seg.getPositionAt(0);
    assert.ok(Math.abs(pos.x) < 0.001, `expected x=0, got ${pos.x}`);
    assert.ok(Math.abs(pos.z) < 0.001, `expected z=0, got ${pos.z}`);
});

test('TurnSegment: getPositionAt(1) returns exit', () => {
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const pos = seg.getPositionAt(1);
    assert.ok(Math.abs(pos.x - 10) < 0.001, `expected x=10, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - 10) < 0.001, `expected z=10, got ${pos.z}`);
});

test('TurnSegment: midpoint is on the curve (not the control point)', () => {
    // For quadratic Bezier: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const pos = seg.getPositionAt(0.5);
    const expectedX = 0.25 * 0 + 0.5 * 10 + 0.25 * 10;  // 7.5
    const expectedZ = 0.25 * 0 + 0.5 * 0 + 0.25 * 10;   // 2.5
    assert.ok(Math.abs(pos.x - expectedX) < 0.001, `expected x=${expectedX}, got ${pos.x}`);
    assert.ok(Math.abs(pos.z - expectedZ) < 0.001, `expected z=${expectedZ}, got ${pos.z}`);
});

test('TurnSegment: arc length > chord length for 90-degree turn', () => {
    // A 90-degree turn: chord length = sqrt(10^2 + 10^2) = 14.14
    // Arc should be longer than the chord
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const chord = Math.sqrt(100 + 100);
    assert.ok(seg.length > chord, `arc ${seg.length} should exceed chord ${chord}`);
});

test('TurnSegment: arc length < sum of control legs', () => {
    // Arc should be shorter than going entry→control + control→exit
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const legs = 10 + 10; // entry→control + control→exit
    assert.ok(seg.length < legs, `arc ${seg.length} should be less than legs ${legs}`);
});

test('TurnSegment: degenerate Bezier (straight through) has correct length', () => {
    // When control is midpoint: should approximate a straight line
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 50, z: 0 }, { x: 100, z: 0 });
    assert.ok(Math.abs(seg.length - 100) < 0.5, `expected ~100, got ${seg.length}`);
});

test('TurnSegment: heading at entry is toward control', () => {
    // Entry going East (control is to the right/east of entry)
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const h = seg.getHeadingAt(0);
    // Tangent at t=0: 2*(P1 - P0) = 2*(10-0, 0-0) = (20, 0) → heading = 0 (East)
    assert.ok(Math.abs(h) < 0.001, `expected heading 0 (East) at entry, got ${h}`);
});

test('TurnSegment: heading at exit is away from control', () => {
    // Exit going South (exit is below control)
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const h = seg.getHeadingAt(1);
    // Tangent at t=1: 2*(P2 - P1) = 2*(0, 10) = (0, 20) → heading = PI/2 (South)
    assert.ok(Math.abs(h - Math.PI / 2) < 0.001, `expected heading PI/2 (South) at exit, got ${h}`);
});

test('TurnSegment: heading rotates smoothly from East to South', () => {
    const seg = new TurnSegment({ x: 0, z: 0 }, { x: 10, z: 0 }, { x: 10, z: 10 });
    const h0 = seg.getHeadingAt(0);     // should be ~0 (East)
    const hMid = seg.getHeadingAt(0.5); // should be ~PI/4 (Southeast)
    const h1 = seg.getHeadingAt(1);     // should be ~PI/2 (South)

    assert.ok(h0 < hMid, `heading should increase: h0=${h0}, hMid=${hMid}`);
    assert.ok(hMid < h1, `heading should increase: hMid=${hMid}, h1=${h1}`);
    assert.ok(Math.abs(hMid - Math.PI / 4) < 0.01, `expected ~PI/4 at midpoint, got ${hMid}`);
});

test('TurnSegment: getEndPoint and getStartPoint', () => {
    const seg = new TurnSegment({ x: 5, z: 10 }, { x: 20, z: 10 }, { x: 20, z: 30 });
    const start = seg.getStartPoint();
    const end = seg.getEndPoint();
    assert.ok(Math.abs(start.x - 5) < 0.001);
    assert.ok(Math.abs(start.z - 10) < 0.001);
    assert.ok(Math.abs(end.x - 20) < 0.001);
    assert.ok(Math.abs(end.z - 30) < 0.001);
});

test('TurnSegment: arc length approximation accuracy', () => {
    // For a quarter circle with radius R, arc length = PI/2 * R
    // Quadratic Bezier approximation of quarter circle:
    // P0 = (R, 0), P1 = (R, R), P2 = (0, R)
    // This isn't a perfect quarter circle, but for known Bezier the
    // arc length should be close to the analytic value.
    const R = 10;
    const seg = new TurnSegment({ x: R, z: 0 }, { x: R, z: R }, { x: 0, z: R });
    // True quarter circle arc = PI/2 * R = 15.708
    // Quadratic Bezier quarter turn arc ≈ 15.2 (slightly different shape)
    // Just verify it's in the right ballpark (between chord and leg sum)
    const chord = Math.sqrt(R * R + R * R);   // 14.14
    const legs = R + R;                        // 20
    assert.ok(seg.length > chord, `arc ${seg.length} should exceed chord ${chord}`);
    assert.ok(seg.length < legs, `arc ${seg.length} should be less than legs ${legs}`);
    // More specific: should be around 14.5-16 for this geometry
    // Quadratic Bezier quarter turn: arc length is ~16.2 (not a perfect circle)
    assert.ok(seg.length > 14 && seg.length < 17,
        `arc length ${seg.length} should be between 14 and 17 for R=${R}`);
});

// ── CarPath ───────────────────────────────────────────────────────

console.log('\n=== CarPath Tests ===');

test('CarPath: empty path returns null for all queries', () => {
    const path = new CarPath();
    assert.strictEqual(path.getPosition(0), null);
    assert.strictEqual(path.getHeading(0), null);
    assert.strictEqual(path.getSegmentAt(0), null);
    assert.strictEqual(path.isInTurn(0), false);
    assert.strictEqual(path.getEndPoint(), null);
    assert.strictEqual(path.totalLength, 0);
});

test('CarPath: single straight segment', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });
    assert.ok(Math.abs(path.totalLength - 100) < 0.001);
    assert.strictEqual(path.segments.length, 1);
});

test('CarPath: getPosition at start, middle, end of straight', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });

    const p0 = path.getPosition(0);
    assert.ok(Math.abs(p0.x) < 0.001, `start x: expected 0, got ${p0.x}`);

    const p50 = path.getPosition(50);
    assert.ok(Math.abs(p50.x - 50) < 0.001, `mid x: expected 50, got ${p50.x}`);

    const p100 = path.getPosition(100);
    assert.ok(Math.abs(p100.x - 100) < 0.001, `end x: expected 100, got ${p100.x}`);
});

test('CarPath: multiple straight segments are continuous', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addStraight({ x: 50, z: 0 }, { x: 50, z: 80 });

    assert.ok(Math.abs(path.totalLength - 130) < 0.001, `expected 130, got ${path.totalLength}`);

    // Position at junction (d = 50)
    const pJunction = path.getPosition(50);
    assert.ok(Math.abs(pJunction.x - 50) < 0.001);
    assert.ok(Math.abs(pJunction.z - 0) < 0.001);

    // Position inside second segment (d = 80)
    const p80 = path.getPosition(80);
    assert.ok(Math.abs(p80.x - 50) < 0.001, `expected x=50, got ${p80.x}`);
    assert.ok(Math.abs(p80.z - 30) < 0.001, `expected z=30, got ${p80.z}`);

    // Position at end
    const pEnd = path.getPosition(130);
    assert.ok(Math.abs(pEnd.x - 50) < 0.001);
    assert.ok(Math.abs(pEnd.z - 80) < 0.001);
});

test('CarPath: straight + turn + straight path', () => {
    const path = new CarPath();
    // 50m East
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    // Turn East→South (90 degrees)
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });
    // 40m South
    path.addStraight({ x: 60, z: 10 }, { x: 60, z: 50 });

    assert.strictEqual(path.segments.length, 3);
    assert.ok(path.totalLength > 100, `total should exceed 100, got ${path.totalLength}`);

    // First segment
    const p25 = path.getPosition(25);
    assert.ok(Math.abs(p25.x - 25) < 0.5);
    assert.ok(Math.abs(p25.z) < 0.5);

    // In the turn
    const turnStart = 50;
    const turnEnd = 50 + path.segments[1].length;
    const turnMid = (turnStart + turnEnd) / 2;
    assert.ok(path.isInTurn(turnMid), 'should be in turn at turn midpoint');
    assert.ok(!path.isInTurn(25), 'should not be in turn at d=25');

    // After the turn
    const afterTurn = turnEnd + 10;
    assert.ok(!path.isInTurn(afterTurn), 'should not be in turn after turn');
});

test('CarPath: getHeading returns correct angles', () => {
    const path = new CarPath();
    // East
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    // South
    path.addStraight({ x: 50, z: 0 }, { x: 50, z: 50 });

    // East heading
    const hEast = path.getHeading(25);
    assert.ok(Math.abs(hEast) < 0.01, `expected 0 (East), got ${hEast}`);

    // South heading
    const hSouth = path.getHeading(75);
    assert.ok(Math.abs(hSouth - Math.PI / 2) < 0.01, `expected PI/2 (South), got ${hSouth}`);
});

test('CarPath: isInTurn correctly identifies segment types', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });
    path.addStraight({ x: 60, z: 10 }, { x: 60, z: 50 });

    assert.strictEqual(path.isInTurn(25), false, 'first straight is not a turn');
    assert.strictEqual(path.isInTurn(50 + path.segments[1].length / 2), true, 'middle is a turn');

    const afterTurnD = 50 + path.segments[1].length + 5;
    assert.strictEqual(path.isInTurn(afterTurnD), false, 'last straight is not a turn');
});

test('CarPath: getSegmentAt returns correct segment info', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });
    path.addStraight({ x: 100, z: 0 }, { x: 100, z: 50 });

    const info1 = path.getSegmentAt(30);
    assert.strictEqual(info1.index, 0);
    assert.ok(Math.abs(info1.t - 0.3) < 0.001, `expected t=0.3, got ${info1.t}`);
    assert.ok(Math.abs(info1.localD - 30) < 0.001);

    const info2 = path.getSegmentAt(120);
    assert.strictEqual(info2.index, 1);
    assert.ok(Math.abs(info2.localD - 20) < 0.001);
    assert.ok(Math.abs(info2.t - 0.4) < 0.001, `expected t=0.4, got ${info2.t}`);
});

test('CarPath: getSegmentAt clamps to path bounds', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });

    // Before start
    const before = path.getSegmentAt(-10);
    assert.strictEqual(before.index, 0);
    assert.ok(Math.abs(before.t) < 0.001, 'should clamp to start');

    // Past end
    const after = path.getSegmentAt(200);
    assert.strictEqual(after.index, 0);
    assert.ok(Math.abs(after.t - 1) < 0.001, 'should clamp to end');
});

// ── trimBefore ────────────────────────────────────────────────────

console.log('\n=== trimBefore Tests ===');

test('trimBefore: removes segments behind d', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });    // 0-50
    path.addStraight({ x: 50, z: 0 }, { x: 100, z: 0 });   // 50-100
    path.addStraight({ x: 100, z: 0 }, { x: 150, z: 0 });  // 100-150

    assert.strictEqual(path.segments.length, 3);

    const removed = path.trimBefore(60);
    assert.strictEqual(removed, 1, 'should remove 1 segment');
    assert.strictEqual(path.segments.length, 2);
});

test('trimBefore: keeps at least one segment', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });

    const removed = path.trimBefore(100);
    assert.strictEqual(removed, 0, 'should not remove last segment');
    assert.strictEqual(path.segments.length, 1);
});

test('trimBefore: removes multiple segments', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 20, z: 0 });    // 0-20
    path.addStraight({ x: 20, z: 0 }, { x: 40, z: 0 });   // 20-40
    path.addStraight({ x: 40, z: 0 }, { x: 60, z: 0 });   // 40-60
    path.addStraight({ x: 60, z: 0 }, { x: 80, z: 0 });   // 60-80

    const removed = path.trimBefore(50);
    assert.strictEqual(removed, 2, 'should remove 2 segments (0-20 and 20-40)');
    assert.strictEqual(path.segments.length, 2);
});

test('trimBefore: preserves d validity for remaining segments', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });    // 0-50
    path.addStraight({ x: 50, z: 0 }, { x: 100, z: 0 });   // 50-100
    path.addStraight({ x: 100, z: 0 }, { x: 150, z: 0 });  // 100-150

    path.trimBefore(60);

    // d=100 should still work (now in the second remaining segment)
    const pos = path.getPosition(100);
    assert.ok(pos !== null, 'should get position at d=100');
    assert.ok(Math.abs(pos.x - 100) < 0.5, `expected x=100, got ${pos.x}`);
});

test('trimBefore: no removal when d is before first segment end', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addStraight({ x: 50, z: 0 }, { x: 100, z: 0 });

    const removed = path.trimBefore(30);
    assert.strictEqual(removed, 0, 'should remove nothing');
    assert.strictEqual(path.segments.length, 2);
});

// ── getEndPoint / getStartPoint ───────────────────────────────────

console.log('\n=== Endpoint Tests ===');

test('getEndPoint: returns last segment endpoint', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });

    const end = path.getEndPoint();
    assert.ok(Math.abs(end.x - 60) < 0.001);
    assert.ok(Math.abs(end.z - 10) < 0.001);
});

test('getStartPoint: returns first segment start', () => {
    const path = new CarPath();
    path.addStraight({ x: 5, z: 10 }, { x: 50, z: 0 });
    path.addStraight({ x: 50, z: 0 }, { x: 100, z: 0 });

    const start = path.getStartPoint();
    assert.ok(Math.abs(start.x - 5) < 0.001);
    assert.ok(Math.abs(start.z - 10) < 0.001);
});

// ── remainingDistance ──────────────────────────────────────────────

console.log('\n=== Remaining Distance Tests ===');

test('remainingDistance: correct at various points', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });

    assert.ok(Math.abs(path.remainingDistance(0) - 100) < 0.001);
    assert.ok(Math.abs(path.remainingDistance(50) - 50) < 0.001);
    assert.ok(Math.abs(path.remainingDistance(100) - 0) < 0.001);
    assert.ok(Math.abs(path.remainingDistance(150) - 0) < 0.001);
});

// ── samplePositions ───────────────────────────────────────────────

console.log('\n=== Sample Positions Tests ===');

test('samplePositions: returns correct number of points', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 100, z: 0 });

    const samples = path.samplePositions(0, 100, 25);
    // 0, 25, 50, 75, 100 = 5 points (100 is both a step point and end point)
    assert.ok(samples.length >= 4 && samples.length <= 6,
        `expected 4-6 points, got ${samples.length}`);
});

test('samplePositions: first and last points are start and end', () => {
    const path = new CarPath();
    path.addStraight({ x: 10, z: 20 }, { x: 60, z: 20 });

    const samples = path.samplePositions(0, 50, 10);
    assert.ok(Math.abs(samples[0].x - 10) < 0.5, 'first sample near start');
    const last = samples[samples.length - 1];
    assert.ok(Math.abs(last.x - 60) < 0.5, 'last sample near end');
});

// ── fromRoute ─────────────────────────────────────────────────────

console.log('\n=== fromRoute Tests ===');

test('fromRoute: builds path from 2-node route', () => {
    const network = {
        nodes: {
            'n0': { id: 'n0', x: 0, z: 0 },
            'n1': { id: 'n1', x: 100, z: 0 },
        },
    };

    const path = CarPath.fromRoute(['n0', 'n1'], network, 2);
    assert.ok(path.segments.length >= 1, `expected at least 1 segment, got ${path.segments.length}`);
    assert.ok(path.totalLength > 90, `path should be near 100m, got ${path.totalLength}`);
});

test('fromRoute: builds path from 3-node route with turn', () => {
    const network = {
        nodes: {
            'n0': { id: 'n0', x: 0, z: 0 },
            'n1': { id: 'n1', x: 100, z: 0 },
            'n2': { id: 'n2', x: 100, z: 100 },
        },
    };

    const path = CarPath.fromRoute(['n0', 'n1', 'n2'], network, 2);
    // Should have: straight + turn + straight = 3 segments
    assert.ok(path.segments.length >= 3, `expected 3+ segments, got ${path.segments.length}`);
    // Total length should be > 180m (two ~100m legs)
    assert.ok(path.totalLength > 180, `total should be > 180, got ${path.totalLength}`);
});

test('fromRoute: path endpoints are near intersections', () => {
    const network = {
        nodes: {
            'n0': { id: 'n0', x: 0, z: 0 },
            'n1': { id: 'n1', x: 100, z: 0 },
        },
    };

    const path = CarPath.fromRoute(['n0', 'n1'], network, 2);
    const start = path.getStartPoint();
    const end = path.getEndPoint();

    // Start should be near n0 (offset by laneOffset)
    assert.ok(Math.abs(start.x - 0) < 5, `start x near 0, got ${start.x}`);
    // End should be near n1 (offset by laneOffset)
    assert.ok(Math.abs(end.x - 100) < 5, `end x near 100, got ${end.x}`);
});

test('fromRoute: returns empty path for single node', () => {
    const network = { nodes: { 'n0': { id: 'n0', x: 0, z: 0 } } };
    const path = CarPath.fromRoute(['n0'], network, 2);
    assert.strictEqual(path.segments.length, 0);
    assert.strictEqual(path.totalLength, 0);
});

test('fromRoute: returns empty path for missing nodes', () => {
    const network = { nodes: {} };
    const path = CarPath.fromRoute(['n0', 'n1'], network, 2);
    assert.strictEqual(path.segments.length, 0);
});

test('fromRoute: handles 4-node route (two turns)', () => {
    const network = {
        nodes: {
            'n0': { id: 'n0', x: 0, z: 0 },
            'n1': { id: 'n1', x: 100, z: 0 },
            'n2': { id: 'n2', x: 100, z: 100 },
            'n3': { id: 'n3', x: 0, z: 100 },
        },
    };

    const path = CarPath.fromRoute(['n0', 'n1', 'n2', 'n3'], network, 2);
    // straight + (turn + straight) + (turn + straight) = 5 segments
    assert.ok(path.segments.length >= 5, `expected 5+ segments, got ${path.segments.length}`);

    // Should have turns
    let turnCount = 0;
    for (const seg of path.segments) {
        if (seg.type === 'turn') turnCount++;
    }
    assert.strictEqual(turnCount, 2, `expected 2 turns, got ${turnCount}`);
});

// ── computeTurnControlFromHeadings ────────────────────────────────

console.log('\n=== computeTurnControlFromHeadings Tests ===');

test('computeTurnControlFromHeadings: East to South gives L-corner', () => {
    const entry = { x: 0, z: 0 };
    const exit = { x: 10, z: 10 };
    const control = computeTurnControlFromHeadings(entry, exit, 0, Math.PI / 2);

    // East heading (0 rad) extended from entry, South heading (PI/2) extended backward from exit
    // Should meet at approximately (10, 0) — the L-corner
    assert.ok(Math.abs(control.x - 10) < 0.5, `expected control x~10, got ${control.x}`);
    assert.ok(Math.abs(control.z - 0) < 0.5, `expected control z~0, got ${control.z}`);
});

test('computeTurnControlFromHeadings: parallel headings give midpoint', () => {
    const entry = { x: 0, z: 0 };
    const exit = { x: 100, z: 0 };
    const control = computeTurnControlFromHeadings(entry, exit, 0, 0);

    assert.ok(Math.abs(control.x - 50) < 0.5, `expected x~50, got ${control.x}`);
    assert.ok(Math.abs(control.z - 0) < 0.5, `expected z~0, got ${control.z}`);
});

// ── Continuity & Integration ──────────────────────────────────────

console.log('\n=== Continuity & Integration Tests ===');

test('Path is continuous at segment boundaries', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });
    path.addStraight({ x: 60, z: 10 }, { x: 60, z: 50 });

    // Check continuity at first boundary (d = 50)
    const p1End = path.getPosition(50 - 0.001);
    const p2Start = path.getPosition(50 + 0.001);
    const gap1 = Math.sqrt((p2Start.x - p1End.x) ** 2 + (p2Start.z - p1End.z) ** 2);
    assert.ok(gap1 < 0.1, `gap at first boundary: ${gap1} should be < 0.1`);

    // Check continuity at second boundary
    const turnLen = path.segments[1].length;
    const boundary2 = 50 + turnLen;
    const p2End = path.getPosition(boundary2 - 0.001);
    const p3Start = path.getPosition(boundary2 + 0.001);
    const gap2 = Math.sqrt((p3Start.x - p2End.x) ** 2 + (p3Start.z - p2End.z) ** 2);
    assert.ok(gap2 < 0.1, `gap at second boundary: ${gap2} should be < 0.1`);
});

test('Heading is approximately continuous at segment boundaries', () => {
    const path = new CarPath();
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });

    // At the boundary (d = 50), the straight heading (East = 0) should match
    // the turn entry heading (also East = 0)
    const hBefore = path.getHeading(50 - 0.001);
    const hAfter = path.getHeading(50 + 0.001);
    const hDiff = Math.abs(hAfter - hBefore);
    assert.ok(hDiff < 0.05, `heading discontinuity at boundary: ${hDiff} should be < 0.05`);
});

test('Full U-turn path: East → South → West', () => {
    const path = new CarPath();
    // East 50m
    path.addStraight({ x: 0, z: 0 }, { x: 50, z: 0 });
    // Turn East → South
    path.addTurn({ x: 50, z: 0 }, { x: 60, z: 0 }, { x: 60, z: 10 });
    // South 30m
    path.addStraight({ x: 60, z: 10 }, { x: 60, z: 40 });
    // Turn South → West
    path.addTurn({ x: 60, z: 40 }, { x: 60, z: 50 }, { x: 50, z: 50 });
    // West 50m
    path.addStraight({ x: 50, z: 50 }, { x: 0, z: 50 });

    assert.strictEqual(path.segments.length, 5);

    // Car at start: heading East
    const hStart = path.getHeading(0);
    assert.ok(Math.abs(hStart) < 0.1, `start heading should be East (0), got ${hStart}`);

    // Car at end: heading West
    const hEnd = path.getHeading(path.totalLength - 1);
    assert.ok(Math.abs(Math.abs(hEnd) - Math.PI) < 0.1,
        `end heading should be West (PI), got ${hEnd}`);
});

// ── Summary ───────────────────────────────────────────────────────

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
