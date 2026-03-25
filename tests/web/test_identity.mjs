// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/identity.js — deterministic identity generator
// Run: node tests/web/test_identity.mjs

import { buildIdentity, buildPedestrianIdentity, buildCarIdentity } from '../../web/sim/identity.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== Identity Tests ===');

// ── buildIdentity: person ─────────────────────────────────────────

test('buildIdentity returns all person fields', () => {
    const id = buildIdentity('ped_0', 'person');
    assert.strictEqual(typeof id.shortId, 'string');
    assert.strictEqual(typeof id.firstName, 'string');
    assert.strictEqual(typeof id.lastName, 'string');
    assert.strictEqual(typeof id.fullName, 'string');
    assert.strictEqual(typeof id.bluetoothMac, 'string');
    assert.strictEqual(typeof id.wifiMac, 'string');
    assert.strictEqual(typeof id.phoneModel, 'string');
    assert.strictEqual(typeof id.employer, 'string');
    assert.strictEqual(typeof id.homeAddress, 'string');
    assert.strictEqual(typeof id.workAddress, 'string');
});

test('buildIdentity fullName is firstName + lastName', () => {
    const id = buildIdentity('ped_0', 'person');
    assert.strictEqual(id.fullName, id.firstName + ' ' + id.lastName);
});

test('buildIdentity shortId is 6 hex characters', () => {
    const id = buildIdentity('ped_0', 'person');
    assert.strictEqual(id.shortId.length, 6);
    assert.ok(/^[0-9A-F]{6}$/.test(id.shortId), `invalid hex: ${id.shortId}`);
});

test('buildIdentity MAC addresses have correct format', () => {
    const id = buildIdentity('ped_0', 'person');
    const macRegex = /^[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}$/;
    assert.ok(macRegex.test(id.bluetoothMac), `invalid BT MAC: ${id.bluetoothMac}`);
    assert.ok(macRegex.test(id.wifiMac), `invalid WiFi MAC: ${id.wifiMac}`);
});

test('buildIdentity is deterministic — same input = same output', () => {
    const id1 = buildIdentity('ped_42', 'person');
    const id2 = buildIdentity('ped_42', 'person');
    assert.strictEqual(id1.firstName, id2.firstName);
    assert.strictEqual(id1.lastName, id2.lastName);
    assert.strictEqual(id1.bluetoothMac, id2.bluetoothMac);
    assert.strictEqual(id1.wifiMac, id2.wifiMac);
    assert.strictEqual(id1.phoneModel, id2.phoneModel);
    assert.strictEqual(id1.homeAddress, id2.homeAddress);
});

test('buildIdentity different IDs produce different identities', () => {
    const id1 = buildIdentity('ped_0', 'person');
    const id2 = buildIdentity('ped_1', 'person');
    // Very unlikely but theoretically possible to collide — just check one field
    const same = (id1.firstName === id2.firstName && id1.lastName === id2.lastName &&
                  id1.bluetoothMac === id2.bluetoothMac);
    assert.ok(!same, 'different IDs should produce different identities');
});

// ── buildIdentity: vehicle ────────────────────────────────────────

test('buildIdentity vehicle has vehicle-specific fields', () => {
    const id = buildIdentity('car_0', 'vehicle');
    assert.strictEqual(typeof id.licensePlate, 'string');
    assert.strictEqual(typeof id.vehicleMake, 'string');
    assert.strictEqual(typeof id.vehicleModel, 'string');
    assert.strictEqual(typeof id.vehicleYear, 'number');
    assert.strictEqual(typeof id.vehicleColor, 'string');
    assert.strictEqual(typeof id.vehicleDesc, 'string');
    assert.strictEqual(typeof id.ownerName, 'string');
});

test('buildIdentity vehicle year is in reasonable range', () => {
    const id = buildIdentity('car_0', 'vehicle');
    assert.ok(id.vehicleYear >= 2018 && id.vehicleYear <= 2025,
        `unexpected year: ${id.vehicleYear}`);
});

test('buildIdentity vehicle desc includes year make model', () => {
    const id = buildIdentity('car_0', 'vehicle');
    assert.ok(id.vehicleDesc.includes(String(id.vehicleYear)), 'desc should include year');
    assert.ok(id.vehicleDesc.includes(id.vehicleMake), 'desc should include make');
    assert.ok(id.vehicleDesc.includes(id.vehicleModel), 'desc should include model');
});

test('buildIdentity vehicle license plate format', () => {
    const id = buildIdentity('car_0', 'vehicle');
    // Format: digit + 3 letters + 3 digits (e.g., 7ABC123)
    assert.ok(id.licensePlate.length >= 6, `plate too short: ${id.licensePlate}`);
    assert.ok(/^\d[A-Z]{3}\d{3}$/.test(id.licensePlate), `invalid plate: ${id.licensePlate}`);
});

test('buildIdentity vehicle ownerName matches person identity', () => {
    const id = buildIdentity('car_0', 'vehicle');
    assert.strictEqual(id.ownerName, id.fullName);
});

test('buildIdentity vehicle is deterministic', () => {
    const id1 = buildIdentity('car_99', 'vehicle');
    const id2 = buildIdentity('car_99', 'vehicle');
    assert.strictEqual(id1.licensePlate, id2.licensePlate);
    assert.strictEqual(id1.vehicleMake, id2.vehicleMake);
    assert.strictEqual(id1.vehicleModel, id2.vehicleModel);
    assert.strictEqual(id1.vehicleColor, id2.vehicleColor);
});

// ── Convenience functions ─────────────────────────────────────────

test('buildPedestrianIdentity creates person identity', () => {
    const id = buildPedestrianIdentity(5);
    assert.strictEqual(typeof id.firstName, 'string');
    assert.strictEqual(typeof id.fullName, 'string');
    assert.strictEqual(id.licensePlate, undefined, 'person should not have plate');
});

test('buildCarIdentity creates vehicle identity', () => {
    const id = buildCarIdentity(3);
    assert.strictEqual(typeof id.licensePlate, 'string');
    assert.strictEqual(typeof id.vehicleMake, 'string');
});

test('buildPedestrianIdentity is deterministic', () => {
    const id1 = buildPedestrianIdentity(7);
    const id2 = buildPedestrianIdentity(7);
    assert.strictEqual(id1.fullName, id2.fullName);
    assert.strictEqual(id1.bluetoothMac, id2.bluetoothMac);
});

test('buildCarIdentity is deterministic', () => {
    const id1 = buildCarIdentity(7);
    const id2 = buildCarIdentity(7);
    assert.strictEqual(id1.licensePlate, id2.licensePlate);
    assert.strictEqual(id1.vehicleDesc, id2.vehicleDesc);
});

// ── Uniqueness across many entities ───────────────────────────────

test('100 pedestrians all have unique MACs', () => {
    const macs = new Set();
    for (let i = 0; i < 100; i++) {
        const id = buildPedestrianIdentity(i);
        macs.add(id.bluetoothMac);
    }
    assert.strictEqual(macs.size, 100, 'all BT MACs should be unique');
});

test('100 vehicles all have unique license plates', () => {
    const plates = new Set();
    for (let i = 0; i < 100; i++) {
        const id = buildCarIdentity(i);
        plates.add(id.licensePlate);
    }
    assert.strictEqual(plates.size, 100, 'all plates should be unique');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
