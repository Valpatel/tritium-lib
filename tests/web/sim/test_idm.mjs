// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { idmAcceleration, idmFreeFlow, idmStep, IDM_DEFAULTS } from '../../../web/sim/idm.js';
import assert from 'node:assert';
import { describe, it } from 'node:test';

describe('IDM', () => {
    it('free flow accelerates toward desired speed', () => {
        const acc = idmFreeFlow(0, IDM_DEFAULTS);
        assert.ok(acc > 0, `Expected positive acceleration, got ${acc}`);
        assert.ok(acc <= IDM_DEFAULTS.a, `Should not exceed max acceleration`);
    });

    it('free flow decelerates when above desired speed', () => {
        const acc = idmFreeFlow(IDM_DEFAULTS.v0 * 1.5, IDM_DEFAULTS);
        assert.ok(acc < 0, `Expected negative acceleration when above v0`);
    });

    it('free flow gives zero acceleration at desired speed', () => {
        const acc = idmFreeFlow(IDM_DEFAULTS.v0, IDM_DEFAULTS);
        assert.ok(Math.abs(acc) < 0.01, `Expected ~0 at desired speed, got ${acc}`);
    });

    it('brakes when close to leader', () => {
        const acc = idmAcceleration(10, 3, 0, IDM_DEFAULTS);
        assert.ok(acc < 0, `Expected braking when close to stopped leader, got ${acc}`);
    });

    it('accelerates when far from leader', () => {
        const acc = idmAcceleration(5, 100, 12, IDM_DEFAULTS);
        assert.ok(acc > 0, `Expected acceleration with large gap, got ${acc}`);
    });

    it('idmStep advances position', () => {
        const { v, ds } = idmStep(10, 1, 0.1);
        assert.ok(v > 10, `Speed should increase with positive acceleration`);
        assert.ok(ds > 0, `Distance should be positive`);
    });

    it('idmStep clamps speed to zero', () => {
        const { v } = idmStep(0.5, -10, 0.1);
        assert.strictEqual(v, 0, `Speed should not go negative`);
    });

    it('acceleration is clamped to physical limits', () => {
        // Very close to stopped leader at high speed — should brake hard but not exceed -9
        const acc = idmAcceleration(30, 0.5, 0, IDM_DEFAULTS);
        assert.ok(acc >= -9.0, `Should not exceed -9 m/s^2 braking, got ${acc}`);
    });
});
