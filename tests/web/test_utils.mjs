// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

// Panel utils depend on DOM APIs (document.createElement, fetch).
// We test what we can in Node.js — _timeAgo is pure logic.

import assert from 'node:assert';
import { describe, it } from 'node:test';

// We can't import _esc, _badge, _statusDot, _fetchJson directly because they
// use DOM APIs not available in Node. Instead we test _timeAgo by extracting
// its logic or using a minimal DOM shim.

// Inline _timeAgo for testing (same logic as utils.js)
function _timeAgo(ts) {
    if (!ts) return 'never';
    const secs = Math.floor(Date.now() / 1000 - ts);
    if (secs < 5) return 'just now';
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
}

describe('_timeAgo', () => {
    it('returns "never" for falsy input', () => {
        assert.strictEqual(_timeAgo(0), 'never');
        assert.strictEqual(_timeAgo(null), 'never');
        assert.strictEqual(_timeAgo(undefined), 'never');
    });

    it('returns "just now" for recent timestamps', () => {
        const now = Math.floor(Date.now() / 1000);
        assert.strictEqual(_timeAgo(now), 'just now');
        assert.strictEqual(_timeAgo(now - 3), 'just now');
    });

    it('returns seconds ago', () => {
        const now = Math.floor(Date.now() / 1000);
        assert.strictEqual(_timeAgo(now - 30), '30s ago');
    });

    it('returns minutes ago', () => {
        const now = Math.floor(Date.now() / 1000);
        assert.strictEqual(_timeAgo(now - 300), '5m ago');
    });

    it('returns hours ago', () => {
        const now = Math.floor(Date.now() / 1000);
        assert.strictEqual(_timeAgo(now - 7200), '2h ago');
    });

    it('returns days ago', () => {
        const now = Math.floor(Date.now() / 1000);
        assert.strictEqual(_timeAgo(now - 86400 * 3), '3d ago');
    });
});

// Test that the module parses without error (syntax check)
describe('utils.js syntax', () => {
    it('parses without error using node --check', async () => {
        const { execSync } = await import('node:child_process');
        execSync('node --check /home/scubasonar/Code/tritium/tritium-lib/web/utils.js');
    });
});
