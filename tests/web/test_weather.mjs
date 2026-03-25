// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/weather.js — CityWeather
// Run: node tests/web/test_weather.mjs

import { CityWeather } from '../../web/sim/weather.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== CityWeather Tests ===');

// ── Constructor ───────────────────────────────────────────────────

test('constructor sets sensible defaults', () => {
    const w = new CityWeather();
    assert.strictEqual(w.hour, 7);
    assert.strictEqual(w.weather, 'clear');
    assert.strictEqual(w.isNight, false);
    assert.strictEqual(typeof w.skyColor, 'number');
    assert.strictEqual(w.speedMultiplier, 1.0);
    assert.strictEqual(w.headwayMultiplier, 1.0);
});

// ── Time of day classification ────────────────────────────────────

test('update: midnight is night', () => {
    const w = new CityWeather();
    w.update(0, 0.1);
    assert.strictEqual(w.isNight, true);
    assert.strictEqual(w.isDusk, false);
    assert.strictEqual(w.isDawn, false);
});

test('update: 3am is night', () => {
    const w = new CityWeather();
    w.update(3, 0.1);
    assert.strictEqual(w.isNight, true);
});

test('update: 6am is dawn', () => {
    const w = new CityWeather();
    w.update(6, 0.1);
    assert.strictEqual(w.isDawn, true);
    assert.strictEqual(w.isNight, false);
});

test('update: noon is day (not dawn/dusk/night)', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    assert.strictEqual(w.isNight, false);
    assert.strictEqual(w.isDusk, false);
    assert.strictEqual(w.isDawn, false);
});

test('update: 6pm is dusk', () => {
    const w = new CityWeather();
    w.update(18, 0.1);
    assert.strictEqual(w.isDusk, true);
    assert.strictEqual(w.isNight, false);
});

test('update: 22h is night', () => {
    const w = new CityWeather();
    w.update(22, 0.1);
    assert.strictEqual(w.isNight, true);
});

// ── Lighting and effects ──────────────────────────────────────────

test('night has lower ambient intensity than day', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    const dayAmbient = w.ambientIntensity;
    w.update(2, 0.1);
    const nightAmbient = w.ambientIntensity;
    assert.ok(nightAmbient < dayAmbient, `night ${nightAmbient} should be < day ${dayAmbient}`);
});

test('night has higher window emissive than day', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    const dayEmissive = w.windowEmissive;
    w.update(2, 0.1);
    const nightEmissive = w.windowEmissive;
    assert.ok(nightEmissive > dayEmissive, `night emissive ${nightEmissive} > day ${dayEmissive}`);
});

test('headlights on at night', () => {
    const w = new CityWeather();
    w.update(2, 0.1);
    assert.strictEqual(w.headlightsOn, true);
});

test('headlights off during clear day', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    assert.strictEqual(w.headlightsOn, false);
});

test('streetLights on at night', () => {
    const w = new CityWeather();
    w.update(22, 0.1);
    assert.strictEqual(w.streetLightsOn, true);
});

test('streetLights off during day', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    assert.strictEqual(w.streetLightsOn, false);
});

// ── Weather effects on driving ────────────────────────────────────

test('rain reduces speed multiplier', () => {
    const w = new CityWeather();
    w.weather = 'rain';
    w.update(12, 0.1);
    assert.ok(w.speedMultiplier < 1.0, `rain speed mult should be < 1: ${w.speedMultiplier}`);
    assert.ok(w.headwayMultiplier > 1.0, `rain headway mult should be > 1: ${w.headwayMultiplier}`);
});

test('fog reduces speed more than rain', () => {
    const w = new CityWeather();
    w.weather = 'fog';
    w.update(12, 0.1);
    const fogSpeed = w.speedMultiplier;

    w.weather = 'rain';
    w.update(12, 0.1);
    const rainSpeed = w.speedMultiplier;

    assert.ok(fogSpeed < rainSpeed, `fog speed ${fogSpeed} should be < rain speed ${rainSpeed}`);
});

test('clear weather has no speed reduction', () => {
    const w = new CityWeather();
    w.weather = 'clear';
    w.update(12, 0.1);
    assert.strictEqual(w.speedMultiplier, 1.0);
    assert.strictEqual(w.headwayMultiplier, 1.0);
});

// ── Fog density ───────────────────────────────────────────────────

test('fog weather has highest fog density', () => {
    const w = new CityWeather();
    w.weather = 'fog';
    w.update(12, 0.1);
    const fogDensity = w.fogDensity;

    w.weather = 'rain';
    w.update(12, 0.1);
    const rainFog = w.fogDensity;

    w.weather = 'clear';
    w.update(12, 0.1);
    const clearFog = w.fogDensity;

    assert.ok(fogDensity > rainFog, 'fog density > rain density');
    assert.ok(rainFog > clearFog, 'rain density > clear density');
});

// ── Sky color ─────────────────────────────────────────────────────

test('skyColor is a valid hex integer', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    assert.ok(w.skyColor >= 0 && w.skyColor <= 0xFFFFFF, `invalid sky color: ${w.skyColor}`);
});

test('night sky is darker than day sky', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    const dayColor = w.skyColor;
    w.update(2, 0.1);
    const nightColor = w.skyColor;
    // Sum of RGB channels as brightness proxy
    const dayBright = ((dayColor >> 16) & 0xFF) + ((dayColor >> 8) & 0xFF) + (dayColor & 0xFF);
    const nightBright = ((nightColor >> 16) & 0xFF) + ((nightColor >> 8) & 0xFF) + (nightColor & 0xFF);
    assert.ok(nightBright < dayBright, `night brightness ${nightBright} should be < day ${dayBright}`);
});

// ── toString ──────────────────────────────────────────────────────

test('toString includes time and weather', () => {
    const w = new CityWeather();
    w.update(14.5, 0.1);
    const str = w.toString();
    assert.ok(str.includes('14'), `should include hour: ${str}`);
    assert.ok(str.includes('30'), `should include minutes: ${str}`);
    assert.ok(str.includes('DAY'), `should include period: ${str}`);
    assert.ok(str.includes('CLEAR'), `should include weather: ${str}`);
});

test('toString shows NIGHT at midnight', () => {
    const w = new CityWeather();
    w.update(0, 0.1);
    const str = w.toString();
    assert.ok(str.includes('NIGHT'), `expected NIGHT: ${str}`);
});

// ── applyToScene (mock) ───────────────────────────────────────────

test('applyToScene applies values to mock scene objects', () => {
    const w = new CityWeather();
    w.update(12, 0.1);

    let bgHex = null;
    const scene = {
        background: { setHex(v) { bgHex = v; } },
        fog: { density: 0 },
    };
    const ambient = { intensity: 0 };
    const sun = { intensity: 0 };

    w.applyToScene(scene, ambient, sun);
    assert.strictEqual(bgHex, w.skyColor);
    assert.strictEqual(scene.fog.density, w.fogDensity);
    assert.strictEqual(ambient.intensity, w.ambientIntensity);
    assert.strictEqual(sun.intensity, w.sunIntensity);
});

test('applyToScene handles null arguments gracefully', () => {
    const w = new CityWeather();
    w.update(12, 0.1);
    // Should not throw with null/undefined scene components
    w.applyToScene({}, null, null);
    w.applyToScene({ background: null, fog: null }, null, null);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
