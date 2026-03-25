// Tests for tritium-lib/web/map/effects/
// Run: node tests/web/map/test_effects.mjs

import { CombatEffects, DEFAULT_WEAPON_VFX } from '../../../web/map/effects/base.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== Combat Effects Tests ===');

test('CombatEffects constructor with defaults', () => {
    const fx = new CombatEffects(null);
    assert.strictEqual(fx.showTracers, true);
    assert.strictEqual(fx.showExplosions, true);
    assert.strictEqual(fx.activeCount, 0);
});

test('CombatEffects constructor with options', () => {
    const fx = new CombatEffects(null, { showTracers: false, showExplosions: false });
    assert.strictEqual(fx.showTracers, false);
    assert.strictEqual(fx.showExplosions, false);
});

test('DEFAULT_WEAPON_VFX has expected presets', () => {
    assert(DEFAULT_WEAPON_VFX.bullet);
    assert(DEFAULT_WEAPON_VFX.missile);
    assert(DEFAULT_WEAPON_VFX.laser);
    assert.strictEqual(typeof DEFAULT_WEAPON_VFX.bullet.color, 'number');
    assert(DEFAULT_WEAPON_VFX.bullet.speed > 0);
});

test('getWeaponVFX returns preset', () => {
    const fx = new CombatEffects(null);
    const vfx = fx.getWeaponVFX('missile');
    assert.strictEqual(vfx.color, 0xff2200);
    assert(vfx.explosionSize > 0);
});

test('getWeaponVFX falls back to bullet for unknown', () => {
    const fx = new CombatEffects(null);
    const vfx = fx.getWeaponVFX('railgun');
    assert.strictEqual(vfx, DEFAULT_WEAPON_VFX.bullet);
});

test('setWeaponPresets overrides', () => {
    const fx = new CombatEffects(null);
    fx.setWeaponPresets({ railgun: { color: 0x0000ff, speed: 500 } });
    const vfx = fx.getWeaponVFX('railgun');
    assert.strictEqual(vfx.color, 0x0000ff);
});

test('addEffect and animate lifecycle', () => {
    const fx = new CombatEffects(null);
    const now = performance.now();
    fx.addEffect({ alive: true, endTime: now + 100, update() {} });
    assert.strictEqual(fx.activeCount, 1);
    fx.animate(now + 200); // past endTime — marks dead
    fx.animate(now + 201); // second pass removes dead effects
    assert.strictEqual(fx.activeCount, 0);
});

test('clearAll removes all effects', () => {
    const fx = new CombatEffects(null);
    fx.addEffect({ alive: true, update() {} });
    fx.addEffect({ alive: true, update() {} });
    assert.strictEqual(fx.activeCount, 2);
    fx.clearAll();
    assert.strictEqual(fx.activeCount, 0);
});

test('destroy cleans up', () => {
    const fx = new CombatEffects(null);
    fx.addEffect({ alive: true, update() {} });
    fx.destroy();
    assert.strictEqual(fx.activeCount, 0);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
