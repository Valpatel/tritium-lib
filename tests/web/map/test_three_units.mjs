// Tests for tritium-lib/web/map/three-units/
// Run: node tests/web/map/test_three_units.mjs

import { Base3DUnit } from '../../../web/map/three-units/base.js';
import { TurretModel } from '../../../web/map/three-units/turret.js';
import { DroneModel } from '../../../web/map/three-units/drone.js';
import { RoverModel } from '../../../web/map/three-units/rover.js';
import { PersonModel } from '../../../web/map/three-units/person.js';
import { TankModel } from '../../../web/map/three-units/tank.js';
import { registerModel, getModel, allModelTypes } from '../../../web/map/three-units/index.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Minimal THREE mock
const THREE = {
    Group: class { constructor() { this.children = []; this.position = {x:0,y:0,z:0,set(x,y,z){this.x=x;this.y=y;this.z=z}}; this.rotation = {x:0,y:0,z:0}; this.scale = {set(){}}; } add(c) { this.children.push(c); } traverse(fn) { fn(this); this.children.forEach(c => { fn(c); if (c.traverse) c.traverse(fn); }); } remove() {} },
    Mesh: class { constructor(g, m) { this.geometry = g; this.material = m; this.position = {x:0,y:0,z:0,set(x,y,z){this.x=x;this.y=y;this.z=z}}; this.rotation = {x:0,y:0,z:0}; this.scale = {set(){}}; } lookAt() {} },
    BoxGeometry: class { dispose() {} },
    SphereGeometry: class { dispose() {} },
    CylinderGeometry: class { dispose() {} },
    RingGeometry: class { dispose() {} },
    MeshLambertMaterial: class { dispose() {} },
    MeshBasicMaterial: class { dispose() {} },
    BufferGeometry: class { setAttribute() {} dispose() {} },
    BufferAttribute: class {},
    Line: class { constructor() { this.geometry = { attributes: { position: { needsUpdate: false } }, dispose() {} }; this.material = { dispose() {} }; } },
    LineBasicMaterial: class { dispose() {} },
    DoubleSide: 2,
};

console.log('=== 3D Unit Model Tests ===\n');

// Base
test('Base3DUnit has typeId generic', () => {
    assert.strictEqual(Base3DUnit.typeId, 'generic');
});

test('Base3DUnit.build returns group', () => {
    const u = new Base3DUnit();
    const g = u.build(THREE);
    assert(g);
    assert(g.children.length > 0);
    assert.strictEqual(u.group, g);
});

test('Base3DUnit.dispose marks disposed', () => {
    const u = new Base3DUnit();
    u.build(THREE);
    u.dispose();
    assert.strictEqual(u.disposed, true);
});

// Each model type
const models = [
    { Cls: TurretModel, id: 'turret', minParts: 3 },
    { Cls: DroneModel, id: 'drone', minParts: 4 },
    { Cls: RoverModel, id: 'rover', minParts: 3 },
    { Cls: PersonModel, id: 'person', minParts: 1 },
    { Cls: TankModel, id: 'tank', minParts: 3 },
];

for (const { Cls, id, minParts } of models) {
    test(`${id}: has correct typeId`, () => {
        assert.strictEqual(Cls.typeId, id);
    });

    test(`${id}: build creates mesh with ${minParts}+ parts`, () => {
        const u = new Cls();
        const g = u.build(THREE);
        assert(g.children.length >= minParts, `Expected ${minParts}+ children, got ${g.children.length}`);
    });

    test(`${id}: dispose works`, () => {
        const u = new Cls();
        u.build(THREE);
        u.dispose();
        assert.strictEqual(u.disposed, true);
    });
}

// Drone animation
test('drone: animate spins rotors', () => {
    const u = new DroneModel();
    u.build(THREE);
    const before = u._rotors.map(r => r.rotation.y);
    u.animate(0.016);
    const after = u._rotors.map(r => r.rotation.y);
    assert(after.some((v, i) => v !== before[i]), 'rotors should have moved');
});

// Registry
test('registry has 5 built-in models', () => {
    assert.strictEqual(allModelTypes().length, 5);
});

test('registry.getModel returns correct class', () => {
    assert.strictEqual(getModel('turret'), TurretModel);
    assert.strictEqual(getModel('drone'), DroneModel);
});

test('custom model can be registered', () => {
    class GraphlingModel extends Base3DUnit { static typeId = 'graphling'; }
    registerModel(GraphlingModel);
    assert.strictEqual(getModel('graphling'), GraphlingModel);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
