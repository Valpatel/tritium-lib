// Tests for tritium-lib/web/map/asset-types/
// Run: node tests/web/map/test_asset_types.mjs

import { BaseAssetType } from '../../../web/map/asset-types/base.js';
import { CameraAssetType } from '../../../web/map/asset-types/camera.js';
import { BLESensorAssetType } from '../../../web/map/asset-types/ble-sensor.js';
import { MotionSensorAssetType } from '../../../web/map/asset-types/motion-sensor.js';
import { MeshRadioAssetType } from '../../../web/map/asset-types/mesh-radio.js';
import { assetTypeRegistry } from '../../../web/map/asset-types/registry.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== Asset Type Tests ===');

// BaseAssetType
test('BaseAssetType has typeId generic', () => {
    assert.strictEqual(BaseAssetType.typeId, 'generic');
});

test('BaseAssetType.getDefaults returns object with expected keys', () => {
    const d = BaseAssetType.getDefaults();
    assert.strictEqual(d.asset_type, 'fixed');
    assert(d.coverage_radius_meters > 0);
    assert(Array.isArray(d.capabilities));
});

test('BaseAssetType.getPopupHtml returns HTML string', () => {
    const html = BaseAssetType.getPopupHtml({ name: 'Test', coverage_radius_meters: 10 });
    assert(html.includes('Test'));
    assert(html.includes('10'));
});

// Camera
test('CameraAssetType is directional cone', () => {
    assert.strictEqual(CameraAssetType.typeId, 'camera');
    assert.strictEqual(CameraAssetType.coverageShape, 'cone');
    assert.strictEqual(CameraAssetType.defaultFov, 90);
    assert.strictEqual(CameraAssetType.color, '#00f0ff');
});

test('Camera defaults include video capability', () => {
    const d = CameraAssetType.getDefaults();
    assert(d.capabilities.includes('video'));
    assert.strictEqual(d.coverage_cone_angle, 90);
});

// BLE Sensor
test('BLESensorAssetType is omni circle', () => {
    assert.strictEqual(BLESensorAssetType.typeId, 'ble_sensor');
    assert.strictEqual(BLESensorAssetType.coverageShape, 'circle');
    assert.strictEqual(BLESensorAssetType.defaultRange, 15);
});

// Motion Sensor
test('MotionSensorAssetType is directional cone', () => {
    assert.strictEqual(MotionSensorAssetType.typeId, 'motion_sensor');
    assert.strictEqual(MotionSensorAssetType.coverageShape, 'cone');
    assert.strictEqual(MotionSensorAssetType.defaultFov, 110);
});

// Mesh Radio
test('MeshRadioAssetType has 500m range', () => {
    assert.strictEqual(MeshRadioAssetType.typeId, 'mesh_radio');
    assert.strictEqual(MeshRadioAssetType.defaultRange, 500);
    assert.strictEqual(MeshRadioAssetType.coverageShape, 'circle');
});

// Registry
test('registry has 4 built-in types', () => {
    assert.strictEqual(assetTypeRegistry.size, 4);
});

test('registry.get returns correct type', () => {
    const cam = assetTypeRegistry.get('camera');
    assert.strictEqual(cam, CameraAssetType);
});

test('registry.all returns all types', () => {
    const all = assetTypeRegistry.all();
    assert.strictEqual(all.length, 4);
    const ids = all.map(t => t.typeId).sort();
    assert.deepStrictEqual(ids, ['ble_sensor', 'camera', 'mesh_radio', 'motion_sensor']);
});

test('registry.typeIds returns string array', () => {
    const ids = assetTypeRegistry.typeIds();
    assert(ids.includes('camera'));
    assert(ids.includes('ble_sensor'));
});

// Custom type registration
test('custom type can be registered', () => {
    class LidarType extends BaseAssetType {
        static typeId = 'lidar';
        static label = 'LIDAR';
        static color = '#aa44ff';
        static defaultRange = 50;
    }
    const before = assetTypeRegistry.size;
    assetTypeRegistry.register(LidarType);
    assert.strictEqual(assetTypeRegistry.size, before + 1);
    assert.strictEqual(assetTypeRegistry.get('lidar'), LidarType);
});

test('resolveForAsset matches by asset_class', () => {
    const T = assetTypeRegistry.resolveForAsset({ asset_class: 'observation' });
    assert.strictEqual(T, CameraAssetType);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
