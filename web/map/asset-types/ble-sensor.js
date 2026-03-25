// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { BaseAssetType } from './base.js';

export class BLESensorAssetType extends BaseAssetType {
    static typeId = 'ble_sensor';
    static label = 'BLE Scanner';
    static icon = 'S';
    static color = '#05ffa1';
    static defaultRange = 15;
    static coverageShape = 'circle';
    static defaultFov = 360;
    static defaultHeight = 2.5;
    static defaultMounting = 'wall';
    static assetClass = 'sensor';
    static defaultCapabilities = ['ble_scan', 'wifi_probe'];
}
