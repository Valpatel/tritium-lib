// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { BaseAssetType } from './base.js';

export class MotionSensorAssetType extends BaseAssetType {
    static typeId = 'motion_sensor';
    static label = 'Motion Detector';
    static icon = 'M';
    static color = '#ff8800';
    static defaultRange = 8;
    static coverageShape = 'cone';
    static defaultFov = 110;
    static defaultHeight = 2.0;
    static defaultMounting = 'wall';
    static assetClass = 'sensor';
    static defaultCapabilities = ['pir_motion', 'alert'];
}
