// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { BaseAssetType } from './base.js';

export class CameraAssetType extends BaseAssetType {
    static typeId = 'camera';
    static label = 'Camera';
    static icon = 'C';
    static color = '#00f0ff';
    static defaultRange = 30;
    static coverageShape = 'cone';
    static defaultFov = 90;
    static defaultHeight = 4.0;
    static defaultMounting = 'pole';
    static assetClass = 'observation';
    static defaultCapabilities = ['video', 'yolo', 'recording'];
}
