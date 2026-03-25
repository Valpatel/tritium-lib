// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { BaseAssetType } from './base.js';

export class MeshRadioAssetType extends BaseAssetType {
    static typeId = 'mesh_radio';
    static label = 'Mesh Radio';
    static icon = 'R';
    static color = '#fcee0a';
    static defaultRange = 500;
    static coverageShape = 'circle';
    static defaultFov = 360;
    static defaultHeight = 12.0;
    static defaultMounting = 'pole';
    static assetClass = 'relay';
    static defaultCapabilities = ['meshtastic', 'gps'];
}
