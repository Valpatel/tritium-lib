// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * 3D unit models — each in its own file. Register custom models via the registry.
 */

export { Base3DUnit } from './base.js';
export { TurretModel } from './turret.js';
export { DroneModel } from './drone.js';
export { RoverModel } from './rover.js';
export { PersonModel } from './person.js';
export { TankModel } from './tank.js';

import { TurretModel } from './turret.js';
import { DroneModel } from './drone.js';
import { RoverModel } from './rover.js';
import { PersonModel } from './person.js';
import { TankModel } from './tank.js';

/** Registry of 3D model classes by typeId. */
const modelRegistry = new Map();

export function registerModel(ModelClass) {
    modelRegistry.set(ModelClass.typeId, ModelClass);
}

export function getModel(typeId) {
    return modelRegistry.get(typeId);
}

export function allModelTypes() {
    return [...modelRegistry.keys()];
}

// Register built-in models
registerModel(TurretModel);
registerModel(DroneModel);
registerModel(RoverModel);
registerModel(PersonModel);
registerModel(TankModel);
