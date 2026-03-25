// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

// City simulation modules — re-exported for convenience.

export { IDM_DEFAULTS, ROAD_SPEEDS, idmAcceleration, idmFreeFlow, idmStep } from './idm.js';
export { MOBIL_DEFAULTS, findNeighborsInLane, evaluateLaneChange, decideLaneChange } from './mobil.js';
export { SpatialGrid } from './spatial-grid.js';
export { RoadNetwork } from './road-network.js';
export { SimVehicle } from './vehicle.js';
export { SimPedestrian, PED_ACTIVITY, PED_COLORS } from './pedestrian.js';
export { TrafficController, TrafficControllerManager } from './traffic-controller.js';
export { ProtestEngine } from './protest-engine.js';
export { PHASES } from './protest-scenario.js';
export { generateDailyRoutine, getCurrentGoal, getNextGoal, randomRole } from './daily-routine.js';
export { SimClock, ScheduleExecutor } from './schedule-executor.js';
export { CityWeather } from './weather.js';
export { buildIdentity, buildPedestrianIdentity, buildCarIdentity } from './identity.js';
export { generateProceduralCity } from './procedural-city.js';
