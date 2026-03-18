/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Shared configuration, constants, and mutable state.
*/

import * as THREE from 'three';

// =========================================================================
// Color Constants
// =========================================================================
export const CYAN    = 0x00f0ff;
export const MAGENTA = 0xff2a6d;
export const GREEN   = 0x05ffa1;
export const YELLOW  = 0xfcee0a;
export const VOID_BG = 0x0a0a0f;

// =========================================================================
// City Grid Constants
// =========================================================================
export const BLOCK_W = 60, BLOCK_H = 50;
export const GRID_COLS = 8, GRID_ROWS = 6;
export const ROAD_W = 16, SIDEWALK_W = 3;
export const CITY_W = GRID_COLS * (BLOCK_W + ROAD_W) + ROAD_W;
export const CITY_H = GRID_ROWS * (BLOCK_H + ROAD_W) + ROAD_W;

// =========================================================================
// Entity Counts
// =========================================================================
export const NUM_PEDESTRIANS = 25;
export const NUM_CARS = 12;
export const NUM_TAXIS = 3;
export const NUM_PROTESTORS = 20;
export const NUM_POLICE = 10;
export const NUM_FIRE_TRUCKS = 0;
export const NUM_POLICE_VANS = 0;

// =========================================================================
// Plaza
// =========================================================================
export const PLAZA_X = CITY_W / 2;
export const PLAZA_Z = CITY_H / 2;

// =========================================================================
// Building Colors
// =========================================================================
export const BLDG_HOUSE   = 0x8B6914;
export const BLDG_OFFICE  = 0x888888;
export const BLDG_SHOP    = 0xD2B48C;
export const BLDG_POLICE  = 0x4466AA;
export const BLDG_HOSPITAL = 0xEEEEEE;
export const BLDG_ROOF    = 0x333333;

// =========================================================================
// Person Dimensions
// =========================================================================
export const BODY_W = 1.0, BODY_H = 1.5, BODY_D = 0.6;
export const HEAD_SIZE = 0.7;
export const PERSON_SCALE = 0.9;

// =========================================================================
// Role Colors
// =========================================================================
export const COLOR_CIVILIAN  = 0x33AA55;
export const COLOR_PROTESTOR = 0xCC2222;
export const COLOR_POLICE    = 0x2244CC;

// =========================================================================
// Skin Tones
// =========================================================================
export const SKIN_TONES = [0xFFDBAC, 0xE5A073, 0xC68642, 0x8D5524, 0xF1C27D, 0xFFCD94];

// =========================================================================
// Supply & Economy Constants
// =========================================================================
export const SUPPLY_TEAR_GAS_MAX = 20;
export const SUPPLY_RUBBER_BULLETS_MAX = 100;
export const SUPPLY_MOLOTOVS_MAX = 15;
export const COST_TEAR_GAS = 500;
export const COST_RUBBER_BULLET = 50;
export const REWARD_ARREST = 200;
export const COST_FIRE = 1000;
export const COST_INJURY = 2000;

// =========================================================================
// Comms & EW Constants
// =========================================================================
export const COMMS_RANGE = 40;
export const EW_JAM_DURATION = 15;
export const EW_JAM_CHECK_INTERVAL = 30;
export const EW_JAM_CHANCE = 0.1;

// =========================================================================
// Detection / Fog of War Constants
// =========================================================================
export const FOG_CELL_SIZE = 20;
export const FOG_COLS = Math.ceil(CITY_W / FOG_CELL_SIZE);
export const FOG_ROWS = Math.ceil(CITY_H / FOG_CELL_SIZE);
export const DETECTION_RANGE = 40;

// =========================================================================
// Territory Control Constants
// =========================================================================
export const TERRITORY_COLS = 4;
export const TERRITORY_ROWS = 3;

// =========================================================================
// Traffic Constants
// =========================================================================
export const TRAFFIC_CYCLE = 8;
export const LANE_OFFSET = 3.0;

// =========================================================================
// InstancedMesh Limits
// =========================================================================
export const MAX_BUILDINGS = 300;
export const MAX_WINDOWS = 3000;
export const MAX_DOORS = 300;
export const MAX_TREES = 200;
export const MAX_LAMPS = 130;
export const MAX_LAMP_LIGHTS = 6;
export const MAX_PEOPLE = 180;
export const MAX_RINGS = 40;
export const MAX_SHIELDS = 15;
export const MAX_CARS_INST = 45;
export const MAX_PARTICLES = 500;
export const MAX_BARRICADES = 8;
export const MAX_TRACERS = 20;
export const MAX_FLASHES = 8;
export const MAX_PROJECTILES = 12;
export const MAX_INJURED_MARKERS = 30;
export const MAX_DETECTION_LINES = 200;
export const MAX_COMMS_LINKS = 50;
export const MAX_IEDS = 6;
export const RAIN_COUNT = 500;
export const MAX_OBJECTIVES = 3;

// =========================================================================
// Robot Constants
// =========================================================================
export const NUM_ROBOT_CARS = 3;
export const LIDAR_RAYS = 12;
export const LIDAR_RANGE = 15;
export const ROBOT_TRAIL_LENGTH = 20;

// =========================================================================
// Other Constants
// =========================================================================
export const CANAL_ROW = 2;
export const CANAL_Z = CANAL_ROW * (BLOCK_H + ROAD_W) + ROAD_W / 2;
export const CANAL_W = 12;
export const FOG_UPDATE_INTERVAL = 1.0;
export const TERRITORY_UPDATE_INTERVAL = 2.0;
export const COMPASS_UPDATE_INTERVAL = 0.1;
export const NARRATION_MAX_VISIBLE = 4;
export const NARRATION_FADE_TIME = 8;
export const ARC_ANGLE = Math.PI / 3;
export const ARC_RANGE = 30;
export const ARC_SEGMENTS = 4;
export const STRIPS_PER_CW = 6;
export const CW_PER_INT = 4;
export const REPLAY_MAX_FRAMES = 3000;
export const REPLAY_CAPTURE_STEP = 1 / 10;
export const SIM_STEP = 1 / 20;

// =========================================================================
// Campaign Constants
// =========================================================================
export const CAMPAIGN_PHASES = [null,
  {name:'Morning Protest',objective:'Monitor situation',duration:60,color:'phase1'},
  {name:'Riot Response',objective:'Contain riot',color:'phase2'},
  {name:'Aftermath',objective:'Secure area',color:'phase3'}];
export const CAMPAIGN_TITLE = 'City Crisis';

// =========================================================================
// Achievement Definitions
// =========================================================================
export const achievementDefs = {
  first_responder: { name: 'First Responder', desc: 'First ambulance dispatched', points: 100 },
  firebreaker: { name: 'Firebreaker', desc: 'Fire truck extinguishes first fire', points: 150 },
  peacekeeper: { name: 'Peacekeeper', desc: '5 arrests without injuries', points: 200 },
  iron_line: { name: 'Iron Line', desc: 'Police hold line for 60 seconds', points: 250 },
  crowd_control: { name: 'Crowd Control', desc: 'Tear gas disperses 10+ protestors', points: 175 },
  under_fire: { name: 'Under Fire', desc: 'Survive 5 molotov attacks', points: 300 },
};

// =========================================================================
// RNG
// =========================================================================
export const rng = () => Math.random();

// =========================================================================
// Shared Reusable THREE.js Objects (for matrix/color setting)
// =========================================================================
export const _mat4 = new THREE.Matrix4();
export const _pos = new THREE.Vector3();
export const _quat = new THREE.Quaternion();
export const _scale = new THREE.Vector3();
export const _color = new THREE.Color();
export const _euler = new THREE.Euler();
export const _skyA = new THREE.Color();
export const _skyB = new THREE.Color();

// =========================================================================
// Shared Mutable State
// =========================================================================
export const state = {
  // Core sim state
  riotMode: false,
  nightMode: false,
  chaseCam: false,
  chaseTarget: null,
  simTime: 8.0,
  riotPhase: 'PEACEFUL',
  riotTimer: 0,
  debugMode: false,
  splitViewMode: false,

  // Campaign
  campaignPhase: 0,
  campaignTimer: 0,
  campaignComplete: false,

  // Counters
  molotovCount: 0,
  teargasCount: 0,
  rockCount: 0,
  arrestCount: 0,
  injuryCount: 0,

  // Supply logistics
  supplyTearGas: SUPPLY_TEAR_GAS_MAX,
  supplyRubberBullets: SUPPLY_RUBBER_BULLETS_MAX,
  supplyMolotovs: SUPPLY_MOLOTOVS_MAX,

  // Economy
  policeBudget: 10000,
  propertyDamage: 0,

  // Status effects
  activeEffects: [],
  effectsCount: 0,

  // Comms
  commsStatus: 'ACTIVE',
  commsDegradedTimer: 0,
  commsLinkCount: 0,

  // EW
  ewJamActive: false,
  ewJamTimer: 0,
  ewJamX: 0,
  ewJamZ: 0,
  ewJamRadius: 40,
  ewJamCheckTimer: 0,
  ewCounterNarrated: false,

  // Weather
  rainActive: false,
  fogOverride: false,
  lightningCount: 0,
  lightningTimer: 0,

  // Building breach
  breachCooldown: 0,
  buildingsCleared: 0,

  // Artillery
  artilleryPhase: 'IDLE',
  artilleryTimer: 0,
  artilleryTriggered: false,
  artilleryTargets: [],
  artilleryCraters: [],
  artilleryMarkers: [],
  cameraShakeTimer: 0,
  cameraShakeOx: 0,
  cameraShakeOz: 0,
  artillerySmokeTimer: 0,

  // IED
  ieds: [],
  iedPlaceTimer: 0,
  iedPlanted: 0,
  iedDetected: 0,
  iedDetonated: 0,

  // Minimap & Compass
  minimapVisible: true,
  compassTimer: 0,

  // Fog of War
  fogOfWarEnabled: false,
  fogGrid: new Uint8Array(FOG_COLS * FOG_ROWS),
  fogTimer: 0,
  detectedCount: 0,

  // Territory
  territoryZones: [],
  territoryTimer: 0,

  // Objectives
  objectives: [],
  objectiveRotation: 0,
  holdGroundTimer: 0,

  // Soundtrack
  soundtrackState: 'PEACEFUL',

  // Achievements
  totalScore: 0,
  achievementsAwarded: {},
  policeLineHoldTime: 0,
  tearGasDispersedTotal: 0,

  // Traffic
  trafficPhase: 0,

  // Narration
  narrationMessages: [],

  // Kill feed
  killFeed: [],

  // Entity arrays
  pedestrians: [],
  cars: [],
  protestors: [],
  police: [],
  tracers: [],
  explosions: [],
  muzzleFlashes: [],
  buildings: [],
  streetLamps: [],
  fires: [],
  rubbleZones: [],
  projectiles: [],
  ambulances: [],
  injuredQueue: [],
  taxis: [],
  fireTrucks: [],
  policeVans: [],
  injuredOnGround: [],
  medics: [],
  carDrivers: [],
  barricades: [],
  robotCars: [],

  // Building damage
  buildingDamageTimer: 0,

  // Supply convoy
  supplyConvoy: null,
  supplyConvoyTimer: 0,

  // Robot state
  robotLidarAngle: 0,
  robotTotalHits: 0,
  robotIntentStates: ['MOVING', 'MOVING', 'MOVING'],

  // Person slot tracking
  personSlotCount: 0,

  // Car slot tracking
  nextCarSlot: 0,

  // Building/window/door tracking
  buildingIdx: 0,
  windowIdx: 0,
  doorIdx: 0,

  // Tree/lamp tracking
  treeIdx: 0,
  lampIdx: 0,

  // Ring tracking
  ringIdx: 0,

  // Particle pool
  particlePool: [],
  activeParticles: [],

  // Helicopter
  helicopter: null,

  // Hospital
  hospitalBuilding: null,

  // Formation
  formationStatus: 'LINE',

  // Road grid helpers
  hRoads: [],
  vRoads: [],
  intersections: [],

  // House buildings (for car drivers)
  houseBuildings: [],

  // All road vehicles (collision detection)
  _allRoadVehicles: [],

  // Rain drops
  rainDrops: [],

  // =========================================================================
  // Three.js Scene Objects (set during init)
  // =========================================================================
  scene: null,
  camera: null,
  renderer: null,
  controls: null,

  // Lights
  ambientLight: null,
  sunLight: null,
  hemiLight: null,

  // Materials
  roadMat: null,
  sidewalkMat: null,
  windowMat: null,
  rainMat: null,

  // InstancedMesh references
  buildingBodyMesh: null,
  buildingRoofMesh: null,
  windowMesh: null,
  doorIMesh: null,
  trunkMesh: null,
  leafMesh: null,
  lampPostMesh: null,
  lampHeadMesh: null,
  personBodyMesh: null,
  personHeadMesh: null,
  ringMesh: null,
  shieldMesh: null,
  carBodyMesh: null,
  carCabinMesh: null,
  carHLMesh: null,
  carTLMesh: null,
  particleMesh: null,
  barricadeMesh: null,
  rainMesh: null,
  trafficLightMesh: null,
  cwMesh: null,
  objectiveMesh: null,
  iedMesh: null,
  obstacleMarkerMesh: null,

  // Pooled lights
  lampLightPool: [],
  cityGlowLights: [],
  lampHalos: [],
  carHeadlights: [],
  tlLights: [],

  // Special meshes
  formationLine: null,
  formationGeo: null,
  formationPositions: null,
  formationMat: null,
  sectorArcLine: null,
  sectorArcGeo: null,
  sectorArcPositions: null,
  fogMesh: null,
  fogTexture: null,
  fogCanvas: null,
  fogCtx: null,
  detectionLines: null,
  detectionLineGeo: null,
  detectionLinePositions: null,
  commsLines: null,
  commsLineGeo: null,
  commsLinePositions: null,
  commsLineColors: null,
  ewRingMesh: null,
  boltMesh: null,
  boltGeo: null,

  // Boat
  boatGroup: null,
  boatLight: null,
  boatSearchlight: null,
  boatState: { x: 30, dir: 1, speed: 3, flashTimer: 0 },
  wake1: null,
  wake2: null,

  // Ground
  ground: null,
  canalMesh: null,

  // Tracers/flashes/projectiles pools
  tracerPool: [],
  flashPool: [],
  projPool: [],
  projMats: null,

  // Injured markers
  injuredMarkers: [],

  // Robot visuals
  robotLidarCyls: [],
  robotLidarLines: [],
  robotTrails: [],
  robotWarningLights: [],
  robotLabels: [],
  robotPathLines: [],
  robotStopBars: [],

  // Crosswalk helpers
  _cwQ: null,
  _cwQ90: null,

  // Objective defs
  objectiveDefs: [],

  // IED helpers
  _iedDummy: null,
  _iedColor: null,

  // Signal colors
  _hlDefault: null,
  _tlDefault: null,
  _tlBrake: null,
  _hlOrange: null,
  _tlColor: null,

  // Morale helpers
  _moraleColorA: null,
  _moraleColorB: null,
  _healthColorA: null,
  _healthColorB: null,

  // AI panel state
  selectedUnit: null,
  selectedSide: null,
  aiPanel: null,
  aiRaycaster: null,
  aiMouse: null,
  aiPlane: null,
  aiIntersect: null,
  targetLine: null,
  targetLineGeo: null,
  targetLineMat: null,

  // Audio
  audioCtx: null,
  ambientOn: false,
  ambientGain: null,
  ambientNoise: null,
  ambientOsc: null,
  crowdGain: null,
  crowdFilter: null,
  crowdNoise: null,
  rainGain: null,
  rainFilter: null,
  rainNoiseSrc: null,

  // Replay
  replayRecording: false,
  replayFrames: [],
  replayPlayback: false,
  replayIdx: 0,
  replaySpeed: 1,
  replayAccum: 0,
  replayRecordAccum: 0,
  replayRecStartTime: 0,

  // Render loop
  lastTime: 0,
  frameCount: 0,
  fpsDisplay: 0,
  fpsTimer: 0,
  simAccum: 0,
  hudTimer: 0,
  lowFpsCounter: 0,
  perfDegraded: false,
  fpsHistory: new Float32Array(200),
  fpsHistIdx: 0,
  _shadowBaked: false,

  // Performance fix bounding sphere
  _cityBoundingSphere: null,
};

// =========================================================================
// Road Grid Initialization (pure data, no scene dependency)
// =========================================================================
for (let row = 0; row <= GRID_ROWS; row++) {
  state.hRoads.push(row * (BLOCK_H + ROAD_W) + ROAD_W / 2);
}
for (let col = 0; col <= GRID_COLS; col++) {
  state.vRoads.push(col * (BLOCK_W + ROAD_W) + ROAD_W / 2);
}
for (const x of state.vRoads) {
  for (const z of state.hRoads) {
    state.intersections.push({ x, z });
  }
}

// Pre-allocate rain drop positions
for (let i = 0; i < RAIN_COUNT; i++) {
  state.rainDrops.push({
    x: rng() * (CITY_W + 100) - 50,
    y: rng() * 80,
    z: rng() * (CITY_H + 100) - 50,
    speed: 20 + rng() * 25,
  });
}
