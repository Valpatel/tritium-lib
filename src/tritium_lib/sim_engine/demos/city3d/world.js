/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — World generation: scene, buildings, trees, lamps, roads, plaza, sidewalks.
*/

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  VOID_BG, CYAN, CITY_W, CITY_H, BLOCK_W, BLOCK_H, GRID_COLS, GRID_ROWS,
  ROAD_W, SIDEWALK_W, PLAZA_X, PLAZA_Z,
  BLDG_HOUSE, BLDG_OFFICE, BLDG_SHOP, BLDG_POLICE, BLDG_HOSPITAL, BLDG_ROOF,
  MAX_BUILDINGS, MAX_WINDOWS, MAX_DOORS, MAX_TREES, MAX_LAMPS, MAX_LAMP_LIGHTS,
  CANAL_ROW, CANAL_Z, CANAL_W,
  MAX_PEOPLE, MAX_RINGS, MAX_SHIELDS,
  MAX_CARS_INST, MAX_PARTICLES, MAX_BARRICADES,
  MAX_DETECTION_LINES, MAX_COMMS_LINKS,
  BODY_W, BODY_H, BODY_D, HEAD_SIZE, PERSON_SCALE,
  RAIN_COUNT, MAX_OBJECTIVES, MAX_IEDS,
  TERRITORY_COLS, TERRITORY_ROWS,
  FOG_COLS, FOG_ROWS,
  STRIPS_PER_CW, CW_PER_INT,
  NUM_POLICE, NUM_ROBOT_CARS, LIDAR_RAYS, ROBOT_TRAIL_LENGTH,
  ARC_ANGLE, ARC_RANGE, ARC_SEGMENTS,
  MAX_TRACERS, MAX_FLASHES, MAX_PROJECTILES, MAX_INJURED_MARKERS,
} from './config.js';

// =========================================================================
// Scene Setup
// =========================================================================
export function init() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(VOID_BG);
  scene.fog = new THREE.FogExp2(VOID_BG, 0.002);
  state.scene = scene;

  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.5, 2000);
  camera.position.set(CITY_W / 2 + 80, 110, CITY_H / 2 + 120);
  camera.lookAt(CITY_W / 2, 0, CITY_H / 2);
  state.camera = camera;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.4;
  document.body.prepend(renderer.domElement);
  state.renderer = renderer;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(CITY_W / 2, 0, CITY_H / 2);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 30;
  controls.maxDistance = 800;
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.update();
  state.controls = controls;

  // Lights
  const ambientLight = new THREE.AmbientLight(0x8899bb, 0.5);
  scene.add(ambientLight);
  state.ambientLight = ambientLight;

  const sunLight = new THREE.DirectionalLight(0xffeedd, 1.2);
  sunLight.position.set(200, 400, 100);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.left = -CITY_W;
  sunLight.shadow.camera.right = CITY_W;
  sunLight.shadow.camera.top = CITY_H;
  sunLight.shadow.camera.bottom = -CITY_H;
  scene.add(sunLight);
  state.sunLight = sunLight;

  const hemiLight = new THREE.HemisphereLight(0x00f0ff, 0x112244, 0.5);
  scene.add(hemiLight);
  state.hemiLight = hemiLight;

  // =========================================================================
  // Ground
  // =========================================================================
  const groundGeo = new THREE.PlaneGeometry(CITY_W + 100, CITY_H + 100);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(CITY_W / 2, -0.05, CITY_H / 2);
  ground.receiveShadow = true;
  scene.add(ground);
  state.ground = ground;

  // Road materials
  const roadMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.85 });
  const sidewalkMat = new THREE.MeshStandardMaterial({ color: 0x666666, roughness: 0.8 });
  state.roadMat = roadMat;
  state.sidewalkMat = sidewalkMat;

  // =========================================================================
  // Roads — horizontal
  // =========================================================================
  for (let row = 0; row <= GRID_ROWS; row++) {
    if (row === CANAL_ROW) continue;
    const z = row * (BLOCK_H + ROAD_W) + ROAD_W / 2;
    const geo = new THREE.PlaneGeometry(CITY_W, ROAD_W);
    const road = new THREE.Mesh(geo, roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.set(CITY_W / 2, 0.01, z);
    road.receiveShadow = true;
    scene.add(road);
    const lineGeo = new THREE.PlaneGeometry(CITY_W, 0.4);
    const lineMat = new THREE.MeshStandardMaterial({ color: 0x999922, roughness: 0.9 });
    const line = new THREE.Mesh(lineGeo, lineMat);
    line.rotation.x = -Math.PI / 2;
    line.position.set(CITY_W / 2, 0.02, z);
    scene.add(line);
  }

  // Canal
  const canalMesh = new THREE.Mesh(new THREE.PlaneGeometry(CITY_W, CANAL_W),
    new THREE.MeshStandardMaterial({ color: 0x1a3344, roughness: 0.3, metalness: 0.4, transparent: true, opacity: 0.85 }));
  canalMesh.rotation.x = -Math.PI / 2; canalMesh.position.set(CITY_W / 2, -0.02, CANAL_Z); scene.add(canalMesh);
  state.canalMesh = canalMesh;
  for (const side of [-1, 1]) {
    const e = new THREE.Mesh(new THREE.PlaneGeometry(CITY_W, 0.6), new THREE.MeshStandardMaterial({ color: 0x0d1a22 }));
    e.rotation.x = -Math.PI / 2; e.position.set(CITY_W / 2, 0.03, CANAL_Z + side * CANAL_W / 2); scene.add(e);
  }
  for (let col = 0; col <= GRID_COLS; col++) {
    const bx = col * (BLOCK_W + ROAD_W) + ROAD_W / 2;
    const br = new THREE.Mesh(new THREE.BoxGeometry(ROAD_W + 4, 0.8, CANAL_W + 2),
      new THREE.MeshStandardMaterial({ color: 0x666666, roughness: 0.85 }));
    br.position.set(bx, 0.3, CANAL_Z); br.castShadow = true; br.receiveShadow = true; scene.add(br);
  }

  // Patrol boat
  const boatGroup = new THREE.Group();
  boatGroup.add(new THREE.Mesh(new THREE.BoxGeometry(2.5, 1, 6), new THREE.MeshStandardMaterial({ color: 0x223366, roughness: 0.6 })));
  const boatLight = new THREE.PointLight(0x00f0ff, 2, 20);
  boatLight.position.set(0, 1.5, 0); boatGroup.add(boatLight);
  const boatSearchlight = new THREE.SpotLight(0xffffcc, 0, 40, Math.PI / 8, 0.5);
  boatSearchlight.position.set(0, 1.2, 2.5);
  boatSearchlight.target.position.set(0, 0, 12);
  boatGroup.add(boatSearchlight); boatGroup.add(boatSearchlight.target);
  boatGroup.position.set(30, 0.3, CANAL_Z); scene.add(boatGroup);
  state.boatGroup = boatGroup;
  state.boatLight = boatLight;
  state.boatSearchlight = boatSearchlight;

  const wakeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
  const wake1 = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.2, 0.8), wakeMat);
  const wake2 = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.15, 0.6), wakeMat.clone());
  scene.add(wake1); scene.add(wake2);
  state.wake1 = wake1;
  state.wake2 = wake2;

  // Roads — vertical
  for (let col = 0; col <= GRID_COLS; col++) {
    const x = col * (BLOCK_W + ROAD_W) + ROAD_W / 2;
    const geo = new THREE.PlaneGeometry(ROAD_W, CITY_H);
    const road = new THREE.Mesh(geo, roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.set(x, 0.01, CITY_H / 2);
    road.receiveShadow = true;
    scene.add(road);
  }

  // Sidewalks
  for (let row = 0; row <= GRID_ROWS; row++) {
    const z = row * (BLOCK_H + ROAD_W) + ROAD_W / 2;
    for (const side of [-1, 1]) {
      const sz = z + side * (ROAD_W / 2 + SIDEWALK_W / 2);
      const geo = new THREE.PlaneGeometry(CITY_W, SIDEWALK_W);
      const sw = new THREE.Mesh(geo, sidewalkMat);
      sw.rotation.x = -Math.PI / 2;
      sw.position.set(CITY_W / 2, 0.05, sz);
      scene.add(sw);
    }
  }
  for (let col = 0; col <= GRID_COLS; col++) {
    const x = col * (BLOCK_W + ROAD_W) + ROAD_W / 2;
    for (const side of [-1, 1]) {
      const sx = x + side * (ROAD_W / 2 + SIDEWALK_W / 2);
      const geo = new THREE.PlaneGeometry(SIDEWALK_W, CITY_H);
      const sw = new THREE.Mesh(geo, sidewalkMat);
      sw.rotation.x = -Math.PI / 2;
      sw.position.set(sx, 0.05, CITY_H / 2);
      scene.add(sw);
    }
  }

  // =========================================================================
  // Buildings — InstancedMesh
  // =========================================================================
  const buildingBodyGeo = new THREE.BoxGeometry(1, 1, 1);
  const buildingBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.0 });
  const buildingBodyMesh = new THREE.InstancedMesh(buildingBodyGeo, buildingBodyMat, MAX_BUILDINGS);
  buildingBodyMesh.castShadow = true;
  buildingBodyMesh.receiveShadow = true;
  buildingBodyMesh.count = 0;
  scene.add(buildingBodyMesh);
  state.buildingBodyMesh = buildingBodyMesh;

  const buildingRoofGeo = new THREE.BoxGeometry(1, 1, 1);
  const buildingRoofMat = new THREE.MeshStandardMaterial({ color: BLDG_ROOF, roughness: 0.9 });
  const buildingRoofMesh = new THREE.InstancedMesh(buildingRoofGeo, buildingRoofMat, MAX_BUILDINGS);
  buildingRoofMesh.castShadow = true;
  buildingRoofMesh.count = 0;
  scene.add(buildingRoofMesh);
  state.buildingRoofMesh = buildingRoofMesh;

  // Windows
  const windowGeo = new THREE.PlaneGeometry(1.5, 1.2);
  const windowMat = new THREE.MeshStandardMaterial({
    color: 0xffdd66, emissive: 0xffdd44, emissiveIntensity: 0.15,
    transparent: true, opacity: 0.7,
  });
  const windowMesh = new THREE.InstancedMesh(windowGeo, windowMat, MAX_WINDOWS);
  windowMesh.count = 0;
  scene.add(windowMesh);
  state.windowMesh = windowMesh;
  state.windowMat = windowMat;

  // Doors
  const doorGeo = new THREE.PlaneGeometry(1, 1);
  const doorMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
  const doorIMesh = new THREE.InstancedMesh(doorGeo, doorMat, MAX_DOORS);
  doorIMesh.count = 0;
  scene.add(doorIMesh);
  state.doorIMesh = doorIMesh;

  // =========================================================================
  // Building placement helper
  // =========================================================================
  function addInstancedBuilding(x, z, w, d, h, type) {
    if (state.buildingIdx >= MAX_BUILDINGS) return;

    let color;
    switch (type) {
      case 'house':    color = BLDG_HOUSE; break;
      case 'office':   color = BLDG_OFFICE; break;
      case 'shop':     color = BLDG_SHOP; break;
      case 'police':   color = BLDG_POLICE; break;
      case 'hospital': color = BLDG_HOSPITAL; break;
      default:         color = BLDG_OFFICE;
    }

    _pos.set(x, h / 2, z);
    _quat.identity();
    _scale.set(w, h, d);
    _mat4.compose(_pos, _quat, _scale);
    buildingBodyMesh.setMatrixAt(state.buildingIdx, _mat4);
    buildingBodyMesh.setColorAt(state.buildingIdx, _color.set(color));

    _pos.set(x, h + 0.25, z);
    _scale.set(w + 0.2, 0.5, d + 0.2);
    _mat4.compose(_pos, _quat, _scale);
    buildingRoofMesh.setMatrixAt(state.buildingIdx, _mat4);

    state.buildingIdx++;
    buildingBodyMesh.count = state.buildingIdx;
    buildingRoofMesh.count = state.buildingIdx;

    // Door
    if (state.doorIdx < MAX_DOORS) {
      const doorW = Math.min(2, w * 0.3);
      const doorH = Math.min(3, h * 0.5);
      let bestFaceDist = Infinity;
      let doorX = x, doorZ = z + d / 2 + 0.05, doorRotY = 0;
      for (const hz of state.hRoads) {
        const fd = Math.abs((z + d / 2) - hz);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x; doorZ = z + d / 2 + 0.05; doorRotY = 0; }
      }
      for (const hz of state.hRoads) {
        const fd = Math.abs((z - d / 2) - hz);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x; doorZ = z - d / 2 - 0.05; doorRotY = Math.PI; }
      }
      for (const vx of state.vRoads) {
        const fd = Math.abs((x + w / 2) - vx);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x + w / 2 + 0.05; doorZ = z; doorRotY = Math.PI / 2; }
      }
      for (const vx of state.vRoads) {
        const fd = Math.abs((x - w / 2) - vx);
        if (fd < bestFaceDist) { bestFaceDist = fd; doorX = x - w / 2 - 0.05; doorZ = z; doorRotY = -Math.PI / 2; }
      }
      _pos.set(doorX, doorH / 2, doorZ);
      _euler.set(0, doorRotY, 0);
      _quat.setFromEuler(_euler);
      _scale.set(doorW, doorH, 1);
      _mat4.compose(_pos, _quat, _scale);
      doorIMesh.setMatrixAt(state.doorIdx, _mat4);
      state.doorIdx++;
      doorIMesh.count = state.doorIdx;
    }

    // Windows
    const startWinIdx = state.windowIdx;
    const floors = Math.floor(h / 4);
    for (let f = 1; f <= floors; f++) {
      const wy = f * 3.5;
      if (wy > h - 1) break;
      for (const side of [1, -1]) {
        const numWins = Math.floor(w / 4);
        for (let wi = 0; wi < numWins; wi++) {
          if (rng() < 0.3 || state.windowIdx >= MAX_WINDOWS) continue;
          const wx = x - w / 2 + 2 + wi * 4;
          _pos.set(wx, wy, z + side * (d / 2 + 0.05));
          _euler.set(0, side < 0 ? Math.PI : 0, 0);
          _quat.setFromEuler(_euler);
          _scale.set(1, 1, 1);
          _mat4.compose(_pos, _quat, _scale);
          windowMesh.setMatrixAt(state.windowIdx, _mat4);
          const winColor = rng() < 0.7 ? 0xffdd66 : 0x66ddff;
          windowMesh.setColorAt(state.windowIdx, _color.set(winColor));
          state.windowIdx++;
        }
      }
      for (const side of [1, -1]) {
        const numWins = Math.floor(d / 4);
        for (let wi = 0; wi < numWins; wi++) {
          if (rng() < 0.4 || state.windowIdx >= MAX_WINDOWS) continue;
          const wz = z - d / 2 + 2 + wi * 4;
          _pos.set(x + side * (w / 2 + 0.05), wy, wz);
          _euler.set(0, side * Math.PI / 2, 0);
          _quat.setFromEuler(_euler);
          _scale.set(1, 1, 1);
          _mat4.compose(_pos, _quat, _scale);
          windowMesh.setMatrixAt(state.windowIdx, _mat4);
          const winColor = rng() < 0.7 ? 0xffdd66 : 0x66ddff;
          windowMesh.setColorAt(state.windowIdx, _color.set(winColor));
          state.windowIdx++;
        }
      }
    }
    windowMesh.count = state.windowIdx;

    const bldgData = { h, type, x, z, w, d, idx: state.buildingIdx - 1, winStart: startWinIdx, winEnd: state.windowIdx, cleared: false, damage: 0, origColor: color };
    state.buildings.push(bldgData);

    if (type === 'hospital') {
      state.hospitalBuilding = bldgData;
      const crossH = new THREE.Mesh(
        new THREE.PlaneGeometry(3, 1),
        new THREE.MeshBasicMaterial({ color: 0xff0000 })
      );
      crossH.position.set(x, h * 0.7, z + d / 2 + 0.1);
      scene.add(crossH);
      const crossV = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 3),
        new THREE.MeshBasicMaterial({ color: 0xff0000 })
      );
      crossV.position.set(x, h * 0.7, z + d / 2 + 0.12);
      scene.add(crossV);
    }
  }

  // =========================================================================
  // Trees
  // =========================================================================
  const trunkGeo = new THREE.BoxGeometry(1, 5, 1);
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5D3A1A });
  const trunkMesh = new THREE.InstancedMesh(trunkGeo, trunkMat, MAX_TREES);
  trunkMesh.castShadow = true;
  trunkMesh.count = 0;
  scene.add(trunkMesh);
  state.trunkMesh = trunkMesh;

  const leafGeo = new THREE.BoxGeometry(4, 4, 4);
  const leafMat = new THREE.MeshStandardMaterial({ color: 0x228B22 });
  const leafMesh = new THREE.InstancedMesh(leafGeo, leafMat, MAX_TREES);
  leafMesh.castShadow = true;
  leafMesh.count = 0;
  scene.add(leafMesh);
  state.leafMesh = leafMesh;

  function addTree(tx, tz) {
    if (state.treeIdx >= MAX_TREES) return;
    _quat.identity();
    _scale.set(1, 1, 1);
    _pos.set(tx, 2.5, tz);
    _mat4.compose(_pos, _quat, _scale);
    trunkMesh.setMatrixAt(state.treeIdx, _mat4);
    _pos.set(tx, 6, tz);
    _mat4.compose(_pos, _quat, _scale);
    leafMesh.setMatrixAt(state.treeIdx, _mat4);
    state.treeIdx++;
    trunkMesh.count = state.treeIdx;
    leafMesh.count = state.treeIdx;
  }

  // =========================================================================
  // Street Lamps
  // =========================================================================
  const lampPostGeo = new THREE.BoxGeometry(0.3, 7, 0.3);
  const lampPostMat = new THREE.MeshStandardMaterial({ color: 0x666666 });
  const lampPostMesh = new THREE.InstancedMesh(lampPostGeo, lampPostMat, MAX_LAMPS);
  lampPostMesh.count = 0;
  scene.add(lampPostMesh);
  state.lampPostMesh = lampPostMesh;

  const lampHeadGeo = new THREE.BoxGeometry(1.2, 0.5, 1.2);
  const lampHeadMat = new THREE.MeshBasicMaterial({ color: 0xffdd88, transparent: true, opacity: 0.8 });
  const lampHeadMesh = new THREE.InstancedMesh(lampHeadGeo, lampHeadMat, MAX_LAMPS);
  lampHeadMesh.count = 0;
  scene.add(lampHeadMesh);
  state.lampHeadMesh = lampHeadMesh;

  // =========================================================================
  // Place buildings, trees, lamps
  // =========================================================================
  const centerCol = Math.floor(GRID_COLS / 2);
  const centerRow = Math.floor(GRID_ROWS / 2);

  for (let row = 0; row < GRID_ROWS; row++) {
    for (let col = 0; col < GRID_COLS; col++) {
      const bx = col * (BLOCK_W + ROAD_W) + ROAD_W + BLOCK_W / 2;
      const bz = row * (BLOCK_H + ROAD_W) + ROAD_W + BLOCK_H / 2;

      if (col === centerCol && row === centerRow) {
        const plazaGeo = new THREE.PlaneGeometry(BLOCK_W - 4, BLOCK_H - 4);
        const plazaMat = new THREE.MeshStandardMaterial({ color: 0x777777, roughness: 0.7 });
        const plaza = new THREE.Mesh(plazaGeo, plazaMat);
        plaza.rotation.x = -Math.PI / 2;
        plaza.position.set(bx, 0.06, bz);
        scene.add(plaza);
        const fountGeo = new THREE.CylinderGeometry(3, 4, 1.5, 12);
        const fountMat = new THREE.MeshStandardMaterial({ color: 0x555566 });
        const fount = new THREE.Mesh(fountGeo, fountMat);
        fount.position.set(bx, 0.75, bz);
        fount.castShadow = true;
        scene.add(fount);
        continue;
      }

      const count = 2 + Math.floor(rng() * 3);
      for (let b = 0; b < count; b++) {
        const typeRoll = rng();
        let type;
        if (col === centerCol + 1 && row === centerRow && b === 0) {
          type = 'police';
        } else if (col === centerCol - 1 && row === centerRow + 1 && b === 0 && !state.hospitalBuilding) {
          type = 'hospital';
        } else if (typeRoll < 0.35) {
          type = 'house';
        } else if (typeRoll < 0.65) {
          type = 'office';
        } else {
          type = 'shop';
        }

        const isOffice = type === 'office';
        const h = isOffice ? 15 + rng() * 30 : 6 + rng() * 10;
        const w = 8 + rng() * 12;
        const d = 8 + rng() * 12;
        const ox = (rng() - 0.5) * (BLOCK_W - w - 4);
        const oz = (rng() - 0.5) * (BLOCK_H - d - 4);
        addInstancedBuilding(bx + ox, bz + oz, w, d, h, type);
      }

      const numTrees = Math.floor(rng() * 3);
      for (let t = 0; t < numTrees; t++) {
        const tx = bx + (rng() - 0.5) * BLOCK_W * 0.8;
        const tz = bz + (rng() - 0.5) * BLOCK_H * 0.8;
        let insideBuilding = false;
        for (const bldg of state.buildings) {
          if (Math.abs(tx - bldg.x) < (bldg.w / 2 + 2) && Math.abs(tz - bldg.z) < (bldg.d / 2 + 2)) {
            insideBuilding = true;
            break;
          }
        }
        if (insideBuilding) continue;
        addTree(tx, tz);
      }
    }
  }

  // Force hospital if not placed
  if (!state.hospitalBuilding) {
    const hx = 2 * (BLOCK_W + ROAD_W) + ROAD_W + BLOCK_W / 2;
    const hz = 4 * (BLOCK_H + ROAD_W) + ROAD_W + BLOCK_H / 2;
    addInstancedBuilding(hx, hz, 14, 12, 10, 'hospital');
  }

  // Finalize instanced meshes
  buildingBodyMesh.instanceMatrix.needsUpdate = true;
  if (buildingBodyMesh.instanceColor) buildingBodyMesh.instanceColor.needsUpdate = true;
  buildingRoofMesh.instanceMatrix.needsUpdate = true;
  windowMesh.instanceMatrix.needsUpdate = true;
  if (windowMesh.instanceColor) windowMesh.instanceColor.needsUpdate = true;
  doorIMesh.instanceMatrix.needsUpdate = true;
  trunkMesh.instanceMatrix.needsUpdate = true;
  leafMesh.instanceMatrix.needsUpdate = true;

  // =========================================================================
  // Street lamp placement
  // =========================================================================
  const LAMP_OFFSET = ROAD_W / 2 + 1;
  for (let row = 0; row <= GRID_ROWS; row++) {
    const z = row * (BLOCK_H + ROAD_W) + ROAD_W / 2;
    for (let col = 0; col <= GRID_COLS; col++) {
      const x = col * (BLOCK_W + ROAD_W) + ROAD_W / 2;
      const corners = [
        { dx: -LAMP_OFFSET, dz: -LAMP_OFFSET },
        { dx:  LAMP_OFFSET, dz:  LAMP_OFFSET },
      ];
      for (const c of corners) {
        if (state.lampIdx >= MAX_LAMPS) break;
        const lx = x + c.dx;
        const lz = z + c.dz;
        _pos.set(lx, 3.5, lz);
        _quat.identity();
        _scale.set(1, 1, 1);
        _mat4.compose(_pos, _quat, _scale);
        lampPostMesh.setMatrixAt(state.lampIdx, _mat4);
        _pos.set(lx, 7.2, lz);
        _mat4.compose(_pos, _quat, _scale);
        lampHeadMesh.setMatrixAt(state.lampIdx, _mat4);
        state.lampIdx++;
        lampPostMesh.count = state.lampIdx;
        lampHeadMesh.count = state.lampIdx;
        state.streetLamps.push({ x: lx, z: lz, light: null });
      }
    }
  }

  // Pooled lamp lights
  for (let i = 0; i < MAX_LAMP_LIGHTS; i++) {
    const ll = new THREE.PointLight(0xffdd88, 0, 50);
    ll.position.set(0, 7, 0);
    scene.add(ll);
    state.lampLightPool.push(ll);
  }

  lampPostMesh.instanceMatrix.needsUpdate = true;
  lampHeadMesh.instanceMatrix.needsUpdate = true;

  // Night city glow
  const glowPositions = [
    [CITY_W * 0.25, 20, CITY_H * 0.25],
    [CITY_W * 0.75, 20, CITY_H * 0.25],
    [CITY_W * 0.25, 20, CITY_H * 0.75],
    [CITY_W * 0.75, 20, CITY_H * 0.75],
  ];
  for (const gp of glowPositions) {
    const gl = new THREE.PointLight(0xffcc66, 0, 120);
    gl.position.set(gp[0], gp[1], gp[2]);
    scene.add(gl);
    state.cityGlowLights.push(gl);
  }

  // Lamp halos
  const haloCanvas = document.createElement('canvas');
  haloCanvas.width = 64; haloCanvas.height = 64;
  const haloCtx = haloCanvas.getContext('2d');
  const haloGrad = haloCtx.createRadialGradient(32, 32, 0, 32, 32, 32);
  haloGrad.addColorStop(0, 'rgba(255,221,136,0.6)');
  haloGrad.addColorStop(0.4, 'rgba(255,221,136,0.15)');
  haloGrad.addColorStop(1, 'rgba(255,221,136,0)');
  haloCtx.fillStyle = haloGrad;
  haloCtx.fillRect(0, 0, 64, 64);
  const haloTexture = new THREE.CanvasTexture(haloCanvas);
  const haloMat = new THREE.SpriteMaterial({
    map: haloTexture, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  for (const lamp of state.streetLamps) {
    const halo = new THREE.Sprite(haloMat.clone());
    halo.scale.set(8, 8, 1);
    halo.position.set(lamp.x, 7.5, lamp.z);
    scene.add(halo);
    state.lampHalos.push(halo);
  }

  // Store house buildings for car drivers
  state.houseBuildings = state.buildings.filter(b => b.type === 'house' || b.type === 'shop');

  // Return addInstancedBuilding for any later use
  return { addInstancedBuilding };
}

export function updateLampLights(camX, camZ, isNight) {
  const sorted = state.streetLamps.map((l, i) => ({
    idx: i, dist: (l.x - camX) ** 2 + (l.z - camZ) ** 2, x: l.x, z: l.z
  })).sort((a, b) => a.dist - b.dist);

  for (let i = 0; i < MAX_LAMP_LIGHTS; i++) {
    if (i < sorted.length) {
      state.lampLightPool[i].position.set(sorted[i].x, 7, sorted[i].z);
      state.lampLightPool[i].intensity = isNight ? 4.0 : 0.2;
      state.lampLightPool[i].distance = isNight ? 50 : 30;
    }
  }
}
