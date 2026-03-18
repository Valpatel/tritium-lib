/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Day/night cycle, rain, lightning, fog, sky colors.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, _skyA, _skyB, rng,
  CITY_W, CITY_H, RAIN_COUNT, VOID_BG,
} from './config.js';

export function getSkyColor(h) {
  if (h >= 7 && h < 17) return 0x88aacc;
  if (h >= 19 || h < 5) return 0x0a0a0f;
  if (h >= 5 && h < 6) { _skyA.set(0x0a0a2e); _skyB.set(0xff8844); _skyA.lerp(_skyB, h - 5); return _skyA.getHex(); }
  if (h >= 6 && h < 7) { _skyA.set(0xff8844); _skyB.set(0x4488cc); _skyA.lerp(_skyB, h - 6); return _skyA.getHex(); }
  if (h >= 17 && h < 18) { _skyA.set(0x4488cc); _skyB.set(0xff6622); _skyA.lerp(_skyB, h - 17); return _skyA.getHex(); }
  _skyA.set(0xff6622); _skyB.set(0x2a1a3e); _skyA.lerp(_skyB, h - 18); return _skyA.getHex();
}

export function initRain() {
  const scene = state.scene;
  const rainGeo = new THREE.BoxGeometry(0.08, 1.5, 0.08);
  const rainMat = new THREE.MeshStandardMaterial({
    color: 0xccddff, emissive: 0x667799, emissiveIntensity: 0.4,
    transparent: true, opacity: 0.55, depthWrite: false,
  });
  const rainMesh = new THREE.InstancedMesh(rainGeo, rainMat, RAIN_COUNT);
  rainMesh.count = 0;
  rainMesh.renderOrder = 15;
  scene.add(rainMesh);
  state.rainMesh = rainMesh;
  state.rainMat = rainMat;

  // Lightning bolt
  const boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(24), 3));
  const boltMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
  const boltMesh = new THREE.LineSegments(boltGeo, boltMat);
  boltMesh.visible = false;
  boltMesh.renderOrder = 20;
  scene.add(boltMesh);
  state.boltMesh = boltMesh;
  state.boltGeo = boltGeo;
}

export function updateRain(dt) {
  if (!state.rainActive) {
    state.rainMesh.count = 0;
    return;
  }
  state.rainMesh.count = RAIN_COUNT;
  for (let i = 0; i < RAIN_COUNT; i++) {
    const drop = state.rainDrops[i];
    drop.y -= drop.speed * dt;
    if (drop.y < 0) {
      drop.y = 80;
      drop.x = rng() * (CITY_W + 100) - 50;
      drop.z = rng() * (CITY_H + 100) - 50;
      drop.speed = 20 + rng() * 25;
    }
    _pos.set(drop.x, drop.y, drop.z);
    _quat.identity();
    const yStretch = 1.0 + (drop.speed - 20) / 25.0;
    _scale.set(1, yStretch, 1);
    _mat4.compose(_pos, _quat, _scale);
    state.rainMesh.setMatrixAt(i, _mat4);
  }
  state.rainMesh.instanceMatrix.needsUpdate = true;
}

export function updateWeather(dt, isNight) {
  const { scene, renderer, rainActive, rainMat, roadMat, windowMat, lampHalos, cityGlowLights } = state;
  const dayFactor = isNight ? 0.2 : 1.0;
  const rainDim = rainActive ? 0.7 : 1.0;
  state.ambientLight.intensity = (0.3 + dayFactor * 0.4) * rainDim;
  state.sunLight.intensity = dayFactor * 1.2 * rainDim;
  state.hemiLight.intensity = (0.3 + dayFactor * 0.3) * rainDim;

  let baseFogDensity = isNight ? 0.004 : 0.0015;
  if (rainActive) baseFogDensity = Math.max(baseFogDensity, 0.004);
  if (state.fogOverride) baseFogDensity = 0.008;
  scene.fog.density = baseFogDensity;

  const skyColor = rainActive ? 0x0a0e1a : getSkyColor(state.simTime);
  scene.background.set(skyColor);
  scene.fog.color.set(skyColor);

  // Lightning
  if (state.lightningTimer > 0) {
    state.lightningTimer--;
    if (state.lightningTimer === 0) { state.boltMesh.visible = false; renderer.toneMappingExposure = isNight ? 0.9 : 1.4; }
  } else if (rainActive && (state.riotPhase === 'RIOT' || state.riotPhase === 'DISPERSAL' || state.riotPhase === 'COMBAT') && Math.random() < 0.02 * dt) {
    state.lightningCount++;
    state.lightningTimer = 3;
    renderer.toneMappingExposure = 4.0;
    const bx = rng() * CITY_W, bz = rng() * CITY_H, pos = state.boltGeo.attributes.position;
    let cy = 80;
    for (let s = 0; s < 4; s++) {
      pos.setXYZ(s * 2, bx + (rng() - 0.5) * 8, cy, bz + (rng() - 0.5) * 8);
      cy -= 15 + rng() * 10;
      pos.setXYZ(s * 2 + 1, bx + (rng() - 0.5) * 12, Math.max(cy, 0), bz + (rng() - 0.5) * 12);
    }
    pos.needsUpdate = true;
    state.boltMesh.visible = true;
    try {
      // Audio is imported statically at module level; use state.audioCtx directly
      if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = state.audioCtx;
      const sz = Math.floor(ctx.sampleRate * 0.6);
      const buf = ctx.createBuffer(1, sz, ctx.sampleRate), d = buf.getChannelData(0);
      for (let i = 0; i < sz; i++) { const t = i / ctx.sampleRate; d[i] = (Math.sin(6.28 * 30 * t) * 0.5 + (Math.random() * 2 - 1) * 0.5) * Math.exp(-t * 4) * 0.2; }
      const src = ctx.createBufferSource(); src.buffer = buf;
      const g = ctx.createGain(); g.gain.value = 0.08;
      src.connect(g).connect(ctx.destination); src.start();
    } catch(e) {}
  } else {
    renderer.toneMappingExposure = isNight ? 0.9 : 1.4;
  }

  // Lamp halos
  const haloOpacity = isNight ? 0.7 : 0.0;
  for (const halo of lampHalos) {
    halo.material.opacity = haloOpacity;
    halo.visible = isNight;
  }

  // City glow
  for (const gl of cityGlowLights) {
    gl.intensity = isNight ? 2.5 : 0;
  }

  // Wet roads
  const wetRoad = rainActive && isNight;
  if (roadMat) {
    roadMat.color.set(wetRoad ? 0x222233 : 0x444444);
    roadMat.metalness = wetRoad ? 0.4 : 0.0;
    roadMat.roughness = wetRoad ? 0.5 : 0.85;
  }

  if (rainMat) rainMat.emissiveIntensity = isNight ? 0.7 : 0.3;

  // Window glow
  if (windowMat) windowMat.emissiveIntensity = isNight ? 0.8 : 0.15;

  // Window flicker
  if (!window._lastWinFlicker) window._lastWinFlicker = 0;
  window._lastWinFlicker += dt;
  if (window._lastWinFlicker > 10 && isNight && state.windowMesh && state.windowMesh.instanceColor) {
    window._lastWinFlicker = 0;
    for (let wi = 0; wi < state.windowMesh.count; wi++) {
      if (rng() < 0.3) {
        state.windowMesh.setColorAt(wi, _color.set(0x111111));
      } else {
        state.windowMesh.setColorAt(wi, _color.set(0xffdd66));
      }
    }
    state.windowMesh.instanceColor.needsUpdate = true;
  }
}
