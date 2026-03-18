/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — HUD updates, kill feed, minimap, compass, debug overlay, AI decision panel.
*/

import * as THREE from 'three';
import {
  state, _color, rng,
  CITY_W, CITY_H, BLOCK_W, BLOCK_H, GRID_COLS, GRID_ROWS, ROAD_W,
  SUPPLY_TEAR_GAS_MAX, MAX_PARTICLES,
  COMMS_RANGE, NUM_ROBOT_CARS, CAMPAIGN_PHASES,
} from './config.js';
import { dist2d } from './people.js';
import { getSkyColor } from './weather.js';
import { updateSoundtrackState } from './systems.js';

// =========================================================================
// HUD Update
// =========================================================================
export function updateHUD() {
  const hours = Math.floor(state.simTime);
  const mins = Math.floor((state.simTime - hours) * 60);
  document.getElementById('clock').textContent =
    `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;

  document.getElementById('civ-count').textContent =
    state.pedestrians.filter(p => p.alive).length + state.cars.filter(c => c.alive).length +
    state.carDrivers.filter(cd => cd.alive).length + state.taxis.length;
  document.getElementById('protestor-count').textContent = state.protestors.filter(p => p.alive && !p.arrested).length;
  document.getElementById('police-count').textContent = state.police.filter(p => p.alive).length;
  document.getElementById('arrest-count').textContent = state.arrestCount;
  document.getElementById('robot-count').textContent = state.robotCars.filter(r => r.alive).length;

  document.getElementById('molotov-count').textContent = state.molotovCount;
  document.getElementById('teargas-count').textContent = state.teargasCount;
  document.getElementById('rock-count').textContent = state.rockCount;
  document.getElementById('injury-count').textContent = state.injuryCount;
  document.getElementById('score-count').textContent = state.totalScore;
  document.getElementById('barricade-count').textContent =
    `${state.barricades.filter(b => b.active).length}/${state.barricades.length}`;

  const alivePolice = state.police.filter(p => p.alive);
  const policeMoraleAvg = alivePolice.length > 0 ? alivePolice.reduce((s, p) => s + (p.morale || 1), 0) / alivePolice.length : 0;
  const aliveProtestors = state.protestors.filter(p => p.alive && !p.arrested);
  const protestorMoraleAvg = aliveProtestors.length > 0 ? aliveProtestors.reduce((s, p) => s + (p.morale || 0.8), 0) / aliveProtestors.length : 0;
  const pmEl = document.getElementById('police-morale');
  if (pmEl) pmEl.textContent = Math.round(policeMoraleAvg * 100) + '%';
  const prMEl = document.getElementById('protestor-morale');
  if (prMEl) prMEl.textContent = Math.round(protestorMoraleAvg * 100) + '%';

  const panicPct = aliveProtestors.length > 0 ? Math.round(aliveProtestors.filter(p => p.fleeing).length / aliveProtestors.length * 100) : 0;
  const clEl = document.getElementById('crowd-clusters');
  if (clEl) {
    const visited = new Set();
    let clusters = 0;
    for (const p of aliveProtestors) {
      if (visited.has(p)) continue;
      clusters++;
      const stack = [p]; visited.add(p);
      while (stack.length) { const c = stack.pop(); for (const p2 of aliveProtestors) { if (!visited.has(p2) && dist2d(c, p2) < 15) { visited.add(p2); stack.push(p2); } } }
    }
    clEl.textContent = clusters + ' clusters';
  }
  const plEl = document.getElementById('panic-level');
  if (plEl) plEl.textContent = panicPct + '%';

  updateSoundtrackState();

  const phaseEl = document.getElementById('phase-indicator');
  phaseEl.textContent = state.riotPhase;
  phaseEl.className = state.riotPhase === 'RIOT' ? 'riot' :
                      state.riotPhase === 'TENSION' ? 'tension' :
                      state.riotPhase === 'DISPERSAL' ? 'dispersal' : '';

  // Campaign phase HUD
  const cpEl = document.getElementById('campaign-phase');
  if (state.campaignPhase > 0 && state.campaignPhase <= 3) {
    const cp = CAMPAIGN_PHASES[state.campaignPhase];
    cpEl.textContent = 'PHASE ' + state.campaignPhase + ': ' + cp.name.toUpperCase();
    cpEl.className = cp.color;
    cpEl.style.display = 'inline-block';
  } else { cpEl.style.display = 'none'; }

  // Kill feed
  const feed = document.getElementById('kill-feed');
  feed.innerHTML = state.killFeed.map((k) => {
    k.age += 1 / 60;
    const cls = k.age > 5 ? 'kill-entry fade' : 'kill-entry';
    let color = '#fcee0a';
    if (k.text.includes('Police')) color = '#00f0ff';
    if (k.text.includes('Protestor')) color = '#05ffa1';
    if (k.text.includes('molotov') || k.text.includes('RIOT')) color = '#ff2a6d';
    if (k.text.includes('Civilian')) color = '#fcee0a';
    if (k.text.includes('helicopter')) color = '#00f0ff';
    if (k.text.includes('Ambulance') || k.text.includes('ambulance')) color = '#ffffff';
    if (k.text.includes('hospital') || k.text.includes('Patient')) color = '#ff6666';
    if (k.text.includes('Fire truck') || k.text.includes('fire')) color = '#ff4400';
    if (k.text.includes('Taxi')) color = '#ffcc00';
    if (k.text.includes('van') || k.text.includes('deployed')) color = '#4488ff';
    return `<div class="${cls}" style="color:${color}">${k.text}</div>`;
  }).join('');

  while (state.killFeed.length > 0 && state.killFeed[state.killFeed.length - 1].age > 12) state.killFeed.pop();

  // Weather HUD
  let weatherText = '';
  if (state.rainActive) weatherText += '<span style="color:#aaccff">RAIN</span> ';
  if (state.fogOverride) weatherText += '<span style="color:#999999">FOG</span> ';
  if (state.fogOfWarEnabled) weatherText += '<span style="color:#00f0ff">Intel FOW</span> ';
  const weatherEl = document.getElementById('weather-hud');
  if (weatherEl) weatherEl.innerHTML = weatherText;

  // Objectives HUD
  const objEl = document.getElementById('objectives-hud');
  if (objEl && state.riotMode) {
    let objHtml = '';
    for (const obj of state.objectiveDefs) {
      if (obj.objectiveStatus === 'inactive') continue;
      let color = '#fcee0a';
      if (obj.objectiveStatus === 'complete') color = '#05ffa1';
      if (obj.objectiveStatus === 'failed') color = '#ff2a6d';
      const statusIcon = obj.objectiveStatus === 'complete' ? '[DONE]' : obj.objectiveStatus === 'failed' ? '[FAIL]' : '[...]';
      objHtml += '<div style="color:' + color + '">' + statusIcon + ' ' + obj.name + '</div>';
    }
    objEl.innerHTML = objHtml;
  } else if (objEl) { objEl.innerHTML = ''; }

  // Split-view panels
  if (state.splitViewMode) {
    const ap = state.police.filter(p => p.alive);
    const pmAvg = ap.length > 0 ? Math.round(ap.reduce((s,p) => s + (p.morale||1), 0) / ap.length * 100) : 0;
    document.getElementById('sv-police').innerHTML =
      '<b>POLICE FORCES</b><br>Officers: ' + ap.length +
      '<br>Morale: ' + pmAvg + '%<br>Budget: $' + state.policeBudget.toLocaleString() +
      '<br>Tear Gas: ' + state.supplyTearGas + '/' + SUPPLY_TEAR_GAS_MAX;
    const pr = state.protestors.filter(p => p.alive && !p.arrested);
    const prAvg = pr.length > 0 ? Math.round(pr.reduce((s,p) => s + (p.morale||0.8), 0) / pr.length * 100) : 0;
    document.getElementById('sv-protest').innerHTML =
      '<b>PROTEST FORCES</b><br>Protestors: ' + pr.length +
      '<br>Morale: ' + prAvg + '%<br>Molotovs: ' + state.molotovCount +
      '<br>Rocks: ' + state.rockCount;
  }
}

// =========================================================================
// Minimap
// =========================================================================
export function drawMinimap() {
  const minimapCanvas = document.getElementById('minimap-canvas');
  if (!minimapCanvas || !state.minimapVisible) return;
  const ctx = minimapCanvas.getContext('2d');
  const MINIMAP_W = 180, MINIMAP_H = 120;
  const sx = MINIMAP_W / CITY_W;
  const sz = MINIMAP_H / CITY_H;

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, MINIMAP_W, MINIMAP_H);

  ctx.strokeStyle = '#333333'; ctx.lineWidth = 1;
  for (let r = 0; r <= GRID_ROWS; r++) {
    const y = r * (BLOCK_H + ROAD_W) * sz;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(MINIMAP_W, y); ctx.stroke();
  }
  for (let c = 0; c <= GRID_COLS; c++) {
    const mx = c * (BLOCK_W + ROAD_W) * sx;
    ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, MINIMAP_H); ctx.stroke();
  }

  ctx.fillStyle = '#1a1a2e';
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const bx = ROAD_W + c * (BLOCK_W + ROAD_W);
      const bz = ROAD_W + r * (BLOCK_H + ROAD_W);
      ctx.fillRect(bx * sx, bz * sz, BLOCK_W * sx, BLOCK_H * sz);
    }
  }

  function drawDot(ex, ez, color, size) {
    ctx.fillStyle = color;
    ctx.fillRect(ex * sx - size / 2, ez * sz - size / 2, size, size);
  }

  for (const car of state.cars) { if (car.alive !== false) drawDot(car.x, car.z, car.isRobot ? '#00f0ff' : '#888888', 3); }
  for (const taxi of state.taxis) { if (taxi.alive !== false) drawDot(taxi.x, taxi.z, '#fcee0a', 3); }
  for (const amb of state.ambulances) { if (amb.alive !== false) drawDot(amb.x, amb.z, '#ffffff', 3); }
  for (const ft of state.fireTrucks) { if (ft.alive !== false) drawDot(ft.x, ft.z, '#ff4400', 3); }
  for (const pv of state.policeVans) { if (pv.alive !== false) drawDot(pv.x, pv.z, '#4466ff', 3); }
  if (state.supplyConvoy && state.supplyConvoy.alive) drawDot(state.supplyConvoy.x, state.supplyConvoy.z, '#225522', 4);
  for (const ped of state.pedestrians) { if (ped.alive !== false) drawDot(ped.x, ped.z, '#666666', 2); }
  for (const pol of state.police) { if (pol.alive !== false) drawDot(pol.x, pol.z, '#00f0ff', 2); }
  for (const pr of state.protestors) { if (pr.alive !== false) drawDot(pr.x, pr.z, state.riotMode ? '#ff2a6d' : '#05ffa1', 2); }
  for (const ied of state.ieds) {
    if (!ied.armed) continue;
    ctx.beginPath();
    ctx.arc(ied.x * sx, ied.z * sz, ied.detected ? 4 : 2, 0, Math.PI * 2);
    ctx.fillStyle = ied.detected ? '#ff0000' : '#ff2a6d44';
    ctx.fill();
  }

  const tgt = state.controls.target;
  const dd = state.camera.position.distanceTo(tgt);
  const vFov = state.camera.fov * Math.PI / 180;
  const viewH = 2 * Math.tan(vFov / 2) * dd;
  const viewW = viewH * state.camera.aspect;
  const vpL = Math.max(0, (tgt.x - viewW / 2) * sx);
  const vpT = Math.max(0, (tgt.z - viewH / 2) * sz);
  const vpW = Math.min(MINIMAP_W - vpL, viewW * sx);
  const vpH = Math.min(MINIMAP_H - vpT, viewH * sz);
  ctx.strokeStyle = '#ffffff88'; ctx.lineWidth = 1;
  ctx.strokeRect(vpL, vpT, vpW, vpH);
  ctx.strokeStyle = '#00f0ff33';
  ctx.strokeRect(0, 0, MINIMAP_W, MINIMAP_H);
}

// =========================================================================
// Compass
// =========================================================================
export function updateCompass() {
  const compassEl = document.getElementById('compass-hud');
  if (!compassEl || !state.controls) return;
  const azimuth = state.controls.getAzimuthalAngle();
  const deg = ((azimuth * 180 / Math.PI) + 360) % 360;
  let primary;
  if (deg >= 315 || deg < 45) primary = 'S';
  else if (deg >= 45 && deg < 135) primary = 'W';
  else if (deg >= 135 && deg < 225) primary = 'N';
  else primary = 'E';
  const directions = ['N', 'E', 'S', 'W'];
  let html = '';
  for (const d of directions) {
    if (d === primary) html += '<span class="compass-active">' + d + '</span> ';
    else html += d + ' ';
  }
  compassEl.innerHTML = html.trim();
}

// =========================================================================
// FPS Graph
// =========================================================================
export function drawFPSGraph() {
  const fpsGraphCanvas = document.getElementById('fps-graph');
  if (!fpsGraphCanvas) return;
  const fpsGraphCtx = fpsGraphCanvas.getContext('2d');
  const w = 200, h = 60;
  fpsGraphCtx.clearRect(0, 0, w, h);
  fpsGraphCtx.strokeStyle = '#05ffa122';
  fpsGraphCtx.beginPath();
  for (let y of [15, 30, 45]) { fpsGraphCtx.moveTo(0, y); fpsGraphCtx.lineTo(w, y); }
  fpsGraphCtx.stroke();
  fpsGraphCtx.fillStyle = '#05ffa166';
  fpsGraphCtx.font = '8px monospace';
  fpsGraphCtx.fillText('60', 1, 13);
  fpsGraphCtx.fillText('30', 1, 33);
  fpsGraphCtx.fillText('0', 1, 58);
  fpsGraphCtx.strokeStyle = '#05ffa1';
  fpsGraphCtx.lineWidth = 1;
  fpsGraphCtx.beginPath();
  for (let i = 0; i < 200; i++) {
    const idx = (state.fpsHistIdx + i) % 200;
    const fps = state.fpsHistory[idx];
    const x = i;
    const y = h - (fps / 60) * h;
    if (i === 0) fpsGraphCtx.moveTo(x, y);
    else fpsGraphCtx.lineTo(x, y);
  }
  fpsGraphCtx.stroke();
  const curFps = state.fpsHistory[(state.fpsHistIdx + 199) % 200];
  if (curFps > 0 && curFps < 30) {
    fpsGraphCtx.fillStyle = '#ff2a6d44';
    fpsGraphCtx.fillRect(0, 0, w, h);
  }
}
