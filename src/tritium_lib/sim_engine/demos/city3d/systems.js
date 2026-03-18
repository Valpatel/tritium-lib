/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Supply logistics, economy/score, achievements, territory control,
           objectives, status effects, fog of war, comms, EW jamming.
*/

import * as THREE from 'three';
import {
  state, _mat4, _pos, _quat, _scale, _color, _euler, rng,
  CITY_W, CITY_H, PLAZA_X, PLAZA_Z,
  SUPPLY_TEAR_GAS_MAX, SUPPLY_RUBBER_BULLETS_MAX, SUPPLY_MOLOTOVS_MAX,
  COMMS_RANGE, EW_JAM_DURATION, EW_JAM_CHECK_INTERVAL, EW_JAM_CHANCE,
  FOG_COLS, FOG_ROWS, FOG_CELL_SIZE, FOG_UPDATE_INTERVAL, DETECTION_RANGE,
  TERRITORY_COLS, TERRITORY_ROWS, TERRITORY_UPDATE_INTERVAL,
  achievementDefs,
} from './config.js';
import { addNarration, formatGrid, hasLineOfSight, dist2d } from './people.js';

// =========================================================================
// Achievement System
// =========================================================================
export function awardAchievement(id) {
  if (state.achievementsAwarded[id]) return;
  const def = achievementDefs[id];
  if (!def) return;
  state.achievementsAwarded[id] = true;
  state.totalScore += def.points;
  showAchievementToast(def.name, def.desc, def.points);
}

function showAchievementToast(name, desc, points) {
  const container = document.getElementById('achievement-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'achievement-toast';
  toast.innerHTML =
    '<span class="ach-icon">&#9733;</span>' +
    '<span class="ach-name">' + name + '</span>' +
    '<span class="ach-score">+' + points + '</span>' +
    '<span class="ach-desc">' + desc + '</span>';
  container.appendChild(toast);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { toast.classList.add('show'); });
  });
  setTimeout(() => {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 500);
  }, 5000);
}

export function checkAchievements(dt) {
  if (!state.riotMode) return;
  if (state.riotPhase === 'RIOT' && state.police.length > 0) {
    const holdingCount = state.police.filter(p => p.alive && p.holdLine).length;
    if (holdingCount >= Math.floor(state.police.length * 0.7)) {
      state.policeLineHoldTime += dt;
      if (state.policeLineHoldTime >= 60) awardAchievement('iron_line');
    }
  }
  if (state.arrestCount >= 5 && state.injuryCount === 0) awardAchievement('peacekeeper');
}

// =========================================================================
// Soundtrack State
// =========================================================================
export function updateSoundtrackState() {
  let newState;
  switch (state.riotPhase) {
    case 'PEACEFUL': newState = 'PEACEFUL'; break;
    case 'TENSION':  newState = 'TENSION'; break;
    case 'RIOT':     newState = 'COMBAT'; break;
    case 'DISPERSAL': newState = 'AFTERMATH'; break;
    default: newState = 'PEACEFUL';
  }
  state.soundtrackState = newState;
  const el = document.getElementById('soundtrack-indicator');
  if (!el) return;
  const stateMap = {
    PEACEFUL:  { text: '\u266B PEACEFUL', cls: 'peaceful' },
    TENSION:   { text: '\u266B TENSION', cls: 'tension' },
    COMBAT:    { text: '\u266B COMBAT', cls: 'combat' },
    AFTERMATH: { text: '\u266B AFTERMATH', cls: 'aftermath' },
  };
  const info = stateMap[state.soundtrackState] || stateMap.PEACEFUL;
  el.textContent = info.text;
  el.className = info.cls;
}

// =========================================================================
// Status Effects
// =========================================================================
export function addStatusEffect(target, type, duration) {
  for (const eff of state.activeEffects) {
    if (eff.target === target && eff.type === type) {
      eff.timer = Math.max(eff.timer, duration);
      return;
    }
  }
  state.activeEffects.push({ target, type, timer: duration });
}

export function updateStatusEffects(dt) {
  state.effectsCount = 0;
  for (let i = state.activeEffects.length - 1; i >= 0; i--) {
    const eff = state.activeEffects[i];
    eff.timer -= dt;
    if (eff.timer <= 0) {
      if (eff.type === 'GAS' && eff.target) eff.target.gasAffected = false;
      if (eff.type === 'STUNNED' && eff.target) eff.target.stunnedTimer = 0;
      state.activeEffects.splice(i, 1);
    } else {
      state.effectsCount++;
    }
  }
  for (const pr of state.protestors) {
    if (!pr.alive) continue;
    if (pr.stunnedTimer > 0) {
      pr.stunnedTimer -= dt;
      if (pr.stunnedTimer <= 0) pr.stunnedTimer = 0;
    }
  }
  const gasClouds = state.activeParticles.filter(p => p.isTearGas);
  for (const pol of state.police) {
    if (!pol.alive) continue;
    let inGas = false;
    for (const gas of gasClouds) {
      const dx = pol.x - gas.x;
      const dz = pol.z - gas.z;
      if (dx * dx + dz * dz < 100 && hasLineOfSight(gas.x, gas.z, pol.x, pol.z)) { inGas = true; break; }
    }
    if (inGas && !pol.gasAffected) {
      pol.gasAffected = true;
      addStatusEffect(pol, 'GAS', 4.0);
    }
  }
  for (const entry of state.injuredOnGround) {
    if (entry.person && entry.person.alive && entry.person.injured) {
      let has = false;
      for (const eff of state.activeEffects) {
        if (eff.target === entry.person && eff.type === 'BLEEDING') { has = true; break; }
      }
      if (!has) addStatusEffect(entry.person, 'BLEEDING', entry.timer);
    }
  }
}

// =========================================================================
// Supply HUD
// =========================================================================
export function updateSupplyHUD() {
  const panel = document.getElementById('supply-panel');
  if (!panel) return;
  panel.style.display = state.riotMode ? 'block' : 'none';
  if (!state.riotMode) return;

  const tgPct = (state.supplyTearGas / SUPPLY_TEAR_GAS_MAX) * 100;
  const tgBar = document.getElementById('supply-teargas');
  if (tgBar) {
    tgBar.style.width = tgPct + '%';
    tgBar.className = 'supply-bar-fill police' + (tgPct < 20 ? ' supply-low' : '');
  }
  const tgCount = document.getElementById('supply-teargas-count');
  if (tgCount) tgCount.textContent = state.supplyTearGas;

  const rbPct = (state.supplyRubberBullets / SUPPLY_RUBBER_BULLETS_MAX) * 100;
  const rbBar = document.getElementById('supply-rubber');
  if (rbBar) {
    rbBar.style.width = rbPct + '%';
    rbBar.className = 'supply-bar-fill police' + (rbPct < 20 ? ' supply-low' : '');
  }
  const rbCount = document.getElementById('supply-rubber-count');
  if (rbCount) rbCount.textContent = state.supplyRubberBullets;

  const mlPct = (state.supplyMolotovs / SUPPLY_MOLOTOVS_MAX) * 100;
  const mlBar = document.getElementById('supply-molotov');
  if (mlBar) {
    mlBar.style.width = mlPct + '%';
    mlBar.className = 'supply-bar-fill protestor' + (mlPct < 20 ? ' supply-low' : '');
  }
  const mlCount = document.getElementById('supply-molotov-count');
  if (mlCount) mlCount.textContent = state.supplyMolotovs;

  const rkBar = document.getElementById('supply-rock');
  if (rkBar) { rkBar.style.width = '100%'; rkBar.className = 'supply-bar-fill unlimited'; }

  const convoyRow = document.getElementById('supply-convoy-row');
  const convoyBar = document.getElementById('supply-convoy-bar');
  const convoyEta = document.getElementById('supply-convoy-eta');
  if (convoyRow) {
    const active = state.supplyConvoy && state.supplyConvoy.alive;
    convoyRow.style.display = (state.riotPhase === 'RIOT' || state.riotPhase === 'DISPERSAL' || active) ? 'flex' : 'none';
    if (active && convoyBar) {
      const pct = state.supplyConvoy.phase === 'driving' ? (state.supplyConvoy.waypointIdx / Math.max(1, state.supplyConvoy.waypoints.length)) * 100 : 100;
      convoyBar.style.width = pct + '%';
    } else if (convoyBar) { convoyBar.style.width = '0%'; }
    if (convoyEta) convoyEta.textContent = active ? 'EN ROUTE' : Math.max(0, Math.ceil(60 - state.supplyConvoyTimer)) + 's';
  }

  const budgetEl = document.getElementById('budget-display');
  if (budgetEl) {
    const sign = state.policeBudget >= 0 ? '' : '-';
    const abs = Math.abs(state.policeBudget);
    budgetEl.textContent = sign + '$' + abs.toLocaleString();
    budgetEl.className = 'budget-val ' + (state.policeBudget >= 0 ? 'budget-positive' : 'budget-negative');
  }
}

// =========================================================================
// EW Jamming
// =========================================================================
export function updateEWJamming(dt) {
  state.ewJamCheckTimer += dt;
  if (state.ewJamActive) {
    state.ewJamTimer -= dt;
    const remaining = Math.max(0, state.ewJamTimer);
    if (remaining < 5 && !state.ewCounterNarrated) { state.ewCounterNarrated = true; addNarration('ew_counter'); }
    const scale = remaining < 5 ? remaining / 5 : 1;
    if (state.ewRingMesh) {
      state.ewRingMesh.scale.set(scale, 1, scale);
      state.ewRingMesh.position.set(state.ewJamX, 0.12, state.ewJamZ);
      state.ewRingMesh.visible = true;
    }
    state.commsStatus = 'JAMMED';
    if (state.ewJamTimer <= 0) { state.ewJamActive = false; if (state.ewRingMesh) state.ewRingMesh.visible = false; }
  } else {
    if (state.ewRingMesh) state.ewRingMesh.visible = false;
    if (state.riotPhase === 'RIOT' && state.ewJamCheckTimer >= EW_JAM_CHECK_INTERVAL) {
      state.ewJamCheckTimer = 0;
      if (rng() < EW_JAM_CHANCE) {
        state.ewJamActive = true; state.ewJamTimer = EW_JAM_DURATION; state.ewCounterNarrated = false;
        state.ewJamX = PLAZA_X + (rng() - 0.5) * 20; state.ewJamZ = PLAZA_Z + (rng() - 0.5) * 20;
        addNarration('ew_jam_start', { grid: formatGrid(state.ewJamX, state.ewJamZ) });
      }
    }
  }
}

// =========================================================================
// Comms Network
// =========================================================================
export function updateCommsNetwork(dt) {
  state.commsDegradedTimer = Math.max(0, state.commsDegradedTimer - dt);
  let nearFire = false;
  for (const cop of state.police) {
    if (!cop.alive) continue;
    for (const f of state.fires) {
      const fd = Math.sqrt((cop.x - f.x) ** 2 + (cop.z - f.z) ** 2);
      if (fd < 15) { nearFire = true; break; }
    }
    if (nearFire) break;
  }
  if (nearFire) state.commsDegradedTimer = 5;
  state.commsStatus = state.commsDegradedTimer > 0 ? 'DEGRADED' : 'ACTIVE';
  const el = document.getElementById('comms-status');
  if (el) {
    el.textContent = state.commsStatus;
    el.style.color = state.commsStatus === 'JAMMED' ? '#ff2a6d' : state.commsStatus === 'ACTIVE' ? '#05ffa1' : '#fcee0a';
  }

  if (state.commsLines) state.commsLines.visible = state.debugMode;
  if (!state.debugMode) return;
  let idx = 0;
  state.commsLinkCount = 0;
  const aliveCops = state.police.filter(p => p.alive);
  const maxLinks = state.commsLinePositions ? state.commsLinePositions.length / 6 : 0;
  for (let i = 0; i < aliveCops.length && idx < maxLinks; i++) {
    for (let j = i + 1; j < aliveCops.length && idx < maxLinks; j++) {
      const a = aliveCops[i], b = aliveCops[j];
      const dd = Math.sqrt((a.x - b.x) ** 2 + (a.z - b.z) ** 2);
      if (dd > COMMS_RANGE) continue;
      const bright = 1 - dd / COMMS_RANGE;
      const base = idx * 6;
      state.commsLinePositions[base] = a.x; state.commsLinePositions[base+1] = 3; state.commsLinePositions[base+2] = a.z;
      state.commsLinePositions[base+3] = b.x; state.commsLinePositions[base+4] = 3; state.commsLinePositions[base+5] = b.z;
      state.commsLineColors[base] = 0; state.commsLineColors[base+1] = bright * 0.94; state.commsLineColors[base+2] = bright;
      state.commsLineColors[base+3] = 0; state.commsLineColors[base+4] = bright * 0.94; state.commsLineColors[base+5] = bright;
      idx++;
      state.commsLinkCount++;
    }
  }
  if (state.commsLineGeo) {
    state.commsLineGeo.setDrawRange(0, idx * 2);
    state.commsLineGeo.attributes.position.needsUpdate = true;
    state.commsLineGeo.attributes.color.needsUpdate = true;
  }
}

// =========================================================================
// Fog of War
// =========================================================================
export function updateFogOfWar(dt) {
  state.fogTimer += dt;
  if (state.fogTimer < FOG_UPDATE_INTERVAL) return;
  state.fogTimer = 0;
  if (!state.fogOfWarEnabled) {
    if (state.fogMesh) state.fogMesh.visible = false;
    if (state.detectionLines) state.detectionLines.visible = false;
    return;
  }
  if (state.fogMesh) state.fogMesh.visible = true;
  for (let i = 0; i < state.fogGrid.length; i++) {
    if (state.fogGrid[i] === 2) state.fogGrid[i] = 1;
  }
  const revealCells = Math.ceil(DETECTION_RANGE / FOG_CELL_SIZE);
  for (const cop of state.police) {
    if (!cop.alive) continue;
    const cx = Math.floor(cop.x / FOG_CELL_SIZE);
    const cz = Math.floor(cop.z / FOG_CELL_SIZE);
    for (let dz = -revealCells; dz <= revealCells; dz++) {
      for (let dx = -revealCells; dx <= revealCells; dx++) {
        const gx = cx + dx;
        const gz = cz + dz;
        if (gx < 0 || gx >= FOG_COLS || gz < 0 || gz >= FOG_ROWS) continue;
        const worldX = (gx + 0.5) * FOG_CELL_SIZE;
        const worldZ = (gz + 0.5) * FOG_CELL_SIZE;
        const dd = Math.sqrt((worldX - cop.x) ** 2 + (worldZ - cop.z) ** 2);
        if (dd <= DETECTION_RANGE) state.fogGrid[gz * FOG_COLS + gx] = 2;
      }
    }
  }
  if (state.fogCtx) {
    const imgData = state.fogCtx.createImageData(FOG_COLS, FOG_ROWS);
    for (let i = 0; i < state.fogGrid.length; i++) {
      const base = i * 4;
      const s = state.fogGrid[i];
      if (s === 0) { imgData.data[base]=5; imgData.data[base+1]=5; imgData.data[base+2]=10; imgData.data[base+3]=Math.round(0.7*255); }
      else if (s === 1) { imgData.data[base]=5; imgData.data[base+1]=5; imgData.data[base+2]=15; imgData.data[base+3]=Math.round(0.3*255); }
      else { imgData.data[base]=0; imgData.data[base+1]=0; imgData.data[base+2]=0; imgData.data[base+3]=0; }
    }
    state.fogCtx.putImageData(imgData, 0, 0);
    if (state.fogTexture) state.fogTexture.needsUpdate = true;
  }

  state.detectedCount = 0;
  let lineIdx = 0;
  if (state.debugMode && state.fogOfWarEnabled && state.detectionLinePositions) {
    for (const cop of state.police) {
      if (!cop.alive) continue;
      for (const prot of state.protestors) {
        if (!prot.alive || prot.arrested) continue;
        const ddx = prot.x - cop.x;
        const ddz = prot.z - cop.z;
        const dd = Math.sqrt(ddx * ddx + ddz * ddz);
        if (dd <= DETECTION_RANGE && lineIdx < state.detectionLinePositions.length / 6 && hasLineOfSight(cop.x, cop.z, prot.x, prot.z)) {
          const lb = lineIdx * 6;
          state.detectionLinePositions[lb]=cop.x; state.detectionLinePositions[lb+1]=1.5; state.detectionLinePositions[lb+2]=cop.z;
          state.detectionLinePositions[lb+3]=prot.x; state.detectionLinePositions[lb+4]=1.5; state.detectionLinePositions[lb+5]=prot.z;
          lineIdx++;
          state.detectedCount++;
        }
      }
    }
  }
  if (state.detectionLineGeo) {
    state.detectionLineGeo.setDrawRange(0, lineIdx * 2);
    state.detectionLineGeo.attributes.position.needsUpdate = true;
  }
  if (state.detectionLines) state.detectionLines.visible = state.debugMode && state.fogOfWarEnabled && lineIdx > 0;
}

// =========================================================================
// Territory Control
// =========================================================================
export function updateTerritoryControl(dt) {
  state.territoryTimer += dt;
  if (state.territoryTimer < TERRITORY_UPDATE_INTERVAL) return;
  state.territoryTimer = 0;
  const showZones = state.riotMode;
  for (const zone of state.territoryZones) {
    zone.plane.visible = showZones;
    zone.border.visible = showZones;
    if (!showZones) continue;
    zone.policeInZone = 0;
    zone.protestorsInZone = 0;
    for (const pol of state.police) {
      if (!pol.alive) continue;
      if (pol.x >= zone.minX && pol.x <= zone.maxX && pol.z >= zone.minZ && pol.z <= zone.maxZ) zone.policeInZone++;
    }
    for (const pr of state.protestors) {
      if (!pr.alive || pr.arrested || pr.fleeing) continue;
      if (pr.x >= zone.minX && pr.x <= zone.maxX && pr.z >= zone.minZ && pr.z <= zone.maxZ) zone.protestorsInZone++;
    }
    if (zone.policeInZone > zone.protestorsInZone && zone.policeInZone > 0) {
      zone.owner = 'police'; zone.plane.material.color.setHex(0x05ffa1); zone.plane.material.opacity = 0.2;
    } else if (zone.protestorsInZone > zone.policeInZone && zone.protestorsInZone > 0) {
      zone.owner = 'protestor'; zone.plane.material.color.setHex(0xff2a6d); zone.plane.material.opacity = 0.2;
    } else {
      zone.owner = 'neutral'; zone.plane.material.color.setHex(0x888888); zone.plane.material.opacity = 0.15;
    }
  }
}

// =========================================================================
// Objectives
// =========================================================================
export function updateObjectives(dt) {
  if (!state.riotMode) { if (state.objectiveMesh) state.objectiveMesh.count = 0; return; }
  state.objectiveRotation += dt * 0.8;
  if (state.objectiveMesh) state.objectiveMesh.count = state.objectiveDefs.length;
  const securePlaza = state.objectiveDefs[0];
  if (securePlaza && securePlaza.objectiveStatus === 'active') {
    let policeNearPlaza = false;
    for (const pol of state.police) {
      if (!pol.alive) continue;
      if (dist2d(pol, { x: PLAZA_X, z: PLAZA_Z }) < 20) { policeNearPlaza = true; break; }
    }
    if (policeNearPlaza) {
      securePlaza.objectiveStatus = 'complete';
      securePlaza.objectiveColor = 0x05ffa1;
    }
  }
  const holdGround = state.objectiveDefs[1];
  if (holdGround && holdGround.objectiveStatus === 'active') {
    let plazaZone = null;
    for (const zone of state.territoryZones) {
      if (PLAZA_X >= zone.minX && PLAZA_X <= zone.maxX && PLAZA_Z >= zone.minZ && PLAZA_Z <= zone.maxZ) { plazaZone = zone; break; }
    }
    if (plazaZone && plazaZone.owner === 'protestor') {
      state.holdGroundTimer += dt;
      if (state.holdGroundTimer >= 60) { holdGround.objectiveStatus = 'complete'; holdGround.objectiveColor = 0x05ffa1; }
    } else { state.holdGroundTimer = Math.max(0, state.holdGroundTimer - dt * 0.5); }
  }
  const protHosp = state.objectiveDefs[2];
  if (protHosp && protHosp.objectiveStatus === 'active' && state.hospitalBuilding) {
    for (const f of state.fires) {
      if (dist2d(f, state.hospitalBuilding) < 30) {
        protHosp.objectiveStatus = 'failed'; protHosp.objectiveColor = 0xff2a6d; break;
      }
    }
  }
  if (state.objectiveMesh) {
    for (let i = 0; i < state.objectiveDefs.length; i++) {
      const obj = state.objectiveDefs[i];
      _euler.set(0, state.objectiveRotation, 0);
      _quat.setFromEuler(_euler);
      _pos.set(obj.x, 15, obj.z);
      _scale.set(1, 1, 1);
      _mat4.compose(_pos, _quat, _scale);
      state.objectiveMesh.setMatrixAt(i, _mat4);
      state.objectiveMesh.setColorAt(i, _color.set(obj.objectiveColor));
    }
    state.objectiveMesh.instanceMatrix.needsUpdate = true;
    if (state.objectiveMesh.instanceColor) state.objectiveMesh.instanceColor.needsUpdate = true;
  }
}

// =========================================================================
// Narration Panel
// =========================================================================
export function updateNarrationPanel(dt) {
  const panel = document.getElementById('narration-panel');
  if (!panel) return;
  for (let i = state.narrationMessages.length - 1; i >= 0; i--) {
    state.narrationMessages[i].age += dt;
    if (state.narrationMessages[i].age > 10) state.narrationMessages.splice(i, 1);
  }
  if (state.narrationMessages.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  panel.innerHTML = state.narrationMessages.map(m => {
    const fading = m.age > 8 ? ' fading' : '';
    return '<div class="narration-msg' + fading + '">' + m.html + '</div>';
  }).join('');
}
