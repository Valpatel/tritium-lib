/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — NPC/Vehicle click-to-inspect panel with identity display.

  Raycasts against personBodyMesh and carBodyMesh on click, finds the
  nearest entity, and shows a cyberpunk-styled identity tooltip panel.
*/

import * as THREE from 'three';
import { state, _color } from './config.js';
import { buildIdentity } from '../../../js/sim/identity.js';

// =========================================================================
// Identity Cache — computed once per entity, stored for reuse
// =========================================================================
const identityCache = new Map();

function getIdentity(entityType, index) {
  const key = entityType + '_' + index;
  if (!identityCache.has(key)) {
    identityCache.set(key, buildIdentity(key, entityType));
  }
  return identityCache.get(key);
}

// =========================================================================
// Panel DOM — created once, shown/hidden as needed
// =========================================================================
let panel = null;
let selectedEntity = null;
let selectedEntityType = null;

function createPanel() {
  panel = document.createElement('div');
  panel.id = 'inspect-panel';
  panel.style.cssText = [
    'position:absolute',
    'display:none',
    'z-index:50',
    'background:rgba(10,10,15,0.94)',
    'border:1px solid #00f0ff',
    'border-radius:4px',
    'padding:10px 14px',
    'font-family:Courier New,monospace',
    'font-size:12px',
    'color:#ccc',
    'line-height:1.7',
    'min-width:260px',
    'max-width:320px',
    'pointer-events:none',
    'box-shadow:0 0 16px #00f0ff22,inset 0 0 8px #00f0ff08',
    'backdrop-filter:blur(4px)',
  ].join(';');
  document.getElementById('hud').appendChild(panel);
  return panel;
}

function showPanel(x, y, html) {
  if (!panel) createPanel();
  panel.innerHTML = html;
  panel.style.display = 'block';

  // Position near click, but keep on screen
  const pw = panel.offsetWidth || 280;
  const ph = panel.offsetHeight || 200;
  let px = x + 16;
  let py = y - ph / 2;
  if (px + pw > window.innerWidth - 8) px = x - pw - 16;
  if (py < 8) py = 8;
  if (py + ph > window.innerHeight - 8) py = window.innerHeight - ph - 8;
  panel.style.left = px + 'px';
  panel.style.top = py + 'px';
}

function hidePanel() {
  if (panel) panel.style.display = 'none';
  selectedEntity = null;
  selectedEntityType = null;
}

// =========================================================================
// HTML builders
// =========================================================================

function personHTML(identity, extraInfo) {
  const morale = extraInfo.morale != null ? extraInfo.morale : 1;
  const moraleColor = morale > 0.6 ? '#05ffa1' : morale > 0.3 ? '#fcee0a' : '#ff2a6d';
  const roleLabel = extraInfo.role || 'CIVILIAN';
  const roleColor = roleLabel === 'POLICE' ? '#00f0ff' :
    roleLabel === 'PROTESTOR' ? '#ff2a6d' : '#05ffa1';

  return `
    <div style="color:${roleColor};font-weight:bold;font-size:15px;margin-bottom:4px;
      text-shadow:0 0 6px ${roleColor}44">${identity.fullName}</div>
    <div style="color:#666;font-size:10px;margin-bottom:6px">${roleLabel} | ID: ${identity.shortId}</div>
    <div style="border-top:1px solid #00f0ff33;padding-top:6px">
      <span style="color:#fcee0a99">Phone:</span>
      <span style="color:#fcee0a">${identity.phoneModel}</span><br>
      <span style="color:#fcee0a99">BLE:</span>
      <span style="color:#05ffa1;font-size:11px">${identity.bluetoothMac}</span><br>
      <span style="color:#fcee0a99">WiFi:</span>
      <span style="color:#05ffa1;font-size:11px">${identity.wifiMac}</span><br>
      <span style="color:#fcee0a99">Employer:</span>
      <span style="color:#ccc">${identity.employer}</span><br>
      <span style="color:#fcee0a99">Home:</span>
      <span style="color:#ccc;font-size:11px">${identity.homeAddress}</span>
    </div>
    <div style="margin-top:6px;border-top:1px solid #00f0ff33;padding-top:4px">
      <span style="color:#fcee0a99">Morale:</span>
      <div style="display:inline-block;width:80px;height:6px;background:#1a1a2e;
        border-radius:2px;border:1px solid #ffffff22;vertical-align:middle;margin-left:4px">
        <div style="height:100%;width:${(morale * 100)}%;background:${moraleColor};
          border-radius:2px"></div>
      </div>
    </div>`;
}

function vehicleHTML(identity) {
  const plateColor = '#fcee0a';

  return `
    <div style="color:#00f0ff;font-weight:bold;font-size:14px;margin-bottom:2px;
      text-shadow:0 0 6px #00f0ff44">${identity.vehicleDesc}</div>
    <div style="color:${plateColor};font-size:13px;font-weight:bold;margin-bottom:6px;
      letter-spacing:2px">${identity.vehicleColor}</div>
    <div style="border-top:1px solid #00f0ff33;padding-top:6px">
      <span style="color:#fcee0a99">Plate:</span>
      <span style="color:#fcee0a;font-weight:bold;letter-spacing:1px">${identity.licensePlate}</span><br>
      <span style="color:#fcee0a99">TPMS:</span>
      <span style="color:#05ffa1">315 MHz (4 sensors)</span><br>
      <span style="color:#fcee0a99">Owner:</span>
      <span style="color:#ccc">${identity.ownerName}</span><br>
      <span style="color:#fcee0a99">Phone:</span>
      <span style="color:#ccc">${identity.phoneModel}</span><br>
      <span style="color:#fcee0a99">BLE:</span>
      <span style="color:#05ffa1;font-size:11px">${identity.bluetoothMac}</span>
    </div>`;
}

// =========================================================================
// Raycasting
// =========================================================================
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const intersectPt = new THREE.Vector3();

/**
 * Initialize the click-to-inspect system. Call after all meshes are created.
 */
export function initInspect() {
  if (!panel) createPanel();

  state.renderer.domElement.addEventListener('click', onClickInspect);
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') hidePanel();
  });
}

function onClickInspect(ev) {
  mouse.x = (ev.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(ev.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, state.camera);
  raycaster.ray.intersectPlane(groundPlane, intersectPt);
  if (!intersectPt) { hidePanel(); return; }

  const clickX = intersectPt.x;
  const clickZ = intersectPt.z;

  // Search pedestrians
  let bestDist = 25; // 5^2
  let bestEntity = null;
  let bestType = null;
  let bestIndex = -1;

  for (let i = 0; i < state.pedestrians.length; i++) {
    const p = state.pedestrians[i];
    if (!p.alive) continue;
    const dd = (p.x - clickX) ** 2 + (p.z - clickZ) ** 2;
    if (dd < bestDist) { bestDist = dd; bestEntity = p; bestType = 'person'; bestIndex = i; }
  }

  // Search protestors
  for (let i = 0; i < state.protestors.length; i++) {
    const p = state.protestors[i];
    if (!p.alive || p.arrested) continue;
    const dd = (p.x - clickX) ** 2 + (p.z - clickZ) ** 2;
    if (dd < bestDist) { bestDist = dd; bestEntity = p; bestType = 'protestor'; bestIndex = i; }
  }

  // Search police
  for (let i = 0; i < state.police.length; i++) {
    const p = state.police[i];
    if (!p.alive) continue;
    const dd = (p.x - clickX) ** 2 + (p.z - clickZ) ** 2;
    if (dd < bestDist) { bestDist = dd; bestEntity = p; bestType = 'police'; bestIndex = i; }
  }

  // Search cars
  for (let i = 0; i < state.cars.length; i++) {
    const c = state.cars[i];
    if (c.alive === false) continue;
    const dd = (c.x - clickX) ** 2 + (c.z - clickZ) ** 2;
    if (dd < bestDist) { bestDist = dd; bestEntity = c; bestType = 'car'; bestIndex = i; }
  }

  // Search taxis
  for (let i = 0; i < state.taxis.length; i++) {
    const c = state.taxis[i];
    if (c.alive === false) continue;
    const dd = (c.x - clickX) ** 2 + (c.z - clickZ) ** 2;
    if (dd < bestDist) { bestDist = dd; bestEntity = c; bestType = 'taxi'; bestIndex = i; }
  }

  if (!bestEntity) { hidePanel(); return; }

  selectedEntity = bestEntity;
  selectedEntityType = bestType;

  // Build identity and show panel
  let html;
  if (bestType === 'car' || bestType === 'taxi') {
    const prefix = bestType === 'taxi' ? 'taxi' : 'car';
    const identity = getIdentity('vehicle', bestIndex);
    html = vehicleHTML(identity);
  } else {
    // Person types: map to a stable entity ID
    let entityPrefix;
    if (bestType === 'protestor') entityPrefix = 'protestor';
    else if (bestType === 'police') entityPrefix = 'police';
    else entityPrefix = 'ped';

    const identity = getIdentity('person', entityPrefix + '_' + bestIndex);
    const role = bestType === 'police' ? 'POLICE' :
      bestType === 'protestor' ? 'PROTESTOR' : 'CIVILIAN';
    const morale = bestEntity.morale != null ? bestEntity.morale : 1;
    html = personHTML(identity, { role, morale });
  }

  showPanel(ev.clientX, ev.clientY, html);
}

/**
 * Update the inspect panel position if following a selected entity.
 * Call from the render loop if you want the panel to track the entity.
 */
export function updateInspect() {
  if (!selectedEntity || !panel || panel.style.display === 'none') return;

  // Auto-hide if entity died or was arrested
  if (selectedEntity.alive === false ||
    (selectedEntity.arrested === true)) {
    hidePanel();
    return;
  }
}

/**
 * Dismiss the inspect panel (e.g. from ESC handler).
 */
export function dismissInspect() {
  hidePanel();
}
