/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Keyboard handlers, camera controls, playback speed.
*/

import { state, rng } from './config.js';
import { ensureAudio, initAmbientAudio } from './audio.js';
import { toggleRecording, togglePlayback } from './replay.js';

// Note: The actual keyboard handler and riot start/stop logic that references
// many other modules (spawnProtestors, spawnPolice, createHelicopter, etc.)
// is wired up in the main HTML module script to avoid circular imports.
// This file exports the handler setup function that the main module calls.

export function setupInputHandlers(handlers) {
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      if (handlers.onSpace) handlers.onSpace();
    }
    if (e.code === 'KeyC') {
      state.chaseCam = !state.chaseCam;
      if (state.chaseCam) {
        state.controls.enabled = false;
        const all = [...state.pedestrians.filter(p => p.alive), ...state.protestors.filter(p => p.alive), ...state.police.filter(p => p.alive)];
        state.chaseTarget = all.length > 0 ? all[Math.floor(rng() * all.length)] : null;
      } else {
        state.controls.enabled = true;
      }
    }
    if (e.code === 'KeyN') { state.nightMode = !state.nightMode; }
    if (e.code === 'KeyD') {
      state.debugMode = !state.debugMode;
      document.getElementById('debug-overlay').classList.toggle('visible', state.debugMode);
      document.getElementById('debug-canvas').classList.toggle('visible', state.debugMode);
    }
    if (e.code === 'KeyR') {
      state.rainActive = !state.rainActive;
      if (!state.rainActive && state.rainMesh) state.rainMesh.count = 0;
    }
    if (e.code === 'KeyS') {
      state.ambientOn = !state.ambientOn;
      if (state.ambientOn && !state.ambientGain) initAmbientAudio();
      if (!state.ambientOn && state.ambientGain) state.ambientGain.gain.value = 0;
    }
    if (e.code === 'KeyF') { state.fogOverride = !state.fogOverride; }
    if (e.code === 'KeyI') {
      state.fogOfWarEnabled = !state.fogOfWarEnabled;
      if (!state.fogOfWarEnabled) state.fogGrid.fill(0);
    }
    if (e.code === 'KeyM') {
      state.minimapVisible = !state.minimapVisible;
      document.getElementById('minimap-canvas').style.display = state.minimapVisible ? 'block' : 'none';
    }
    if (e.code === 'Escape') {
      if (handlers.onDeselect) handlers.onDeselect();
    }
    if (e.code === 'KeyK') { toggleRecording(); }
    if (e.code === 'KeyL') { togglePlayback(); }
    if (e.code === 'KeyA') {
      ensureAudio();
      if (handlers.onArtillery) handlers.onArtillery();
    }
    if (e.code === 'KeyP') { if (handlers.onAdvanceCampaign) handlers.onAdvanceCampaign(); }
    if (e.code === 'Digit2') {
      state.splitViewMode = !state.splitViewMode;
      document.getElementById('split-view').classList.toggle('active', state.splitViewMode);
    }
  });

  // Resize handler
  window.addEventListener('resize', () => {
    state.camera.aspect = innerWidth / innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(innerWidth, innerHeight);
  });
}
