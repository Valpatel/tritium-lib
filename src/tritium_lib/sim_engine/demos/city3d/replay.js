/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Recording and playback of simulation state.
*/

import { state, REPLAY_MAX_FRAMES, REPLAY_CAPTURE_STEP } from './config.js';
import { updatePersonInstance } from './people.js';
import { updateCarInstance } from './vehicles.js';

export function captureReplayFrame() {
  if (state.replayFrames.length >= REPLAY_MAX_FRAMES) {
    toggleRecording();
    return;
  }
  const frame = {
    simTime: state.simTime,
    tick: state.replayFrames.length,
    pedestrians: state.pedestrians.filter(p => p.alive).map(p => ({
      slot: p.slot, x: p.x, z: p.z, rotY: p.rotY,
      bodyColor: p.bodyColor, headColor: p.headColor, scale: p.scale
    })),
    cars: state.cars.filter(c => c.alive).map(c => ({
      idx: c.idx, x: c.x, z: c.z, horizontal: c.horizontal, dir: c.dir,
      bodyColor: c.bodyColor, cabinColor: c.cabinColor
    })),
    protestors: state.protestors.filter(p => p.alive).map(p => ({
      slot: p.slot, x: p.x, z: p.z, rotY: p.rotY,
      bodyColor: p.bodyColor, headColor: p.headColor, scale: p.scale
    })),
    police: state.police.filter(p => p.alive).map(p => ({
      slot: p.slot, x: p.x, z: p.z, rotY: p.rotY,
      bodyColor: p.bodyColor, headColor: p.headColor, scale: p.scale
    })),
    taxis: state.taxis.filter(t => t.alive !== false).map(t => ({
      idx: t.idx, x: t.x, z: t.z, horizontal: t.horizontal, dir: t.dir,
      bodyColor: t.bodyColor, cabinColor: t.cabinColor
    })),
    carDrivers: state.carDrivers.filter(cd => cd.alive !== false).map(cd => ({
      personSlot: cd.personSlot, x: cd.x, z: cd.z, rotY: cd.rotY,
      bodyColor: cd.bodyColor, headColor: cd.headColor, scale: cd.scale,
      phase: cd.phase
    })),
  };
  state.replayFrames.push(frame);
}

export function applyReplayFrame(frame) {
  for (const p of frame.pedestrians) {
    updatePersonInstance(p.slot, p.x, 0, p.z, p.rotY, p.bodyColor, p.headColor, p.scale);
  }
  for (const c of frame.cars) {
    const rotY = c.horizontal ? (c.dir > 0 ? 0 : Math.PI) : (c.dir > 0 ? Math.PI / 2 : -Math.PI / 2);
    updateCarInstance(c.idx, c.x, 0, c.z, rotY, c.bodyColor, c.cabinColor);
  }
  for (const p of frame.protestors) {
    updatePersonInstance(p.slot, p.x, 0, p.z, p.rotY, p.bodyColor, p.headColor, p.scale);
  }
  for (const p of frame.police) {
    updatePersonInstance(p.slot, p.x, 0, p.z, p.rotY, p.bodyColor, p.headColor, p.scale);
  }
  for (const t of frame.taxis) {
    const rotY = t.horizontal ? (t.dir > 0 ? 0 : Math.PI) : (t.dir > 0 ? Math.PI / 2 : -Math.PI / 2);
    updateCarInstance(t.idx, t.x, 0, t.z, rotY, t.bodyColor, t.cabinColor);
  }
  for (const cd of frame.carDrivers) {
    if (cd.phase === 'walking_to_car' || cd.phase === 'walking_to_dest') {
      updatePersonInstance(cd.personSlot, cd.x, 0, cd.z, cd.rotY, cd.bodyColor, cd.headColor, cd.scale);
    }
  }
}

export function toggleRecording() {
  const btn = document.getElementById('rec-btn');
  const timer = document.getElementById('rec-timer');
  if (state.replayPlayback) return;
  state.replayRecording = !state.replayRecording;
  if (state.replayRecording) {
    state.replayFrames = [];
    state.replayRecordAccum = 0;
    state.replayRecStartTime = performance.now();
    btn.classList.add('recording');
  } else {
    btn.classList.remove('recording');
    timer.style.display = 'none';
    if (state.replayFrames.length > 0) showPlaybackBar();
  }
}

function showPlaybackBar() {
  document.getElementById('playback-bar').classList.add('visible');
  updatePlaybackHUD();
}

function hidePlaybackBar() {
  document.getElementById('playback-bar').classList.remove('visible');
}

export function togglePlayback() {
  if (state.replayFrames.length === 0) return;
  if (state.replayRecording) toggleRecording();
  state.replayPlayback = !state.replayPlayback;
  state.replayAccum = 0;
  if (state.replayPlayback) {
    if (state.replayIdx >= state.replayFrames.length - 1) state.replayIdx = 0;
    showPlaybackBar();
  }
  document.getElementById('pb-play').textContent = state.replayPlayback ? 'PAUSE' : 'PLAY';
}

export function stopPlayback() {
  state.replayPlayback = false;
  state.replayIdx = 0;
  document.getElementById('pb-play').textContent = 'PLAY';
  hidePlaybackBar();
}

export function setPlaybackSpeed(speed) {
  state.replaySpeed = speed;
  const buttons = document.getElementById('playback-bar').querySelectorAll('button');
  buttons.forEach(b => {
    if (b.textContent.endsWith('x') && parseFloat(b.textContent) === speed) b.classList.add('active');
    else if (b.textContent.endsWith('x')) b.classList.remove('active');
  });
}

export function seekPlayback(event) {
  if (state.replayFrames.length === 0) return;
  const bar = document.getElementById('playback-scrubber');
  const rect = bar.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  state.replayIdx = Math.floor(pct * (state.replayFrames.length - 1));
  applyReplayFrame(state.replayFrames[state.replayIdx]);
  updatePlaybackHUD();
}

export function updatePlaybackHUD() {
  const total = state.replayFrames.length;
  if (total === 0) return;
  const pct = total > 1 ? (state.replayIdx / (total - 1)) * 100 : 0;
  document.getElementById('playback-progress').style.width = pct + '%';
  document.getElementById('playback-frame').textContent = (state.replayIdx + 1) + '/' + total;
}

export function updateRecordingTimer() {
  if (!state.replayRecording) return;
  const elapsed = (performance.now() - state.replayRecStartTime) / 1000;
  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60);
  const timer = document.getElementById('rec-timer');
  timer.textContent = String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  timer.style.display = 'inline';
}
