/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  City3D — Audio context, sound synthesis, soundtrack, ambient soundscape.
*/

import { state, rng } from './config.js';

export function ensureAudio() {
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return state.audioCtx;
}

export function playGunshot(x, z) {
  const ctx = ensureAudio();
  const bufSize = ctx.sampleRate * 0.08;
  const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    const env = Math.exp(-i / (bufSize * 0.15));
    data[i] = (Math.random() * 2 - 1) * env * 0.4;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = 0.12;
  src.connect(gain).connect(ctx.destination);
  src.start();
}

export function playCrash(x, z) {
  const ctx = ensureAudio();
  const bufSize = Math.floor(ctx.sampleRate * 0.12);
  const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    const env = Math.exp(-i / (bufSize * 0.2));
    data[i] = (Math.random() * 2 - 1) * env * 0.5;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = 0.6;
  const gain = ctx.createGain();
  gain.gain.value = 0.15;
  src.connect(gain).connect(ctx.destination);
  src.start();
}

export function playExplosion(x, z) {
  const ctx = ensureAudio();
  const dur = 0.5;
  const bufSize = Math.floor(ctx.sampleRate * dur);
  const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    const t = i / ctx.sampleRate;
    const env = Math.exp(-t * 6);
    const low = Math.sin(2 * Math.PI * 40 * t + Math.sin(2 * Math.PI * 15 * t) * 3);
    data[i] = (low * 0.6 + (Math.random() * 2 - 1) * 0.4) * env * 0.3;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = 0.15;
  src.connect(gain).connect(ctx.destination);
  src.start();
}

export function playHiss(x, z) {
  const ctx = ensureAudio();
  const dur = 0.8;
  const bufSize = Math.floor(ctx.sampleRate * dur);
  const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    const t = i / ctx.sampleRate;
    const env = Math.exp(-t * 2);
    data[i] = (Math.random() * 2 - 1) * env * 0.15;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = 0.1;
  src.connect(gain).connect(ctx.destination);
  src.start();
}

export function playSiren() {
  const ctx = ensureAudio();
  const dur = 1.2;
  const bufSize = Math.floor(ctx.sampleRate * dur);
  const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    const t = i / ctx.sampleRate;
    const freq = 600 + 400 * Math.sin(2 * Math.PI * 2 * t);
    data[i] = Math.sin(2 * Math.PI * freq * t) * 0.08 * Math.exp(-t * 0.5);
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = 0.06;
  src.connect(gain).connect(ctx.destination);
  src.start();
}

export function playHorn() {
  const ctx = ensureAudio(), o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = 440; g.gain.value = 0.07;
  o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 0.15);
}

export function initAmbientAudio() {
  const ctx = ensureAudio();
  state.ambientGain = ctx.createGain(); state.ambientGain.gain.value = 0.03; state.ambientGain.connect(ctx.destination);
  state.ambientOsc = ctx.createOscillator(); state.ambientOsc.frequency.value = 80; state.ambientOsc.type = 'sine';
  const oscGain = ctx.createGain(); oscGain.gain.value = 0.4; state.ambientOsc.connect(oscGain).connect(state.ambientGain);
  const nBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), nd = nBuf.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  state.ambientNoise = ctx.createBufferSource(); state.ambientNoise.buffer = nBuf; state.ambientNoise.loop = true;
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 300;
  state.ambientNoise.connect(lp).connect(state.ambientGain); state.ambientOsc.start(); state.ambientNoise.start();
  state.crowdGain = ctx.createGain(); state.crowdGain.gain.value = 0; state.crowdGain.connect(ctx.destination);
  state.crowdFilter = ctx.createBiquadFilter(); state.crowdFilter.type = 'bandpass'; state.crowdFilter.frequency.value = 300; state.crowdFilter.Q.value = 1;
  const cBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), cd2 = cBuf.getChannelData(0);
  for (let i = 0; i < cd2.length; i++) cd2[i] = Math.random() * 2 - 1;
  state.crowdNoise = ctx.createBufferSource(); state.crowdNoise.buffer = cBuf; state.crowdNoise.loop = true;
  state.crowdNoise.connect(state.crowdFilter).connect(state.crowdGain); state.crowdNoise.start();
  state.rainGain = ctx.createGain(); state.rainGain.gain.value = 0; state.rainGain.connect(ctx.destination);
  state.rainFilter = ctx.createBiquadFilter(); state.rainFilter.type = 'highpass'; state.rainFilter.frequency.value = 4000;
  state.rainNoiseSrc = ctx.createBufferSource(); state.rainNoiseSrc.buffer = nBuf; state.rainNoiseSrc.loop = true;
  state.rainNoiseSrc.connect(state.rainFilter).connect(state.rainGain); state.rainNoiseSrc.start();
}

export function updateAmbientAudio(dt) {
  if (!state.ambientOn || !state.ambientGain) return;
  const dayVol = (state.simTime > 7 && state.simTime < 20) ? 0.04 : 0.015;
  state.ambientGain.gain.linearRampToValueAtTime(dayVol, (state.audioCtx ? state.audioCtx.currentTime : 0) + 0.1);
  const isRiot = state.riotPhase === 'RIOT';
  const crowdVol = isRiot ? Math.min(0.06, state.protestors.length * 0.003) : 0;
  state.crowdGain.gain.linearRampToValueAtTime(crowdVol, (state.audioCtx.currentTime) + 0.1);
  const rVol = state.rainActive ? 0.035 : 0;
  state.rainGain.gain.linearRampToValueAtTime(rVol, (state.audioCtx.currentTime) + 0.1);
  for (const car of state.cars) { if (car.stuckTimer > 0 && Math.random() < 0.02 * dt) playHorn(); }
}
