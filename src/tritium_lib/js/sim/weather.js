// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Weather simulation logic — pure math, no Three.js.
 *
 * Computes sky color, day/night factors, fog density from sim state.
 * Renderers use these values to update scene visuals.
 */

/**
 * Get sky color hex for a given hour (0-24).
 * Returns a plain integer color (e.g., 0x88aacc).
 */
export function getSkyColorHex(hour) {
  if (hour >= 7 && hour < 17) return 0x88aacc;
  if (hour >= 19 || hour < 5) return 0x0a0a0f;
  // Dawn/dusk transitions return interpolated values
  // These are approximations — the renderer can do smooth lerp if needed
  if (hour >= 5 && hour < 6) return lerpColor(0x0a0a2e, 0xff8844, hour - 5);
  if (hour >= 6 && hour < 7) return lerpColor(0xff8844, 0x4488cc, hour - 6);
  if (hour >= 17 && hour < 18) return lerpColor(0x4488cc, 0xff6622, hour - 17);
  return lerpColor(0xff6622, 0x2a1a3e, hour - 18);
}

/**
 * Compute weather parameters from state for use by renderers.
 */
export function computeWeather(simState) {
  const isNight = simState.isNight;
  const rain = simState.weather.rain;
  const dayFactor = isNight ? 0.2 : 1.0;
  const rainDim = rain ? 0.7 : 1.0;

  let baseFogDensity = isNight ? 0.004 : 0.0015;
  if (rain) baseFogDensity = Math.max(baseFogDensity, 0.004);
  if (simState.weather.fog) baseFogDensity = 0.008;

  const skyColor = rain ? 0x0a0e1a : getSkyColorHex(simState.simTime);

  return {
    isNight,
    dayFactor,
    rainDim,
    ambientIntensity: (0.3 + dayFactor * 0.4) * rainDim,
    sunIntensity: dayFactor * 1.2 * rainDim,
    hemiIntensity: (0.3 + dayFactor * 0.3) * rainDim,
    fogDensity: baseFogDensity,
    skyColor,
    toneMappingExposure: isNight ? 0.9 : 1.4,
    haloOpacity: isNight ? 0.7 : 0.0,
    windowEmissiveIntensity: isNight ? 0.8 : 0.15,
    rainEmissiveIntensity: isNight ? 0.7 : 0.3,
  };
}

/**
 * Check if lightning should strike this frame.
 * Returns true if a new bolt should be spawned.
 */
export function shouldLightning(simState, dt) {
  if (!simState.weather.rain) return false;
  const phase = simState.phase;
  if (phase !== 'RIOT' && phase !== 'DISPERSAL' && phase !== 'COMBAT') return false;
  return Math.random() < 0.02 * dt;
}

// =========================================================================
// Internal helpers
// =========================================================================

function lerpColor(a, b, t) {
  const ar = (a >> 16) & 0xff, ag = (a >> 8) & 0xff, ab = a & 0xff;
  const br = (b >> 16) & 0xff, bg = (b >> 8) & 0xff, bb = b & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return (r << 16) | (g << 8) | bl;
}
