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
  if (hour >= 8 && hour < 17) return 0x87CEEB;       // day: sky blue
  if (hour >= 21 || hour < 4) return 0x0a0a18;       // night: dark blue-black
  // Dawn (4-8): dark → warm peach → blue
  if (hour >= 4 && hour < 5.5) return lerpColor(0x0a0a18, 0x1a1a3e, (hour - 4) / 1.5);
  if (hour >= 5.5 && hour < 6.5) return lerpColor(0x1a1a3e, 0xc4956a, (hour - 5.5));
  if (hour >= 6.5 && hour < 8) return lerpColor(0xc4956a, 0x87CEEB, (hour - 6.5) / 1.5);
  // Dusk (17-21): blue → warm peach → deep blue → dark
  if (hour >= 17 && hour < 18.5) return lerpColor(0x87CEEB, 0xc4856a, (hour - 17) / 1.5);
  if (hour >= 18.5 && hour < 19.5) return lerpColor(0xc4856a, 0x3a2a4e, (hour - 18.5));
  return lerpColor(0x3a2a4e, 0x0a0a18, (hour - 19.5) / 1.5);
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
