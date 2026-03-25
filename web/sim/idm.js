// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * IDM — Intelligent Driver Model (Treiber 2000).
 *
 * Computes acceleration for a vehicle based on its speed, the gap to
 * the vehicle ahead, and the speed difference. Pure math, no rendering.
 *
 * a_IDM = a * [1 - (v/v0)^delta - (s*(v, Δv) / s)^2]
 * s*(v, Δv) = s0 + max(0, v*T + v*Δv / (2*sqrt(a*b)))
 */

/**
 * @typedef {Object} IDMParams
 * @property {number} v0 — desired speed (m/s)
 * @property {number} a — max acceleration (m/s²)
 * @property {number} b — comfortable deceleration (m/s²)
 * @property {number} s0 — minimum gap when stopped (m)
 * @property {number} T — safe time headway (s)
 * @property {number} delta — acceleration exponent (default 4)
 */

/** Default IDM parameters for a car on a residential road. */
export const IDM_DEFAULTS = {
    v0: 12,    // ~43 km/h
    a: 1.4,
    b: 2.0,
    s0: 2.0,
    T: 1.5,
    delta: 4,
};

/** Speed defaults by road class (m/s). */
export const ROAD_SPEEDS = {
    motorway: 30,      // 108 km/h
    trunk: 25,         // 90 km/h
    primary: 18,       // 65 km/h
    secondary: 15,     // 54 km/h
    tertiary: 13,      // 47 km/h
    residential: 10,   // 36 km/h
    service: 5,        // 18 km/h
    unclassified: 10,
    living_street: 5,
};

/**
 * Compute IDM acceleration.
 *
 * @param {number} v — current speed (m/s)
 * @param {number} gap — bumper-to-bumper distance to leader (m)
 * @param {number} vLeader — leader speed (m/s)
 * @param {IDMParams} params — IDM parameters
 * @returns {number} acceleration (m/s²), negative = braking
 */
export function idmAcceleration(v, gap, vLeader, params) {
    const { v0, a, b, s0, T, delta } = params;

    // Free-road term
    const freeRoad = 1 - Math.pow(v / v0, delta);

    // Desired gap
    const dv = v - vLeader;
    const sStar = s0 + Math.max(0, v * T + (v * dv) / (2 * Math.sqrt(a * b)));

    // Interaction term — clamp gap to prevent division by zero
    const interaction = (sStar / Math.max(gap, 0.5)) ** 2;

    // Clamp to physical limits (-9 m/s² ≈ 1g braking)
    return Math.max(-9.0, Math.min(a, a * (freeRoad - interaction)));
}

/**
 * Free-flow acceleration (no leader ahead).
 */
export function idmFreeFlow(v, params) {
    const { v0, a, delta } = params;
    return a * (1 - Math.pow(v / v0, delta));
}

/**
 * Update speed and position using IDM.
 *
 * @param {number} v — current speed
 * @param {number} acc — IDM acceleration
 * @param {number} dt — timestep
 * @returns {{ v: number, ds: number }} new speed and distance traveled
 */
export function idmStep(v, acc, dt) {
    const newV = Math.max(0, v + acc * dt);
    const ds = Math.max(0, v * dt + 0.5 * acc * dt * dt);
    return { v: newV, ds };
}
