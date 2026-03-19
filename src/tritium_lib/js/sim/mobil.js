// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * MOBIL Lane Change Model — Minimizing Overall Braking Induced by Lane Changes.
 *
 * MOBIL decides whether a lane change is both SAFE and BENEFICIAL:
 * - Safety: the new follower can still brake within comfortable limits
 * - Incentive: the benefit to me outweighs the cost to others (weighted by politeness)
 *
 * This module works with IDM — it uses IDM accelerations to evaluate
 * the consequences of a lane change for all affected vehicles.
 *
 * Reference: Kesting, Treiber, Helbing (2007)
 * "General Lane-Changing Model MOBIL for Car-Following Models"
 *
 * This is a pure math module with zero rendering dependencies.
 */

import { idmAcceleration } from './idm.js';

/**
 * Default MOBIL parameters.
 * @typedef {Object} MOBILParams
 * @property {number} politeness - Politeness factor (0=selfish, 0.5=realistic). Default 0.3
 * @property {number} threshold - Acceleration threshold to prevent frantic hopping (m/s²). Default 0.2
 * @property {number} bSafe - Maximum safe deceleration for new follower (m/s²). Default 4.0
 * @property {number} minGap - Minimum gap required in target lane (m). Default 5.0
 */
export const MOBIL_DEFAULTS = {
    politeness: 0.3,
    threshold: 0.2,
    bSafe: 4.0,
    minGap: 5.0,
};

/**
 * Find the nearest vehicle ahead and behind in a specific lane on a specific road.
 *
 * @param {number} u - Position of the subject vehicle
 * @param {Object} road - Road edge
 * @param {number} lane - Lane index
 * @param {Array} allCars - All vehicles (real + virtual)
 * @param {Object} excludeCar - Vehicle to exclude (self)
 * @returns {{ ahead: Object|null, aheadGap: number, behind: Object|null, behindGap: number }}
 */
export function findNeighborsInLane(u, road, lane, allCars, excludeCar) {
    let aheadGap = Infinity, behindGap = Infinity;
    let ahead = null, behind = null;

    for (const car of allCars) {
        if (car === excludeCar) continue;
        if (car.road !== road) continue;
        if (car.lane !== lane) continue;

        const gap = car.u - u;
        if (gap > 0 && gap < aheadGap) {
            aheadGap = gap;
            ahead = car;
        } else if (gap < 0 && -gap < behindGap) {
            behindGap = -gap;
            behind = car;
        }
    }

    // Subtract car lengths for bumper-to-bumper gaps
    if (ahead) {
        aheadGap = Math.max(0.1, aheadGap - (excludeCar.length || 4) - (ahead.length || 4));
    }
    if (behind) {
        behindGap = Math.max(0.1, behindGap - (excludeCar.length || 4) - (behind.length || 4));
    }

    return { ahead, aheadGap, behind, behindGap };
}

/**
 * Evaluate whether a lane change is safe and beneficial using MOBIL.
 *
 * @param {Object} car - The vehicle considering a lane change
 * @param {number} targetLane - Target lane index
 * @param {Array} allCars - All vehicles on the same road
 * @param {MOBILParams} [params] - MOBIL parameters
 * @returns {{ shouldChange: boolean, incentive: number, reason: string }}
 */
export function evaluateLaneChange(car, targetLane, allCars, params = MOBIL_DEFAULTS) {
    const { politeness, threshold, bSafe, minGap } = params;
    const idmP = car.idmParams;

    // Current situation: me and my current follower
    const currentNeighbors = findNeighborsInLane(car.u, car.road, car.lane, allCars, car);
    const a_c = idmAcceleration(
        car.speed,
        currentNeighbors.aheadGap,
        currentNeighbors.ahead ? currentNeighbors.ahead.speed : car.speed,
        idmP
    );

    // Target lane situation
    const targetNeighbors = findNeighborsInLane(car.u, car.road, targetLane, allCars, car);

    // Safety check: is there enough gap in target lane?
    if (targetNeighbors.aheadGap < minGap || targetNeighbors.behindGap < minGap) {
        return { shouldChange: false, incentive: -Infinity, reason: 'insufficient_gap' };
    }

    // My acceleration in target lane (after changing)
    const a_c_prime = idmAcceleration(
        car.speed,
        targetNeighbors.aheadGap,
        targetNeighbors.ahead ? targetNeighbors.ahead.speed : car.speed,
        idmP
    );

    // New follower's acceleration in target lane (after I insert myself)
    // The new follower is the vehicle behind me in the target lane
    const newFollower = targetNeighbors.behind;
    if (newFollower) {
        const newFollowerIdm = newFollower.idmParams || idmP;

        // New follower's current acceleration (before my change)
        const a_n = idmAcceleration(
            newFollower.speed,
            targetNeighbors.behindGap + (car.length || 4) + (newFollower.length || 4), // gap to car ahead of me
            targetNeighbors.ahead ? targetNeighbors.ahead.speed : newFollower.speed,
            newFollowerIdm
        );

        // New follower's acceleration after I insert (gap is now to me)
        const a_n_prime = idmAcceleration(
            newFollower.speed,
            targetNeighbors.behindGap,
            car.speed,
            newFollowerIdm
        );

        // SAFETY CRITERION: new follower must not brake harder than bSafe
        if (a_n_prime < -bSafe) {
            return { shouldChange: false, incentive: -Infinity, reason: 'unsafe_new_follower' };
        }

        // Old follower's situation (follower in my current lane)
        const oldFollower = currentNeighbors.behind;
        let a_o = 0, a_o_prime = 0;
        if (oldFollower) {
            const oldFollowerIdm = oldFollower.idmParams || idmP;
            a_o = idmAcceleration(
                oldFollower.speed,
                currentNeighbors.behindGap,
                car.speed,
                oldFollowerIdm
            );
            // After I leave, old follower's new leader is my current leader
            const newGapForOld = currentNeighbors.behindGap + (car.length || 4) + currentNeighbors.aheadGap;
            a_o_prime = idmAcceleration(
                oldFollower.speed,
                newGapForOld,
                currentNeighbors.ahead ? currentNeighbors.ahead.speed : oldFollower.speed,
                oldFollowerIdm
            );
        }

        // INCENTIVE CRITERION:
        // My advantage - politeness × (disadvantage to others) > threshold
        const myAdvantage = a_c_prime - a_c;
        const othersDisadvantage = (a_n - a_n_prime) + (a_o - a_o_prime);
        const incentive = myAdvantage - politeness * othersDisadvantage;

        return {
            shouldChange: incentive > threshold,
            incentive,
            reason: incentive > threshold ? 'beneficial' : 'insufficient_incentive',
        };
    }

    // No new follower — only check my advantage
    const incentive = a_c_prime - a_c;
    return {
        shouldChange: incentive > threshold,
        incentive,
        reason: incentive > threshold ? 'beneficial_empty_lane' : 'insufficient_incentive',
    };
}

/**
 * Decide lane change direction for a vehicle.
 * Checks both left and right lanes (if available), picks the better option.
 *
 * @param {Object} car - The vehicle
 * @param {Array} allCars - All vehicles on the road
 * @param {MOBILParams} [params] - MOBIL parameters
 * @returns {{ direction: 'left'|'right'|null, targetLane: number|null, incentive: number }}
 */
export function decideLaneChange(car, allCars, params = MOBIL_DEFAULTS) {
    const n = car.road.numLanesPerDir || 2;
    const isForward = car.lane < n;
    const currentLane = car.lane;

    // Determine adjacent lanes in the same direction
    let leftLane = null, rightLane = null;

    if (isForward) {
        if (currentLane > 0) leftLane = currentLane - 1;       // more toward center
        if (currentLane < n - 1) rightLane = currentLane + 1;  // more toward edge
    } else {
        if (currentLane > n) leftLane = currentLane - 1;
        if (currentLane < 2 * n - 1) rightLane = currentLane + 1;
    }

    let bestDirection = null;
    let bestLane = null;
    let bestIncentive = -Infinity;

    if (leftLane !== null) {
        const result = evaluateLaneChange(car, leftLane, allCars, params);
        if (result.shouldChange && result.incentive > bestIncentive) {
            bestDirection = 'left';
            bestLane = leftLane;
            bestIncentive = result.incentive;
        }
    }

    if (rightLane !== null) {
        const result = evaluateLaneChange(car, rightLane, allCars, params);
        if (result.shouldChange && result.incentive > bestIncentive) {
            bestDirection = 'right';
            bestLane = rightLane;
            bestIncentive = result.incentive;
        }
    }

    return {
        direction: bestDirection,
        targetLane: bestLane,
        incentive: bestIncentive,
    };
}

/**
 * Lane change animation state.
 * During a lane change, the vehicle smoothly transitions between lanes.
 *
 * @param {number} fromLane - Starting lane
 * @param {number} toLane - Target lane
 * @param {number} duration - Duration of lane change in seconds (default 2.0)
 * @returns {Object} Lane change state
 */
export function createLaneChangeState(fromLane, toLane, duration = 2.0) {
    return {
        fromLane,
        toLane,
        t: 0,
        duration,
        active: true,
    };
}

/**
 * Update lane change animation and return interpolated lane position.
 *
 * @param {Object} state - Lane change state from createLaneChangeState
 * @param {number} dt - Time step in seconds
 * @returns {{ lane: number, complete: boolean }}
 */
export function updateLaneChange(state, dt) {
    state.t += dt / state.duration;

    if (state.t >= 1) {
        state.active = false;
        return { lane: state.toLane, complete: true };
    }

    // Smooth interpolation using sine curve (slow at start/end, fast in middle)
    const smoothT = 0.5 - 0.5 * Math.cos(state.t * Math.PI);
    const interpolatedLane = state.fromLane + (state.toLane - state.fromLane) * smoothT;

    return { lane: interpolatedLane, complete: false };
}
