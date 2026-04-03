# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MOBIL -- Minimizing Overall Braking Induced by Lane Changes.

Evaluates whether a lane change is SAFE and BENEFICIAL using IDM
accelerations.  Safety: new follower in target lane can brake within
comfortable limits.  Incentive: driver's gain outweighs weighted cost to
others.

This is the Python port of ``web/sim/mobil.js``.

Reference: Kesting, Treiber, Helbing (2007)
"General Lane-Changing Model MOBIL for Car-Following Models"

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tritium_lib.sim_engine.idm import IDMParams, idm_acceleration

if TYPE_CHECKING:
    from tritium_lib.sim_engine.traffic import TrafficVehicle


@dataclass
class MOBILParams:
    """Parameters for the MOBIL lane-change model.

    Attributes
    ----------
    politeness : float
        Consideration for other drivers.  0 = selfish, 1 = altruistic.
    threshold : float
        Minimum acceleration incentive to trigger a lane change (m/s^2).
    b_safe : float
        Maximum safe deceleration for the new follower (m/s^2, positive).
    min_gap : float
        Minimum gap required in the target lane (m).
    """

    politeness: float = 0.3
    threshold: float = 0.2    # m/s^2
    b_safe: float = 4.0       # m/s^2
    min_gap: float = 5.0      # m


MOBIL_DEFAULTS = MOBILParams()


@dataclass
class LaneNeighbors:
    """Nearest vehicle ahead and behind in a specific lane."""

    ahead: TrafficVehicle | None = None
    ahead_gap: float = float("inf")
    behind: TrafficVehicle | None = None
    behind_gap: float = float("inf")


def find_neighbors_in_lane(
    car: TrafficVehicle,
    target_lane: int,
    nearby_vehicles: list[TrafficVehicle],
) -> LaneNeighbors:
    """Find the nearest vehicle ahead and behind in a specific lane on the
    same edge.

    Parameters
    ----------
    car : TrafficVehicle
        The subject vehicle.
    target_lane : int
        Lane index to search.
    nearby_vehicles : list[TrafficVehicle]
        Vehicles on the same edge.

    Returns
    -------
    LaneNeighbors
        Nearest ahead and behind vehicles with bumper-to-bumper gaps.
    """
    result = LaneNeighbors()

    for other in nearby_vehicles:
        if other is car:
            continue
        if other.edge_id != car.edge_id:
            continue
        if other.direction != car.direction:
            continue
        if other.lane_idx != target_lane:
            continue

        gap = (other.u - car.u) * car.direction
        if gap > 0 and gap < result.ahead_gap:
            result.ahead_gap = gap
            result.ahead = other
        elif gap < 0 and -gap < result.behind_gap:
            result.behind_gap = -gap
            result.behind = other

    # Subtract vehicle half-lengths for bumper-to-bumper gaps
    if result.ahead is not None:
        result.ahead_gap = max(
            0.1,
            result.ahead_gap - car.length / 2 - result.ahead.length / 2,
        )
    if result.behind is not None:
        result.behind_gap = max(
            0.1,
            result.behind_gap - car.length / 2 - result.behind.length / 2,
        )

    return result


@dataclass
class LaneChangeResult:
    """Result of a MOBIL lane-change evaluation."""

    should_change: bool
    incentive: float
    reason: str


def evaluate_lane_change(
    car: TrafficVehicle,
    target_lane: int,
    nearby_vehicles: list[TrafficVehicle],
    params: MOBILParams | None = None,
) -> LaneChangeResult:
    """Evaluate whether a lane change to *target_lane* is safe and
    beneficial.

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle considering a lane change.
    target_lane : int
        Target lane index.
    nearby_vehicles : list[TrafficVehicle]
        All vehicles on the same edge.
    params : MOBILParams, optional
        MOBIL parameters.  Defaults to :data:`MOBIL_DEFAULTS`.

    Returns
    -------
    LaneChangeResult
    """
    if params is None:
        params = MOBIL_DEFAULTS

    idm_p = car.idm

    # Current lane neighbors
    cur = find_neighbors_in_lane(car, car.lane_idx, nearby_vehicles)

    # My current acceleration
    a_c = idm_acceleration(
        car.speed,
        cur.ahead_gap,
        cur.ahead.speed if cur.ahead else car.speed,
        idm_p,
    )

    # Target lane neighbors
    tgt = find_neighbors_in_lane(car, target_lane, nearby_vehicles)

    # Gap check
    if tgt.ahead_gap < params.min_gap or tgt.behind_gap < params.min_gap:
        return LaneChangeResult(
            should_change=False,
            incentive=float("-inf"),
            reason="insufficient_gap",
        )

    # My acceleration in target lane
    a_c_prime = idm_acceleration(
        car.speed,
        tgt.ahead_gap,
        tgt.ahead.speed if tgt.ahead else car.speed,
        idm_p,
    )

    # New follower in target lane
    new_follower = tgt.behind
    if new_follower is not None:
        nf_idm = new_follower.idm

        # New follower's current acceleration (before lane change)
        nf_current_gap = (
            tgt.behind_gap
            + car.length
            + (tgt.ahead_gap if tgt.ahead_gap < float("inf") else 100.0)
        )
        a_n = idm_acceleration(
            new_follower.speed,
            nf_current_gap,
            tgt.ahead.speed if tgt.ahead else new_follower.speed,
            nf_idm,
        )

        # New follower's acceleration after we insert
        a_n_prime = idm_acceleration(
            new_follower.speed,
            tgt.behind_gap,
            car.speed,
            nf_idm,
        )

        # Safety criterion
        if a_n_prime < -params.b_safe:
            return LaneChangeResult(
                should_change=False,
                incentive=float("-inf"),
                reason="unsafe_new_follower",
            )

        # Old follower in current lane
        a_o = 0.0
        a_o_prime = 0.0
        old_follower = cur.behind
        if old_follower is not None:
            of_idm = old_follower.idm
            a_o = idm_acceleration(
                old_follower.speed,
                cur.behind_gap,
                car.speed,
                of_idm,
            )
            # After we leave, old follower's leader becomes our current leader
            new_gap_for_old = cur.behind_gap + car.length + cur.ahead_gap
            a_o_prime = idm_acceleration(
                old_follower.speed,
                new_gap_for_old,
                cur.ahead.speed if cur.ahead else old_follower.speed,
                of_idm,
            )

        # Incentive criterion
        my_advantage = a_c_prime - a_c
        others_disadvantage = (a_n - a_n_prime) + (a_o - a_o_prime)
        incentive = my_advantage - params.politeness * others_disadvantage

        return LaneChangeResult(
            should_change=incentive > params.threshold,
            incentive=incentive,
            reason="beneficial" if incentive > params.threshold else "insufficient_incentive",
        )

    # No new follower -- empty lane, only check my advantage
    incentive = a_c_prime - a_c
    return LaneChangeResult(
        should_change=incentive > params.threshold,
        incentive=incentive,
        reason="beneficial_empty_lane" if incentive > params.threshold else "insufficient_incentive",
    )


@dataclass
class LaneChangeDecision:
    """Best lane change direction for a vehicle."""

    direction: str | None = None      # "left" | "right" | None
    target_lane: int | None = None
    incentive: float = float("-inf")


def decide_lane_change(
    car: TrafficVehicle,
    nearby_vehicles: list[TrafficVehicle],
    num_lanes: int = 1,
    params: MOBILParams | None = None,
) -> LaneChangeDecision:
    """Decide the best lane change direction for a vehicle.

    Checks both adjacent lanes (left and right) and picks the one with
    the highest MOBIL incentive, if any passes the threshold.

    Parameters
    ----------
    car : TrafficVehicle
        The vehicle considering a lane change.
    nearby_vehicles : list[TrafficVehicle]
        Vehicles on the same edge.
    num_lanes : int
        Number of lanes per direction on this edge.
    params : MOBILParams, optional
        MOBIL parameters.

    Returns
    -------
    LaneChangeDecision
    """
    if num_lanes <= 1:
        return LaneChangeDecision()

    best = LaneChangeDecision()
    current_lane = car.lane_idx

    # Check left (lower lane index)
    if current_lane > 0:
        result = evaluate_lane_change(car, current_lane - 1, nearby_vehicles, params)
        if result.should_change and result.incentive > best.incentive:
            best = LaneChangeDecision(
                direction="left",
                target_lane=current_lane - 1,
                incentive=result.incentive,
            )

    # Check right (higher lane index)
    if current_lane < num_lanes - 1:
        result = evaluate_lane_change(car, current_lane + 1, nearby_vehicles, params)
        if result.should_change and result.incentive > best.incentive:
            best = LaneChangeDecision(
                direction="right",
                target_lane=current_lane + 1,
                incentive=result.incentive,
            )

    return best
