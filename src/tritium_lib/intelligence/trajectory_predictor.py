# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TrajectoryPredictor — predict where a target is heading.

Combines Kalman filter state estimation, road-network constraints,
learned daily routines, and group (convoy/flock) movement to produce
multi-horizon trajectory predictions with confidence scores.

Integrates with:
  - KalmanPredictor   — velocity + acceleration state estimation
  - StreetGraph       — road-constrained path projection
  - BehavioralPatternLearner — learned daily routines and frequent zones
  - ConvoyDetector    — group / flock movement

Usage::

    from tritium_lib.tracking import TargetHistory, StreetGraph
    from tritium_lib.intelligence.trajectory_predictor import TrajectoryPredictor

    history = TargetHistory()
    predictor = TrajectoryPredictor(history=history)

    # Feed positions, then predict:
    preds = predictor.predict("ble_aa:bb:cc", horizon_minutes=5)
    dest = predictor.predict_destination("ble_aa:bb:cc")
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Prediction:
    """A single predicted future position with confidence and time horizon."""

    x: float
    y: float
    horizon_minutes: float
    confidence: float          # 0.0 to 1.0
    cone_radius_m: float       # uncertainty radius in meters
    heading_deg: float = 0.0   # predicted heading (compass, 0=north)
    speed_mps: float = 0.0     # predicted speed in m/s
    model: str = ""            # which model produced this prediction
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "horizon_minutes": round(self.horizon_minutes, 2),
            "confidence": round(self.confidence, 3),
            "cone_radius_m": round(self.cone_radius_m, 1),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DestinationPrediction:
    """A predicted destination (endpoint) for a target."""

    x: float
    y: float
    confidence: float        # 0.0 to 1.0
    label: str = ""          # e.g. "home", "work", "zone_3"
    estimated_arrival_s: float = 0.0  # seconds until arrival
    model: str = ""          # which model produced this
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "confidence": round(self.confidence, 3),
            "label": self.label,
            "estimated_arrival_s": round(self.estimated_arrival_s, 1),
            "model": self.model,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Prediction model ABC
# ---------------------------------------------------------------------------

class PredictionModel(ABC):
    """Base class for pluggable trajectory prediction algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model."""

    @abstractmethod
    def predict(
        self,
        target_id: str,
        horizon_minutes: float,
        context: PredictionContext,
    ) -> list[Prediction]:
        """Produce predictions for a target at the given horizon.

        Args:
            target_id: Unique target identifier.
            horizon_minutes: How far ahead to predict (in minutes).
            context: Shared prediction context with history, graphs, etc.

        Returns:
            List of Prediction objects (may be empty if model cannot predict).
        """

    def predict_destination(
        self,
        target_id: str,
        context: PredictionContext,
    ) -> DestinationPrediction | None:
        """Predict the most likely destination (endpoint) for a target.

        Default implementation returns None. Override in subclasses.
        """
        return None


# ---------------------------------------------------------------------------
# Prediction context — shared state passed to models
# ---------------------------------------------------------------------------

@dataclass
class PredictionContext:
    """Shared context passed to all prediction models.

    Bundles references to external systems so models do not need to
    import or hold direct references themselves.
    """

    history: Any = None                # TargetHistory
    street_graph: Any = None           # StreetGraph
    behavioral_learner: Any = None     # BehavioralPatternLearner
    convoy_detector: Any = None        # ConvoyDetector
    kalman_states: dict = field(default_factory=dict)  # target_id -> KalmanState


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------

# Minimum number of trail points for any prediction
_MIN_TRAIL_POINTS = 3
# Minimum speed (m/s) for a target to be considered moving
_MIN_SPEED_MPS = 0.3
# Base confidence for a 1-minute prediction (exponential decay with horizon)
_BASE_CONFIDENCE = 0.85
# Cone growth rate (meters per minute)
_CONE_GROWTH_RATE = 10.0


class LinearExtrapolation(PredictionModel):
    """Simple velocity extrapolation from recent movement history.

    Fits a velocity vector from the last N positions and projects
    forward linearly. No road constraints or behavioral awareness.
    """

    @property
    def name(self) -> str:
        return "linear"

    def predict(
        self,
        target_id: str,
        horizon_minutes: float,
        context: PredictionContext,
    ) -> list[Prediction]:
        if context.history is None:
            return []

        trail = context.history.get_trail(target_id, max_points=20)
        if len(trail) < _MIN_TRAIL_POINTS:
            return []

        # Use last two points for velocity
        x0, y0, t0 = trail[-2]
        x1, y1, t1 = trail[-1]
        dt = t1 - t0
        if dt <= 0:
            return []

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = math.hypot(vx, vy)
        if speed < _MIN_SPEED_MPS:
            return []

        heading = math.degrees(math.atan2(vx, vy)) % 360

        dt_s = horizon_minutes * 60.0
        pred_x = x1 + vx * dt_s
        pred_y = y1 + vy * dt_s

        confidence = _BASE_CONFIDENCE * math.exp(-0.1 * horizon_minutes)
        confidence = max(0.05, confidence)

        # Velocity variance widens the cone
        cone = _CONE_GROWTH_RATE * horizon_minutes
        if len(trail) >= 4:
            speeds = []
            for i in range(1, len(trail)):
                dti = trail[i][2] - trail[i - 1][2]
                if dti > 0:
                    speeds.append(
                        math.hypot(
                            trail[i][0] - trail[i - 1][0],
                            trail[i][1] - trail[i - 1][1],
                        ) / dti
                    )
            if speeds:
                mean_s = sum(speeds) / len(speeds)
                var_s = sum((s - mean_s) ** 2 for s in speeds) / len(speeds)
                cone += math.sqrt(var_s) * dt_s * 0.5

        return [Prediction(
            x=pred_x,
            y=pred_y,
            horizon_minutes=horizon_minutes,
            confidence=confidence,
            cone_radius_m=cone,
            heading_deg=heading,
            speed_mps=speed,
            model=self.name,
        )]


class RoadConstrained(PredictionModel):
    """Predict along the road network using StreetGraph.

    Snaps the current position to the nearest road node, then walks
    the road graph in the current heading direction for the predicted
    distance, returning a position constrained to the road network.
    """

    @property
    def name(self) -> str:
        return "road_constrained"

    def predict(
        self,
        target_id: str,
        horizon_minutes: float,
        context: PredictionContext,
    ) -> list[Prediction]:
        if context.history is None or context.street_graph is None:
            return []

        sg = context.street_graph
        if sg.graph is None:
            return []

        trail = context.history.get_trail(target_id, max_points=20)
        if len(trail) < _MIN_TRAIL_POINTS:
            return []

        # Current position and velocity
        x0, y0, t0 = trail[-2]
        x1, y1, t1 = trail[-1]
        dt = t1 - t0
        if dt <= 0:
            return []

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = math.hypot(vx, vy)
        if speed < _MIN_SPEED_MPS:
            return []

        heading = math.degrees(math.atan2(vx, vy)) % 360

        # Distance to travel
        travel_dist = speed * horizon_minutes * 60.0

        # Snap to nearest node
        start_node, snap_dist = sg.nearest_node(x1, y1)
        if start_node is None or snap_dist > 50.0:
            # Too far from road — fall back to empty
            return []

        # Walk the graph in the direction of travel
        pos = sg._node_positions.get(start_node)
        if pos is None:
            return []

        walked = self._walk_graph(sg, start_node, heading, travel_dist)
        if walked is None:
            return []

        pred_x, pred_y = walked

        # Road-constrained predictions have tighter cones
        confidence = _BASE_CONFIDENCE * math.exp(-0.07 * horizon_minutes)
        confidence = max(0.05, min(0.95, confidence))
        cone = _CONE_GROWTH_RATE * 0.5 * horizon_minutes  # tighter than linear

        return [Prediction(
            x=pred_x,
            y=pred_y,
            horizon_minutes=horizon_minutes,
            confidence=confidence,
            cone_radius_m=cone,
            heading_deg=heading,
            speed_mps=speed,
            model=self.name,
            metadata={"snap_dist_m": round(snap_dist, 1)},
        )]

    def _walk_graph(
        self,
        sg: Any,
        start_node: int,
        heading_deg: float,
        distance_m: float,
    ) -> tuple[float, float] | None:
        """Walk along graph edges in the given heading for the given distance.

        Greedily chooses the neighbor node most aligned with the heading
        at each step.
        """
        import networkx as nx

        current = start_node
        remaining = distance_m
        visited = {current}

        heading_rad = math.radians(heading_deg)
        dir_x = math.sin(heading_rad)
        dir_y = math.cos(heading_rad)

        while remaining > 0:
            pos = sg._node_positions.get(current)
            if pos is None:
                break

            # Find the neighbor most aligned with the travel direction
            best_neighbor = None
            best_alignment = -2.0  # cos similarity, -1 to 1
            best_dist = 0.0

            for neighbor in sg.graph.neighbors(current):
                if neighbor in visited:
                    continue
                npos = sg._node_positions.get(neighbor)
                if npos is None:
                    continue

                dx = npos[0] - pos[0]
                dy = npos[1] - pos[1]
                edge_len = math.hypot(dx, dy)
                if edge_len < 0.1:
                    continue

                # Cosine similarity with heading direction
                cos_sim = (dx * dir_x + dy * dir_y) / edge_len
                if cos_sim > best_alignment:
                    best_alignment = cos_sim
                    best_neighbor = neighbor
                    best_dist = edge_len

            if best_neighbor is None or best_alignment < 0.0:
                # No forward-facing neighbor — stop here
                break

            if best_dist >= remaining:
                # Interpolate along the last edge
                frac = remaining / best_dist
                pos_cur = sg._node_positions[current]
                pos_next = sg._node_positions[best_neighbor]
                final_x = pos_cur[0] + frac * (pos_next[0] - pos_cur[0])
                final_y = pos_cur[1] + frac * (pos_next[1] - pos_cur[1])
                return (final_x, final_y)

            remaining -= best_dist
            visited.add(best_neighbor)
            current = best_neighbor

        # Return the position we stopped at
        final_pos = sg._node_positions.get(current)
        return final_pos

    def predict_destination(
        self,
        target_id: str,
        context: PredictionContext,
    ) -> DestinationPrediction | None:
        """Predict destination by walking the road graph until a dead-end
        or a likely stop point (intersection with many branches)."""
        if context.history is None or context.street_graph is None:
            return None

        sg = context.street_graph
        if sg.graph is None:
            return None

        trail = context.history.get_trail(target_id, max_points=20)
        if len(trail) < _MIN_TRAIL_POINTS:
            return None

        x0, y0, t0 = trail[-2]
        x1, y1, t1 = trail[-1]
        dt = t1 - t0
        if dt <= 0:
            return None

        speed = math.hypot((x1 - x0) / dt, (y1 - y0) / dt)
        if speed < _MIN_SPEED_MPS:
            return None

        heading = math.degrees(math.atan2((x1 - x0) / dt, (y1 - y0) / dt)) % 360

        start_node, snap_dist = sg.nearest_node(x1, y1)
        if start_node is None or snap_dist > 50.0:
            return None

        # Walk further (up to 30 min of travel)
        max_dist = speed * 30 * 60.0
        walked = self._walk_graph(sg, start_node, heading, max_dist)
        if walked is None:
            return None

        dest_x, dest_y = walked
        dist_to_dest = math.hypot(dest_x - x1, dest_y - y1)
        eta_s = dist_to_dest / speed if speed > 0 else 0.0

        return DestinationPrediction(
            x=dest_x,
            y=dest_y,
            confidence=0.3,  # low — pure road extrapolation
            label="road_endpoint",
            estimated_arrival_s=eta_s,
            model=self.name,
        )


class RoutineAware(PredictionModel):
    """Predict based on learned daily patterns from BehavioralPatternLearner.

    Uses the target's learned route and frequent zones to predict where
    it is likely heading. If the target has a learned route and its current
    position is on that route, project forward along the route waypoints.
    If the target has frequent zones, predict the most likely destination.
    """

    @property
    def name(self) -> str:
        return "routine"

    def predict(
        self,
        target_id: str,
        horizon_minutes: float,
        context: PredictionContext,
    ) -> list[Prediction]:
        if context.behavioral_learner is None or context.history is None:
            return []

        learner = context.behavioral_learner
        trail = context.history.get_trail(target_id, max_points=20)
        if len(trail) < _MIN_TRAIL_POINTS:
            return []

        x_now, y_now = trail[-1][0], trail[-1][1]

        # Get learned route
        route = None
        if hasattr(learner, '_routes'):
            route = learner._routes.get(target_id)

        if route is None or len(route.waypoints) < 2:
            return []

        # Find nearest waypoint on the learned route
        best_idx = -1
        best_dist = float("inf")
        for i, wp in enumerate(route.waypoints):
            d = math.hypot(wp.x - x_now, wp.y - y_now)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx < 0 or best_dist > 50.0:
            # Too far from learned route
            return []

        # Estimate speed from trail
        x0, y0, t0 = trail[-2]
        x1, y1, t1 = trail[-1]
        dt = t1 - t0
        if dt <= 0:
            return []
        speed = math.hypot((x1 - x0) / dt, (y1 - y0) / dt)
        if speed < _MIN_SPEED_MPS:
            return []

        # Walk forward along route waypoints for the predicted distance
        travel_dist = speed * horizon_minutes * 60.0
        pred_pos = self._walk_route(route.waypoints, best_idx, travel_dist)
        if pred_pos is None:
            return []

        pred_x, pred_y = pred_pos
        heading = math.degrees(math.atan2(pred_x - x_now, pred_y - y_now)) % 360

        # Higher confidence when near a learned route
        confidence = _BASE_CONFIDENCE * math.exp(-0.05 * horizon_minutes)
        confidence *= max(0.3, 1.0 - best_dist / 50.0)
        confidence = max(0.05, min(0.95, confidence))

        # Cone is tighter along learned route
        cone = _CONE_GROWTH_RATE * 0.4 * horizon_minutes
        # Widen by route waypoint std
        if best_idx < len(route.waypoints):
            wp = route.waypoints[best_idx]
            cone += math.hypot(wp.std_x, wp.std_y)

        return [Prediction(
            x=pred_x,
            y=pred_y,
            horizon_minutes=horizon_minutes,
            confidence=confidence,
            cone_radius_m=cone,
            heading_deg=heading,
            speed_mps=speed,
            model=self.name,
            metadata={
                "route_waypoint_idx": best_idx,
                "dist_from_route_m": round(best_dist, 1),
            },
        )]

    def _walk_route(
        self,
        waypoints: list,
        start_idx: int,
        distance_m: float,
    ) -> tuple[float, float] | None:
        """Walk forward along route waypoints for the given distance."""
        remaining = distance_m
        for i in range(start_idx, len(waypoints) - 1):
            wp_a = waypoints[i]
            wp_b = waypoints[i + 1]
            seg_len = math.hypot(wp_b.x - wp_a.x, wp_b.y - wp_a.y)
            if seg_len < 0.01:
                continue
            if seg_len >= remaining:
                frac = remaining / seg_len
                return (
                    wp_a.x + frac * (wp_b.x - wp_a.x),
                    wp_a.y + frac * (wp_b.y - wp_a.y),
                )
            remaining -= seg_len

        # Reached end of route — return last waypoint
        if waypoints:
            last = waypoints[-1]
            return (last.x, last.y)
        return None

    def predict_destination(
        self,
        target_id: str,
        context: PredictionContext,
    ) -> DestinationPrediction | None:
        """Predict destination from frequent zones and learned route endpoint."""
        if context.behavioral_learner is None or context.history is None:
            return None

        learner = context.behavioral_learner
        trail = context.history.get_trail(target_id, max_points=10)
        if len(trail) < 2:
            return None

        x_now, y_now = trail[-1][0], trail[-1][1]

        # Check frequent zones — find the nearest unvisited one
        zones = []
        if hasattr(learner, '_zones'):
            zones = learner._zones.get(target_id, [])

        # Estimate speed
        x0, y0, t0 = trail[-2]
        x1, y1, t1 = trail[-1]
        dt = t1 - t0
        speed = math.hypot((x1 - x0) / dt, (y1 - y0) / dt) if dt > 0 else 0.0

        # Also check route endpoint
        route = None
        if hasattr(learner, '_routes'):
            route = learner._routes.get(target_id)

        candidates: list[DestinationPrediction] = []

        # Frequent zones as destination candidates
        for zone in zones:
            dist = math.hypot(zone.center_x - x_now, zone.center_y - y_now)
            if dist < zone.radius:
                # Already in this zone — skip
                continue
            eta = dist / speed if speed > _MIN_SPEED_MPS else 0.0
            conf = min(0.9, zone.visit_count / 20.0)  # More visits = higher conf
            candidates.append(DestinationPrediction(
                x=zone.center_x,
                y=zone.center_y,
                confidence=conf,
                label=zone.label or "frequent_zone",
                estimated_arrival_s=eta,
                model=self.name,
            ))

        # Route endpoint as destination candidate
        if route and route.waypoints:
            last_wp = route.waypoints[-1]
            dist = math.hypot(last_wp.x - x_now, last_wp.y - y_now)
            if dist > 5.0:  # Not already there
                eta = dist / speed if speed > _MIN_SPEED_MPS else 0.0
                candidates.append(DestinationPrediction(
                    x=last_wp.x,
                    y=last_wp.y,
                    confidence=0.5,
                    label="route_endpoint",
                    estimated_arrival_s=eta,
                    model=self.name,
                ))

        if not candidates:
            return None

        # Return highest-confidence candidate
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[0]


class FlockAware(PredictionModel):
    """Predict based on group movement (convoy / flock).

    If the target is part of an active convoy, use the convoy's average
    heading and speed to predict the target's future position. Convoy
    membership implies coordinated movement, so the prediction is
    constrained to the group trajectory.
    """

    @property
    def name(self) -> str:
        return "flock"

    def predict(
        self,
        target_id: str,
        horizon_minutes: float,
        context: PredictionContext,
    ) -> list[Prediction]:
        if context.convoy_detector is None or context.history is None:
            return []

        trail = context.history.get_trail(target_id, max_points=10)
        if len(trail) < 2:
            return []

        x_now, y_now = trail[-1][0], trail[-1][1]

        # Find convoy this target belongs to
        convoy = self._find_convoy(target_id, context.convoy_detector)
        if convoy is None:
            return []

        # Use convoy average speed and heading
        avg_speed = convoy.get("speed_avg_mps", 0.0)
        avg_heading = convoy.get("heading_avg_deg", 0.0)

        if avg_speed < _MIN_SPEED_MPS:
            return []

        dt_s = horizon_minutes * 60.0
        heading_rad = math.radians(avg_heading)
        pred_x = x_now + avg_speed * math.sin(heading_rad) * dt_s
        pred_y = y_now + avg_speed * math.cos(heading_rad) * dt_s

        member_count = len(convoy.get("member_target_ids", []))
        # Confidence rises with more convoy members
        confidence = _BASE_CONFIDENCE * math.exp(-0.06 * horizon_minutes)
        confidence *= min(1.0, member_count / 5.0)
        confidence = max(0.05, min(0.95, confidence))

        # Cone accounts for convoy spread
        heading_var = convoy.get("heading_variance_deg", 10.0)
        speed_var = convoy.get("speed_variance_mps", 0.5)
        cone = _CONE_GROWTH_RATE * 0.6 * horizon_minutes
        cone += heading_var * 0.5  # heading uncertainty widens cone
        cone += speed_var * dt_s * 0.3  # speed uncertainty

        return [Prediction(
            x=pred_x,
            y=pred_y,
            horizon_minutes=horizon_minutes,
            confidence=confidence,
            cone_radius_m=cone,
            heading_deg=avg_heading,
            speed_mps=avg_speed,
            model=self.name,
            metadata={
                "convoy_id": convoy.get("convoy_id", ""),
                "member_count": member_count,
            },
        )]

    def _find_convoy(self, target_id: str, convoy_detector: Any) -> dict | None:
        """Find the active convoy that contains the given target."""
        try:
            convoys = convoy_detector.get_active_convoys()
        except Exception:
            return None

        for convoy in convoys:
            members = convoy.get("member_target_ids", [])
            if target_id in members:
                return convoy
        return None


# ---------------------------------------------------------------------------
# TrajectoryPredictor — orchestrates multiple models
# ---------------------------------------------------------------------------

class TrajectoryPredictor:
    """Predict future positions for a target using multiple models.

    Combines results from all registered prediction models, selecting
    the best prediction at each horizon based on confidence.

    Parameters
    ----------
    history:
        TargetHistory instance for position trails.
    street_graph:
        Optional StreetGraph for road-constrained predictions.
    behavioral_learner:
        Optional BehavioralPatternLearner for routine-based predictions.
    convoy_detector:
        Optional ConvoyDetector for group movement predictions.
    models:
        Optional list of PredictionModel instances. If None, all four
        built-in models are registered.
    """

    def __init__(
        self,
        history: Any = None,
        street_graph: Any = None,
        behavioral_learner: Any = None,
        convoy_detector: Any = None,
        models: list[PredictionModel] | None = None,
    ) -> None:
        self._context = PredictionContext(
            history=history,
            street_graph=street_graph,
            behavioral_learner=behavioral_learner,
            convoy_detector=convoy_detector,
        )
        if models is not None:
            self._models = list(models)
        else:
            self._models = [
                LinearExtrapolation(),
                RoadConstrained(),
                RoutineAware(),
                FlockAware(),
            ]

    @property
    def context(self) -> PredictionContext:
        """Access the shared prediction context."""
        return self._context

    @property
    def models(self) -> list[PredictionModel]:
        """List of registered prediction models."""
        return list(self._models)

    def add_model(self, model: PredictionModel) -> None:
        """Register an additional prediction model."""
        self._models.append(model)

    def remove_model(self, name: str) -> bool:
        """Remove a prediction model by name. Returns True if removed."""
        before = len(self._models)
        self._models = [m for m in self._models if m.name != name]
        return len(self._models) < before

    def predict(
        self,
        target_id: str,
        horizon_minutes: float = 5.0,
        models: list[str] | None = None,
    ) -> list[Prediction]:
        """Predict future positions for a target at the given horizon.

        Runs all registered models (or a subset if ``models`` is given)
        and returns predictions sorted by confidence (highest first).

        Args:
            target_id: Unique target identifier.
            horizon_minutes: How far ahead to predict (in minutes).
            models: Optional list of model names to use. None = all.

        Returns:
            List of Prediction objects, sorted by confidence descending.
        """
        active_models = self._models
        if models is not None:
            model_set = set(models)
            active_models = [m for m in self._models if m.name in model_set]

        all_predictions: list[Prediction] = []
        for model in active_models:
            try:
                preds = model.predict(target_id, horizon_minutes, self._context)
                all_predictions.extend(preds)
            except Exception as e:
                log.warning(
                    "Model %s failed for target %s: %s",
                    model.name, target_id, e,
                )

        # Sort by confidence, highest first
        all_predictions.sort(key=lambda p: p.confidence, reverse=True)
        return all_predictions

    def predict_best(
        self,
        target_id: str,
        horizon_minutes: float = 5.0,
    ) -> Prediction | None:
        """Return the single highest-confidence prediction.

        Convenience wrapper around ``predict()`` that returns just
        the top prediction, or None if no model produced a result.
        """
        preds = self.predict(target_id, horizon_minutes)
        return preds[0] if preds else None

    def predict_multi_horizon(
        self,
        target_id: str,
        horizons: list[float] | None = None,
    ) -> dict[float, list[Prediction]]:
        """Predict at multiple time horizons.

        Args:
            target_id: Unique target identifier.
            horizons: List of horizons in minutes. Default: [1, 5, 15].

        Returns:
            Dict mapping horizon_minutes -> list of Predictions.
        """
        if horizons is None:
            horizons = [1.0, 5.0, 15.0]

        results: dict[float, list[Prediction]] = {}
        for h in horizons:
            results[h] = self.predict(target_id, h)
        return results

    def predict_destination(
        self,
        target_id: str,
    ) -> DestinationPrediction | None:
        """Predict the most likely endpoint for a target.

        Queries all models that support destination prediction and
        returns the highest-confidence result.
        """
        candidates: list[DestinationPrediction] = []
        for model in self._models:
            try:
                dest = model.predict_destination(target_id, self._context)
                if dest is not None:
                    candidates.append(dest)
            except Exception as e:
                log.warning(
                    "Model %s destination prediction failed for %s: %s",
                    model.name, target_id, e,
                )

        if not candidates:
            return None

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[0]

    def predict_all(
        self,
        target_ids: list[str],
        horizon_minutes: float = 5.0,
    ) -> dict[str, list[Prediction]]:
        """Predict for multiple targets at once.

        Args:
            target_ids: List of target identifiers.
            horizon_minutes: How far ahead to predict.

        Returns:
            Dict mapping target_id -> list of Predictions (only for targets
            with at least one prediction).
        """
        results: dict[str, list[Prediction]] = {}
        for tid in target_ids:
            preds = self.predict(tid, horizon_minutes)
            if preds:
                results[tid] = preds
        return results

    def get_model_names(self) -> list[str]:
        """Return names of all registered models."""
        return [m.name for m in self._models]

    def stats(self) -> dict[str, Any]:
        """Return diagnostic stats about the predictor."""
        return {
            "model_count": len(self._models),
            "model_names": self.get_model_names(),
            "has_history": self._context.history is not None,
            "has_street_graph": self._context.street_graph is not None,
            "has_behavioral_learner": self._context.behavioral_learner is not None,
            "has_convoy_detector": self._context.convoy_detector is not None,
        }
