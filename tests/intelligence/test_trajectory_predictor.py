# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TrajectoryPredictor — trajectory prediction module.

Covers:
  - Prediction / DestinationPrediction data classes
  - LinearExtrapolation model
  - RoadConstrained model
  - RoutineAware model
  - FlockAware model
  - TrajectoryPredictor orchestrator (predict, predict_best, predict_destination, etc.)
  - Edge cases: insufficient data, stationary targets, missing context
"""

import math
import time

import pytest

from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.intelligence.trajectory_predictor import (
    DestinationPrediction,
    FlockAware,
    LinearExtrapolation,
    Prediction,
    PredictionContext,
    PredictionModel,
    RoadConstrained,
    RoutineAware,
    TrajectoryPredictor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history_moving_east(
    target_id: str = "t1",
    speed_mps: float = 10.0,
    n_points: int = 10,
    dt: float = 1.0,
    start_time: float = 1000.0,
) -> TargetHistory:
    """Create a TargetHistory with a target moving due east (+X) at constant speed."""
    history = TargetHistory()
    for i in range(n_points):
        x = i * speed_mps * dt
        y = 0.0
        t = start_time + i * dt
        history.record(target_id, (x, y), timestamp=t)
    return history


def _make_history_moving_north(
    target_id: str = "t1",
    speed_mps: float = 5.0,
    n_points: int = 10,
    dt: float = 1.0,
    start_time: float = 1000.0,
) -> TargetHistory:
    """Create a TargetHistory with a target moving due north (+Y)."""
    history = TargetHistory()
    for i in range(n_points):
        x = 0.0
        y = i * speed_mps * dt
        t = start_time + i * dt
        history.record(target_id, (x, y), timestamp=t)
    return history


def _make_history_stationary(
    target_id: str = "t1",
    n_points: int = 10,
) -> TargetHistory:
    """Create a TargetHistory with a stationary target."""
    history = TargetHistory()
    for i in range(n_points):
        history.record(target_id, (50.0, 50.0), timestamp=1000.0 + i)
    return history


class _FakeStreetGraph:
    """Minimal fake StreetGraph for testing RoadConstrained model."""

    def __init__(self):
        nx = pytest.importorskip("networkx", reason="networkx not installed")
        self.graph = nx.Graph()
        self._node_positions = {}

        # Build a simple L-shaped road: east then north
        # Nodes 0-5 going east (x=0..500, y=0)
        # Nodes 5-10 going north (x=500, y=0..500)
        for i in range(11):
            if i <= 5:
                pos = (i * 100.0, 0.0)
            else:
                pos = (500.0, (i - 5) * 100.0)
            self._node_positions[i] = pos
            self.graph.add_node(i, x=pos[0], y=pos[1])

        for i in range(10):
            p1 = self._node_positions[i]
            p2 = self._node_positions[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.graph.add_edge(i, i + 1, weight=dist, road_class="residential")

    def nearest_node(self, x, y):
        best_id = None
        best_dist = float("inf")
        for nid, pos in self._node_positions.items():
            d = math.hypot(x - pos[0], y - pos[1])
            if d < best_dist:
                best_dist = d
                best_id = nid
        return (best_id, best_dist)


class _FakeLearnedWaypoint:
    """Minimal waypoint for RoutineAware tests."""

    def __init__(self, x, y, std_x=1.0, std_y=1.0):
        self.x = x
        self.y = y
        self.std_x = std_x
        self.std_y = std_y
        self.observation_count = 10


class _FakeLearnedRoute:
    """Minimal learned route for RoutineAware tests."""

    def __init__(self, target_id, waypoints):
        self.target_id = target_id
        self.waypoints = waypoints
        self.total_observations = 100
        self.mean_duration_s = 300.0
        self.std_duration_s = 30.0
        self.last_updated = time.time()


class _FakeFrequentZone:
    """Minimal frequent zone for RoutineAware destination tests."""

    def __init__(self, cx, cy, radius=20.0, visits=10, label=""):
        self.center_x = cx
        self.center_y = cy
        self.radius = radius
        self.visit_count = visits
        self.total_dwell_s = 3600.0
        self.label = label


class _FakeBehavioralLearner:
    """Minimal behavioral learner for RoutineAware tests."""

    def __init__(self):
        self._routes = {}
        self._zones = {}


class _FakeConvoyDetector:
    """Minimal convoy detector for FlockAware tests."""

    def __init__(self, convoys=None):
        self._convoys = convoys or []

    def get_active_convoys(self):
        return self._convoys


# ---------------------------------------------------------------------------
# Prediction data class tests
# ---------------------------------------------------------------------------

class TestPredictionDataClass:
    def test_prediction_fields(self):
        p = Prediction(x=10.0, y=20.0, horizon_minutes=5.0,
                       confidence=0.8, cone_radius_m=15.0,
                       heading_deg=90.0, speed_mps=10.0, model="test")
        assert p.x == 10.0
        assert p.y == 20.0
        assert p.horizon_minutes == 5.0
        assert p.confidence == 0.8
        assert p.model == "test"

    def test_prediction_to_dict(self):
        p = Prediction(x=10.123, y=20.456, horizon_minutes=5.0,
                       confidence=0.812, cone_radius_m=15.34,
                       heading_deg=90.56, speed_mps=10.789, model="linear")
        d = p.to_dict()
        assert d["x"] == 10.12
        assert d["y"] == 20.46
        assert d["confidence"] == 0.812
        assert d["model"] == "linear"
        assert "metadata" in d

    def test_prediction_default_metadata(self):
        p = Prediction(x=0, y=0, horizon_minutes=1.0,
                       confidence=0.5, cone_radius_m=5.0)
        assert p.metadata == {}
        assert p.model == ""

    def test_destination_prediction_fields(self):
        dp = DestinationPrediction(
            x=100.0, y=200.0, confidence=0.7,
            label="home", estimated_arrival_s=600.0, model="routine",
        )
        assert dp.label == "home"
        assert dp.estimated_arrival_s == 600.0

    def test_destination_prediction_to_dict(self):
        dp = DestinationPrediction(
            x=100.5, y=200.3, confidence=0.75,
            label="work", estimated_arrival_s=1200.5, model="routine",
        )
        d = dp.to_dict()
        assert d["x"] == 100.5
        assert d["label"] == "work"
        assert d["estimated_arrival_s"] == 1200.5


# ---------------------------------------------------------------------------
# LinearExtrapolation tests
# ---------------------------------------------------------------------------

class TestLinearExtrapolation:
    def test_basic_east_prediction(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        preds = model.predict("t1", 1.0, ctx)
        assert len(preds) == 1
        p = preds[0]
        # At 10 m/s for 1 minute = 600m further east
        expected_x = 90.0 + 10.0 * 60.0  # last pos + velocity * 60s
        assert abs(p.x - expected_x) < 1.0
        assert abs(p.y) < 1.0
        assert p.model == "linear"

    def test_basic_north_prediction(self):
        history = _make_history_moving_north("t1", speed_mps=5.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        preds = model.predict("t1", 1.0, ctx)
        assert len(preds) == 1
        p = preds[0]
        expected_y = 45.0 + 5.0 * 60.0
        assert abs(p.y - expected_y) < 1.0

    def test_stationary_returns_empty(self):
        history = _make_history_stationary("t1")
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_insufficient_data_returns_empty(self):
        history = TargetHistory()
        history.record("t1", (10.0, 20.0), timestamp=1000.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_no_history_returns_empty(self):
        ctx = PredictionContext(history=None)
        model = LinearExtrapolation()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_confidence_decays_with_horizon(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        p1 = model.predict("t1", 1.0, ctx)[0]
        p15 = model.predict("t1", 15.0, ctx)[0]
        assert p1.confidence > p15.confidence

    def test_cone_grows_with_horizon(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        p1 = model.predict("t1", 1.0, ctx)[0]
        p15 = model.predict("t1", 15.0, ctx)[0]
        assert p15.cone_radius_m > p1.cone_radius_m

    def test_heading_east(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        p = model.predict("t1", 1.0, ctx)[0]
        # Moving east = heading ~90 degrees
        assert 80.0 < p.heading_deg < 100.0

    def test_speed_preserved(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        ctx = PredictionContext(history=history)
        model = LinearExtrapolation()
        p = model.predict("t1", 1.0, ctx)[0]
        assert abs(p.speed_mps - 10.0) < 0.5

    def test_model_name(self):
        model = LinearExtrapolation()
        assert model.name == "linear"


# ---------------------------------------------------------------------------
# RoadConstrained tests
# ---------------------------------------------------------------------------

class TestRoadConstrained:
    def test_basic_road_prediction(self):
        """Target near road node 0, moving east, should follow road."""
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)
        sg = _FakeStreetGraph()
        ctx = PredictionContext(history=history, street_graph=sg)
        model = RoadConstrained()
        preds = model.predict("t1", 1.0, ctx)
        assert len(preds) == 1
        p = preds[0]
        assert p.model == "road_constrained"
        # Should be somewhere along the east road
        assert p.x > 40.0
        assert p.y >= -5.0  # stays near y=0

    def test_no_graph_returns_empty(self):
        history = _make_history_moving_east("t1")
        ctx = PredictionContext(history=history, street_graph=None)
        model = RoadConstrained()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_graph_no_nodes_returns_empty(self):
        history = _make_history_moving_east("t1")
        sg = _FakeStreetGraph()
        sg.graph = None  # simulate unloaded graph
        ctx = PredictionContext(history=history, street_graph=sg)
        model = RoadConstrained()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_too_far_from_road_returns_empty(self):
        """Target far from any road node should return empty."""
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 5000.0), timestamp=1000.0 + i)
        sg = _FakeStreetGraph()
        ctx = PredictionContext(history=history, street_graph=sg)
        model = RoadConstrained()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_road_cone_tighter_than_linear(self):
        """Road-constrained predictions should have tighter cone."""
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)
        sg = _FakeStreetGraph()
        ctx_road = PredictionContext(history=history, street_graph=sg)
        ctx_lin = PredictionContext(history=history)

        road_model = RoadConstrained()
        lin_model = LinearExtrapolation()

        road_preds = road_model.predict("t1", 5.0, ctx_road)
        lin_preds = lin_model.predict("t1", 5.0, ctx_lin)

        if road_preds and lin_preds:
            assert road_preds[0].cone_radius_m < lin_preds[0].cone_radius_m

    def test_predict_destination_road(self):
        """Test road-based destination prediction."""
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)
        sg = _FakeStreetGraph()
        ctx = PredictionContext(history=history, street_graph=sg)
        model = RoadConstrained()
        dest = model.predict_destination("t1", ctx)
        # Should get some destination along the road
        if dest is not None:
            assert dest.model == "road_constrained"
            assert dest.confidence > 0

    def test_model_name(self):
        model = RoadConstrained()
        assert model.name == "road_constrained"


# ---------------------------------------------------------------------------
# RoutineAware tests
# ---------------------------------------------------------------------------

class TestRoutineAware:
    def _make_routine_context(self, target_id="t1"):
        """Create context with a learned east-bound route."""
        history = TargetHistory()
        for i in range(10):
            history.record(target_id, (i * 10.0, 0.0), timestamp=1000.0 + i)

        learner = _FakeBehavioralLearner()
        waypoints = [_FakeLearnedWaypoint(x=i * 50.0, y=0.0) for i in range(20)]
        learner._routes[target_id] = _FakeLearnedRoute(target_id, waypoints)

        return PredictionContext(history=history, behavioral_learner=learner)

    def test_basic_routine_prediction(self):
        ctx = self._make_routine_context()
        model = RoutineAware()
        preds = model.predict("t1", 1.0, ctx)
        assert len(preds) == 1
        p = preds[0]
        assert p.model == "routine"
        assert p.confidence > 0

    def test_no_learner_returns_empty(self):
        history = _make_history_moving_east("t1")
        ctx = PredictionContext(history=history, behavioral_learner=None)
        model = RoutineAware()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_no_route_returns_empty(self):
        history = _make_history_moving_east("t1")
        learner = _FakeBehavioralLearner()  # no routes
        ctx = PredictionContext(history=history, behavioral_learner=learner)
        model = RoutineAware()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_far_from_route_returns_empty(self):
        """Target far from its learned route should return empty."""
        history = TargetHistory()
        for i in range(10):
            # Moving at y=5000, far from route at y=0
            history.record("t1", (i * 10.0, 5000.0), timestamp=1000.0 + i)

        learner = _FakeBehavioralLearner()
        waypoints = [_FakeLearnedWaypoint(x=i * 50.0, y=0.0) for i in range(20)]
        learner._routes["t1"] = _FakeLearnedRoute("t1", waypoints)

        ctx = PredictionContext(history=history, behavioral_learner=learner)
        model = RoutineAware()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_routine_destination_from_zones(self):
        """Destination prediction from frequent zones."""
        history = TargetHistory()
        for i in range(10):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)

        learner = _FakeBehavioralLearner()
        learner._zones["t1"] = [
            _FakeFrequentZone(500.0, 0.0, radius=20.0, visits=15, label="work"),
        ]
        # Need a route too for completeness, but zones are the focus
        waypoints = [_FakeLearnedWaypoint(x=i * 50.0, y=0.0) for i in range(20)]
        learner._routes["t1"] = _FakeLearnedRoute("t1", waypoints)

        ctx = PredictionContext(history=history, behavioral_learner=learner)
        model = RoutineAware()
        dest = model.predict_destination("t1", ctx)
        assert dest is not None
        assert dest.model == "routine"
        assert dest.label in ("work", "frequent_zone", "route_endpoint")

    def test_routine_cone_tighter_than_linear(self):
        ctx = self._make_routine_context()
        routine_model = RoutineAware()
        linear_model = LinearExtrapolation()

        r_preds = routine_model.predict("t1", 5.0, ctx)
        l_preds = linear_model.predict("t1", 5.0, ctx)

        if r_preds and l_preds:
            assert r_preds[0].cone_radius_m < l_preds[0].cone_radius_m

    def test_model_name(self):
        model = RoutineAware()
        assert model.name == "routine"


# ---------------------------------------------------------------------------
# FlockAware tests
# ---------------------------------------------------------------------------

class TestFlockAware:
    def test_basic_flock_prediction(self):
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)

        convoy_data = {
            "convoy_id": "conv_abc",
            "member_target_ids": ["t1", "t2", "t3"],
            "speed_avg_mps": 10.0,
            "heading_avg_deg": 90.0,  # east
            "heading_variance_deg": 5.0,
            "speed_variance_mps": 0.5,
        }
        detector = _FakeConvoyDetector([convoy_data])
        ctx = PredictionContext(history=history, convoy_detector=detector)
        model = FlockAware()
        preds = model.predict("t1", 1.0, ctx)
        assert len(preds) == 1
        p = preds[0]
        assert p.model == "flock"
        assert abs(p.heading_deg - 90.0) < 0.1
        assert p.metadata.get("convoy_id") == "conv_abc"

    def test_not_in_convoy_returns_empty(self):
        history = _make_history_moving_east("t1")
        detector = _FakeConvoyDetector([])
        ctx = PredictionContext(history=history, convoy_detector=detector)
        model = FlockAware()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_no_detector_returns_empty(self):
        history = _make_history_moving_east("t1")
        ctx = PredictionContext(history=history, convoy_detector=None)
        model = FlockAware()
        preds = model.predict("t1", 1.0, ctx)
        assert preds == []

    def test_flock_confidence_scales_with_members(self):
        """More convoy members should increase confidence."""
        history = TargetHistory()
        for i in range(5):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)

        small_convoy = {
            "convoy_id": "conv_s",
            "member_target_ids": ["t1", "t2", "t3"],
            "speed_avg_mps": 10.0,
            "heading_avg_deg": 90.0,
            "heading_variance_deg": 5.0,
            "speed_variance_mps": 0.5,
        }
        big_convoy = {
            "convoy_id": "conv_b",
            "member_target_ids": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "speed_avg_mps": 10.0,
            "heading_avg_deg": 90.0,
            "heading_variance_deg": 5.0,
            "speed_variance_mps": 0.5,
        }

        ctx_small = PredictionContext(
            history=history, convoy_detector=_FakeConvoyDetector([small_convoy])
        )
        ctx_big = PredictionContext(
            history=history, convoy_detector=_FakeConvoyDetector([big_convoy])
        )

        model = FlockAware()
        p_small = model.predict("t1", 1.0, ctx_small)[0]
        p_big = model.predict("t1", 1.0, ctx_big)[0]
        assert p_big.confidence >= p_small.confidence

    def test_model_name(self):
        model = FlockAware()
        assert model.name == "flock"


# ---------------------------------------------------------------------------
# TrajectoryPredictor orchestrator tests
# ---------------------------------------------------------------------------

class TestTrajectoryPredictor:
    def test_default_models(self):
        tp = TrajectoryPredictor()
        names = tp.get_model_names()
        assert "linear" in names
        assert "road_constrained" in names
        assert "routine" in names
        assert "flock" in names

    def test_custom_models(self):
        tp = TrajectoryPredictor(models=[LinearExtrapolation()])
        assert tp.get_model_names() == ["linear"]

    def test_add_model(self):
        tp = TrajectoryPredictor(models=[])
        assert len(tp.models) == 0
        tp.add_model(LinearExtrapolation())
        assert len(tp.models) == 1

    def test_remove_model(self):
        tp = TrajectoryPredictor()
        assert tp.remove_model("linear") is True
        assert "linear" not in tp.get_model_names()
        assert tp.remove_model("nonexistent") is False

    def test_predict_moving_target(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        preds = tp.predict("t1", horizon_minutes=5.0)
        # At minimum, linear should produce a prediction
        assert len(preds) >= 1
        assert all(p.confidence > 0 for p in preds)

    def test_predict_stationary_returns_empty(self):
        history = _make_history_stationary("t1")
        tp = TrajectoryPredictor(history=history)
        preds = tp.predict("t1", horizon_minutes=5.0)
        assert preds == []

    def test_predict_sorted_by_confidence(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        preds = tp.predict("t1", horizon_minutes=5.0)
        if len(preds) >= 2:
            for i in range(len(preds) - 1):
                assert preds[i].confidence >= preds[i + 1].confidence

    def test_predict_best(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        best = tp.predict_best("t1", horizon_minutes=5.0)
        assert best is not None
        assert best.confidence > 0

    def test_predict_best_no_data(self):
        tp = TrajectoryPredictor(history=TargetHistory())
        best = tp.predict_best("nonexistent", horizon_minutes=5.0)
        assert best is None

    def test_predict_multi_horizon(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        results = tp.predict_multi_horizon("t1", horizons=[1.0, 5.0, 15.0])
        assert 1.0 in results
        assert 5.0 in results
        assert 15.0 in results

    def test_predict_multi_horizon_default(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        results = tp.predict_multi_horizon("t1")
        assert len(results) == 3

    def test_predict_destination_no_models_return_none(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        # Linear and flock don't support destination, road/routine need
        # more context. With just history, destination may be None.
        dest = tp.predict_destination("t1")
        # We just check it doesn't crash
        assert dest is None or isinstance(dest, DestinationPrediction)

    def test_predict_all_targets(self):
        history = TargetHistory()
        for i in range(10):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)
            history.record("t2", (0.0, i * 5.0), timestamp=1000.0 + i)
        tp = TrajectoryPredictor(history=history)
        results = tp.predict_all(["t1", "t2"], horizon_minutes=5.0)
        assert "t1" in results
        assert "t2" in results

    def test_predict_filter_by_model_name(self):
        history = _make_history_moving_east("t1", speed_mps=10.0)
        tp = TrajectoryPredictor(history=history)
        preds = tp.predict("t1", horizon_minutes=5.0, models=["linear"])
        assert all(p.model == "linear" for p in preds)

    def test_stats(self):
        history = _make_history_moving_east("t1")
        tp = TrajectoryPredictor(history=history)
        s = tp.stats()
        assert s["model_count"] == 4
        assert s["has_history"] is True
        assert s["has_street_graph"] is False

    def test_context_accessible(self):
        history = _make_history_moving_east("t1")
        tp = TrajectoryPredictor(history=history)
        assert tp.context.history is history

    def test_predict_with_road_and_routine(self):
        """Full integration: history + road graph + learned route."""
        history = TargetHistory()
        for i in range(10):
            history.record("t1", (i * 10.0, 0.0), timestamp=1000.0 + i)

        sg = _FakeStreetGraph()

        learner = _FakeBehavioralLearner()
        waypoints = [_FakeLearnedWaypoint(x=i * 50.0, y=0.0) for i in range(20)]
        learner._routes["t1"] = _FakeLearnedRoute("t1", waypoints)

        tp = TrajectoryPredictor(
            history=history,
            street_graph=sg,
            behavioral_learner=learner,
        )
        preds = tp.predict("t1", horizon_minutes=1.0)
        # Should get predictions from linear, road, and routine
        models_used = {p.model for p in preds}
        assert "linear" in models_used
        # road_constrained and routine may also contribute
        assert len(preds) >= 1


# ---------------------------------------------------------------------------
# PredictionModel ABC tests
# ---------------------------------------------------------------------------

class TestPredictionModelABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            PredictionModel()

    def test_default_predict_destination_returns_none(self):
        """Custom model that doesn't override predict_destination returns None."""
        class _MinimalModel(PredictionModel):
            @property
            def name(self):
                return "minimal"

            def predict(self, target_id, horizon_minutes, context):
                return []

        model = _MinimalModel()
        ctx = PredictionContext()
        assert model.predict_destination("t1", ctx) is None


# ---------------------------------------------------------------------------
# Import from intelligence package
# ---------------------------------------------------------------------------

class TestImportFromPackage:
    def test_import_from_intelligence(self):
        from tritium_lib.intelligence import (
            TrajectoryPredictor,
            Prediction,
            DestinationPrediction,
            PredictionModel,
            PredictionContext,
            LinearExtrapolation,
            RoadConstrained,
            RoutineAware,
            FlockAware,
        )
        assert TrajectoryPredictor is not None
        assert Prediction is not None
