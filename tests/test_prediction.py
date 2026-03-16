# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TargetPrediction model."""

from tritium_lib.models.prediction import (
    PredictedPosition,
    TargetPrediction,
)


def test_predicted_position_roundtrip():
    p = PredictedPosition(lat=40.0, lng=-74.0, time_offset_sec=30.0, confidence=0.9, radius_m=12.5)
    d = p.to_dict()
    p2 = PredictedPosition.from_dict(d)
    assert p2.lat == 40.0
    assert p2.time_offset_sec == 30.0
    assert p2.confidence == 0.9


def test_prediction_defaults():
    pred = TargetPrediction(target_id="ble_aa:bb:cc")
    assert pred.timestamp is not None
    assert pred.model == "linear"
    assert pred.max_prediction_time_sec == 0.0
    assert pred.min_confidence == 0.0


def test_linear_predictions_heading_north():
    pred = TargetPrediction(
        target_id="ble_test",
        current_lat=40.0,
        current_lng=-74.0,
        heading_deg=0.0,  # north
        speed_mps=10.0,
    )
    pred.generate_linear_predictions(steps=3, interval_sec=10.0)
    assert len(pred.predicted_positions) == 3
    # Heading north: lat should increase
    for p in pred.predicted_positions:
        assert p.lat > 40.0
    # Each step further north
    assert pred.predicted_positions[2].lat > pred.predicted_positions[1].lat
    assert pred.predicted_positions[1].lat > pred.predicted_positions[0].lat


def test_linear_predictions_heading_east():
    pred = TargetPrediction(
        target_id="ble_test",
        current_lat=40.0,
        current_lng=-74.0,
        heading_deg=90.0,  # east
        speed_mps=5.0,
    )
    pred.generate_linear_predictions(steps=2, interval_sec=60.0)
    assert len(pred.predicted_positions) == 2
    # Heading east: lng should increase
    for p in pred.predicted_positions:
        assert p.lng > -74.0


def test_confidence_decays():
    pred = TargetPrediction(
        target_id="t1",
        current_lat=0.0,
        current_lng=0.0,
        heading_deg=45.0,
        speed_mps=1.0,
        confidence_decay_rate=0.005,
    )
    pred.generate_linear_predictions(steps=5, interval_sec=30.0)
    confs = [p.confidence for p in pred.predicted_positions]
    # Each subsequent confidence should be lower
    for i in range(1, len(confs)):
        assert confs[i] < confs[i - 1]
    assert pred.min_confidence == confs[-1]


def test_radius_grows():
    pred = TargetPrediction(
        target_id="t1",
        current_lat=0.0,
        current_lng=0.0,
        heading_deg=0.0,
        speed_mps=10.0,
    )
    pred.generate_linear_predictions(steps=4, interval_sec=30.0, base_radius_m=2.0)
    radii = [p.radius_m for p in pred.predicted_positions]
    for i in range(1, len(radii)):
        assert radii[i] >= radii[i - 1]


def test_max_prediction_time():
    pred = TargetPrediction(target_id="t1")
    pred.generate_linear_predictions(steps=3, interval_sec=60.0)
    assert pred.max_prediction_time_sec == 180.0


def test_generate_clears_previous():
    pred = TargetPrediction(target_id="t1", speed_mps=1.0, heading_deg=0.0)
    pred.generate_linear_predictions(steps=5)
    assert len(pred.predicted_positions) == 5
    pred.generate_linear_predictions(steps=2)
    assert len(pred.predicted_positions) == 2


def test_roundtrip():
    pred = TargetPrediction(
        target_id="mesh_node7",
        current_lat=35.5,
        current_lng=-120.3,
        heading_deg=180.0,
        speed_mps=3.0,
        model="kalman",
    )
    pred.generate_linear_predictions(steps=3, interval_sec=10.0)
    d = pred.to_dict()
    pred2 = TargetPrediction.from_dict(d)
    assert pred2.target_id == "mesh_node7"
    assert pred2.model == "kalman"
    assert len(pred2.predicted_positions) == 3
    assert pred2.heading_deg == 180.0
