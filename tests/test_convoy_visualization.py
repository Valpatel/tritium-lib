# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ConvoyVisualization model."""

from tritium_lib.models.convoy_visualization import (
    ConvoyFormationType,
    ConvoyVisualization,
    LatLng,
)


def test_default_creation():
    cv = ConvoyVisualization(convoy_id="c1")
    assert cv.convoy_id == "c1"
    assert cv.target_ids == []
    assert cv.heading_degrees == 0.0
    assert cv.speed_estimate == 0.0
    assert cv.formation_type == ConvoyFormationType.CLUSTER
    assert cv.confidence == 0.0
    assert cv.bounding_box == []
    assert cv.color == "#fcee0a"


def test_member_count():
    cv = ConvoyVisualization(
        convoy_id="c1",
        target_ids=["t1", "t2", "t3"],
    )
    assert cv.member_count == 3


def test_has_bounding_box():
    cv = ConvoyVisualization(convoy_id="c1")
    assert cv.has_bounding_box is False

    cv2 = ConvoyVisualization(
        convoy_id="c2",
        bounding_box=[
            LatLng(lat=1.0, lng=2.0),
            LatLng(lat=3.0, lng=4.0),
            LatLng(lat=5.0, lng=6.0),
        ],
    )
    assert cv2.has_bounding_box is True


def test_speed_kmh():
    cv = ConvoyVisualization(convoy_id="c1", speed_estimate=10.0)
    assert abs(cv.speed_kmh - 36.0) < 0.01


def test_heading_label():
    cases = [
        (0.0, "N"),
        (45.0, "NE"),
        (90.0, "E"),
        (135.0, "SE"),
        (180.0, "S"),
        (225.0, "SW"),
        (270.0, "W"),
        (315.0, "NW"),
        (350.0, "N"),
    ]
    for deg, expected in cases:
        cv = ConvoyVisualization(convoy_id="c1", heading_degrees=deg)
        assert cv.heading_label() == expected, f"heading={deg} expected={expected} got={cv.heading_label()}"


def test_formation_types():
    assert ConvoyFormationType.COLUMN.value == "column"
    assert ConvoyFormationType.PARALLEL.value == "parallel"
    assert ConvoyFormationType.CLUSTER.value == "cluster"


def test_to_dict():
    cv = ConvoyVisualization(
        convoy_id="c1",
        target_ids=["t1", "t2", "t3"],
        heading_degrees=90.0,
        speed_estimate=5.0,
        formation_type=ConvoyFormationType.COLUMN,
        confidence=0.85,
        bounding_box=[
            LatLng(lat=40.0, lng=-74.0),
            LatLng(lat=40.1, lng=-74.0),
            LatLng(lat=40.1, lng=-73.9),
        ],
        label="Convoy Alpha",
    )
    d = cv.to_dict()
    assert d["convoy_id"] == "c1"
    assert d["target_ids"] == ["t1", "t2", "t3"]
    assert d["heading_degrees"] == 90.0
    assert d["formation_type"] == "column"
    assert d["confidence"] == 0.85
    assert len(d["bounding_box"]) == 3
    assert d["bounding_box"][0] == {"lat": 40.0, "lng": -74.0}
    assert d["member_count"] == 3
    assert abs(d["speed_kmh"] - 18.0) < 0.01
    assert d["heading_label"] == "E"
    assert d["label"] == "Convoy Alpha"


def test_from_dict():
    data = {
        "convoy_id": "c2",
        "target_ids": ["a", "b", "c", "d"],
        "heading_degrees": 225.0,
        "speed_estimate": 2.5,
        "formation_type": "parallel",
        "confidence": 0.6,
        "bounding_box": [
            {"lat": 1.0, "lng": 2.0},
            {"lat": 3.0, "lng": 4.0},
            {"lat": 5.0, "lng": 6.0},
        ],
        "label": "Group B",
        "color": "#ff2a6d",
    }
    cv = ConvoyVisualization.from_dict(data)
    assert cv.convoy_id == "c2"
    assert cv.member_count == 4
    assert cv.formation_type == ConvoyFormationType.PARALLEL
    assert cv.confidence == 0.6
    assert len(cv.bounding_box) == 3
    assert cv.bounding_box[0].lat == 1.0
    assert cv.color == "#ff2a6d"


def test_round_trip():
    original = ConvoyVisualization(
        convoy_id="rt1",
        target_ids=["x1", "x2", "x3"],
        heading_degrees=180.0,
        speed_estimate=8.0,
        formation_type=ConvoyFormationType.PARALLEL,
        confidence=0.75,
        bounding_box=[
            LatLng(lat=10.0, lng=20.0),
            LatLng(lat=10.1, lng=20.0),
            LatLng(lat=10.1, lng=20.1),
            LatLng(lat=10.0, lng=20.1),
        ],
    )
    d = original.to_dict()
    restored = ConvoyVisualization.from_dict(d)
    assert restored.convoy_id == original.convoy_id
    assert restored.target_ids == original.target_ids
    assert restored.heading_degrees == original.heading_degrees
    assert restored.formation_type == original.formation_type
    assert len(restored.bounding_box) == 4


def test_latlng_model():
    p = LatLng(lat=51.5, lng=-0.1)
    assert p.lat == 51.5
    assert p.lng == -0.1


def test_empty_bounding_box_from_dict():
    data = {"convoy_id": "empty"}
    cv = ConvoyVisualization.from_dict(data)
    assert cv.bounding_box == []
    assert cv.has_bounding_box is False
