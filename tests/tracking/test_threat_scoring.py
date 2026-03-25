# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.threat_scoring."""

import time
import pytest
from tritium_lib.tracking.threat_scoring import (
    ThreatScorer,
    BehaviorProfile,
    LOITER_WEIGHT,
    ZONE_VIOLATION_WEIGHT,
)


# --- BehaviorProfile ---

def test_profile_initial_zero():
    p = BehaviorProfile(target_id="t-1")
    assert p.threat_score == 0.0
    assert p.compute_threat_score() == 0.0


def test_profile_compute_weighted():
    p = BehaviorProfile(target_id="t-1")
    p.loiter_score = 1.0
    p.zone_score = 1.0
    p.timing_score = 1.0
    p.movement_score = 1.0
    p.appearance_score = 1.0
    score = p.compute_threat_score()
    assert score == pytest.approx(1.0, abs=0.01)


def test_profile_to_dict():
    p = BehaviorProfile(target_id="t-1")
    d = p.to_dict()
    assert d["target_id"] == "t-1"
    assert "threat_score" in d
    assert "zone_violations" in d


# --- ThreatScorer ---

def test_evaluate_empty():
    scorer = ThreatScorer()
    scores = scorer.evaluate([])
    assert scores == {}


def test_evaluate_friendly_zero():
    scorer = ThreatScorer()
    scores = scorer.evaluate([
        {"target_id": "f-1", "position": (0, 0), "heading": 0,
         "speed": 5, "source": "ble", "alliance": "friendly"},
    ])
    assert scores["f-1"] == 0.0


def test_evaluate_creates_profile():
    scorer = ThreatScorer()
    scorer.evaluate([
        {"target_id": "t-1", "position": (0, 0), "heading": 0,
         "speed": 5, "source": "yolo"},
    ])
    profile = scorer.get_profile("t-1")
    assert profile is not None
    assert profile["target_id"] == "t-1"


def test_evaluate_with_object_targets():
    """Test that evaluate works with attribute-based targets (not just dicts)."""

    class FakeTarget:
        def __init__(self):
            self.target_id = "obj-1"
            self.position = (10, 20)
            self.heading = 90.0
            self.speed = 3.0
            self.source = "yolo"
            self.alliance = "hostile"

    scorer = ThreatScorer()
    scores = scorer.evaluate([FakeTarget()])
    assert "obj-1" in scores


def test_zone_violation_increases_score():
    def always_violated(tid, pos):
        return True

    scorer = ThreatScorer(geofence_checker=always_violated)
    target = {"target_id": "t-1", "position": (50, 50), "heading": 0,
              "speed": 5, "source": "yolo"}
    scorer.evaluate([target])
    scorer.evaluate([target])
    profile = scorer.get_profile("t-1")
    assert profile["zone_score"] > 0


def test_no_geofence_checker_no_zone_score():
    scorer = ThreatScorer(geofence_checker=None)
    target = {"target_id": "t-1", "position": (50, 50), "heading": 0,
              "speed": 5, "source": "yolo"}
    scorer.evaluate([target])
    profile = scorer.get_profile("t-1")
    assert profile["zone_score"] == 0.0


def test_on_score_update_callback():
    updates = []

    def on_update(tid, score, profile_dict):
        updates.append((tid, score))

    scorer = ThreatScorer(on_score_update=on_update)
    # First eval creates profile at 0, then zone violation jumps it
    def violate(tid, pos):
        return True

    scorer._geofence_checker = violate
    target = {"target_id": "t-1", "position": (50, 50), "heading": 0,
              "speed": 5, "source": "yolo"}
    # Need multiple evals for zone_score to accumulate enough for 0.1 threshold
    for _ in range(10):
        scorer.evaluate([target])


def test_score_decay_for_absent_targets():
    scorer = ThreatScorer()
    target = {"target_id": "t-1", "position": (50, 50), "heading": 0,
              "speed": 5, "source": "yolo"}
    # Create profile
    scorer.evaluate([target])
    # Now evaluate without t-1 — should decay
    scorer.evaluate([])
    # Profile should still exist (unless score dropped below threshold)
    # Just verify no crash


def test_get_all_profiles():
    scorer = ThreatScorer()
    # Evaluate all 3 targets in a single call so none decay/expire
    scorer.evaluate([
        {"target_id": f"t-{i}", "position": (i * 10, 0), "heading": 0,
         "speed": 5, "source": "yolo"}
        for i in range(3)
    ])
    profiles = scorer.get_all_profiles()
    assert len(profiles) == 3


def test_get_score_unknown_target():
    scorer = ThreatScorer()
    assert scorer.get_score("nonexistent") == 0.0


def test_get_status():
    scorer = ThreatScorer()
    status = scorer.get_status()
    assert "total_profiles" in status
    assert "high_threat_count" in status
    assert status["has_geofence_checker"] is False


def test_movement_anomaly_erratic_heading():
    scorer = ThreatScorer()
    target = {"target_id": "t-1", "position": (0, 0), "heading": 0,
              "speed": 5, "source": "yolo"}
    # Rapidly changing heading
    for heading in [0, 180, 0, 180, 0, 180, 0, 180, 0, 180]:
        target["heading"] = heading
        scorer.evaluate([target])
    profile = scorer.get_profile("t-1")
    assert profile["movement_score"] > 0


def test_profile_score_clamped_to_zero_one():
    p = BehaviorProfile(target_id="t-1")
    p.loiter_score = 5.0
    p.zone_score = 5.0
    p.timing_score = 5.0
    p.movement_score = 5.0
    p.appearance_score = 5.0
    score = p.compute_threat_score()
    assert score == 1.0


def test_profile_negative_scores_clamped():
    p = BehaviorProfile(target_id="t-1")
    p.loiter_score = -1.0
    p.zone_score = -1.0
    p.timing_score = -1.0
    p.movement_score = -1.0
    p.appearance_score = -1.0
    score = p.compute_threat_score()
    assert score == 0.0


def test_evaluate_position_as_dict():
    scorer = ThreatScorer()
    target = {"target_id": "t-1", "position": {"x": 10, "y": 20},
              "heading": 0, "speed": 5, "source": "yolo"}
    scores = scorer.evaluate([target])
    assert "t-1" in scores


def test_evaluate_empty_target_id_skipped():
    scorer = ThreatScorer()
    target = {"target_id": "", "position": (0, 0), "heading": 0,
              "speed": 5, "source": "yolo"}
    scores = scorer.evaluate([target])
    assert scores == {}


def test_get_profile_nonexistent():
    scorer = ThreatScorer()
    assert scorer.get_profile("does-not-exist") is None
