# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.convoy_detector."""

import time
import pytest
from tritium_lib.tracking.convoy_detector import (
    ConvoyDetector,
    TargetMotion,
    MIN_CONVOY_MEMBERS,
)


class FakeHistory:
    """Mock TargetHistory that returns deterministic trails."""

    def __init__(self):
        self._trails = {}

    def add_trail(self, target_id, points):
        self._trails[target_id] = points

    def get_target_ids(self):
        return list(self._trails.keys())

    def get_trail(self, target_id, max_points=10):
        trail = self._trails.get(target_id, [])
        return trail[-max_points:]


class FakeEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, data):
        self.events.append((topic, data))


def make_convoy_history(n_targets=4, base_time=1000.0):
    """Create a FakeHistory with n targets moving in the same direction."""
    history = FakeHistory()
    for i in range(n_targets):
        trail = []
        for t in range(5):
            # All targets moving NE at similar speed, within 50m of each other
            x = 10.0 * i + t * 5.0
            y = 10.0 * i + t * 5.0
            trail.append((x, y, base_time + t))
        history.add_trail(f"target-{i}", trail)
    return history


# --- Basic detection ---

def test_detect_convoy_from_comoving_targets():
    history = make_convoy_history(4)
    bus = FakeEventBus()
    det = ConvoyDetector(history=history, event_bus=bus)
    convoys = det.analyze()
    assert len(convoys) >= 1
    assert len(convoys[0]["member_target_ids"]) >= MIN_CONVOY_MEMBERS


def test_no_convoy_too_few_targets():
    history = FakeHistory()
    # Only 2 targets - below minimum
    for i in range(2):
        trail = [(i * 10 + t * 5, t * 5, 1000 + t) for t in range(5)]
        history.add_trail(f"target-{i}", trail)
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    assert len(convoys) == 0


def test_no_convoy_different_directions():
    history = FakeHistory()
    # Targets moving in different directions
    history.add_trail("t-0", [(t * 5, 0, 1000 + t) for t in range(5)])  # east
    history.add_trail("t-1", [(0, t * 5, 1000 + t) for t in range(5)])  # north
    history.add_trail("t-2", [(-t * 5, 0, 1000 + t) for t in range(5)])  # west
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    assert len(convoys) == 0


def test_no_convoy_stationary_targets():
    history = FakeHistory()
    for i in range(4):
        trail = [(i * 10, i * 10, 1000 + t) for t in range(5)]  # no movement
        history.add_trail(f"t-{i}", trail)
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    assert len(convoys) == 0


def test_no_history_returns_empty():
    det = ConvoyDetector(history=None)
    assert det.analyze() == []


# --- Event publishing ---

def test_convoy_detected_event():
    history = make_convoy_history(4)
    bus = FakeEventBus()
    det = ConvoyDetector(history=history, event_bus=bus)
    det.analyze()
    topics = [e[0] for e in bus.events]
    assert "convoy_detected" in topics


def test_no_event_bus_no_crash():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history, event_bus=None)
    convoys = det.analyze()
    assert len(convoys) >= 1


# --- Convoy state ---

def test_get_active_convoys():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history)
    det.analyze()
    active = det.get_active_convoys()
    assert len(active) >= 1
    assert active[0]["status"] == "active"


def test_get_summary():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history)
    det.analyze()
    summary = det.get_summary()
    assert summary.active_convoys >= 1
    assert summary.total_members >= MIN_CONVOY_MEMBERS


# --- Suspicious score ---

def test_suspicious_score_range():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    assert len(convoys) >= 1
    score = convoys[0]["suspicious_score"]
    assert 0.0 <= score <= 1.0


# --- Model conversion ---

def test_to_convoy_model():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    model = det.to_convoy_model(convoys[0])
    assert model.convoy_id == convoys[0]["convoy_id"]
    assert len(model.member_target_ids) >= MIN_CONVOY_MEMBERS


def test_to_visualization():
    history = make_convoy_history(4)
    det = ConvoyDetector(history=history)
    convoys = det.analyze()
    viz = det.to_visualization(convoys[0])
    assert "Convoy" in viz.label
    assert viz.color == "#fcee0a"


# --- Static math helpers ---

def test_circular_mean():
    # Mean of 0 and 360 should be near 0 (not 180)
    mean = ConvoyDetector._circular_mean([350, 10])
    assert mean < 10 or mean > 350


def test_variance():
    v = ConvoyDetector._variance([5, 5, 5])
    assert v == 0.0
    v2 = ConvoyDetector._variance([0, 10])
    assert v2 == 25.0
