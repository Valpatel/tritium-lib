# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.correlation_strategies."""

import time
import math
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.target_tracker import TrackedTarget
from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.dossier import DossierStore
from tritium_lib.tracking.correlation_strategies import (
    StrategyScore,
    SpatialStrategy,
    TemporalStrategy,
    SignalPatternStrategy,
    WiFiProbeStrategy,
    DossierStrategy,
)


def _make_target(
    target_id: str,
    source: str,
    position: tuple[float, float] = (0.0, 0.0),
    last_seen: float | None = None,
    asset_type: str = "person",
) -> TrackedTarget:
    now = last_seen if last_seen is not None else time.monotonic()
    return TrackedTarget(
        target_id=target_id,
        name=target_id,
        alliance="unknown",
        asset_type=asset_type,
        position=position,
        source=source,
        last_seen=now,
        first_seen=now,
        confirming_sources={source},
    )


# --- StrategyScore ---

class TestStrategyScore:
    def test_fields(self):
        s = StrategyScore(strategy_name="test", score=0.5, detail="info")
        assert s.strategy_name == "test"
        assert s.score == 0.5
        assert s.detail == "info"


# --- SpatialStrategy ---

class TestSpatialStrategy:
    def test_zero_distance(self):
        s = SpatialStrategy(radius=5.0)
        t1 = _make_target("a", "ble", position=(10.0, 10.0))
        t2 = _make_target("b", "yolo", position=(10.0, 10.0))
        result = s.evaluate(t1, t2)
        assert result.score > 0.9

    def test_at_radius_boundary(self):
        s = SpatialStrategy(radius=5.0)
        t1 = _make_target("a", "ble", position=(0.0, 0.0))
        t2 = _make_target("b", "yolo", position=(5.0, 0.0))
        result = s.evaluate(t1, t2)
        # At exactly radius, score is small but non-zero due to 1.1x margin
        assert result.score < 0.15

    def test_beyond_radius(self):
        s = SpatialStrategy(radius=5.0)
        t1 = _make_target("a", "ble", position=(0.0, 0.0))
        t2 = _make_target("b", "yolo", position=(100.0, 0.0))
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_within_radius(self):
        s = SpatialStrategy(radius=10.0)
        t1 = _make_target("a", "ble", position=(0.0, 0.0))
        t2 = _make_target("b", "yolo", position=(3.0, 4.0))
        result = s.evaluate(t1, t2)
        assert 0.0 < result.score < 1.0

    def test_custom_radius(self):
        s = SpatialStrategy(radius=100.0)
        t1 = _make_target("a", "ble", position=(0.0, 0.0))
        t2 = _make_target("b", "yolo", position=(50.0, 0.0))
        result = s.evaluate(t1, t2)
        assert result.score > 0.0


# --- TemporalStrategy ---

class TestTemporalStrategy:
    def _build_history_with_movement(
        self,
        target_id: str,
        start_pos: tuple[float, float],
        heading_deg: float,
        speed: float,
        history: TargetHistory,
        n_points: int = 5,
        start_t: float = 100.0,
    ):
        rad = math.radians(heading_deg)
        dx_per_s = speed * math.sin(rad)
        dy_per_s = speed * math.cos(rad)
        for i in range(n_points):
            t = start_t + i
            x = start_pos[0] + dx_per_s * i
            y = start_pos[1] + dy_per_s * i
            history.record(target_id, (x, y), timestamp=t)

    def test_insufficient_history(self):
        h = TargetHistory()
        h.record("a", (0, 0), timestamp=100.0)
        s = TemporalStrategy(history=h)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0
        assert "insufficient" in result.detail

    def test_co_movement_same_heading(self):
        h = TargetHistory()
        self._build_history_with_movement("a", (0, 0), 90.0, 5.0, h)
        self._build_history_with_movement("b", (1, 0), 90.0, 5.0, h)
        s = TemporalStrategy(history=h)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score > 0.5

    def test_opposite_headings(self):
        h = TargetHistory()
        self._build_history_with_movement("a", (0, 0), 0.0, 5.0, h)
        self._build_history_with_movement("b", (0, 50), 180.0, 5.0, h)
        s = TemporalStrategy(history=h)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        # Heading score is 0 but speed match gives 0.4 (60% heading + 40% speed)
        assert result.score <= 0.4

    def test_both_stationary(self):
        h = TargetHistory()
        for i in range(5):
            h.record("a", (10.0, 10.0), timestamp=100.0 + i)
            h.record("b", (20.0, 20.0), timestamp=100.0 + i)
        s = TemporalStrategy(history=h)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_compute_heading_single_point(self):
        assert TemporalStrategy._compute_heading([(0, 0, 0)]) == 0.0

    def test_compute_speed_single_point(self):
        assert TemporalStrategy._compute_speed([(0, 0, 0)]) == 0.0


# --- SignalPatternStrategy ---

class TestSignalPatternStrategy:
    def test_same_source_returns_zero(self):
        s = SignalPatternStrategy()
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "ble", last_seen=now)
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_simultaneous_different_sources(self):
        s = SignalPatternStrategy()
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "yolo", last_seen=now)
        result = s.evaluate(t1, t2)
        assert result.score > 0.9

    def test_outside_window(self):
        s = SignalPatternStrategy(appearance_window=5.0)
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "yolo", last_seen=now - 10.0)
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_ble_yolo_boost(self):
        s = SignalPatternStrategy()
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "yolo", last_seen=now - 1.0)
        result_boosted = s.evaluate(t1, t2)

        t3 = _make_target("c", "wifi", last_seen=now)
        t4 = _make_target("d", "mesh", last_seen=now - 1.0)
        result_normal = s.evaluate(t3, t4)

        # BLE+YOLO pair should get a boost
        assert result_boosted.score >= result_normal.score


# --- WiFiProbeStrategy ---

class TestWiFiProbeStrategy:
    def test_non_ble_wifi_pair_returns_zero(self):
        s = WiFiProbeStrategy()
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_ble_wifi_probe_pair(self):
        s = WiFiProbeStrategy()
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "wifi_probe", last_seen=now)
        result = s.evaluate(t1, t2)
        assert result.score > 0.9

    def test_ble_wifi_probe_outside_window(self):
        s = WiFiProbeStrategy(max_window=5.0)
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t2 = _make_target("b", "wifi_probe", last_seen=now - 10.0)
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_same_observer_boost(self):
        s = WiFiProbeStrategy()
        now = time.monotonic()
        t1 = _make_target("a", "ble", last_seen=now)
        t1.observer_id = "node_1"
        t2 = _make_target("b", "wifi_probe", last_seen=now)
        t2.observer_id = "node_1"
        result = s.evaluate(t1, t2)
        assert result.score > 0.9


# --- DossierStrategy ---

class TestDossierStrategy:
    def test_no_prior_association(self):
        store = DossierStore()
        s = DossierStrategy(dossier_store=store)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_known_association(self):
        store = DossierStore()
        store.create_or_update("a", "ble", "b", "yolo", confidence=0.8)
        s = DossierStrategy(dossier_store=store)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        assert result.score >= 0.7

    def test_different_dossiers_returns_zero(self):
        store = DossierStore()
        store.create_or_update("a", "ble", "x", "yolo", confidence=0.5)
        store.create_or_update("b", "ble", "y", "yolo", confidence=0.5)
        s = DossierStrategy(dossier_store=store)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "ble")
        result = s.evaluate(t1, t2)
        assert result.score == 0.0

    def test_repeated_correlations_increase_score(self):
        store = DossierStore()
        store.create_or_update("a", "ble", "b", "yolo", confidence=0.5)
        store.create_or_update("a", "ble", "b", "yolo", confidence=0.6)
        store.create_or_update("a", "ble", "b", "yolo", confidence=0.7)
        s = DossierStrategy(dossier_store=store)
        t1 = _make_target("a", "ble")
        t2 = _make_target("b", "yolo")
        result = s.evaluate(t1, t2)
        # correlation_count should be 3, so score = min(1.0, 0.7 + 0.1 * 3) = 1.0
        assert result.score >= 0.9
