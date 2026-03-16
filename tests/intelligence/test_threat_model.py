# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ThreatModel — unified threat assessment engine."""
from __future__ import annotations

import time

import pytest

from tritium_lib.intelligence.threat_model import (
    DEFAULT_SIGNAL_WEIGHTS,
    ThreatAssessment,
    ThreatLevel,
    ThreatModel,
    ThreatSignal,
    score_to_threat_level,
)


class TestThreatSignal:
    """ThreatSignal dataclass."""

    def test_default_values(self):
        sig = ThreatSignal()
        assert sig.signal_type == ""
        assert sig.score == 0.0
        assert sig.ttl_seconds == 0.0
        assert sig.target_id == ""

    def test_not_expired_no_ttl(self):
        sig = ThreatSignal(timestamp=time.time() - 9999, ttl_seconds=0.0)
        assert not sig.is_expired()

    def test_expired_with_ttl(self):
        sig = ThreatSignal(timestamp=time.time() - 100, ttl_seconds=50)
        assert sig.is_expired()

    def test_not_expired_within_ttl(self):
        sig = ThreatSignal(timestamp=time.time(), ttl_seconds=3600)
        assert not sig.is_expired()

    def test_to_dict(self):
        sig = ThreatSignal(
            signal_type="behavior",
            score=0.75,
            source="loiter_detector",
            detail="Loitered 10min",
            target_id="ble_test",
        )
        d = sig.to_dict()
        assert d["signal_type"] == "behavior"
        assert d["score"] == 0.75
        assert d["source"] == "loiter_detector"
        assert d["target_id"] == "ble_test"


class TestThreatAssessment:
    """ThreatAssessment dataclass."""

    def test_default_values(self):
        a = ThreatAssessment()
        assert a.composite_score == 0.0
        assert a.threat_level == ThreatLevel.GREEN
        assert a.signal_count == 0

    def test_to_dict(self):
        a = ThreatAssessment(
            target_id="t1",
            composite_score=0.65,
            threat_level=ThreatLevel.RED,
            sub_scores={"behavior": 0.8},
            signal_count=3,
        )
        d = a.to_dict()
        assert d["threat_level"] == "RED"
        assert d["composite_score"] == 0.65
        assert d["sub_scores"]["behavior"] == 0.8


class TestThreatLevel:
    """ThreatLevel enum."""

    def test_values(self):
        assert ThreatLevel.GREEN.value == "GREEN"
        assert ThreatLevel.CRITICAL.value == "CRITICAL"

    def test_score_to_level(self):
        assert score_to_threat_level(0.0) == ThreatLevel.GREEN
        assert score_to_threat_level(0.1) == ThreatLevel.GREEN
        assert score_to_threat_level(0.2) == ThreatLevel.YELLOW
        assert score_to_threat_level(0.4) == ThreatLevel.ORANGE
        assert score_to_threat_level(0.6) == ThreatLevel.RED
        assert score_to_threat_level(0.8) == ThreatLevel.CRITICAL
        assert score_to_threat_level(1.0) == ThreatLevel.CRITICAL


class TestThreatModel:
    """ThreatModel core functionality."""

    def test_empty_assessment(self):
        model = ThreatModel()
        a = model.assess("unknown_target")
        assert a.composite_score == 0.0
        assert a.threat_level == ThreatLevel.GREEN
        assert a.signal_count == 0

    def test_single_signal(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior",
            score=0.8,
            target_id="t1",
            source="loiter",
        ))
        a = model.assess("t1")
        assert a.signal_count == 1
        assert a.composite_score > 0.0
        assert a.sub_scores["behavior"] == 0.8

    def test_multiple_signal_types(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.6, target_id="t1",
        ))
        model.add_signal(ThreatSignal(
            signal_type="threat_feed", score=0.9, target_id="t1",
        ))
        model.add_signal(ThreatSignal(
            signal_type="zone_violation", score=0.5, target_id="t1",
        ))
        a = model.assess("t1")
        assert a.signal_count == 3
        assert a.sub_scores["behavior"] == 0.6
        assert a.sub_scores["threat_feed"] == 0.9
        assert a.sub_scores["zone_violation"] == 0.5
        # Composite should be weighted average
        assert 0.0 < a.composite_score < 1.0

    def test_max_score_per_type(self):
        """Multiple signals of same type: max is used."""
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.3, target_id="t1",
        ))
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.9, target_id="t1",
        ))
        a = model.assess("t1")
        assert a.sub_scores["behavior"] == 0.9

    def test_score_clamping(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=1.5, target_id="t1",
        ))
        model.add_signal(ThreatSignal(
            signal_type="classification", score=-0.5, target_id="t2",
        ))
        a1 = model.assess("t1")
        assert a1.sub_scores["behavior"] <= 1.0
        a2 = model.assess("t2")
        assert a2.sub_scores["classification"] >= 0.0

    def test_expired_signals_filtered(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior",
            score=0.9,
            target_id="t1",
            timestamp=time.time() - 200,
            ttl_seconds=100,
        ))
        a = model.assess("t1")
        assert a.signal_count == 0
        assert a.composite_score == 0.0

    def test_empty_target_id_ignored(self):
        model = ThreatModel()
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.5, target_id=""))
        assert len(model._signals) == 0

    def test_assess_all(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.3, target_id="t1"))
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.9, target_id="t2"))
        results = model.assess_all()
        assert len(results) == 2
        # Sorted by composite score descending
        assert results[0].target_id == "t2"

    def test_clear_signals(self):
        model = ThreatModel()
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.5, target_id="t1"))
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.6, target_id="t1"))
        removed = model.clear_signals("t1")
        assert removed == 2
        a = model.assess("t1")
        assert a.signal_count == 0

    def test_clear_all(self):
        model = ThreatModel()
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.5, target_id="t1"))
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.6, target_id="t2"))
        total = model.clear_all()
        assert total == 2

    def test_get_signals(self):
        model = ThreatModel()
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.5, target_id="t1"))
        model.add_signal(ThreatSignal(signal_type="threat_feed", score=0.9, target_id="t1"))
        all_sigs = model.get_signals("t1")
        assert len(all_sigs) == 2
        behavior_sigs = model.get_signals("t1", signal_type="behavior")
        assert len(behavior_sigs) == 1

    def test_get_targets_above(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.1, target_id="low"))
        model.add_signal(ThreatSignal(signal_type="threat_feed", score=1.0, target_id="high"))
        above = model.get_targets_above(0.2)
        assert "high" in above
        assert "low" not in above

    def test_get_stats(self):
        model = ThreatModel()
        model.add_signal(ThreatSignal(signal_type="behavior", score=0.5, target_id="t1"))
        stats = model.get_stats()
        assert stats["total_targets"] == 1
        assert stats["total_signals"] == 1
        assert stats["decay_enabled"] is True
        assert "weights" in stats

    def test_max_signals_per_target(self):
        model = ThreatModel(max_signals_per_target=5)
        for i in range(10):
            model.add_signal(ThreatSignal(
                signal_type="behavior", score=0.1 * i, target_id="t1",
                timestamp=time.time() + i,
            ))
        with model._lock:
            assert len(model._signals["t1"]) <= 5

    def test_top_signals_in_assessment(self):
        model = ThreatModel(decay_enabled=False)
        for i in range(7):
            model.add_signal(ThreatSignal(
                signal_type="behavior",
                score=0.1 * (i + 1),
                target_id="t1",
                source=f"src_{i}",
            ))
        a = model.assess("t1")
        assert len(a.top_signals) <= 5
        # Top signal should have highest score
        assert a.top_signals[0]["score"] >= a.top_signals[-1]["score"]

    def test_decay_reduces_score(self):
        """Time decay should reduce effective score of old signals."""
        model = ThreatModel(decay_enabled=True)
        # Signal from 2 hours ago
        model.add_signal(ThreatSignal(
            signal_type="behavior",
            score=1.0,
            target_id="t1",
            timestamp=time.time() - 7200,
        ))
        a = model.assess("t1")
        # After 2 hours (2 half-lives), effective score should be ~0.25
        assert a.sub_scores["behavior"] < 0.5

    def test_to_dict(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.5, target_id="t1",
        ))
        d = model.to_dict()
        assert "stats" in d
        assert "assessments" in d
        assert len(d["assessments"]) == 1

    def test_custom_weights(self):
        weights = {"behavior": 1.0}
        model = ThreatModel(weights=weights, decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.7, target_id="t1",
        ))
        a = model.assess("t1")
        assert abs(a.composite_score - 0.7) < 0.01

    def test_threat_level_critical(self):
        model = ThreatModel(decay_enabled=False)
        # Max out all signal types
        for sig_type in DEFAULT_SIGNAL_WEIGHTS:
            model.add_signal(ThreatSignal(
                signal_type=sig_type, score=1.0, target_id="t1",
            ))
        a = model.assess("t1")
        assert a.threat_level == ThreatLevel.CRITICAL

    def test_cache_invalidation_on_new_signal(self):
        model = ThreatModel(decay_enabled=False)
        model.add_signal(ThreatSignal(
            signal_type="behavior", score=0.3, target_id="t1",
        ))
        a1 = model.assess("t1")
        model.add_signal(ThreatSignal(
            signal_type="threat_feed", score=0.9, target_id="t1",
        ))
        a2 = model.assess("t1")
        assert a2.composite_score > a1.composite_score
