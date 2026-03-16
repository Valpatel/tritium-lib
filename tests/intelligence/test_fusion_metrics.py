# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for FusionMetrics — correlation pipeline health tracking."""

import pytest

from tritium_lib.intelligence.fusion_metrics import FusionMetrics, StrategyMetric


class TestStrategyMetric:
    """Unit tests for StrategyMetric dataclass."""

    def test_defaults(self):
        sm = StrategyMetric(name="spatial")
        assert sm.accuracy == 0.0
        assert sm.contribution_rate == 0.0
        assert sm.avg_score == 0.0

    def test_accuracy_calculation(self):
        sm = StrategyMetric(name="spatial", confirmed=8, rejected=2)
        assert sm.accuracy == pytest.approx(0.8, abs=0.01)

    def test_contribution_rate(self):
        sm = StrategyMetric(name="temporal", evaluations=10, contributed=7)
        assert sm.contribution_rate == pytest.approx(0.7, abs=0.01)

    def test_avg_score(self):
        sm = StrategyMetric(name="signal", evaluations=4, total_score=2.0)
        assert sm.avg_score == pytest.approx(0.5, abs=0.01)

    def test_to_dict(self):
        sm = StrategyMetric(name="dossier", evaluations=10, contributed=5,
                            confirmed=3, rejected=1, total_score=4.0)
        d = sm.to_dict()
        assert d["name"] == "dossier"
        assert d["evaluations"] == 10
        assert d["accuracy"] == pytest.approx(0.75, abs=0.01)


class TestFusionMetrics:
    """Unit tests for FusionMetrics."""

    def test_record_fusion(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.85,
                        [("spatial", 0.9), ("temporal", 0.3)])
        status = m.get_status()
        assert status["total_fusions"] == 1
        assert len(status["strategies"]) == 2

    def test_source_pair_stats(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.8)
        m.record_fusion("camera", "ble", 0.7)  # same pair reversed
        m.record_fusion("wifi", "camera", 0.6)

        pairs = m.get_source_pair_stats()
        # ble+camera should have count 2 (normalized order)
        assert pairs.get("ble+camera", 0) == 2
        assert pairs.get("camera+wifi", 0) == 1

    def test_record_feedback_confirmed(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.85,
                        [("spatial", 0.9)],
                        primary_id="ble_abc", secondary_id="cam_123")

        result = m.record_feedback("ble_abc", "cam_123", confirmed=True)
        assert result is True
        assert m.get_confirmation_rate() == 1.0

    def test_record_feedback_rejected(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.85,
                        [("spatial", 0.9)],
                        primary_id="ble_abc", secondary_id="cam_123")

        m.record_feedback("ble_abc", "cam_123", confirmed=False)
        assert m.get_confirmation_rate() == 0.0

    def test_feedback_reversed_ids(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.85,
                        primary_id="ble_abc", secondary_id="cam_123")

        result = m.record_feedback("cam_123", "ble_abc", confirmed=True)
        assert result is True

    def test_feedback_unknown_ids(self):
        m = FusionMetrics()
        result = m.record_feedback("unknown1", "unknown2", confirmed=True)
        assert result is False

    def test_hourly_rate(self):
        m = FusionMetrics(window_seconds=3600.0)
        for _ in range(10):
            m.record_fusion("ble", "camera", 0.8)
        rate = m.get_hourly_rate()
        assert rate == pytest.approx(10.0, abs=0.5)

    def test_strategy_evaluation_standalone(self):
        m = FusionMetrics()
        m.record_strategy_evaluation("spatial", 0.8)
        m.record_strategy_evaluation("spatial", 0.6)
        m.record_strategy_evaluation("temporal", 0.0)

        perf = m.get_strategy_performance()
        assert len(perf) == 2
        spatial = next(p for p in perf if p["name"] == "spatial")
        assert spatial["evaluations"] == 2
        assert spatial["contributed"] == 2

    def test_weight_recommendations_with_feedback(self):
        m = FusionMetrics()
        for i in range(10):
            m.record_fusion("ble", "camera", 0.8,
                            [("spatial", 0.9), ("temporal", 0.3)],
                            primary_id=f"a{i}", secondary_id=f"b{i}")
            m.record_feedback(f"a{i}", f"b{i}", confirmed=(i < 8))

        recs = m.get_strategy_weights_recommendation()
        assert "spatial" in recs
        assert "temporal" in recs
        assert abs(sum(recs.values()) - 1.0) < 0.01

    def test_weight_recommendations_insufficient_feedback(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.8,
                        [("spatial", 0.9), ("temporal", 0.3)])
        recs = m.get_strategy_weights_recommendation()
        # Should still return something based on contribution rate
        assert len(recs) == 2

    def test_get_status_complete(self):
        m = FusionMetrics()
        m.record_fusion("ble", "camera", 0.85,
                        [("spatial", 0.9), ("temporal", 0.3)])
        status = m.get_status()

        # Verify all expected keys are present
        assert "total_fusions" in status
        assert "confirmation_rate" in status
        assert "hourly_rate" in status
        assert "source_pairs" in status
        assert "strategies" in status
        assert "window_fusions" in status

    def test_event_pruning(self):
        m = FusionMetrics()
        m._max_events = 10  # Small limit for testing
        for i in range(20):
            m.record_fusion("ble", "camera", 0.8)
        assert len(m._events) == 10
        assert m._total_fusions == 20
