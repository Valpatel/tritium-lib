# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for feature vector models."""

import pytest
from datetime import datetime, timezone

from tritium_lib.models.feature_vector import (
    AggregatedFeatures,
    ClassificationFeedback,
    EdgeIntelligenceMetrics,
    FeatureSource,
    FeatureVector,
)


class TestFeatureVector:
    """Tests for FeatureVector model."""

    def test_create_minimal(self):
        fv = FeatureVector(source_id="node-01")
        assert fv.source_id == "node-01"
        assert fv.features == {}
        assert fv.version == 1
        assert fv.source_type == FeatureSource.BLE

    def test_create_with_features(self):
        fv = FeatureVector(
            source_id="node-01",
            mac="AA:BB:CC:DD:EE:FF",
            features={"oui_hash": 0.42, "name_length": 8.0, "rssi_near_pct": 0.3},
            version=1,
        )
        assert fv.mac == "AA:BB:CC:DD:EE:FF"
        assert fv.features["oui_hash"] == pytest.approx(0.42)
        assert len(fv.features) == 3

    def test_feature_list_sorted(self):
        fv = FeatureVector(
            source_id="n1",
            features={"c": 3.0, "a": 1.0, "b": 2.0},
        )
        assert fv.feature_list() == [1.0, 2.0, 3.0]

    def test_feature_list_with_keys(self):
        fv = FeatureVector(
            source_id="n1",
            features={"x": 10.0, "y": 20.0, "z": 30.0},
        )
        assert fv.feature_list(["z", "x"]) == [30.0, 10.0]

    def test_feature_list_missing_key(self):
        fv = FeatureVector(
            source_id="n1",
            features={"x": 10.0},
        )
        assert fv.feature_list(["x", "missing"]) == [10.0, 0.0]

    def test_source_types(self):
        for src in FeatureSource:
            fv = FeatureVector(source_id="n1", source_type=src)
            assert fv.source_type == src

    def test_serialization_roundtrip(self):
        fv = FeatureVector(
            source_id="edge-42",
            mac="11:22:33:44:55:66",
            features={"oui_hash": 0.5, "name_length": 12.0},
            source_type=FeatureSource.WIFI,
        )
        data = fv.model_dump()
        fv2 = FeatureVector(**data)
        assert fv2.source_id == fv.source_id
        assert fv2.features == fv.features
        assert fv2.source_type == FeatureSource.WIFI


class TestAggregatedFeatures:
    """Tests for AggregatedFeatures model."""

    def test_create_empty(self):
        af = AggregatedFeatures(mac="AA:BB:CC:DD:EE:FF")
        assert af.mac == "AA:BB:CC:DD:EE:FF"
        assert af.vectors == []
        assert af.node_count == 0

    def test_compute_mean_empty(self):
        af = AggregatedFeatures(mac="XX")
        assert af.compute_mean() == {}

    def test_compute_mean_single(self):
        fv = FeatureVector(source_id="n1", features={"a": 4.0, "b": 8.0})
        af = AggregatedFeatures(mac="XX", vectors=[fv])
        mean = af.compute_mean()
        assert mean["a"] == pytest.approx(4.0)
        assert mean["b"] == pytest.approx(8.0)

    def test_compute_mean_multiple(self):
        fv1 = FeatureVector(source_id="n1", features={"a": 2.0, "b": 6.0})
        fv2 = FeatureVector(source_id="n2", features={"a": 4.0, "b": 10.0})
        af = AggregatedFeatures(mac="XX", vectors=[fv1, fv2])
        mean = af.compute_mean()
        assert mean["a"] == pytest.approx(3.0)
        assert mean["b"] == pytest.approx(8.0)

    def test_compute_mean_sparse_features(self):
        fv1 = FeatureVector(source_id="n1", features={"a": 2.0})
        fv2 = FeatureVector(source_id="n2", features={"b": 10.0})
        af = AggregatedFeatures(mac="XX", vectors=[fv1, fv2])
        mean = af.compute_mean()
        assert mean["a"] == pytest.approx(2.0)
        assert mean["b"] == pytest.approx(10.0)


class TestClassificationFeedback:
    """Tests for ClassificationFeedback model."""

    def test_create(self):
        fb = ClassificationFeedback(
            mac="AA:BB:CC:DD:EE:FF",
            predicted_type="phone",
            confidence=0.92,
        )
        assert fb.mac == "AA:BB:CC:DD:EE:FF"
        assert fb.predicted_type == "phone"
        assert fb.confidence == pytest.approx(0.92)
        assert fb.confirmed_by == "ml_classifier"

    def test_custom_confirmed_by(self):
        fb = ClassificationFeedback(
            mac="XX",
            predicted_type="watch",
            confidence=0.8,
            confirmed_by="operator",
        )
        assert fb.confirmed_by == "operator"


class TestEdgeIntelligenceMetrics:
    """Tests for EdgeIntelligenceMetrics model."""

    def test_create_defaults(self):
        m = EdgeIntelligenceMetrics(node_id="edge-01")
        assert m.node_id == "edge-01"
        assert m.total_devices_seen == 0
        assert m.accuracy_rate == 0.0
        assert m.last_feedback_ts is None

    def test_create_with_values(self):
        m = EdgeIntelligenceMetrics(
            node_id="edge-01",
            total_devices_seen=42,
            devices_classified=30,
            feedback_received=25,
            accuracy_rate=0.85,
            feature_vectors_sent=100,
        )
        assert m.total_devices_seen == 42
        assert m.devices_classified == 30
        assert m.accuracy_rate == pytest.approx(0.85)


class TestImports:
    """Test that models are importable from the top-level package."""

    def test_import_from_models(self):
        from tritium_lib.models import (
            AggregatedFeatures,
            ClassificationFeedback,
            EdgeIntelligenceMetrics,
            FeatureSource,
            FeatureVector,
        )
        assert FeatureVector is not None
        assert ClassificationFeedback is not None
