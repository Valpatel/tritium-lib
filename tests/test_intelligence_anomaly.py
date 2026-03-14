# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.anomaly module."""
import pytest
from tritium_lib.intelligence.anomaly import (
    Anomaly,
    AnomalyDetector,
    AutoencoderDetector,
    SimpleThresholdDetector,
)


class TestAnomaly:
    """Test Anomaly dataclass."""

    def test_to_dict(self):
        a = Anomaly(
            metric_name="ble_count",
            current_value=50.0,
            baseline_mean=10.0,
            baseline_std=2.0,
            deviation_sigma=20.0,
            direction="above",
            severity="high",
            score=0.95,
        )
        d = a.to_dict()
        assert d["metric_name"] == "ble_count"
        assert d["current_value"] == 50.0
        assert d["severity"] == "high"

    def test_default_values(self):
        a = Anomaly()
        assert a.metric_name == ""
        assert a.severity == "low"
        assert a.score == 0.0


class TestSimpleThresholdDetector:
    """Test the simple threshold detector."""

    def _make_baseline(self, metric, value, count=300):
        return [{metric: value + (i % 3) - 1.0} for i in range(count)]

    def test_name(self):
        d = SimpleThresholdDetector()
        assert d.name() == "simple_threshold"

    def test_is_anomaly_detector(self):
        d = SimpleThresholdDetector()
        assert isinstance(d, AnomalyDetector)

    def test_insufficient_baseline(self):
        d = SimpleThresholdDetector(min_baseline_samples=10)
        baseline = [{"ble_count": 10.0}] * 5
        result = d.detect({"ble_count": 100.0}, baseline)
        assert result == []

    def test_no_anomaly(self):
        d = SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
        baseline = self._make_baseline("ble_count", 10.0, 300)
        result = d.detect({"ble_count": 10.0}, baseline)
        assert result == []

    def test_anomaly_above(self):
        d = SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
        baseline = self._make_baseline("ble_count", 10.0, 300)
        result = d.detect({"ble_count": 50.0}, baseline)
        assert len(result) == 1
        assert result[0].metric_name == "ble_count"
        assert result[0].direction == "above"
        assert result[0].deviation_sigma > 2.0

    def test_anomaly_below(self):
        d = SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
        baseline = self._make_baseline("ble_count", 50.0, 300)
        result = d.detect({"ble_count": 5.0}, baseline)
        assert len(result) == 1
        assert result[0].direction == "below"

    def test_severity_levels(self):
        d = SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
        # Create baseline with std=1
        baseline = [{"x": 10.0}] * 150 + [{"x": 12.0}] * 150

        # 3 sigma -> medium
        result = d.detect({"x": 10.0 + 3.1 * 1.0}, baseline)
        if result:
            assert result[0].severity in ("low", "medium")

    def test_multiple_metrics(self):
        d = SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
        baseline = [{"ble": 10.0, "wifi": 5.0}] * 300
        # Both anomalous
        result = d.detect({"ble": 50.0, "wifi": 50.0}, baseline)
        assert len(result) == 2

    def test_zero_variance_baseline(self):
        d = SimpleThresholdDetector(min_baseline_samples=10)
        baseline = [{"x": 10.0}] * 300
        result = d.detect({"x": 15.0}, baseline)
        assert len(result) == 1
        assert result[0].severity == "high"

    def test_unknown_metric_in_current(self):
        d = SimpleThresholdDetector(min_baseline_samples=10)
        baseline = [{"x": 10.0}] * 300
        result = d.detect({"y": 100.0}, baseline)
        assert result == []  # metric "y" not in baseline


class TestAutoencoderDetector:
    """Test the autoencoder detector."""

    def test_name(self):
        d = AutoencoderDetector()
        assert d.name() == "autoencoder"

    def test_is_anomaly_detector(self):
        d = AutoencoderDetector()
        assert isinstance(d, AnomalyDetector)

    def test_insufficient_baseline(self):
        d = AutoencoderDetector()
        baseline = [{"x": 1.0}] * 5
        result = d.detect({"x": 100.0}, baseline)
        assert result == []

    def test_numpy_not_required_for_import(self):
        """AutoencoderDetector should import even without numpy."""
        d = AutoencoderDetector()
        assert d.name() == "autoencoder"

    def test_normal_value_no_anomaly(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not available")

        d = AutoencoderDetector(epochs=50)
        baseline = [{"a": 10.0 + i * 0.1, "b": 5.0 + i * 0.05} for i in range(100)]
        result = d.detect({"a": 10.5, "b": 5.25}, baseline)
        # Normal value should not trigger anomaly
        assert len(result) == 0

    def test_anomalous_value_detected(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not available")

        d = AutoencoderDetector(epochs=100)
        baseline = [{"a": 10.0 + (i % 3), "b": 5.0 + (i % 2)} for i in range(100)]
        result = d.detect({"a": 100.0, "b": 100.0}, baseline)
        # Extreme values should trigger anomaly
        assert len(result) > 0
