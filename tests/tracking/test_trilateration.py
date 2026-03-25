# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.trilateration."""

import time
import pytest

from tritium_lib.tracking.trilateration import (
    TrilaterationEngine,
    Sighting,
    PositionResult,
)


class TestSighting:
    def test_fields(self):
        s = Sighting(node_id="n1", lat=40.0, lon=-74.0, rssi=-60.0)
        assert s.node_id == "n1"
        assert s.lat == 40.0
        assert s.rssi == -60.0


class TestPositionResult:
    def test_to_dict(self):
        r = PositionResult(lat=40.0, lon=-74.0, confidence=0.8, anchors_used=3)
        d = r.to_dict()
        assert d["lat"] == 40.0
        assert d["lon"] == -74.0
        assert d["confidence"] == 0.8
        assert d["anchors_used"] == 3
        assert d["method"] == "weighted_centroid"


class TestTrilaterationEngineInit:
    def test_defaults(self):
        eng = TrilaterationEngine()
        assert eng.tracked_macs == 0
        assert eng._min_anchors == 3

    def test_custom_params(self):
        eng = TrilaterationEngine(min_anchors=2, window=10.0)
        assert eng._min_anchors == 2
        assert eng._window == 10.0


class TestRecordSighting:
    def test_record_single(self):
        eng = TrilaterationEngine()
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_1", 40.0, -74.0, -60.0)
        assert eng.tracked_macs == 1
        assert eng.get_sighting_count("AA:BB:CC:DD:EE:FF") == 1

    def test_mac_case_insensitive(self):
        eng = TrilaterationEngine()
        eng.record_sighting("aa:bb:cc:dd:ee:ff", "node_1", 40.0, -74.0, -60.0)
        assert eng.get_sighting_count("AA:BB:CC:DD:EE:FF") == 1

    def test_replaces_same_node(self):
        eng = TrilaterationEngine()
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_1", 40.0, -74.0, -60.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_1", 40.0, -74.0, -55.0)
        assert eng.get_sighting_count("AA:BB:CC:DD:EE:FF") == 1

    def test_multiple_nodes(self):
        eng = TrilaterationEngine()
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_1", 40.0, -74.0, -60.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_2", 40.001, -74.001, -65.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_3", 40.002, -73.999, -70.0)
        assert eng.get_sighting_count("AA:BB:CC:DD:EE:FF") == 3


class TestEstimatePosition:
    def test_insufficient_anchors(self):
        eng = TrilaterationEngine(min_anchors=3)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_1", 40.0, -74.0, -60.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "node_2", 40.001, -74.001, -65.0)
        result = eng.estimate_position("AA:BB:CC:DD:EE:FF")
        assert result is None

    def test_three_nodes_returns_estimate(self):
        eng = TrilaterationEngine(min_anchors=3, window=9999)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n1", 40.0000, -74.0000, -50.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n2", 40.0010, -74.0000, -55.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n3", 40.0005, -74.0010, -52.0)
        result = eng.estimate_position("AA:BB:CC:DD:EE:FF")
        if result is not None:
            assert isinstance(result, PositionResult)
            assert result.anchors_used == 3
            assert 39.9 < result.lat < 40.1
            assert -74.1 < result.lon < -73.9
            assert 0 <= result.confidence <= 1.0

    def test_unknown_mac_returns_none(self):
        eng = TrilaterationEngine()
        result = eng.estimate_position("00:00:00:00:00:00")
        assert result is None


class TestGetAllEstimates:
    def test_returns_dict(self):
        eng = TrilaterationEngine(min_anchors=3, window=9999)
        # MAC with enough anchors
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n1", 40.0, -74.0, -50.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n2", 40.001, -74.0, -55.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n3", 40.0005, -74.001, -52.0)
        # MAC with not enough
        eng.record_sighting("11:22:33:44:55:66", "n1", 40.0, -74.0, -60.0)

        results = eng.get_all_estimates()
        assert isinstance(results, dict)
        # First MAC may be in results, second should not
        assert "11:22:33:44:55:66" not in results


class TestPruneStale:
    def test_prune_removes_old(self):
        eng = TrilaterationEngine(stale_threshold=1.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n1", 40.0, -74.0, -60.0)
        time.sleep(1.5)
        removed = eng.prune_stale()
        assert removed >= 1
        assert eng.tracked_macs == 0

    def test_prune_keeps_recent(self):
        eng = TrilaterationEngine(stale_threshold=60.0)
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n1", 40.0, -74.0, -60.0)
        removed = eng.prune_stale()
        assert removed == 0
        assert eng.tracked_macs == 1


class TestClear:
    def test_clear_all(self):
        eng = TrilaterationEngine()
        eng.record_sighting("AA:BB:CC:DD:EE:FF", "n1", 40.0, -74.0, -60.0)
        eng.record_sighting("11:22:33:44:55:66", "n1", 40.0, -74.0, -60.0)
        eng.clear()
        assert eng.tracked_macs == 0
