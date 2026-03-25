# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CrowdDynamicsAnalyzer — crowd formation, flow, density, dispersal."""

from __future__ import annotations

import math

import pytest

from tritium_lib.intelligence.crowd_dynamics import (
    CrowdCluster,
    CrowdDynamicsAnalyzer,
    CrowdEvent,
    CrowdEventType,
    CrowdState,
    DensityCell,
    DensityEstimator,
    DispersalDetector,
    FlowAnalyzer,
    FlowVector,
    FormationDetector,
    DEFAULT_CLUSTER_RADIUS_M,
    MIN_CROWD_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_targets(positions: list[tuple[float, float]], prefix: str = "t") -> list[dict]:
    """Build a target list from positions."""
    return [
        {"target_id": f"{prefix}{i}", "position": pos}
        for i, pos in enumerate(positions)
    ]


def _cluster_at(center: tuple[float, float], n: int, spread: float = 2.0) -> list[dict]:
    """Create *n* targets clustered around *center* within *spread* meters."""
    targets = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        r = spread * (0.3 + 0.7 * (i % 3) / 3)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        targets.append({"target_id": f"c{i}", "position": (x, y)})
    return targets


# ============================================================================
# CrowdCluster dataclass
# ============================================================================

class TestCrowdCluster:
    def test_size_property(self):
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        assert c.size == 3

    def test_duration_property(self):
        c = CrowdCluster(cluster_id="c1", member_ids=["a"],
                         center=(0, 0), radius=5.0,
                         first_seen=100.0, last_seen=160.0)
        assert c.duration_s == pytest.approx(60.0)

    def test_duration_zero_when_same(self):
        c = CrowdCluster(cluster_id="c1", member_ids=["a"],
                         center=(0, 0), radius=5.0,
                         first_seen=100.0, last_seen=100.0)
        assert c.duration_s == 0.0

    def test_area_m2(self):
        c = CrowdCluster(cluster_id="c1", member_ids=["a"],
                         center=(0, 0), radius=10.0)
        assert c.area_m2 == pytest.approx(math.pi * 100.0)

    def test_to_dict_has_all_keys(self):
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b"],
                         center=(5.5, 3.2), radius=7.0,
                         first_seen=1.0, last_seen=2.0)
        d = c.to_dict()
        assert d["cluster_id"] == "c1"
        assert d["size"] == 2
        assert "center" in d
        assert "radius" in d
        assert "state" in d
        assert "density" in d
        assert "flow_heading" in d


# ============================================================================
# CrowdEvent dataclass
# ============================================================================

class TestCrowdEvent:
    def test_to_dict(self):
        e = CrowdEvent(
            event_id="e1", event_type=CrowdEventType.FORMATION,
            cluster_id="c1", timestamp=100.0, member_count=5,
        )
        d = e.to_dict()
        assert d["event_type"] == "formation"
        assert d["cluster_id"] == "c1"
        assert d["member_count"] == 5


# ============================================================================
# detect_clusters — basic clustering
# ============================================================================

class TestDetectClusters:
    def test_tight_group_forms_one_cluster(self):
        """Targets within cluster_radius form a single cluster."""
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 1
        assert clusters[0].size == 5

    def test_two_separated_groups(self):
        """Two well-separated groups produce two clusters."""
        group_a = _cluster_at((10, 10), n=4, spread=2.0)
        group_b = _cluster_at((200, 200), n=4, spread=2.0)
        # Give unique IDs
        for i, t in enumerate(group_b):
            t["target_id"] = f"b{i}"
        targets = group_a + group_b
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 2

    def test_too_few_targets_no_cluster(self):
        """Fewer than min_crowd_size targets should not form a cluster."""
        targets = _make_targets([(0, 0), (1, 1)])
        analyzer = CrowdDynamicsAnalyzer(min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 0

    def test_scattered_targets_no_cluster(self):
        """Widely scattered targets should not form a cluster."""
        positions = [(i * 100, i * 100) for i in range(5)]
        targets = _make_targets(positions)
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=10.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 0

    def test_cluster_centroid(self):
        """Cluster centroid should be near the mean position."""
        positions = [(10, 10), (12, 10), (11, 12), (10, 11)]
        targets = _make_targets(positions)
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 1
        cx, cy = clusters[0].center
        assert abs(cx - 10.75) < 0.01
        assert abs(cy - 10.75) < 0.01

    def test_cluster_radius_positive(self):
        """Cluster radius should always be > 0."""
        targets = _cluster_at((50, 50), n=4, spread=3.0)
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 1
        assert clusters[0].radius > 0

    def test_empty_targets(self):
        analyzer = CrowdDynamicsAnalyzer()
        clusters = analyzer.detect_clusters([], timestamp=1.0)
        assert clusters == []

    def test_cluster_ids_are_unique(self):
        """Every cluster gets a unique ID."""
        group_a = _cluster_at((10, 10), n=5, spread=2.0)
        group_b = _cluster_at((200, 200), n=5, spread=2.0)
        for i, t in enumerate(group_b):
            t["target_id"] = f"b{i}"
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        clusters = analyzer.detect_clusters(group_a + group_b, timestamp=1.0)
        ids = [c.cluster_id for c in clusters]
        assert len(set(ids)) == len(ids)


# ============================================================================
# FormationDetector
# ============================================================================

class TestFormationDetector:
    def test_new_cluster_triggers_formation(self):
        detector = FormationDetector(min_size=3)
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        events = detector.detect([c], timestamp=1.0)
        assert len(events) == 1
        assert events[0].event_type == CrowdEventType.FORMATION

    def test_same_cluster_no_repeat(self):
        detector = FormationDetector(min_size=3)
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        detector.detect([c], timestamp=1.0)
        events = detector.detect([c], timestamp=2.0)
        assert len(events) == 0

    def test_too_small_not_detected(self):
        detector = FormationDetector(min_size=3)
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b"],
                         center=(0, 0), radius=5.0)
        events = detector.detect([c], timestamp=1.0)
        assert len(events) == 0

    def test_reset_allows_redetection(self):
        detector = FormationDetector(min_size=3)
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        detector.detect([c], timestamp=1.0)
        detector.reset()
        events = detector.detect([c], timestamp=2.0)
        assert len(events) == 1


# ============================================================================
# DispersalDetector
# ============================================================================

class TestDispersalDetector:
    def test_disappeared_cluster_fires_dispersal(self):
        detector = DispersalDetector()
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        detector.detect([c], timestamp=1.0)
        events = detector.detect([], timestamp=2.0)
        assert len(events) == 1
        assert events[0].event_type == CrowdEventType.DISPERSAL
        assert events[0].previous_count == 3
        assert events[0].member_count == 0

    def test_shrunk_cluster_fires_dispersal(self):
        detector = DispersalDetector(shrink_ratio=0.5)
        c1 = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c", "d", "e", "f"],
                          center=(0, 0), radius=5.0)
        detector.detect([c1], timestamp=1.0)
        c2 = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                          center=(0, 0), radius=5.0)
        events = detector.detect([c2], timestamp=2.0)
        assert len(events) == 1
        assert events[0].event_type == CrowdEventType.DISPERSAL

    def test_stable_cluster_no_event(self):
        detector = DispersalDetector(shrink_ratio=0.5)
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        detector.detect([c], timestamp=1.0)
        events = detector.detect([c], timestamp=2.0)
        assert len(events) == 0

    def test_reset_clears_history(self):
        detector = DispersalDetector()
        c = CrowdCluster(cluster_id="c1", member_ids=["a", "b", "c"],
                         center=(0, 0), radius=5.0)
        detector.detect([c], timestamp=1.0)
        detector.reset()
        # After reset, disappearing should not fire because there's no history
        events = detector.detect([], timestamp=2.0)
        assert len(events) == 0


# ============================================================================
# FlowAnalyzer
# ============================================================================

class TestFlowAnalyzer:
    def test_cluster_flow_north(self):
        """Targets moving north produce heading ~0 degrees."""
        flow = FlowAnalyzer()
        # Record previous positions
        targets_t0 = [
            {"target_id": "a", "position": (10.0, 10.0)},
            {"target_id": "b", "position": (12.0, 10.0)},
        ]
        flow.record_positions(targets_t0, timestamp=0.0)

        # Targets moved north (positive y)
        targets_t1 = [
            {"target_id": "a", "position": (10.0, 20.0)},
            {"target_id": "b", "position": (12.0, 20.0)},
        ]
        cluster = CrowdCluster(
            cluster_id="c1", member_ids=["a", "b"],
            center=(11.0, 20.0), radius=2.0,
        )
        heading, speed = flow.compute_cluster_flow(cluster, targets_t1, timestamp=1.0)
        assert speed > 0
        # atan2(0, 10) = 0 degrees (north in our convention: atan2(vx, vy))
        assert abs(heading - 0.0) < 10.0 or abs(heading - 360.0) < 10.0

    def test_cluster_flow_east(self):
        """Targets moving east (positive x)."""
        flow = FlowAnalyzer()
        targets_t0 = [
            {"target_id": "a", "position": (0.0, 5.0)},
            {"target_id": "b", "position": (0.0, 7.0)},
        ]
        flow.record_positions(targets_t0, timestamp=0.0)

        targets_t1 = [
            {"target_id": "a", "position": (10.0, 5.0)},
            {"target_id": "b", "position": (10.0, 7.0)},
        ]
        cluster = CrowdCluster(
            cluster_id="c1", member_ids=["a", "b"],
            center=(10.0, 6.0), radius=2.0,
        )
        heading, speed = flow.compute_cluster_flow(cluster, targets_t1, timestamp=1.0)
        assert speed > 0
        # atan2(10, 0) = 90 degrees
        assert abs(heading - 90.0) < 10.0

    def test_no_previous_positions_zero_flow(self):
        flow = FlowAnalyzer()
        targets = [{"target_id": "a", "position": (10.0, 10.0)}]
        cluster = CrowdCluster(
            cluster_id="c1", member_ids=["a"],
            center=(10, 10), radius=1.0,
        )
        heading, speed = flow.compute_cluster_flow(cluster, targets, timestamp=1.0)
        assert speed == 0.0

    def test_grid_flow_returns_vectors(self):
        flow = FlowAnalyzer()
        targets_t0 = _make_targets([(10, 10), (20, 10), (30, 10)])
        flow.record_positions(targets_t0, timestamp=0.0)
        targets_t1 = _make_targets([(10, 20), (20, 20), (30, 20)])
        vectors = flow.compute_grid_flow(
            targets_t1, area=(0, 0, 40, 40), timestamp=1.0, resolution=4,
        )
        assert len(vectors) > 0
        for v in vectors:
            assert isinstance(v, FlowVector)
            assert v.speed_mps > 0

    def test_flow_vector_to_dict(self):
        v = FlowVector(x=5.0, y=10.0, heading_deg=90.0, speed_mps=3.5, sample_count=4)
        d = v.to_dict()
        assert d["heading_deg"] == 90.0
        assert d["speed_mps"] == 3.5
        assert d["sample_count"] == 4


# ============================================================================
# DensityEstimator
# ============================================================================

class TestDensityEstimator:
    def test_single_cell_density(self):
        """All targets in one cell should give correct density."""
        estimator = DensityEstimator(cell_size=100.0)
        targets = _make_targets([(5, 5), (6, 5), (5, 6), (7, 7)])
        cells = estimator.estimate(targets, area=(0, 0, 100, 100))
        assert len(cells) == 1
        # 4 targets in a 100x100 cell = 4/10000 = 0.0004
        assert cells[0].count == 4
        assert cells[0].density == pytest.approx(4.0 / 10000.0)

    def test_multiple_cells(self):
        """Targets in different cells produce multiple DensityCells."""
        estimator = DensityEstimator(cell_size=10.0)
        targets = _make_targets([(5, 5), (15, 15), (25, 25)])
        cells = estimator.estimate(targets, area=(0, 0, 30, 30))
        assert len(cells) == 3

    def test_empty_targets(self):
        estimator = DensityEstimator()
        cells = estimator.estimate([], area=(0, 0, 100, 100))
        assert cells == []

    def test_peak_density(self):
        estimator = DensityEstimator(cell_size=10.0)
        # Put 10 targets in one cell, 1 in another
        targets = _make_targets([(5, 5)] * 10 + [(50, 50)])
        peak = estimator.peak_density(targets, area=(0, 0, 100, 100))
        assert peak > 0
        # 10 targets in 10x10 = 0.1 targets/m^2
        assert peak == pytest.approx(10.0 / 100.0)

    def test_average_density(self):
        estimator = DensityEstimator(cell_size=10.0)
        targets = _make_targets([(5, 5)] * 4 + [(55, 55)] * 2)
        avg = estimator.average_density(targets, area=(0, 0, 100, 100))
        # Two cells: 4/100 and 2/100 -> avg = 0.03
        assert avg == pytest.approx((0.04 + 0.02) / 2)

    def test_density_cell_to_dict(self):
        cell = DensityCell(row=2, col=3, x=35.0, y=25.0, count=5, density=0.05)
        d = cell.to_dict()
        assert d["row"] == 2
        assert d["col"] == 3
        assert d["count"] == 5
        assert d["density"] == 0.05

    def test_targets_outside_area_ignored(self):
        estimator = DensityEstimator(cell_size=10.0)
        targets = _make_targets([(-5, -5), (5, 5), (200, 200)])
        cells = estimator.estimate(targets, area=(0, 0, 50, 50))
        total = sum(c.count for c in cells)
        assert total == 1  # only (5,5) is inside


# ============================================================================
# CrowdDynamicsAnalyzer — full update cycle
# ============================================================================

class TestCrowdDynamicsAnalyzer:
    def test_formation_event_on_first_update(self):
        """First update with a crowd should fire a formation event."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        events = analyzer.update(targets, timestamp=1.0)
        formation = [e for e in events if e.event_type == CrowdEventType.FORMATION]
        assert len(formation) == 1
        assert formation[0].member_count == 5

    def test_stable_crowd_no_events(self):
        """Repeated updates with same targets produce no new events."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        events = analyzer.update(targets, timestamp=2.0)
        # No formation (already known), no dispersal, no growth
        assert len(events) == 0

    def test_dispersal_when_crowd_removed(self):
        """Removing all targets should fire a dispersal event."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        events = analyzer.update([], timestamp=2.0)
        dispersal = [e for e in events if e.event_type == CrowdEventType.DISPERSAL]
        assert len(dispersal) >= 1

    def test_growth_event(self):
        """Growing a cluster by 50%+ fires a growth event."""
        analyzer = CrowdDynamicsAnalyzer(
            cluster_radius=15.0, min_crowd_size=3, growth_ratio=1.5,
        )
        # Start with 4 targets
        targets_small = _cluster_at((50, 50), n=4, spread=3.0)
        analyzer.update(targets_small, timestamp=1.0)

        # Grow to 8 targets (100% growth)
        targets_big = _cluster_at((50, 50), n=8, spread=3.0)
        events = analyzer.update(targets_big, timestamp=2.0)
        growth = [e for e in events if e.event_type == CrowdEventType.GROWTH]
        assert len(growth) >= 1

    def test_get_active_clusters(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=6, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        active = analyzer.get_active_clusters()
        assert len(active) == 1
        assert active[0].size == 6

    def test_get_cluster_by_id(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        active = analyzer.get_active_clusters()
        cid = active[0].cluster_id
        cluster = analyzer.get_cluster(cid)
        assert cluster is not None
        assert cluster.cluster_id == cid

    def test_get_cluster_returns_none_for_unknown(self):
        analyzer = CrowdDynamicsAnalyzer()
        assert analyzer.get_cluster("nonexistent") is None

    def test_get_events(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        events = analyzer.get_events(limit=10)
        assert len(events) >= 1

    def test_get_stats(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        stats = analyzer.get_stats()
        assert stats["active_clusters"] == 1
        assert stats["total_crowd_members"] == 5
        assert stats["largest_cluster"] == 5
        assert stats["total_events"] >= 1

    def test_get_density_map(self):
        analyzer = CrowdDynamicsAnalyzer(density_cell_size=20.0)
        targets = _cluster_at((50, 50), n=10, spread=5.0)
        cells = analyzer.get_density_map(targets, area=(0, 0, 100, 100))
        assert len(cells) > 0
        assert all(c.count > 0 for c in cells)

    def test_clear_resets_state(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        assert len(analyzer.get_active_clusters()) == 1
        analyzer.clear()
        assert len(analyzer.get_active_clusters()) == 0
        assert len(analyzer.get_events()) == 0

    def test_cluster_inherits_identity_across_updates(self):
        """A cluster that persists should keep the same ID across updates."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        id1 = analyzer.get_active_clusters()[0].cluster_id

        # Same targets, slight movement
        for t in targets:
            x, y = t["position"]
            t["position"] = (x + 0.5, y + 0.5)
        analyzer.update(targets, timestamp=2.0)
        id2 = analyzer.get_active_clusters()[0].cluster_id
        assert id1 == id2

    def test_flow_computed_on_update(self):
        """After two updates, clusters should have flow data."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=0.0)

        # Move all targets east
        for t in targets:
            x, y = t["position"]
            t["position"] = (x + 10.0, y)
        analyzer.update(targets, timestamp=1.0)

        active = analyzer.get_active_clusters()
        assert len(active) == 1
        assert active[0].flow_speed > 0

    def test_density_set_on_update(self):
        """After update, clusters should have density set."""
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        cluster = analyzer.get_active_clusters()[0]
        assert cluster.density > 0

    def test_multiple_updates_accumulate_events(self):
        analyzer = CrowdDynamicsAnalyzer(cluster_radius=15.0, min_crowd_size=3)
        targets = _cluster_at((50, 50), n=5, spread=3.0)
        analyzer.update(targets, timestamp=1.0)
        analyzer.update([], timestamp=2.0)
        all_events = analyzer.get_events(limit=100)
        types = {e.event_type for e in all_events}
        assert CrowdEventType.FORMATION in types
        assert CrowdEventType.DISPERSAL in types


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_target_no_cluster(self):
        analyzer = CrowdDynamicsAnalyzer(min_crowd_size=3)
        targets = [{"target_id": "solo", "position": (50, 50)}]
        events = analyzer.update(targets, timestamp=1.0)
        assert len(analyzer.get_active_clusters()) == 0

    def test_targets_missing_position_ignored(self):
        analyzer = CrowdDynamicsAnalyzer(min_crowd_size=3)
        targets = [
            {"target_id": "a", "position": (10, 10)},
            {"target_id": "b"},  # no position
            {"target_id": "c", "position": (11, 11)},
            {"target_id": "d", "position": (12, 10)},
        ]
        clusters = analyzer.detect_clusters(targets, timestamp=1.0)
        assert len(clusters) == 1
        assert clusters[0].size == 3  # b excluded

    def test_degenerate_area_density(self):
        """Zero-area bounding box should return empty."""
        estimator = DensityEstimator()
        targets = _make_targets([(5, 5)])
        cells = estimator.estimate(targets, area=(0, 0, 0, 0))
        assert cells == []

    def test_grid_flow_empty_area(self):
        flow = FlowAnalyzer()
        vectors = flow.compute_grid_flow([], area=(0, 0, 0, 0), timestamp=1.0)
        assert vectors == []
