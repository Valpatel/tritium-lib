# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BehaviorCluster model."""

import pytest
from tritium_lib.models.clustering import (
    BehaviorCluster,
    ClusterSummary,
    CommonPattern,
    FormationType,
)


def test_create_cluster():
    cluster = BehaviorCluster(
        cluster_id="clust_001",
        targets=["ble_aabb", "ble_ccdd"],
        centroid_lat=30.2672,
        centroid_lng=-97.7431,
        radius_m=50.0,
        formation_type=FormationType.CONVOY,
        confidence=0.85,
    )
    assert cluster.cluster_id == "clust_001"
    assert cluster.target_count == 2
    assert cluster.formation_type == FormationType.CONVOY
    assert cluster.confidence == 0.85
    assert cluster.created_at is not None
    assert cluster.updated_at is not None


def test_add_remove_target():
    cluster = BehaviorCluster(cluster_id="clust_002")
    assert cluster.target_count == 0

    assert cluster.add_target("ble_001") is True
    assert cluster.target_count == 1

    # Duplicate add returns False
    assert cluster.add_target("ble_001") is False
    assert cluster.target_count == 1

    assert cluster.add_target("ble_002") is True
    assert cluster.target_count == 2

    assert cluster.has_target("ble_001") is True
    assert cluster.has_target("ble_999") is False

    assert cluster.remove_target("ble_001") is True
    assert cluster.target_count == 1
    assert cluster.has_target("ble_001") is False

    # Remove non-existent returns False
    assert cluster.remove_target("ble_999") is False


def test_merge_clusters():
    c1 = BehaviorCluster(
        cluster_id="clust_a",
        targets=["t1", "t2"],
        centroid_lat=30.0,
        centroid_lng=-97.0,
        radius_m=50.0,
        observation_count=10,
        confidence=0.8,
        source_patterns=["pat_1"],
    )
    c2 = BehaviorCluster(
        cluster_id="clust_b",
        targets=["t2", "t3"],
        centroid_lat=30.1,
        centroid_lng=-97.1,
        radius_m=60.0,
        observation_count=10,
        confidence=0.6,
        source_patterns=["pat_2"],
    )
    c1.merge(c2)

    assert c1.target_count == 3
    assert set(c1.targets) == {"t1", "t2", "t3"}
    assert c1.observation_count == 20
    assert c1.centroid_lat == pytest.approx(30.05, abs=0.01)
    assert c1.centroid_lng == pytest.approx(-97.05, abs=0.01)
    assert c1.radius_m == 72.0  # max(50, 60) * 1.2
    assert "pat_1" in c1.source_patterns
    assert "pat_2" in c1.source_patterns


def test_common_pattern():
    cp = CommonPattern(
        speed_min_mps=1.2,
        speed_max_mps=3.5,
        active_hour_start=8,
        active_hour_end=17,
        regularity_score=0.9,
    )
    assert cp.speed_min_mps == 1.2
    assert cp.active_hour_start == 8
    assert cp.regularity_score == 0.9


def test_formation_types():
    assert FormationType.CONVOY.value == "convoy"
    assert FormationType.SWARM.value == "swarm"
    assert FormationType.PATROL.value == "patrol"
    assert FormationType.DISPERSED.value == "dispersed"
    assert FormationType.STATIONARY.value == "stationary"
    assert FormationType.UNKNOWN.value == "unknown"


def test_cluster_summary():
    cluster = BehaviorCluster(
        cluster_id="clust_sum",
        targets=["a", "b", "c"],
        common_pattern=CommonPattern(
            speed_min_mps=1.0,
            speed_max_mps=5.0,
            active_hour_start=6,
            active_hour_end=18,
        ),
        centroid_lat=30.0,
        centroid_lng=-97.0,
        radius_m=75.0,
        formation_type=FormationType.PATROL,
        confidence=0.7,
    )
    summary = ClusterSummary.from_cluster(cluster)
    assert summary.cluster_id == "clust_sum"
    assert summary.target_count == 3
    assert summary.formation_type == FormationType.PATROL
    assert "1.0" in summary.speed_range
    assert "5.0" in summary.speed_range
    assert "06:00" in summary.active_hours
    assert "18:00" in summary.active_hours


def test_cluster_serialization():
    cluster = BehaviorCluster(
        cluster_id="clust_ser",
        targets=["x"],
        formation_type=FormationType.SWARM,
    )
    data = cluster.model_dump()
    assert data["cluster_id"] == "clust_ser"
    assert data["formation_type"] == "swarm"
    assert data["targets"] == ["x"]

    # Round-trip
    restored = BehaviorCluster(**data)
    assert restored.cluster_id == "clust_ser"
    assert restored.formation_type == FormationType.SWARM
