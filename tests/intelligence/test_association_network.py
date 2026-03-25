# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the AssociationNetwork and NetworkAnalyzer modules."""

from __future__ import annotations

import time

import pytest

from tritium_lib.intelligence.association_network import (
    Association,
    AssociationNetwork,
    AssociationSummary,
    EvidenceKind,
    KeyPlayer,
    NetworkAnalyzer,
    TargetGroup,
    WeakLink,
    build_from_comint,
    build_from_tracking,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _build_triangle_network() -> AssociationNetwork:
    """A -> B -> C -> A triangle with varying strengths."""
    net = AssociationNetwork()
    net.add_association("A", "B", "co_location", score=0.9)
    net.add_association("B", "C", "co_location", score=0.7)
    net.add_association("C", "A", "communication", score=0.8)
    return net


def _build_two_clusters_network() -> AssociationNetwork:
    """Two tight clusters connected by a weak bridge: {A,B,C} -- {D,E,F}."""
    net = AssociationNetwork()
    # Cluster 1: A, B, C — strong internal links
    net.add_association("A", "B", "co_location", score=0.9)
    net.add_association("B", "C", "co_location", score=0.9)
    net.add_association("A", "C", "communication", score=0.85)
    # Cluster 2: D, E, F — strong internal links
    net.add_association("D", "E", "co_location", score=0.9)
    net.add_association("E", "F", "communication", score=0.8)
    net.add_association("D", "F", "device_sharing", score=0.85)
    # Weak bridge: C -- D
    net.add_association("C", "D", "travel_pattern", score=0.15)
    return net


# ---------------------------------------------------------------------------
# Association dataclass tests
# ---------------------------------------------------------------------------

class TestAssociation:
    def test_create_basic(self):
        a = Association(target_a="ble_AA", target_b="det_person_1", kind="co_location")
        assert a.target_a == "ble_AA"
        assert a.target_b == "det_person_1"
        assert a.kind == "co_location"
        assert a.score == 0.5  # default

    def test_score_clamped(self):
        a = Association(target_a="A", target_b="B", kind="co_location", score=1.5)
        assert a.score == 1.0
        b = Association(target_a="A", target_b="B", kind="co_location", score=-0.3)
        assert b.score == 0.0

    def test_pair_key_sorted(self):
        a = Association(target_a="Z", target_b="A", kind="co_location")
        assert a.pair_key == ("A", "Z")

    def test_to_dict(self):
        a = Association(target_a="X", target_b="Y", kind="communication", score=0.75)
        d = a.to_dict()
        assert d["target_a"] == "X"
        assert d["target_b"] == "Y"
        assert d["kind"] == "communication"
        assert d["score"] == 0.75

    def test_evidence_stored(self):
        a = Association(
            target_a="A", target_b="B", kind="co_location",
            evidence={"distance_m": 3.5, "zone_id": "z1"},
        )
        assert a.evidence["distance_m"] == 3.5
        assert a.evidence["zone_id"] == "z1"


# ---------------------------------------------------------------------------
# EvidenceKind enum tests
# ---------------------------------------------------------------------------

class TestEvidenceKind:
    def test_all_values(self):
        assert EvidenceKind.CO_LOCATION.value == "co_location"
        assert EvidenceKind.COMMUNICATION.value == "communication"
        assert EvidenceKind.TRAVEL_PATTERN.value == "travel_pattern"
        assert EvidenceKind.DEVICE_SHARING.value == "device_sharing"

    def test_string_coercion(self):
        assert EvidenceKind("co_location") == EvidenceKind.CO_LOCATION


# ---------------------------------------------------------------------------
# AssociationNetwork tests
# ---------------------------------------------------------------------------

class TestAssociationNetwork:
    def test_empty_network(self):
        net = AssociationNetwork()
        assert net.target_count == 0
        assert net.association_count == 0
        assert net.edge_count == 0

    def test_add_association(self):
        net = AssociationNetwork()
        assoc = net.add_association("A", "B", "co_location", score=0.8)
        assert net.target_count == 2
        assert net.association_count == 1
        assert net.edge_count == 1
        assert assoc.target_a == "A"

    def test_add_prebuilt(self):
        net = AssociationNetwork()
        a = Association(target_a="X", target_b="Y", kind="communication", score=0.7)
        net.add(a)
        assert net.association_count == 1
        assert "X" in net.get_targets()

    def test_add_many(self):
        net = AssociationNetwork()
        assocs = [
            Association(target_a="A", target_b="B", kind="co_location"),
            Association(target_a="B", target_b="C", kind="communication"),
        ]
        net.add_many(assocs)
        assert net.association_count == 2
        assert net.target_count == 3

    def test_get_targets(self):
        net = _build_triangle_network()
        targets = net.get_targets()
        assert targets == ["A", "B", "C"]

    def test_get_peers(self):
        net = _build_triangle_network()
        peers_a = net.get_peers("A")
        assert "B" in peers_a
        assert "C" in peers_a
        assert len(peers_a) == 2

    def test_get_associations_between(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.8)
        net.add_association("A", "B", "communication", score=0.6)
        assocs = net.get_associations_between("A", "B")
        assert len(assocs) == 2

    def test_get_associations_for(self):
        net = _build_triangle_network()
        assocs = net.get_associations_for("B")
        assert len(assocs) == 2  # B-A and B-C

    def test_get_associations_for_kind_filter(self):
        net = _build_triangle_network()
        assocs = net.get_associations_for("A", kind="communication")
        # C-A is communication
        assert len(assocs) == 1
        assert assocs[0].kind == "communication"

    def test_get_associations_for_since_filter(self):
        net = AssociationNetwork()
        t1 = 1000.0
        t2 = 2000.0
        net.add_association("A", "B", "co_location", timestamp=t1)
        net.add_association("A", "C", "co_location", timestamp=t2)
        assocs = net.get_associations_for("A", since=1500.0)
        assert len(assocs) == 1

    def test_get_target_stats(self):
        net = _build_triangle_network()
        stats = net.get_target_stats("A")
        assert stats["target_id"] == "A"
        assert stats["peer_count"] == 2
        assert stats["association_count"] == 2
        assert "co_location" in stats["evidence_kinds"]

    def test_strength_basic(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.9)
        s = net.strength("A", "B")
        assert 0.0 < s <= 1.0

    def test_strength_no_association(self):
        net = AssociationNetwork()
        assert net.strength("A", "B") == 0.0

    def test_strength_multiple_observations(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.5)
        s1 = net.strength("A", "B")
        net.add_association("A", "B", "communication", score=0.8)
        s2 = net.strength("A", "B")
        # More observations should increase combined score
        assert s2 > s1

    def test_strength_symmetric(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.8)
        assert net.strength("A", "B") == net.strength("B", "A")

    def test_summarize_pair(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.8, timestamp=100.0)
        net.add_association("A", "B", "communication", score=0.6, timestamp=200.0)
        summary = net.summarize_pair("A", "B")
        assert summary.association_count == 2
        assert "co_location" in summary.kinds
        assert "communication" in summary.kinds
        assert summary.first_seen == 100.0
        assert summary.last_seen == 200.0
        assert 0.0 < summary.combined_score <= 1.0

    def test_summarize_pair_empty(self):
        net = AssociationNetwork()
        summary = net.summarize_pair("X", "Y")
        assert summary.combined_score == 0.0
        assert summary.association_count == 0

    def test_to_graph_dict(self):
        net = _build_triangle_network()
        g = net.to_graph_dict()
        assert g["target_count"] == 3
        assert g["edge_count"] == 3
        assert len(g["nodes"]) == 3
        assert len(g["edges"]) == 3
        # Each edge should have weight, count, kinds
        for edge in g["edges"]:
            assert "weight" in edge
            assert "count" in edge
            assert "kinds" in edge

    def test_prune(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", timestamp=100.0)
        net.add_association("A", "C", "co_location", timestamp=200.0)
        removed = net.prune(before=150.0)
        assert removed == 1
        assert net.association_count == 1
        assert net.target_count == 2  # A and C remain

    def test_clear(self):
        net = _build_triangle_network()
        net.clear()
        assert net.target_count == 0
        assert net.association_count == 0


# ---------------------------------------------------------------------------
# AssociationSummary tests
# ---------------------------------------------------------------------------

class TestAssociationSummary:
    def test_to_dict(self):
        s = AssociationSummary(
            target_a="A", target_b="B",
            combined_score=0.75, association_count=3,
            kinds={"co_location", "communication"},
            first_seen=100.0, last_seen=300.0,
        )
        d = s.to_dict()
        assert d["combined_score"] == 0.75
        assert d["association_count"] == 3
        assert "co_location" in d["kinds"]


# ---------------------------------------------------------------------------
# NetworkAnalyzer tests
# ---------------------------------------------------------------------------

class TestNetworkAnalyzer:
    def test_find_associates_basic(self):
        net = _build_triangle_network()
        analyzer = NetworkAnalyzer(net)
        assocs = analyzer.find_associates("A")
        assert len(assocs) == 2
        # Sorted by strength descending
        assert assocs[0].combined_score >= assocs[1].combined_score

    def test_find_associates_min_score_filter(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.9)
        net.add_association("A", "C", "co_location", score=0.1)
        analyzer = NetworkAnalyzer(net)
        # High min_score should filter out weak associations
        strong = analyzer.find_associates("A", min_score=0.5)
        # Only B should remain (C's combined score should be below threshold)
        assert all(s.combined_score >= 0.5 for s in strong)

    def test_find_associates_limit(self):
        net = AssociationNetwork()
        for i in range(20):
            net.add_association("A", f"T{i}", "co_location", score=0.5)
        analyzer = NetworkAnalyzer(net)
        assocs = analyzer.find_associates("A", limit=5)
        assert len(assocs) == 5

    def test_find_groups_empty(self):
        analyzer = NetworkAnalyzer(AssociationNetwork())
        groups = analyzer.find_groups()
        assert groups == []

    def test_find_groups_single_cluster(self):
        net = _build_triangle_network()
        analyzer = NetworkAnalyzer(net)
        groups = analyzer.find_groups()
        assert len(groups) >= 1
        # All three should be in the same group
        all_members = set()
        for g in groups:
            all_members.update(g.members)
        assert {"A", "B", "C"}.issubset(all_members)

    def test_find_groups_two_clusters(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        # Use a higher min_edge_score to split the weak bridge
        groups = analyzer.find_groups(min_edge_score=0.3)
        assert len(groups) >= 2

    def test_find_groups_min_size(self):
        net = AssociationNetwork()
        net.add_association("A", "B", "co_location", score=0.9)
        # Singleton C should not form a group
        analyzer = NetworkAnalyzer(net)
        groups = analyzer.find_groups(min_size=2)
        for g in groups:
            assert len(g.members) >= 2

    def test_target_group_to_dict(self):
        g = TargetGroup(group_id=1, members={"A", "B"}, internal_score=0.8)
        d = g.to_dict()
        assert d["group_id"] == 1
        assert d["member_count"] == 2

    def test_find_key_players_basic(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        kps = analyzer.find_key_players()
        assert len(kps) > 0
        # Key players should have valid scores
        for kp in kps:
            assert 0.0 <= kp.overall_score <= 1.0
            assert kp.degree > 0

    def test_find_key_players_limit(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        kps = analyzer.find_key_players(limit=2)
        assert len(kps) <= 2

    def test_find_key_players_bridge_nodes_rank_high(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        groups = analyzer.find_groups(min_edge_score=0.3)
        kps = analyzer.find_key_players(groups=groups)
        # C and D are bridge nodes — they should appear in the results
        kp_ids = {kp.target_id for kp in kps}
        assert "C" in kp_ids or "D" in kp_ids

    def test_key_player_to_dict(self):
        kp = KeyPlayer(target_id="X", degree=3, weighted_degree=2.5, overall_score=0.7)
        d = kp.to_dict()
        assert d["target_id"] == "X"
        assert d["degree"] == 3

    def test_find_weak_links(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        weak = analyzer.find_weak_links(max_score=0.5)
        assert len(weak) >= 1
        # The C-D bridge should be found
        pairs = {(w.target_a, w.target_b) for w in weak}
        assert ("C", "D") in pairs or ("D", "C") in pairs

    def test_find_weak_links_bridge_flag(self):
        net = _build_two_clusters_network()
        analyzer = NetworkAnalyzer(net)
        groups = analyzer.find_groups(min_edge_score=0.3)
        weak = analyzer.find_weak_links(max_score=0.5, groups=groups)
        bridges = [w for w in weak if w.is_bridge]
        assert len(bridges) >= 1

    def test_weak_link_to_dict(self):
        w = WeakLink(target_a="A", target_b="B", score=0.2, is_bridge=True)
        d = w.to_dict()
        assert d["is_bridge"] is True
        assert d["score"] == 0.2

    def test_reachable_within(self):
        net = _build_triangle_network()
        analyzer = NetworkAnalyzer(net)
        reached = analyzer.reachable_within("A", max_hops=1)
        assert "B" in reached
        assert "C" in reached

    def test_reachable_within_hop_limit(self):
        net = AssociationNetwork()
        # Linear chain: A - B - C - D
        net.add_association("A", "B", "co_location", score=0.8)
        net.add_association("B", "C", "co_location", score=0.8)
        net.add_association("C", "D", "co_location", score=0.8)
        analyzer = NetworkAnalyzer(net)
        one_hop = analyzer.reachable_within("A", max_hops=1)
        assert "B" in one_hop
        assert "C" not in one_hop
        two_hop = analyzer.reachable_within("A", max_hops=2)
        assert "C" in two_hop
        assert "D" not in two_hop

    def test_reachable_within_min_score_filter(self):
        net = AssociationNetwork()
        # communication kind has weight 0.9, producing a higher combined score
        net.add_association("A", "B", "communication", score=0.9)
        net.add_association("A", "C", "co_location", score=0.05)
        analyzer = NetworkAnalyzer(net)
        # A-B strength should be above 0.4, A-C well below
        reached = analyzer.reachable_within("A", max_hops=1, min_score=0.4)
        assert "B" in reached
        # C's edge is too weak
        assert "C" not in reached

    def test_get_statistics(self):
        net = _build_triangle_network()
        analyzer = NetworkAnalyzer(net)
        stats = analyzer.get_statistics()
        assert stats["target_count"] == 3
        assert stats["edge_count"] == 3
        assert stats["association_count"] == 3
        assert stats["avg_edge_weight"] > 0
        assert stats["avg_peers_per_target"] > 0
        assert "co_location" in stats["evidence_kind_distribution"]

    def test_export(self):
        net = _build_triangle_network()
        analyzer = NetworkAnalyzer(net)
        export = analyzer.export()
        assert "network" in export
        assert "groups" in export
        assert "key_players" in export
        assert "weak_links" in export
        assert "statistics" in export
        assert export["network"]["target_count"] == 3


# ---------------------------------------------------------------------------
# build_from_tracking tests
# ---------------------------------------------------------------------------

class TestBuildFromTracking:
    def test_empty_input(self):
        net = build_from_tracking([])
        assert net.target_count == 0

    def test_single_target(self):
        targets = [{"target_id": "A", "position": (0.0, 0.0), "last_seen": _now()}]
        net = build_from_tracking(targets)
        assert net.association_count == 0

    def test_co_located_targets(self):
        t = _now()
        targets = [
            {"target_id": "A", "position": (10.0, 10.0), "last_seen": t},
            {"target_id": "B", "position": (12.0, 10.0), "last_seen": t + 5},
        ]
        net = build_from_tracking(targets, co_location_distance=10.0)
        assert net.association_count >= 1
        assert net.strength("A", "B") > 0

    def test_too_far_apart(self):
        t = _now()
        targets = [
            {"target_id": "A", "position": (0.0, 0.0), "last_seen": t},
            {"target_id": "B", "position": (100.0, 100.0), "last_seen": t},
        ]
        net = build_from_tracking(targets, co_location_distance=10.0)
        assert net.association_count == 0

    def test_too_far_apart_in_time(self):
        t = _now()
        targets = [
            {"target_id": "A", "position": (0.0, 0.0), "last_seen": t},
            {"target_id": "B", "position": (1.0, 0.0), "last_seen": t + 200},
        ]
        net = build_from_tracking(targets, co_location_time=60.0)
        assert net.association_count == 0

    def test_correlated_ids_create_device_sharing(self):
        targets = [
            {"target_id": "A", "position": (0, 0), "last_seen": 0, "correlated_ids": ["uuid_123"]},
            {"target_id": "B", "position": (50, 50), "last_seen": 0, "correlated_ids": ["uuid_123"]},
        ]
        net = build_from_tracking(targets)
        assocs = net.get_associations_between("A", "B")
        kinds = {a.kind for a in assocs}
        assert EvidenceKind.DEVICE_SHARING.value in kinds

    def test_confirming_sources_create_communication(self):
        targets = [
            {"target_id": "A", "position": (0, 0), "last_seen": 0, "confirming_sources": ["ble", "wifi"]},
            {"target_id": "B", "position": (50, 50), "last_seen": 0, "confirming_sources": ["ble"]},
        ]
        net = build_from_tracking(targets)
        assocs = net.get_associations_between("A", "B")
        kinds = {a.kind for a in assocs}
        assert EvidenceKind.COMMUNICATION.value in kinds

    def test_with_dataclass_targets(self):
        """Test with objects that have attributes instead of dict keys."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeTarget:
            target_id: str
            position: tuple = (0.0, 0.0)
            last_seen: float = 0.0
            source: str = "test"
            asset_type: str = "person"
            correlated_ids: list = field(default_factory=list)
            confirming_sources: set = field(default_factory=set)

        t = _now()
        targets = [
            FakeTarget(target_id="A", position=(5.0, 5.0), last_seen=t),
            FakeTarget(target_id="B", position=(6.0, 5.0), last_seen=t),
        ]
        net = build_from_tracking(targets)
        assert net.association_count >= 1

    def test_co_location_score_decays_with_distance(self):
        t = _now()
        # Close pair
        targets_close = [
            {"target_id": "A", "position": (0.0, 0.0), "last_seen": t},
            {"target_id": "B", "position": (1.0, 0.0), "last_seen": t},
        ]
        net_close = build_from_tracking(targets_close, co_location_distance=10.0)
        # Far pair (but still within range)
        targets_far = [
            {"target_id": "C", "position": (0.0, 0.0), "last_seen": t},
            {"target_id": "D", "position": (9.0, 0.0), "last_seen": t},
        ]
        net_far = build_from_tracking(targets_far, co_location_distance=10.0)
        assert net_close.strength("A", "B") > net_far.strength("C", "D")


# ---------------------------------------------------------------------------
# build_from_comint tests
# ---------------------------------------------------------------------------

class TestBuildFromComint:
    def test_basic_conversion(self):
        """Test conversion from a mock CommNetwork."""
        class MockCommNetwork:
            def to_graph_dict(self):
                return {
                    "nodes": [{"id": "A"}, {"id": "B"}],
                    "edges": [
                        {
                            "source": "A",
                            "target": "B",
                            "weight": 10,
                            "media": {"ble": 7, "wifi": 3},
                        }
                    ],
                }

        net = build_from_comint(MockCommNetwork())
        assert net.target_count == 2
        assert net.association_count == 1
        assocs = net.get_associations_between("A", "B")
        assert len(assocs) == 1
        assert assocs[0].kind == EvidenceKind.COMMUNICATION.value
        assert assocs[0].evidence["primary_medium"] == "ble"

    def test_empty_comint(self):
        class EmptyNetwork:
            def to_graph_dict(self):
                return {"nodes": [], "edges": []}

        net = build_from_comint(EmptyNetwork())
        assert net.target_count == 0

    def test_multiple_edges(self):
        class MultiEdgeNetwork:
            def to_graph_dict(self):
                return {
                    "nodes": [{"id": "X"}, {"id": "Y"}, {"id": "Z"}],
                    "edges": [
                        {"source": "X", "target": "Y", "weight": 5, "media": {"wifi": 5}},
                        {"source": "Y", "target": "Z", "weight": 20, "media": {"ble": 20}},
                    ],
                }

        net = build_from_comint(MultiEdgeNetwork())
        assert net.target_count == 3
        assert net.edge_count == 2
        # Higher weight should produce higher score
        assert net.strength("Y", "Z") > net.strength("X", "Y")


# ---------------------------------------------------------------------------
# Decay and edge cases
# ---------------------------------------------------------------------------

class TestDecay:
    def test_decay_reduces_old_scores(self):
        # With very short decay half-life, old observations should lose weight
        net = AssociationNetwork(decay_hours=0.001)  # ~3.6 seconds
        old_ts = time.time() - 100  # 100 seconds ago
        net.add_association("A", "B", "co_location", score=0.9, timestamp=old_ts)

        net_no_decay = AssociationNetwork(decay_hours=0.0)
        net_no_decay.add_association("A", "B", "co_location", score=0.9, timestamp=old_ts)

        # Decayed should be less than non-decayed
        assert net.strength("A", "B") < net_no_decay.strength("A", "B")


class TestEdgeCases:
    def test_self_association(self):
        net = AssociationNetwork()
        net.add_association("A", "A", "co_location", score=0.5)
        assert net.target_count == 1
        assert net.strength("A", "A") > 0

    def test_large_network(self):
        net = AssociationNetwork()
        for i in range(100):
            net.add_association(f"T{i}", f"T{i+1}", "co_location", score=0.5)
        assert net.target_count == 101
        assert net.edge_count == 100

    def test_analyzer_on_empty_network(self):
        analyzer = NetworkAnalyzer()
        assert analyzer.find_associates("X") == []
        assert analyzer.find_groups() == []
        assert analyzer.find_key_players() == []
        assert analyzer.find_weak_links() == []
        assert analyzer.reachable_within("X") == set()

    def test_find_groups_returns_sorted_by_size(self):
        net = AssociationNetwork()
        # Big group: A, B, C, D
        net.add_association("A", "B", "co_location", score=0.9)
        net.add_association("B", "C", "co_location", score=0.9)
        net.add_association("C", "D", "co_location", score=0.9)
        net.add_association("A", "D", "co_location", score=0.9)
        # Small group: X, Y
        net.add_association("X", "Y", "co_location", score=0.9)
        analyzer = NetworkAnalyzer(net)
        groups = analyzer.find_groups()
        if len(groups) >= 2:
            assert len(groups[0].members) >= len(groups[1].members)
