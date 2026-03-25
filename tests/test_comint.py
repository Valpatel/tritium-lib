# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.comint — communications intelligence module."""

import time
import pytest

from tritium_lib.comint import (
    CommAnalyzer,
    CommLink,
    CommNetwork,
    CommunityResult,
    BridgeEntity,
    TimelineEntry,
)


# ── CommLink ─────────────────────────────────────────────────────────


class TestCommLink:
    """Tests for the CommLink dataclass."""

    def test_basic_creation(self):
        link = CommLink(source="aa:bb:cc:dd:ee:ff", target="11:22:33:44:55:66", medium="ble")
        assert link.source == "AA:BB:CC:DD:EE:FF"
        assert link.target == "11:22:33:44:55:66"
        assert link.medium == "ble"
        assert isinstance(link.timestamp, float)

    def test_uppercases_identifiers(self):
        link = CommLink(source="abc", target="def", medium="wifi")
        assert link.source == "ABC"
        assert link.target == "DEF"

    def test_lowercases_medium(self):
        link = CommLink(source="A", target="B", medium="BLE")
        assert link.medium == "ble"

    def test_strips_content_from_metadata(self):
        link = CommLink(
            source="A",
            target="B",
            medium="meshtastic",
            metadata={
                "content": "secret message",
                "message": "hello",
                "payload": b"\x00",
                "body": "text",
                "text": "yo",
                "rssi": -70,
                "channel": 3,
            },
        )
        assert "content" not in link.metadata
        assert "message" not in link.metadata
        assert "payload" not in link.metadata
        assert "body" not in link.metadata
        assert "text" not in link.metadata
        assert link.metadata["rssi"] == -70
        assert link.metadata["channel"] == 3

    def test_custom_timestamp(self):
        link = CommLink(source="A", target="B", medium="wifi", timestamp=1000.0)
        assert link.timestamp == 1000.0


# ── CommNetwork ──────────────────────────────────────────────────────


class TestCommNetwork:
    """Tests for the CommNetwork graph."""

    def _make_network(self):
        net = CommNetwork(retention_hours=0)
        return net

    def test_empty_network(self):
        net = self._make_network()
        assert net.entity_count == 0
        assert net.link_count == 0

    def test_add_single_link(self):
        net = self._make_network()
        link = CommLink(source="A", target="B", medium="ble", timestamp=100.0)
        net.add_link(link)
        assert net.entity_count == 2
        assert net.link_count == 1

    def test_bidirectional_adjacency(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        peers_a = net.get_peers("A")
        peers_b = net.get_peers("B")
        assert "B" in peers_a
        assert "A" in peers_b

    def test_link_count_accumulates(self):
        net = self._make_network()
        for i in range(5):
            net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0 + i))
        assert net.get_link_count("A", "B") == 5
        assert net.get_link_count("B", "A") == 5

    def test_get_entities(self):
        net = self._make_network()
        net.add_link(CommLink(source="C", target="A", medium="wifi", timestamp=100.0))
        net.add_link(CommLink(source="B", target="C", medium="ble", timestamp=101.0))
        entities = net.get_entities()
        assert entities == ["A", "B", "C"]

    def test_entity_stats(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="A", target="C", medium="wifi", timestamp=200.0))
        stats = net.get_entity_stats("A")
        assert stats["peer_count"] == 2
        assert stats["total_links"] == 2
        assert stats["first_seen"] == 100.0
        assert stats["last_seen"] == 200.0
        assert set(stats["media"]) == {"ble", "wifi"}

    def test_medium_breakdown(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=101.0))
        net.add_link(CommLink(source="A", target="B", medium="wifi", timestamp=102.0))
        breakdown = net.get_medium_breakdown("A", "B")
        assert breakdown["ble"] == 2
        assert breakdown["wifi"] == 1

    def test_get_links_for_entity(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="C", target="D", medium="wifi", timestamp=101.0))
        net.add_link(CommLink(source="A", target="C", medium="ble", timestamp=102.0))
        links = net.get_links_for("A")
        assert len(links) == 2
        assert links[0].timestamp == 100.0
        assert links[1].timestamp == 102.0

    def test_get_links_for_time_filtered(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="A", target="C", medium="ble", timestamp=200.0))
        net.add_link(CommLink(source="A", target="D", medium="ble", timestamp=300.0))
        links = net.get_links_for("A", since=150.0, until=250.0)
        assert len(links) == 1
        assert links[0].target == "C"

    def test_to_graph_dict(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="B", target="C", medium="wifi", timestamp=101.0))
        graph = net.to_graph_dict()
        assert graph["entity_count"] == 3
        assert graph["edge_count"] == 2
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

    def test_no_duplicate_edges_in_graph_dict(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        graph = net.to_graph_dict()
        # Should be one edge, not two (even though adjacency is bidirectional)
        assert graph["edge_count"] == 1

    def test_prune_removes_old_links(self):
        net = CommNetwork(retention_hours=1)
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.add_link(CommLink(source="A", target="C", medium="wifi", timestamp=time.time()))
        removed = net.prune(before=500.0)
        assert removed == 1
        assert net.link_count == 1
        assert net.entity_count == 2  # A and C remain

    def test_clear(self):
        net = self._make_network()
        net.add_link(CommLink(source="A", target="B", medium="ble", timestamp=100.0))
        net.clear()
        assert net.entity_count == 0
        assert net.link_count == 0

    def test_case_insensitive_lookup(self):
        net = self._make_network()
        net.add_link(CommLink(source="aa:bb", target="cc:dd", medium="ble", timestamp=100.0))
        assert net.get_link_count("AA:BB", "CC:DD") == 1
        assert net.get_link_count("aa:bb", "cc:dd") == 1

    def test_add_links_batch(self):
        net = self._make_network()
        links = [
            CommLink(source="A", target="B", medium="ble", timestamp=100.0),
            CommLink(source="B", target="C", medium="wifi", timestamp=101.0),
        ]
        net.add_links(links)
        assert net.link_count == 2
        assert net.entity_count == 3


# ── CommAnalyzer — recording helpers ─────────────────────────────────


class TestCommAnalyzerRecording:
    """Tests for CommAnalyzer convenience recording methods."""

    def test_record_ble_pairing(self):
        analyzer = CommAnalyzer()
        link = analyzer.record_ble_pairing("AA:BB", "CC:DD", timestamp=100.0, rssi=-65)
        assert link.medium == "ble"
        assert link.source == "AA:BB"
        assert link.target == "CC:DD"
        assert link.metadata.get("rssi") == -65
        assert analyzer.network.link_count == 1

    def test_record_wifi_association(self):
        analyzer = CommAnalyzer()
        link = analyzer.record_wifi_association("client_mac", "ap_bssid", timestamp=100.0, ssid="MyWiFi")
        assert link.medium == "wifi"
        assert link.metadata.get("ssid") == "MyWiFi"

    def test_record_mesh_message_strips_content(self):
        analyzer = CommAnalyzer()
        link = analyzer.record_mesh_message(
            "node1", "node2", timestamp=100.0,
            hop_count=3, content="secret", message="hello", text="yo"
        )
        assert link.medium == "meshtastic"
        assert "content" not in link.metadata
        assert "message" not in link.metadata
        assert "text" not in link.metadata
        assert link.metadata.get("hop_count") == 3

    def test_record_espnow_frame(self):
        analyzer = CommAnalyzer()
        link = analyzer.record_espnow_frame("MAC1", "MAC2", timestamp=100.0, rssi=-80)
        assert link.medium == "espnow"
        assert link.metadata.get("rssi") == -80

    def test_record_generic(self):
        analyzer = CommAnalyzer()
        link = CommLink(source="X", target="Y", medium="custom", timestamp=100.0)
        analyzer.record(link)
        assert analyzer.network.link_count == 1


# ── CommAnalyzer — community detection ───────────────────────────────


def _build_two_communities():
    """Build an analyzer with two distinct communication groups.

    Group 1: A-B-C (all talk to each other heavily)
    Group 2: D-E-F (all talk to each other heavily)
    Bridge: C-D have a single link connecting the groups.
    """
    analyzer = CommAnalyzer()
    ts = 1000.0

    # Group 1: A, B, C — heavily interconnected
    for i in range(10):
        analyzer.record_ble_pairing("A", "B", timestamp=ts + i)
        analyzer.record_ble_pairing("B", "C", timestamp=ts + 10 + i)
        analyzer.record_ble_pairing("A", "C", timestamp=ts + 20 + i)

    # Group 2: D, E, F — heavily interconnected
    for i in range(10):
        analyzer.record_wifi_association("D", "E", timestamp=ts + 30 + i)
        analyzer.record_wifi_association("E", "F", timestamp=ts + 40 + i)
        analyzer.record_wifi_association("D", "F", timestamp=ts + 50 + i)

    # Weak bridge: C-D with a single link
    analyzer.record_mesh_message("C", "D", timestamp=ts + 60)

    return analyzer


class TestCommunityDetection:
    """Tests for find_communities()."""

    def test_empty_network_no_communities(self):
        analyzer = CommAnalyzer()
        communities = analyzer.find_communities()
        assert communities == []

    def test_single_pair_is_community(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        communities = analyzer.find_communities(min_size=2)
        assert len(communities) >= 1
        # A and B should be in the same community
        found = False
        for c in communities:
            if "A" in c.members and "B" in c.members:
                found = True
                break
        assert found

    def test_two_distinct_groups(self):
        analyzer = _build_two_communities()
        communities = analyzer.find_communities(min_size=2)

        # Should detect at least 2 communities
        assert len(communities) >= 2

        # Check that the groups are roughly correct
        all_members = set()
        for c in communities:
            all_members.update(c.members)
        assert {"A", "B", "C", "D", "E", "F"}.issubset(all_members)

    def test_community_has_link_count(self):
        analyzer = _build_two_communities()
        communities = analyzer.find_communities(min_size=2)
        for c in communities:
            assert c.link_count > 0

    def test_community_has_primary_medium(self):
        analyzer = _build_two_communities()
        communities = analyzer.find_communities(min_size=2)
        media = {c.primary_medium for c in communities}
        # At least one community should have a detected medium
        assert any(m != "" for m in media)

    def test_community_to_dict(self):
        result = CommunityResult(
            community_id=1,
            members={"A", "B", "C"},
            link_count=5,
            primary_medium="ble",
        )
        d = result.to_dict()
        assert d["community_id"] == 1
        assert d["member_count"] == 3
        assert d["link_count"] == 5
        assert d["primary_medium"] == "ble"
        assert d["members"] == ["A", "B", "C"]

    def test_min_size_filter(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        communities = analyzer.find_communities(min_size=5)
        assert len(communities) == 0


# ── CommAnalyzer — bridge detection ──────────────────────────────────


class TestBridgeDetection:
    """Tests for find_bridges()."""

    def test_no_bridges_in_single_community(self):
        analyzer = CommAnalyzer()
        for i in range(5):
            analyzer.record_ble_pairing("A", "B", timestamp=100.0 + i)
            analyzer.record_ble_pairing("B", "C", timestamp=110.0 + i)
            analyzer.record_ble_pairing("A", "C", timestamp=120.0 + i)
        bridges = analyzer.find_bridges()
        # Tight group, no inter-community bridges
        assert len(bridges) == 0

    def test_bridge_between_two_groups(self):
        analyzer = _build_two_communities()
        bridges = analyzer.find_bridges()
        # C and/or D should be identified as bridges
        bridge_ids = {b.entity_id for b in bridges}
        assert bridge_ids & {"C", "D"}, f"Expected C or D as bridge, got {bridge_ids}"

    def test_bridge_has_score(self):
        analyzer = _build_two_communities()
        bridges = analyzer.find_bridges()
        for b in bridges:
            assert 0.0 <= b.bridge_score <= 1.0

    def test_bridge_to_dict(self):
        bridge = BridgeEntity(
            entity_id="X",
            communities={1, 2},
            bridge_score=0.75,
            link_count=10,
        )
        d = bridge.to_dict()
        assert d["entity_id"] == "X"
        assert d["community_count"] == 2
        assert d["bridge_score"] == 0.75
        assert d["link_count"] == 10


# ── CommAnalyzer — timeline ──────────────────────────────────────────


class TestTimeline:
    """Tests for communication_timeline()."""

    def test_empty_timeline(self):
        analyzer = CommAnalyzer()
        timeline = analyzer.communication_timeline("NOBODY")
        assert timeline == []

    def test_timeline_basic(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_wifi_association("A", "C", timestamp=200.0)
        analyzer.record_mesh_message("D", "A", timestamp=300.0)

        timeline = analyzer.communication_timeline("A")
        assert len(timeline) == 3
        assert timeline[0].peer == "B"
        assert timeline[0].direction == "out"
        assert timeline[0].medium == "ble"
        assert timeline[1].peer == "C"
        assert timeline[1].direction == "out"
        assert timeline[2].peer == "D"
        assert timeline[2].direction == "in"

    def test_timeline_time_filter(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_ble_pairing("A", "C", timestamp=200.0)
        analyzer.record_ble_pairing("A", "D", timestamp=300.0)

        timeline = analyzer.communication_timeline("A", since=150.0, until=250.0)
        assert len(timeline) == 1
        assert timeline[0].peer == "C"

    def test_timeline_entry_to_dict(self):
        entry = TimelineEntry(
            peer="B", medium="ble", timestamp=100.0, direction="out", count=1
        )
        d = entry.to_dict()
        assert d["peer"] == "B"
        assert d["medium"] == "ble"
        assert d["direction"] == "out"

    def test_timeline_chronological_order(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "C", timestamp=300.0)
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_ble_pairing("A", "D", timestamp=200.0)

        timeline = analyzer.communication_timeline("A")
        timestamps = [e.timestamp for e in timeline]
        assert timestamps == sorted(timestamps)


# ── CommAnalyzer — convenience queries ───────────────────────────────


class TestConvenienceQueries:
    """Tests for top_communicators, most_active_pairs, entity_reach."""

    def test_top_communicators(self):
        analyzer = CommAnalyzer()
        # A talks to everyone
        for peer in ["B", "C", "D", "E"]:
            for i in range(3):
                analyzer.record_ble_pairing("A", peer, timestamp=100.0 + i)
        # B only talks to A (already counted above)

        top = analyzer.top_communicators(n=3)
        assert len(top) <= 5  # max entities in network
        # A should be top communicator
        assert top[0]["entity_id"] == "A"

    def test_most_active_pairs(self):
        analyzer = CommAnalyzer()
        for i in range(10):
            analyzer.record_ble_pairing("A", "B", timestamp=100.0 + i)
        for i in range(3):
            analyzer.record_ble_pairing("C", "D", timestamp=200.0 + i)

        pairs = analyzer.most_active_pairs(n=2)
        assert len(pairs) == 2
        # A-B should be the most active
        assert pairs[0]["count"] > pairs[1]["count"]

    def test_entity_reach_one_hop(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_ble_pairing("B", "C", timestamp=101.0)
        analyzer.record_ble_pairing("C", "D", timestamp=102.0)

        reach = analyzer.entity_reach("A", max_hops=1)
        assert "B" in reach
        assert "C" not in reach
        assert "D" not in reach

    def test_entity_reach_two_hops(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_ble_pairing("B", "C", timestamp=101.0)
        analyzer.record_ble_pairing("C", "D", timestamp=102.0)

        reach = analyzer.entity_reach("A", max_hops=2)
        assert "B" in reach
        assert "C" in reach
        assert "D" not in reach

    def test_entity_reach_excludes_self(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        reach = analyzer.entity_reach("A", max_hops=1)
        assert "A" not in reach

    def test_entity_reach_max_hops_clamped(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        # max_hops > 5 should be clamped to 5
        reach = analyzer.entity_reach("A", max_hops=100)
        assert "B" in reach


# ── CommAnalyzer — statistics & export ───────────────────────────────


class TestStatisticsAndExport:
    """Tests for get_statistics() and export()."""

    def test_get_statistics(self):
        analyzer = CommAnalyzer()
        analyzer.record_ble_pairing("A", "B", timestamp=100.0)
        analyzer.record_wifi_association("A", "C", timestamp=101.0)

        stats = analyzer.get_statistics()
        assert stats["entity_count"] == 3
        assert stats["link_count"] == 2
        assert stats["edge_count"] == 2
        assert "ble" in stats["medium_distribution"]
        assert "wifi" in stats["medium_distribution"]
        assert stats["avg_peers_per_entity"] > 0

    def test_export_full(self):
        analyzer = _build_two_communities()
        export = analyzer.export()
        assert "network" in export
        assert "communities" in export
        assert "bridges" in export
        assert "statistics" in export
        assert export["statistics"]["entity_count"] > 0

    def test_empty_statistics(self):
        analyzer = CommAnalyzer()
        stats = analyzer.get_statistics()
        assert stats["entity_count"] == 0
        assert stats["link_count"] == 0
        assert stats["avg_peers_per_entity"] == 0.0
