# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.network_analysis."""

import time

import pytest

from tritium_lib.tracking.network_analysis import (
    COMMON_SSIDS,
    DeviceProfile,
    NetworkAnalyzer,
    ProbeRecord,
)


class TestProbeRecord:
    def test_default_rssi(self):
        rec = ProbeRecord(mac="AA:BB:CC:DD:EE:FF", ssid="Test", timestamp=1000.0)
        assert rec.rssi == -80

    def test_custom_rssi(self):
        rec = ProbeRecord(mac="AA:BB:CC:DD:EE:FF", ssid="Test", timestamp=1000.0, rssi=-50)
        assert rec.rssi == -50


class TestDeviceProfile:
    def test_classify_silent(self):
        p = DeviceProfile(mac="AA:BB:CC:DD:EE:FF")
        assert p._classify() == "silent"

    def test_classify_single_network(self):
        p = DeviceProfile(mac="AA:BB:CC:DD:EE:FF", ssids={"Home"})
        assert p._classify() == "single_network"

    def test_classify_home_user(self):
        p = DeviceProfile(mac="AA:BB:CC:DD:EE:FF", ssids={"Home", "Work"})
        assert p._classify() == "home_user"

    def test_classify_mobile(self):
        p = DeviceProfile(mac="AA:BB:CC:DD:EE:FF", ssids={"A", "B", "C", "D", "E"})
        assert p._classify() == "mobile"

    def test_classify_heavy_traveler(self):
        ssids = {f"net{i}" for i in range(12)}
        p = DeviceProfile(mac="AA:BB:CC:DD:EE:FF", ssids=ssids)
        assert p._classify() == "heavy_traveler"

    def test_to_dict(self):
        p = DeviceProfile(
            mac="AA:BB:CC:DD:EE:FF",
            ssids={"Home"},
            first_seen=100.0,
            last_seen=200.0,
            probe_count=5,
            oui_vendor="Apple",
        )
        d = p.to_dict()
        assert d["mac"] == "AA:BB:CC:DD:EE:FF"
        assert d["ssids"] == ["Home"]
        assert d["ssid_count"] == 1
        assert d["probe_count"] == 5
        assert d["oui_vendor"] == "Apple"
        assert d["device_type"] == "single_network"


class TestNetworkAnalyzer:
    def test_record_probe_basic(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("aa:bb:cc:dd:ee:ff", "TestNet", timestamp=1000.0)
        profile = analyzer.get_device_profile("AA:BB:CC:DD:EE:FF")
        assert profile is not None
        assert profile["mac"] == "AA:BB:CC:DD:EE:FF"
        assert "TestNet" in profile["ssids"]
        assert profile["probe_count"] == 1

    def test_mac_normalized_to_upper(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("aa:bb:cc:dd:ee:ff", "Net1", timestamp=1000.0)
        assert analyzer.get_device_profile("aa:bb:cc:dd:ee:ff") is not None
        # Should work with upper too since get_device_profile normalizes
        assert analyzer.get_device_profile("AA:BB:CC:DD:EE:FF") is not None

    def test_multiple_ssids_per_device(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:FF", "Home", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:FF", "Work", timestamp=1001.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:FF", "Coffee", timestamp=1002.0)
        profile = analyzer.get_device_profile("AA:BB:CC:DD:EE:FF")
        assert profile["ssid_count"] == 3
        assert profile["probe_count"] == 3
        assert profile["device_type"] == "home_user"

    def test_oui_vendor_stored(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:FF", "Net", oui_vendor="Samsung", timestamp=1000.0)
        profile = analyzer.get_device_profile("AA:BB:CC:DD:EE:FF")
        assert profile["oui_vendor"] == "Samsung"

    def test_get_ssid_devices(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "SharedNet", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:02", "SharedNet", timestamp=1001.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:03", "OtherNet", timestamp=1002.0)
        devices = analyzer.get_ssid_devices("SharedNet")
        assert len(devices) == 2
        assert "AA:BB:CC:DD:EE:01" in devices
        assert "AA:BB:CC:DD:EE:02" in devices

    def test_get_correlated_devices(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "CorpNet", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "HomeNet", timestamp=1001.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:02", "CorpNet", timestamp=1002.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:02", "HomeNet", timestamp=1003.0)
        correlated = analyzer.get_correlated_devices("AA:BB:CC:DD:EE:01", min_shared=2)
        assert len(correlated) == 1
        assert correlated[0]["mac"] == "AA:BB:CC:DD:EE:02"
        assert correlated[0]["strength"] == 2

    def test_get_correlated_devices_unknown_mac(self):
        analyzer = NetworkAnalyzer()
        result = analyzer.get_correlated_devices("XX:XX:XX:XX:XX:XX")
        assert result == []

    def test_get_network_graph_basic(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "MyNet", timestamp=1000.0)
        graph = analyzer.get_network_graph(exclude_common=False)
        assert graph["device_count"] == 1
        assert graph["ssid_count"] == 1
        assert len(graph["nodes"]) == 2  # 1 device + 1 SSID
        assert len(graph["edges"]) == 1  # 1 probes_for edge

    def test_get_network_graph_shared_edge(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "CorpNet", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:02", "CorpNet", timestamp=1001.0)
        graph = analyzer.get_network_graph(min_shared_ssids=1, exclude_common=False)
        assert graph["device_edges"] == 1
        # Find the shared_network edge
        shared = [e for e in graph["edges"] if e["type"] == "shared_network"]
        assert len(shared) == 1
        assert shared[0]["strength"] == 1

    def test_common_ssid_filtering(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "xfinitywifi", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "MyUniqueNet", timestamp=1001.0)
        graph = analyzer.get_network_graph(exclude_common=True)
        # xfinitywifi should be filtered out
        ssid_nodes = [n for n in graph["nodes"] if n["type"] == "ssid"]
        ssid_labels = {n["label"] for n in ssid_nodes}
        assert "xfinitywifi" not in ssid_labels
        assert "MyUniqueNet" in ssid_labels

    def test_get_statistics(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net1", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net2", timestamp=1001.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:02", "Net1", timestamp=1002.0)
        stats = analyzer.get_statistics()
        assert stats["total_probes"] == 3
        assert stats["total_devices"] == 2
        assert stats["total_ssids"] == 2
        assert "device_types" in stats
        assert len(stats["top_ssids"]) > 0

    def test_prune(self):
        analyzer = NetworkAnalyzer(retention_hours=1.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net1", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net2", timestamp=2000.0)
        removed = analyzer.prune(before=1500.0)
        assert removed == 1

    def test_clear(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net1", timestamp=1000.0)
        analyzer.clear()
        assert analyzer.get_device_profile("AA:BB:CC:DD:EE:01") is None
        stats = analyzer.get_statistics()
        assert stats["total_probes"] == 0
        assert stats["total_devices"] == 0

    def test_get_device_profile_unknown(self):
        analyzer = NetworkAnalyzer()
        assert analyzer.get_device_profile("XX:XX:XX:XX:XX:XX") is None

    def test_last_seen_updated(self):
        analyzer = NetworkAnalyzer()
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net1", timestamp=1000.0)
        analyzer.record_probe("AA:BB:CC:DD:EE:01", "Net1", timestamp=2000.0)
        profile = analyzer.get_device_profile("AA:BB:CC:DD:EE:01")
        assert profile["first_seen"] == 1000.0
        assert profile["last_seen"] == 2000.0

    def test_common_ssids_constant(self):
        assert isinstance(COMMON_SSIDS, frozenset)
        assert "xfinitywifi" in COMMON_SSIDS
        assert "" in COMMON_SSIDS
