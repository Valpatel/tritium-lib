# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the SQLite-backed BLE sighting store."""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from tritium_lib.store.ble import BleStore


@pytest.fixture
def store():
    """Create an in-memory BleStore for testing."""
    s = BleStore(":memory:")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:
    """Verify the database schema is created on init."""

    def test_tables_exist(self, store: BleStore):
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in tables}
        assert "ble_sightings" in names
        assert "ble_targets" in names
        assert "wifi_sightings" in names
        assert "wifi_targets" in names
        assert "node_positions" in names

    def test_indexes_exist(self, store: BleStore):
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        names = {r["name"] for r in indexes}
        assert "idx_sightings_mac" in names
        assert "idx_sightings_ts" in names
        assert "idx_sightings_node" in names
        assert "idx_wifi_ssid" in names
        assert "idx_wifi_bssid" in names
        assert "idx_wifi_ts" in names
        assert "idx_wifi_node" in names


# ---------------------------------------------------------------------------
# Single sighting recording
# ---------------------------------------------------------------------------

class TestRecordSighting:
    """Test recording individual sightings."""

    def test_returns_row_id(self, store: BleStore):
        row_id = store.record_sighting(
            mac="AA:BB:CC:DD:EE:FF",
            name="TestDevice",
            rssi=-55,
            node_id="node-1",
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_sequential_ids(self, store: BleStore):
        id1 = store.record_sighting(mac="AA:BB:CC:DD:EE:01", name="", rssi=-60, node_id="n1")
        id2 = store.record_sighting(mac="AA:BB:CC:DD:EE:02", name="", rssi=-70, node_id="n1")
        assert id2 == id1 + 1

    def test_optional_fields(self, store: BleStore):
        row_id = store.record_sighting(
            mac="AA:BB:CC:DD:EE:FF",
            name="Dev",
            rssi=-50,
            node_id="n1",
            node_ip="192.168.1.10",
            is_known=True,
            seen_count=5,
        )
        row = store._conn.execute(
            "SELECT * FROM ble_sightings WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["node_ip"] == "192.168.1.10"
        assert row["is_known"] == 1
        assert row["seen_count"] == 5


# ---------------------------------------------------------------------------
# Batch recording
# ---------------------------------------------------------------------------

class TestBatchRecording:
    """Test batch-inserting sightings."""

    def test_batch_insert_count(self, store: BleStore):
        sightings = [
            {"mac": f"AA:BB:CC:DD:EE:{i:02X}", "rssi": -50 - i, "node_id": "n1"}
            for i in range(10)
        ]
        count = store.record_sightings_batch(sightings)
        assert count == 10

    def test_batch_empty(self, store: BleStore):
        assert store.record_sightings_batch([]) == 0

    def test_batch_data_persisted(self, store: BleStore):
        sightings = [
            {"mac": "AA:BB:CC:DD:EE:01", "name": "Dev1", "rssi": -55, "node_id": "n1"},
            {"mac": "AA:BB:CC:DD:EE:02", "name": "Dev2", "rssi": -65, "node_id": "n2"},
        ]
        store.record_sightings_batch(sightings)
        rows = store._conn.execute("SELECT COUNT(*) AS c FROM ble_sightings").fetchone()
        assert rows["c"] == 2


# ---------------------------------------------------------------------------
# get_active_devices
# ---------------------------------------------------------------------------

class TestGetActiveDevices:
    """Test the primary live tracker query."""

    def test_returns_active_only(self, store: BleStore):
        # Insert a recent sighting
        store.record_sighting(mac="AA:BB:CC:DD:EE:01", name="Active", rssi=-50, node_id="n1")

        # Insert an old sighting directly
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        store._conn.execute(
            """INSERT INTO ble_sightings
               (mac, name, rssi, node_id, node_ip, is_known, seen_count, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AA:BB:CC:DD:EE:02", "Old", -70, "n1", "", 0, 1, old_ts),
        )
        store._conn.commit()

        active = store.get_active_devices(timeout_minutes=2)
        macs = {d["mac"] for d in active}
        assert "AA:BB:CC:DD:EE:01" in macs
        assert "AA:BB:CC:DD:EE:02" not in macs

    def test_aggregation_across_nodes(self, store: BleStore):
        mac = "AA:BB:CC:DD:EE:01"
        store.record_sighting(mac=mac, name="Dev", rssi=-55, node_id="n1")
        store.record_sighting(mac=mac, name="Dev", rssi=-65, node_id="n2")
        store.record_sighting(mac=mac, name="Dev", rssi=-45, node_id="n3")

        active = store.get_active_devices(timeout_minutes=2)
        assert len(active) == 1
        dev = active[0]
        assert dev["mac"] == mac
        assert dev["node_count"] == 3
        assert dev["strongest_rssi"] == -45
        assert dev["total_sightings"] == 3
        assert len(dev["nodes"]) == 3

    def test_nodes_contain_rssi(self, store: BleStore):
        """Each node entry should have rssi for trilateration."""
        mac = "AA:BB:CC:DD:EE:01"
        store.record_sighting(mac=mac, name="", rssi=-50, node_id="n1")
        store.record_sighting(mac=mac, name="", rssi=-60, node_id="n2")

        active = store.get_active_devices()
        dev = active[0]
        for node in dev["nodes"]:
            assert "node_id" in node
            assert "rssi" in node
            assert "last_seen" in node

    def test_empty_database(self, store: BleStore):
        assert store.get_active_devices() == []


# ---------------------------------------------------------------------------
# get_device_history
# ---------------------------------------------------------------------------

class TestGetDeviceHistory:
    """Test per-device sighting history."""

    def test_returns_correct_mac(self, store: BleStore):
        store.record_sighting(mac="AA:BB:CC:DD:EE:01", name="", rssi=-50, node_id="n1")
        store.record_sighting(mac="AA:BB:CC:DD:EE:02", name="", rssi=-60, node_id="n1")

        history = store.get_device_history("AA:BB:CC:DD:EE:01")
        assert len(history) == 1
        assert history[0]["mac"] == "AA:BB:CC:DD:EE:01"

    def test_limit(self, store: BleStore):
        mac = "AA:BB:CC:DD:EE:01"
        for _ in range(10):
            store.record_sighting(mac=mac, name="", rssi=-50, node_id="n1")
        history = store.get_device_history(mac, limit=5)
        assert len(history) == 5

    def test_ordered_by_timestamp_desc(self, store: BleStore):
        mac = "AA:BB:CC:DD:EE:01"
        for i in range(5):
            store.record_sighting(mac=mac, name="", rssi=-50 - i, node_id="n1")
        history = store.get_device_history(mac)
        timestamps = [h["timestamp"] for h in history]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Target CRUD
# ---------------------------------------------------------------------------

class TestTargetCrud:
    """Test target tracking operations."""

    def test_add_and_list(self, store: BleStore):
        result = store.add_target("AA:BB:CC:DD:EE:01", "My Phone", color="#FF0000")
        assert result["mac"] == "AA:BB:CC:DD:EE:01"
        assert result["label"] == "My Phone"
        assert result["tracked"] is True
        assert result["color"] == "#FF0000"

        targets = store.list_targets()
        assert len(targets) == 1
        assert targets[0]["mac"] == "AA:BB:CC:DD:EE:01"

    def test_remove_existing(self, store: BleStore):
        store.add_target("AA:BB:CC:DD:EE:01", "Device")
        assert store.remove_target("AA:BB:CC:DD:EE:01") is True
        assert store.list_targets() == []

    def test_remove_nonexistent(self, store: BleStore):
        assert store.remove_target("AA:BB:CC:DD:EE:99") is False

    def test_update_label(self, store: BleStore):
        store.add_target("AA:BB:CC:DD:EE:01", "Old Label")
        assert store.update_target("AA:BB:CC:DD:EE:01", label="New Label") is True
        targets = store.list_targets()
        assert targets[0]["label"] == "New Label"

    def test_update_tracked_flag(self, store: BleStore):
        store.add_target("AA:BB:CC:DD:EE:01", "Device")
        store.update_target("AA:BB:CC:DD:EE:01", tracked=False)
        targets = store.list_targets()
        assert targets[0]["tracked"] == 0

    def test_update_color(self, store: BleStore):
        store.add_target("AA:BB:CC:DD:EE:01", "Device")
        store.update_target("AA:BB:CC:DD:EE:01", color="#00FF00")
        targets = store.list_targets()
        assert targets[0]["color"] == "#00FF00"

    def test_update_nonexistent(self, store: BleStore):
        assert store.update_target("AA:BB:CC:DD:EE:99", label="X") is False

    def test_update_no_fields(self, store: BleStore):
        store.add_target("AA:BB:CC:DD:EE:01", "Device")
        assert store.update_target("AA:BB:CC:DD:EE:01") is False

    def test_list_targets_with_sightings(self, store: BleStore):
        mac = "AA:BB:CC:DD:EE:01"
        store.add_target(mac, "Phone")
        store.record_sighting(mac=mac, name="Phone", rssi=-55, node_id="n1")
        targets = store.list_targets()
        assert targets[0]["last_rssi"] == -55


# ---------------------------------------------------------------------------
# Node position CRUD
# ---------------------------------------------------------------------------

class TestNodePositions:
    """Test sensor node position operations."""

    def test_set_and_get(self, store: BleStore):
        store.set_node_position("n1", x=0.0, y=0.0, lat=33.75, lon=-84.39, label="Front door")
        pos = store.get_node_position("n1")
        assert pos is not None
        assert pos["x"] == 0.0
        assert pos["y"] == 0.0
        assert pos["lat"] == 33.75
        assert pos["lon"] == -84.39
        assert pos["label"] == "Front door"

    def test_get_all_positions(self, store: BleStore):
        store.set_node_position("n1", x=0.0, y=0.0)
        store.set_node_position("n2", x=10.0, y=0.0)
        store.set_node_position("n3", x=5.0, y=8.66)
        positions = store.get_node_positions()
        assert len(positions) == 3
        assert "n1" in positions
        assert "n2" in positions
        assert "n3" in positions

    def test_get_nonexistent(self, store: BleStore):
        assert store.get_node_position("nope") is None

    def test_update_position(self, store: BleStore):
        store.set_node_position("n1", x=0.0, y=0.0)
        store.set_node_position("n1", x=5.0, y=5.0)
        pos = store.get_node_position("n1")
        assert pos["x"] == 5.0
        assert pos["y"] == 5.0

    def test_remove(self, store: BleStore):
        store.set_node_position("n1", x=0.0, y=0.0)
        assert store.remove_node_position("n1") is True
        assert store.get_node_position("n1") is None

    def test_remove_nonexistent(self, store: BleStore):
        assert store.remove_node_position("nope") is False


# ---------------------------------------------------------------------------
# Prune old sightings
# ---------------------------------------------------------------------------

class TestPrune:
    """Test sighting cleanup."""

    def test_prune_removes_old(self, store: BleStore):
        # Insert an old sighting
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        store._conn.execute(
            """INSERT INTO ble_sightings
               (mac, name, rssi, node_id, node_ip, is_known, seen_count, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AA:BB:CC:DD:EE:01", "", -60, "n1", "", 0, 1, old_ts),
        )
        store._conn.commit()

        # Insert a recent sighting
        store.record_sighting(mac="AA:BB:CC:DD:EE:02", name="", rssi=-50, node_id="n1")

        deleted = store.prune_old_sightings(days=7)
        assert deleted == 1

        count = store._conn.execute("SELECT COUNT(*) AS c FROM ble_sightings").fetchone()["c"]
        assert count == 1

    def test_prune_nothing_to_delete(self, store: BleStore):
        store.record_sighting(mac="AA:BB:CC:DD:EE:01", name="", rssi=-50, node_id="n1")
        deleted = store.prune_old_sightings(days=7)
        assert deleted == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    """Test database statistics."""

    def test_empty_stats(self, store: BleStore):
        stats = store.get_stats()
        assert stats["sighting_count"] == 0
        assert stats["device_count"] == 0
        assert stats["target_count"] == 0
        assert stats["node_count"] == 0
        assert stats["oldest_sighting"] is None
        assert stats["newest_sighting"] is None
        assert stats["wifi_sighting_count"] == 0
        assert stats["wifi_network_count"] == 0

    def test_populated_stats(self, store: BleStore):
        store.record_sighting(mac="AA:BB:CC:DD:EE:01", name="", rssi=-50, node_id="n1")
        store.record_sighting(mac="AA:BB:CC:DD:EE:02", name="", rssi=-60, node_id="n1")
        store.add_target("AA:BB:CC:DD:EE:01", "Phone")
        store.set_node_position("n1", x=0.0, y=0.0)

        stats = store.get_stats()
        assert stats["sighting_count"] == 2
        assert stats["device_count"] == 2
        assert stats["target_count"] == 1
        assert stats["node_count"] == 1
        assert stats["oldest_sighting"] is not None
        assert stats["newest_sighting"] is not None
        assert stats["db_size_bytes"] == 0  # in-memory DB

    def test_wifi_sighting_count_in_stats(self, store: BleStore):
        store.record_wifi_sighting(ssid="TestNet", bssid="AA:BB:CC:DD:EE:01", rssi=-50, node_id="n1")
        store.record_wifi_sighting(ssid="TestNet2", bssid="AA:BB:CC:DD:EE:02", rssi=-60, node_id="n1")

        stats = store.get_stats()
        assert stats["wifi_sighting_count"] == 2
        assert stats["wifi_network_count"] == 2


# ---------------------------------------------------------------------------
# get_device_summary
# ---------------------------------------------------------------------------

class TestGetDeviceSummary:
    """Test device summary query."""

    def test_summary(self, store: BleStore):
        store.record_sighting(
            mac="AA:BB:CC:DD:EE:01", name="", rssi=-50, node_id="n1", is_known=True
        )
        store.record_sighting(mac="AA:BB:CC:DD:EE:02", name="", rssi=-60, node_id="n1")
        store.add_target("AA:BB:CC:DD:EE:01", "Phone")

        summary = store.get_device_summary()
        assert summary["total_unique_macs"] == 2
        assert summary["active_count"] == 2
        assert summary["known_count"] == 1
        assert summary["tracked_count"] == 1
        assert summary["sighting_count"] == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify concurrent writes don't corrupt the database."""

    def test_concurrent_writes(self, store: BleStore):
        errors: list[Exception] = []
        num_threads = 5
        writes_per_thread = 20

        def writer(thread_id: int):
            try:
                for i in range(writes_per_thread):
                    store.record_sighting(
                        mac=f"AA:BB:CC:DD:{thread_id:02X}:{i:02X}",
                        name=f"thread-{thread_id}",
                        rssi=-50 - i,
                        node_id=f"n{thread_id}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"

        count = store._conn.execute("SELECT COUNT(*) AS c FROM ble_sightings").fetchone()["c"]
        assert count == num_threads * writes_per_thread


# ---------------------------------------------------------------------------
# WiFi sighting recording
# ---------------------------------------------------------------------------

class TestWifiSighting:
    """Test recording individual WiFi sightings."""

    def test_returns_row_id(self, store: BleStore):
        row_id = store.record_wifi_sighting(
            ssid="TestNetwork",
            bssid="AA:BB:CC:DD:EE:01",
            rssi=-55,
            node_id="n1",
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_verify_data(self, store: BleStore):
        row_id = store.record_wifi_sighting(
            ssid="MyWiFi",
            bssid="AA:BB:CC:DD:EE:02",
            rssi=-42,
            channel=6,
            auth_type="WPA2",
            node_id="n1",
        )
        row = store._conn.execute(
            "SELECT * FROM wifi_sightings WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["ssid"] == "MyWiFi"
        assert row["bssid"] == "AA:BB:CC:DD:EE:02"
        assert row["rssi"] == -42
        assert row["channel"] == 6
        assert row["auth_type"] == "WPA2"
        assert row["node_id"] == "n1"
        assert row["timestamp"] is not None


# ---------------------------------------------------------------------------
# WiFi batch recording
# ---------------------------------------------------------------------------

class TestWifiBatchRecording:
    """Test batch-inserting WiFi sightings."""

    def test_batch_insert_count(self, store: BleStore):
        sightings = [
            {"ssid": f"Net{i}", "bssid": f"AA:BB:CC:DD:EE:{i:02X}", "rssi": -50 - i, "node_id": "n1"}
            for i in range(10)
        ]
        count = store.record_wifi_sightings_batch(sightings)
        assert count == 10

    def test_batch_empty(self, store: BleStore):
        assert store.record_wifi_sightings_batch([]) == 0

    def test_batch_data_persisted(self, store: BleStore):
        sightings = [
            {"ssid": "Net1", "bssid": "AA:BB:CC:DD:EE:01", "rssi": -55, "node_id": "n1"},
            {"ssid": "Net2", "bssid": "AA:BB:CC:DD:EE:02", "rssi": -65, "node_id": "n2"},
        ]
        store.record_wifi_sightings_batch(sightings)
        rows = store._conn.execute("SELECT COUNT(*) AS c FROM wifi_sightings").fetchone()
        assert rows["c"] == 2


# ---------------------------------------------------------------------------
# get_active_wifi_networks
# ---------------------------------------------------------------------------

class TestGetActiveWifiNetworks:
    """Test the active WiFi networks query."""

    def test_returns_active_only(self, store: BleStore):
        store.record_wifi_sighting(
            ssid="ActiveNet", bssid="AA:BB:CC:DD:EE:01", rssi=-50, node_id="n1"
        )

        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        store._conn.execute(
            """INSERT INTO wifi_sightings
               (ssid, bssid, rssi, channel, auth_type, node_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("OldNet", "AA:BB:CC:DD:EE:02", -70, 0, "", "n1", old_ts),
        )
        store._conn.commit()

        active = store.get_active_wifi_networks(timeout_minutes=5)
        bssids = {n["bssid"] for n in active}
        assert "AA:BB:CC:DD:EE:01" in bssids
        assert "AA:BB:CC:DD:EE:02" not in bssids

    def test_aggregation_across_nodes(self, store: BleStore):
        bssid = "AA:BB:CC:DD:EE:01"
        store.record_wifi_sighting(ssid="Net", bssid=bssid, rssi=-55, node_id="n1")
        store.record_wifi_sighting(ssid="Net", bssid=bssid, rssi=-65, node_id="n2")
        store.record_wifi_sighting(ssid="Net", bssid=bssid, rssi=-45, node_id="n3")

        active = store.get_active_wifi_networks(timeout_minutes=5)
        assert len(active) == 1
        net = active[0]
        assert net["bssid"] == bssid
        assert net["node_count"] == 3
        assert net["strongest_rssi"] == -45
        assert len(net["nodes"]) == 3

    def test_empty_database(self, store: BleStore):
        assert store.get_active_wifi_networks() == []


# ---------------------------------------------------------------------------
# WiFi history
# ---------------------------------------------------------------------------

class TestWifiHistory:
    """Test per-BSSID WiFi sighting history."""

    def test_returns_correct_bssid(self, store: BleStore):
        store.record_wifi_sighting(ssid="Net1", bssid="AA:BB:CC:DD:EE:01", rssi=-50, node_id="n1")
        store.record_wifi_sighting(ssid="Net2", bssid="AA:BB:CC:DD:EE:02", rssi=-60, node_id="n1")

        history = store.get_wifi_history("AA:BB:CC:DD:EE:01")
        assert len(history) == 1
        assert history[0]["bssid"] == "AA:BB:CC:DD:EE:01"

    def test_limit(self, store: BleStore):
        bssid = "AA:BB:CC:DD:EE:01"
        for _ in range(10):
            store.record_wifi_sighting(ssid="Net", bssid=bssid, rssi=-50, node_id="n1")
        history = store.get_wifi_history(bssid, limit=5)
        assert len(history) == 5

    def test_ordered_by_timestamp_desc(self, store: BleStore):
        bssid = "AA:BB:CC:DD:EE:01"
        for i in range(5):
            store.record_wifi_sighting(ssid="Net", bssid=bssid, rssi=-50 - i, node_id="n1")
        history = store.get_wifi_history(bssid)
        timestamps = [h["timestamp"] for h in history]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# WiFi target CRUD
# ---------------------------------------------------------------------------

class TestWifiTargetCrud:
    """Test WiFi target tracking operations."""

    def test_add_and_list(self, store: BleStore):
        result = store.add_wifi_target("AA:BB:CC:DD:EE:01", "Office AP", ssid="OfficeNet", color="#FF0000")
        assert result["bssid"] == "AA:BB:CC:DD:EE:01"
        assert result["ssid"] == "OfficeNet"
        assert result["label"] == "Office AP"
        assert result["tracked"] is True
        assert result["color"] == "#FF0000"

        targets = store.list_wifi_targets()
        assert len(targets) == 1
        assert targets[0]["bssid"] == "AA:BB:CC:DD:EE:01"

    def test_remove_existing(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "AP")
        assert store.remove_wifi_target("AA:BB:CC:DD:EE:01") is True
        assert store.list_wifi_targets() == []

    def test_remove_nonexistent(self, store: BleStore):
        assert store.remove_wifi_target("AA:BB:CC:DD:EE:99") is False

    def test_update_label(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "Old Label")
        assert store.update_wifi_target("AA:BB:CC:DD:EE:01", label="New Label") is True
        targets = store.list_wifi_targets()
        assert targets[0]["label"] == "New Label"

    def test_update_ssid(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "AP")
        store.update_wifi_target("AA:BB:CC:DD:EE:01", ssid="UpdatedSSID")
        targets = store.list_wifi_targets()
        assert targets[0]["ssid"] == "UpdatedSSID"

    def test_update_color(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "AP")
        store.update_wifi_target("AA:BB:CC:DD:EE:01", color="#00FF00")
        targets = store.list_wifi_targets()
        assert targets[0]["color"] == "#00FF00"

    def test_update_tracked_flag(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "AP")
        store.update_wifi_target("AA:BB:CC:DD:EE:01", tracked=False)
        targets = store.list_wifi_targets()
        assert targets[0]["tracked"] == 0

    def test_update_nonexistent(self, store: BleStore):
        assert store.update_wifi_target("AA:BB:CC:DD:EE:99", label="X") is False

    def test_update_no_fields(self, store: BleStore):
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "AP")
        assert store.update_wifi_target("AA:BB:CC:DD:EE:01") is False


# ---------------------------------------------------------------------------
# WiFi summary
# ---------------------------------------------------------------------------

class TestWifiSummary:
    """Test WiFi summary query."""

    def test_summary(self, store: BleStore):
        store.record_wifi_sighting(ssid="Net1", bssid="AA:BB:CC:DD:EE:01", rssi=-50, node_id="n1")
        store.record_wifi_sighting(ssid="Net2", bssid="AA:BB:CC:DD:EE:02", rssi=-60, node_id="n1")
        store.add_wifi_target("AA:BB:CC:DD:EE:01", "Office", ssid="Net1")

        summary = store.get_wifi_summary()
        assert summary["total_unique_networks"] == 2
        assert summary["active_count"] == 2
        assert summary["tracked_count"] == 1
        assert summary["sighting_count"] == 2


# ---------------------------------------------------------------------------
# WiFi prune
# ---------------------------------------------------------------------------

class TestWifiPrune:
    """Test WiFi sighting cleanup."""

    def test_prune_removes_old(self, store: BleStore):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        store._conn.execute(
            """INSERT INTO wifi_sightings
               (ssid, bssid, rssi, channel, auth_type, node_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("OldNet", "AA:BB:CC:DD:EE:01", -60, 6, "WPA2", "n1", old_ts),
        )
        store._conn.commit()

        store.record_wifi_sighting(ssid="NewNet", bssid="AA:BB:CC:DD:EE:02", rssi=-50, node_id="n1")

        deleted = store.prune_old_wifi_sightings(days=7)
        assert deleted == 1

        count = store._conn.execute("SELECT COUNT(*) AS c FROM wifi_sightings").fetchone()["c"]
        assert count == 1

    def test_prune_nothing_to_delete(self, store: BleStore):
        store.record_wifi_sighting(ssid="Net", bssid="AA:BB:CC:DD:EE:01", rssi=-50, node_id="n1")
        deleted = store.prune_old_wifi_sightings(days=7)
        assert deleted == 0
