# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for multi-site federation models."""

import time
import pytest

from tritium_lib.models.federation import (
    ConnectionState,
    FederatedSite,
    FederationMessage,
    FederationMessageType,
    SharedTarget,
    SharePolicy,
    SiteConnection,
    SiteRole,
    federation_topic,
    is_message_expired,
)


class TestFederatedSite:
    """Tests for FederatedSite model."""

    def test_defaults(self):
        site = FederatedSite()
        assert site.name == "Unknown Site"
        assert site.role == SiteRole.PEER
        assert site.share_policy == SharePolicy.TARGETS_ONLY
        assert site.mqtt_port == 1883
        assert site.enabled is True
        assert len(site.site_id) > 0

    def test_custom_values(self):
        site = FederatedSite(
            site_id="site-alpha",
            name="Alpha HQ",
            mqtt_host="10.0.0.1",
            mqtt_port=8883,
            role=SiteRole.PRIMARY,
            share_policy=SharePolicy.FULL,
            lat=37.7749,
            lng=-122.4194,
            tags=["hq", "primary"],
        )
        assert site.site_id == "site-alpha"
        assert site.name == "Alpha HQ"
        assert site.mqtt_host == "10.0.0.1"
        assert site.mqtt_port == 8883
        assert site.role == SiteRole.PRIMARY
        assert site.share_policy == SharePolicy.FULL
        assert site.lat == 37.7749
        assert "hq" in site.tags

    def test_json_roundtrip(self):
        site = FederatedSite(name="Test Site", mqtt_host="localhost")
        data = site.model_dump()
        restored = FederatedSite(**data)
        assert restored.name == site.name
        assert restored.site_id == site.site_id
        assert restored.mqtt_host == site.mqtt_host


class TestSiteConnection:
    """Tests for SiteConnection model."""

    def test_defaults(self):
        conn = SiteConnection(site_id="site-1")
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.targets_shared == 0
        assert conn.last_heartbeat is None

    def test_connected_state(self):
        now = time.time()
        conn = SiteConnection(
            site_id="site-1",
            state=ConnectionState.CONNECTED,
            last_heartbeat=now,
            latency_ms=42.5,
            targets_shared=10,
            targets_received=7,
            connected_since=now - 3600,
        )
        assert conn.state == ConnectionState.CONNECTED
        assert conn.latency_ms == 42.5
        assert conn.targets_shared == 10

    def test_all_states(self):
        for state in ConnectionState:
            conn = SiteConnection(site_id="test", state=state)
            assert conn.state == state


class TestSharedTarget:
    """Tests for SharedTarget model."""

    def test_defaults(self):
        target = SharedTarget(
            target_id="ble_aabbccddeeff",
            source_site_id="site-alpha",
        )
        assert target.target_id == "ble_aabbccddeeff"
        assert target.source_site_id == "site-alpha"
        assert target.alliance == "unknown"
        assert target.confidence == 0.5

    def test_full_target(self):
        target = SharedTarget(
            target_id="det_person_1",
            source_site_id="site-bravo",
            name="Suspicious Person",
            entity_type="person",
            classification="person",
            alliance="hostile",
            lat=37.7159,
            lng=-121.896,
            heading=45.0,
            speed=1.5,
            confidence=0.85,
            source="yolo",
            threat_level="high",
            identifiers={"face_id": "face_001"},
            dossier_id="dossier-123",
        )
        assert target.alliance == "hostile"
        assert target.threat_level == "high"
        assert target.lat == 37.7159

    def test_json_roundtrip(self):
        target = SharedTarget(
            target_id="mesh_node_5",
            source_site_id="site-1",
            name="Mesh Node 5",
            source="mesh",
        )
        data = target.model_dump()
        restored = SharedTarget(**data)
        assert restored.target_id == target.target_id
        assert restored.name == target.name


class TestFederationMessage:
    """Tests for FederationMessage model."""

    def test_defaults(self):
        msg = FederationMessage(
            message_type=FederationMessageType.SITE_HEARTBEAT,
            source_site_id="site-1",
        )
        assert msg.message_type == FederationMessageType.SITE_HEARTBEAT
        assert msg.source_site_id == "site-1"
        assert msg.target_site_id == ""
        assert msg.ttl == 3
        assert len(msg.message_id) > 0

    def test_target_update(self):
        msg = FederationMessage(
            message_type=FederationMessageType.TARGET_UPDATE,
            source_site_id="site-alpha",
            target_site_id="site-bravo",
            payload={
                "target_id": "ble_aabbccddeeff",
                "lat": 37.7,
                "lng": -121.9,
            },
        )
        assert msg.message_type == FederationMessageType.TARGET_UPDATE
        assert msg.payload["target_id"] == "ble_aabbccddeeff"

    def test_all_message_types(self):
        for mt in FederationMessageType:
            msg = FederationMessage(
                message_type=mt,
                source_site_id="test",
            )
            assert msg.message_type == mt

    def test_json_roundtrip(self):
        msg = FederationMessage(
            message_type=FederationMessageType.DOSSIER_SYNC,
            source_site_id="s1",
            payload={"dossier_id": "d-123", "signals": []},
        )
        data = msg.model_dump()
        restored = FederationMessage(**data)
        assert restored.message_id == msg.message_id
        assert restored.payload == msg.payload


class TestFederationTopic:
    """Tests for federation_topic utility."""

    def test_heartbeat_topic(self):
        topic = federation_topic("site-alpha", FederationMessageType.SITE_HEARTBEAT)
        assert topic == "tritium/federation/site-alpha/site_heartbeat"

    def test_target_update_topic(self):
        topic = federation_topic("hq", FederationMessageType.TARGET_UPDATE)
        assert topic == "tritium/federation/hq/target_update"

    def test_dossier_sync_topic(self):
        topic = federation_topic("remote-1", FederationMessageType.DOSSIER_SYNC)
        assert topic == "tritium/federation/remote-1/dossier_sync"


class TestIsMessageExpired:
    """Tests for is_message_expired utility."""

    def test_fresh_message_not_expired(self):
        msg = FederationMessage(
            message_type=FederationMessageType.SITE_HEARTBEAT,
            source_site_id="test",
        )
        assert not is_message_expired(msg)

    def test_old_message_expired(self):
        msg = FederationMessage(
            message_type=FederationMessageType.SITE_HEARTBEAT,
            source_site_id="test",
            timestamp=time.time() - 600,  # 10 minutes ago
        )
        assert is_message_expired(msg)

    def test_custom_max_age(self):
        msg = FederationMessage(
            message_type=FederationMessageType.SITE_HEARTBEAT,
            source_site_id="test",
            timestamp=time.time() - 10,  # 10 seconds ago
        )
        assert not is_message_expired(msg, max_age_s=60)
        assert is_message_expired(msg, max_age_s=5)


class TestEnums:
    """Tests for federation enum types."""

    def test_site_role_values(self):
        assert SiteRole.PRIMARY.value == "primary"
        assert SiteRole.SECONDARY.value == "secondary"
        assert SiteRole.PEER.value == "peer"

    def test_share_policy_values(self):
        assert SharePolicy.NONE.value == "none"
        assert SharePolicy.FULL.value == "full"

    def test_connection_state_values(self):
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.ERROR.value == "error"

    def test_message_type_values(self):
        assert FederationMessageType.SITE_ANNOUNCE.value == "site_announce"
        assert FederationMessageType.ACK.value == "ack"
