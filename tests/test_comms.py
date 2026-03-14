# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CommChannel models."""

import pytest
from tritium_lib.models.comms import (
    AuthType,
    ChannelAuth,
    ChannelHealth,
    ChannelInventory,
    ChannelStatus,
    ChannelType,
    CommChannel,
    select_best_channel,
    summarize_channels,
)


class TestCommChannel:
    """Tests for CommChannel model."""

    def test_defaults(self):
        ch = CommChannel()
        assert ch.channel_id  # UUID auto-generated
        assert ch.name == "Unnamed Channel"
        assert ch.channel_type == ChannelType.MQTT
        assert ch.status == ChannelStatus.DISCONNECTED
        assert ch.enabled is True
        assert ch.bytes_sent == 0

    def test_custom_fields(self):
        ch = CommChannel(
            name="Main MQTT",
            channel_type=ChannelType.MQTT,
            endpoint="mqtt.local:1883",
            status=ChannelStatus.CONNECTED,
            priority=10,
            tags=["primary"],
        )
        assert ch.name == "Main MQTT"
        assert ch.endpoint == "mqtt.local:1883"
        assert ch.priority == 10
        assert "primary" in ch.tags

    def test_auth(self):
        ch = CommChannel(
            auth=ChannelAuth(
                auth_type=AuthType.BASIC,
                username="admin",
                password="secret",
            ),
        )
        assert ch.auth.auth_type == AuthType.BASIC
        assert ch.auth.username == "admin"

    def test_serialization(self):
        ch = CommChannel(name="Test", channel_type=ChannelType.TAK)
        d = ch.model_dump()
        assert d["name"] == "Test"
        assert d["channel_type"] == "tak"

    def test_channel_types(self):
        for ct in ChannelType:
            ch = CommChannel(channel_type=ct)
            assert ch.channel_type == ct


class TestChannelHealth:
    """Tests for ChannelHealth model."""

    def test_defaults(self):
        h = ChannelHealth(
            channel_id="ch1",
            channel_type=ChannelType.MQTT,
            status=ChannelStatus.CONNECTED,
        )
        assert h.uptime_pct == 0.0
        assert h.last_activity is None


class TestSummarizeChannels:
    """Tests for summarize_channels utility."""

    def test_empty(self):
        inv = summarize_channels([])
        assert inv.total == 0
        assert inv.connected == 0

    def test_mixed_statuses(self):
        channels = [
            CommChannel(status=ChannelStatus.CONNECTED, channel_type=ChannelType.MQTT),
            CommChannel(status=ChannelStatus.CONNECTED, channel_type=ChannelType.MQTT),
            CommChannel(status=ChannelStatus.DISCONNECTED, channel_type=ChannelType.TAK),
            CommChannel(status=ChannelStatus.ERROR, channel_type=ChannelType.WEBSOCKET),
            CommChannel(status=ChannelStatus.DISABLED, channel_type=ChannelType.SERIAL),
        ]
        inv = summarize_channels(channels)
        assert inv.total == 5
        assert inv.connected == 2
        assert inv.disconnected == 1
        assert inv.error == 1
        assert inv.disabled == 1
        assert inv.by_type["mqtt"] == 2
        assert inv.by_type["tak"] == 1
        assert len(inv.channels) == 5


class TestSelectBestChannel:
    """Tests for select_best_channel utility."""

    def test_no_channels(self):
        assert select_best_channel([]) is None

    def test_no_connected(self):
        channels = [
            CommChannel(status=ChannelStatus.DISCONNECTED),
        ]
        assert select_best_channel(channels) is None

    def test_disabled_excluded(self):
        channels = [
            CommChannel(status=ChannelStatus.CONNECTED, enabled=False),
        ]
        assert select_best_channel(channels) is None

    def test_selects_highest_priority(self):
        channels = [
            CommChannel(name="Low", status=ChannelStatus.CONNECTED, priority=1),
            CommChannel(name="High", status=ChannelStatus.CONNECTED, priority=10),
            CommChannel(name="Mid", status=ChannelStatus.CONNECTED, priority=5),
        ]
        best = select_best_channel(channels)
        assert best is not None
        assert best.name == "High"

    def test_filter_by_type(self):
        channels = [
            CommChannel(name="MQTT", status=ChannelStatus.CONNECTED, channel_type=ChannelType.MQTT, priority=1),
            CommChannel(name="TAK", status=ChannelStatus.CONNECTED, channel_type=ChannelType.TAK, priority=10),
        ]
        best = select_best_channel(channels, channel_type=ChannelType.MQTT)
        assert best is not None
        assert best.name == "MQTT"

    def test_latency_tiebreak(self):
        channels = [
            CommChannel(name="Slow", status=ChannelStatus.CONNECTED, priority=5, latency_ms=100.0),
            CommChannel(name="Fast", status=ChannelStatus.CONNECTED, priority=5, latency_ms=10.0),
        ]
        best = select_best_channel(channels)
        assert best is not None
        assert best.name == "Fast"
