# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the communications and radio simulation module."""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.sim_engine.comms import (
    CommsSimulator,
    Jammer,
    Radio,
    RadioChannel,
    RadioMessage,
    RadioType,
    RADIO_PRESETS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim() -> CommsSimulator:
    """Return a fresh CommsSimulator."""
    return CommsSimulator()


@pytest.fixture
def squad_channel() -> RadioChannel:
    return RadioChannel(channel_id="ch1", frequency_mhz=150.0, name="Squad Net")


@pytest.fixture
def encrypted_channel() -> RadioChannel:
    return RadioChannel(
        channel_id="ch_enc", frequency_mhz=225.0, name="Secure Net",
        encrypted=True, alliance="blue",
    )


@pytest.fixture
def blue_channel() -> RadioChannel:
    return RadioChannel(
        channel_id="ch_blue", frequency_mhz=150.0, name="Blue Net",
        encrypted=False, alliance="blue",
    )


@pytest.fixture
def red_channel() -> RadioChannel:
    return RadioChannel(
        channel_id="ch_red", frequency_mhz=150.0, name="Red Net",
        encrypted=False, alliance="red",
    )


def _make_radio(
    radio_id: str,
    position: tuple[float, float] = (0.0, 0.0),
    channel: str = "ch1",
    preset: str = "squad_radio",
    radio_type: RadioType = RadioType.HANDHELD,
    battery: float = 1.0,
) -> Radio:
    """Helper to build a Radio from a preset."""
    p = RADIO_PRESETS[preset]
    return Radio(
        radio_id=radio_id,
        radio_type=radio_type,
        position=position,
        transmit_power_w=p["transmit_power_w"],
        receive_sensitivity_dbm=p["receive_sensitivity_dbm"],
        range_m=p["range_m"],
        current_channel=channel,
        battery=battery,
    )


# ---------------------------------------------------------------------------
# RadioType enum
# ---------------------------------------------------------------------------

class TestRadioType:
    def test_all_types_exist(self):
        assert RadioType.HANDHELD.value == "handheld"
        assert RadioType.MANPACK.value == "manpack"
        assert RadioType.VEHICLE_MOUNTED.value == "vehicle_mounted"
        assert RadioType.BASE_STATION.value == "base_station"
        assert RadioType.SATELLITE.value == "satellite"

    def test_enum_count(self):
        assert len(RadioType) == 5


# ---------------------------------------------------------------------------
# RadioChannel
# ---------------------------------------------------------------------------

class TestRadioChannel:
    def test_defaults(self):
        ch = RadioChannel(channel_id="c", frequency_mhz=100.0, name="Test")
        assert ch.encrypted is False
        assert ch.alliance == ""

    def test_encrypted(self):
        ch = RadioChannel(channel_id="c", frequency_mhz=100.0, name="Sec", encrypted=True)
        assert ch.encrypted is True


# ---------------------------------------------------------------------------
# Radio
# ---------------------------------------------------------------------------

class TestRadio:
    def test_defaults(self):
        r = _make_radio("r1")
        assert r.is_transmitting is False
        assert r.is_jammed is False
        assert r.battery == 1.0

    def test_custom_position(self):
        r = _make_radio("r1", position=(500.0, 300.0))
        assert r.position == (500.0, 300.0)

    def test_preset_values(self):
        r = _make_radio("r1", preset="platoon_radio")
        assert r.transmit_power_w == 20.0
        assert r.range_m == 10_000.0


# ---------------------------------------------------------------------------
# RadioMessage
# ---------------------------------------------------------------------------

class TestRadioMessage:
    def test_fields(self):
        msg = RadioMessage(
            message_id="m1", sender_id="s1", channel_id="ch1",
            content="hello", message_type="voice",
            timestamp=1000.0, encrypted=False, position=(1.0, 2.0),
        )
        assert msg.content == "hello"
        assert msg.message_type == "voice"
        assert msg.position == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Jammer
# ---------------------------------------------------------------------------

class TestJammer:
    def test_defaults(self):
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0)
        assert j.is_active is True
        assert j.frequencies == []

    def test_frequency_selective(self):
        j = Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0,
            frequencies=[150.0, 225.0],
        )
        assert 150.0 in j.frequencies


# ---------------------------------------------------------------------------
# RADIO_PRESETS
# ---------------------------------------------------------------------------

class TestRadioPresets:
    def test_all_presets_exist(self):
        expected = {"squad_radio", "platoon_radio", "vehicle_radio", "base_station", "satcom"}
        assert set(RADIO_PRESETS.keys()) == expected

    def test_squad_radio_values(self):
        p = RADIO_PRESETS["squad_radio"]
        assert p["transmit_power_w"] == 5.0
        assert p["range_m"] == 2_000.0

    def test_platoon_radio_values(self):
        p = RADIO_PRESETS["platoon_radio"]
        assert p["transmit_power_w"] == 20.0
        assert p["range_m"] == 10_000.0

    def test_vehicle_radio_values(self):
        p = RADIO_PRESETS["vehicle_radio"]
        assert p["transmit_power_w"] == 50.0
        assert p["range_m"] == 30_000.0

    def test_base_station_values(self):
        p = RADIO_PRESETS["base_station"]
        assert p["transmit_power_w"] == 100.0
        assert p["range_m"] == 50_000.0

    def test_satcom_unlimited_range(self):
        p = RADIO_PRESETS["satcom"]
        assert p["range_m"] == float("inf")
        assert p["transmit_power_w"] == 5.0

    def test_all_presets_have_required_keys(self):
        for name, preset in RADIO_PRESETS.items():
            assert "transmit_power_w" in preset, f"{name} missing transmit_power_w"
            assert "receive_sensitivity_dbm" in preset, f"{name} missing receive_sensitivity_dbm"
            assert "range_m" in preset, f"{name} missing range_m"


# ---------------------------------------------------------------------------
# CommsSimulator — add / remove
# ---------------------------------------------------------------------------

class TestSimulatorAddRemove:
    def test_add_radio(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("r1")
        sim.add_radio(r)
        assert "r1" in sim.radios

    def test_remove_radio(self, sim: CommsSimulator):
        sim.add_radio(_make_radio("r1"))
        sim.remove_radio("r1")
        assert "r1" not in sim.radios

    def test_remove_nonexistent_radio(self, sim: CommsSimulator):
        sim.remove_radio("nope")  # should not raise

    def test_add_channel(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        assert "ch1" in sim.channels

    def test_remove_channel(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.remove_channel("ch1")
        assert "ch1" not in sim.channels

    def test_add_jammer(self, sim: CommsSimulator):
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0)
        sim.add_jammer(j)
        assert len(sim.jammers) == 1

    def test_remove_jammer(self, sim: CommsSimulator):
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0)
        sim.add_jammer(j)
        sim.remove_jammer("j1")
        assert len(sim.jammers) == 0

    def test_remove_jammer_nonexistent(self, sim: CommsSimulator):
        sim.remove_jammer("nope")  # should not raise


# ---------------------------------------------------------------------------
# CommsSimulator — jamming
# ---------------------------------------------------------------------------

class TestJamming:
    def test_broadband_jammer_jams_radio(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("r1", position=(100.0, 100.0))
        sim.add_radio(r)
        j = Jammer(jammer_id="j1", position=(100.0, 100.0), radius=500.0, power_w=50.0)
        sim.add_jammer(j)
        assert sim.is_jammed(r) is True

    def test_out_of_range_jammer_no_effect(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("r1", position=(5000.0, 5000.0))
        sim.add_radio(r)
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=50.0)
        sim.add_jammer(j)
        assert sim.is_jammed(r) is False

    def test_frequency_selective_jammer_matches(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)  # 150 MHz
        r = _make_radio("r1", position=(0.0, 0.0))
        sim.add_radio(r)
        j = Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0,
            frequencies=[150.0],
        )
        sim.add_jammer(j)
        assert sim.is_jammed(r) is True

    def test_frequency_selective_jammer_no_match(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)  # 150 MHz
        r = _make_radio("r1", position=(0.0, 0.0))
        sim.add_radio(r)
        j = Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=10.0,
            frequencies=[300.0],
        )
        sim.add_jammer(j)
        assert sim.is_jammed(r) is False

    def test_inactive_jammer_no_effect(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("r1", position=(0.0, 0.0))
        sim.add_radio(r)
        j = Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=50.0,
            is_active=False,
        )
        sim.add_jammer(j)
        assert sim.is_jammed(r) is False

    def test_no_channel_not_jammed(self, sim: CommsSimulator):
        """Radio on a channel that doesn't exist in the sim can't be jammed."""
        r = _make_radio("r1", channel="nonexistent")
        sim.add_radio(r)
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=500.0, power_w=50.0)
        sim.add_jammer(j)
        assert sim.is_jammed(r) is False


# ---------------------------------------------------------------------------
# CommsSimulator — can_communicate
# ---------------------------------------------------------------------------

class TestCanCommunicate:
    def test_same_channel_in_range(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(1000.0, 0.0)))
        assert sim.can_communicate("a", "b") is True

    def test_out_of_range(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(5000.0, 0.0)))  # >2km
        assert sim.can_communicate("a", "b") is False

    def test_different_channels(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        ch2 = RadioChannel(channel_id="ch2", frequency_mhz=300.0, name="Other")
        sim.add_channel(ch2)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0), channel="ch1"))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0), channel="ch2"))
        assert sim.can_communicate("a", "b") is False

    def test_one_jammed(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=50.0, power_w=50.0,
        ))
        assert sim.can_communicate("a", "b") is False

    def test_nonexistent_radio(self, sim: CommsSimulator):
        assert sim.can_communicate("x", "y") is False

    def test_dead_battery(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0), battery=0.0))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        assert sim.can_communicate("a", "b") is False

    def test_satcom_unlimited_range(self, sim: CommsSimulator):
        ch = RadioChannel(channel_id="sat", frequency_mhz=1600.0, name="Sat Link")
        sim.add_channel(ch)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0), channel="sat", preset="satcom"))
        sim.add_radio(_make_radio("b", position=(999999.0, 999999.0), channel="sat", preset="satcom"))
        assert sim.can_communicate("a", "b") is True


# ---------------------------------------------------------------------------
# CommsSimulator — transmit
# ---------------------------------------------------------------------------

class TestTransmit:
    def test_basic_transmit(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(500.0, 0.0)))
        report = sim.transmit("a", "Hello!")
        assert report["success"] is True
        assert "b" in report["recipients"]
        assert report["jammed"] is False
        assert len(sim.message_log) == 1

    def test_transmit_sender_not_found(self, sim: CommsSimulator):
        report = sim.transmit("nope", "Hello!")
        assert report["success"] is False
        assert report["reason"] == "sender_not_found"

    def test_transmit_channel_not_found(self, sim: CommsSimulator):
        r = _make_radio("a", channel="missing")
        sim.add_radio(r)
        report = sim.transmit("a", "Hello!")
        assert report["success"] is False
        assert report["reason"] == "channel_not_found"

    def test_transmit_no_battery(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", battery=0.0))
        report = sim.transmit("a", "Hello!")
        assert report["success"] is False
        assert report["reason"] == "no_battery"

    def test_transmit_sender_jammed(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=50.0,
        ))
        report = sim.transmit("a", "Hello!")
        assert report["success"] is False
        assert report["jammed"] is True
        assert report["reason"] == "sender_jammed"

    def test_transmit_receiver_jammed_excluded(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        # Jammer near b but not a
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(100.0, 0.0), radius=50.0, power_w=50.0,
        ))
        report = sim.transmit("a", "Hello!")
        assert report["success"] is True
        assert "b" not in report["recipients"]

    def test_transmit_out_of_range_excluded(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("far", position=(5000.0, 0.0)))
        report = sim.transmit("a", "Hello!")
        assert report["success"] is True
        assert "far" not in report["recipients"]

    def test_transmit_different_channel_excluded(self, sim: CommsSimulator, squad_channel: RadioChannel):
        ch2 = RadioChannel(channel_id="ch2", frequency_mhz=300.0, name="Other")
        sim.add_channel(squad_channel)
        sim.add_channel(ch2)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0), channel="ch1"))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0), channel="ch2"))
        report = sim.transmit("a", "Hello!")
        assert "b" not in report["recipients"]

    def test_transmit_sets_transmitting(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a"))
        sim.transmit("a", "Tx test")
        assert sim.radios["a"].is_transmitting is True

    def test_message_log_populated(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a"))
        sim.transmit("a", "First")
        sim.transmit("a", "Second")
        assert len(sim.message_log) == 2
        assert sim.message_log[0].content == "First"
        assert sim.message_log[1].content == "Second"

    def test_transmit_message_type(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a"))
        sim.transmit("a", "Alert!", msg_type="alert")
        assert sim.message_log[-1].message_type == "alert"

    def test_transmit_encrypted_channel(self, sim: CommsSimulator, encrypted_channel: RadioChannel):
        sim.add_channel(encrypted_channel)
        sim.add_radio(_make_radio("a", channel="ch_enc"))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0), channel="ch_enc"))
        report = sim.transmit("a", "Secret stuff")
        assert report["success"] is True
        assert sim.message_log[-1].encrypted is True

    def test_multiple_recipients(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        sim.add_radio(_make_radio("c", position=(200.0, 0.0)))
        sim.add_radio(_make_radio("d", position=(300.0, 0.0)))
        report = sim.transmit("a", "Broadcast")
        assert set(report["recipients"]) == {"b", "c", "d"}

    def test_dead_battery_receiver_excluded(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0), battery=0.0))
        report = sim.transmit("a", "Hello!")
        assert "b" not in report["recipients"]


# ---------------------------------------------------------------------------
# CommsSimulator — interception
# ---------------------------------------------------------------------------

class TestInterception:
    def test_unencrypted_intercepted_by_enemy(
        self, sim: CommsSimulator,
        blue_channel: RadioChannel,
        red_channel: RadioChannel,
    ):
        sim.add_channel(blue_channel)
        sim.add_channel(red_channel)
        sim.add_radio(_make_radio("blue1", position=(0.0, 0.0), channel="ch_blue"))
        sim.add_radio(_make_radio("red1", position=(500.0, 0.0), channel="ch_red"))
        report = sim.transmit("blue1", "We are moving north")
        assert report["intercepted"] is True
        assert len(sim.intercepted) == 1

    def test_encrypted_not_intercepted(
        self, sim: CommsSimulator,
        encrypted_channel: RadioChannel,
        red_channel: RadioChannel,
    ):
        sim.add_channel(encrypted_channel)
        sim.add_channel(red_channel)
        sim.add_radio(_make_radio("blue1", position=(0.0, 0.0), channel="ch_enc"))
        sim.add_radio(_make_radio("red1", position=(500.0, 0.0), channel="ch_red"))
        report = sim.transmit("blue1", "Secret orders")
        assert report["intercepted"] is False
        assert len(sim.intercepted) == 0

    def test_no_alliance_no_intercept(self, sim: CommsSimulator, squad_channel: RadioChannel):
        """Channels without alliance tags don't trigger intercept logic."""
        sim.add_channel(squad_channel)  # no alliance
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        report = sim.transmit("a", "test")
        assert report["intercepted"] is False

    def test_enemy_out_of_range_no_intercept(
        self, sim: CommsSimulator,
        blue_channel: RadioChannel,
        red_channel: RadioChannel,
    ):
        sim.add_channel(blue_channel)
        sim.add_channel(red_channel)
        sim.add_radio(_make_radio("blue1", position=(0.0, 0.0), channel="ch_blue"))
        sim.add_radio(_make_radio("red1", position=(50000.0, 0.0), channel="ch_red"))
        report = sim.transmit("blue1", "We are moving north")
        assert report["intercepted"] is False

    def test_same_alliance_no_intercept(self, sim: CommsSimulator):
        ch_a = RadioChannel(channel_id="a", frequency_mhz=150.0, name="A", alliance="blue")
        ch_b = RadioChannel(channel_id="b", frequency_mhz=150.0, name="B", alliance="blue")
        sim.add_channel(ch_a)
        sim.add_channel(ch_b)
        sim.add_radio(_make_radio("r1", position=(0.0, 0.0), channel="a"))
        sim.add_radio(_make_radio("r2", position=(100.0, 0.0), channel="b"))
        report = sim.transmit("r1", "Friendly msg")
        assert report["intercepted"] is False


# ---------------------------------------------------------------------------
# CommsSimulator — tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_battery_drain_while_transmitting(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a")
        r.is_transmitting = True
        sim.add_radio(r)
        initial = r.battery
        sim.tick(100.0)  # 100 seconds
        assert sim.radios["a"].battery < initial

    def test_no_drain_when_not_transmitting(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a")
        r.is_transmitting = False
        sim.add_radio(r)
        sim.tick(100.0)
        assert sim.radios["a"].battery == 1.0

    def test_battery_clamps_at_zero(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a", battery=0.001)
        r.is_transmitting = True
        sim.add_radio(r)
        sim.tick(1000.0)
        assert sim.radios["a"].battery == 0.0

    def test_transmitting_stops_at_zero_battery(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a", battery=0.001)
        r.is_transmitting = True
        sim.add_radio(r)
        sim.tick(1000.0)
        assert sim.radios["a"].is_transmitting is False

    def test_vehicle_no_drain(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("v1", preset="vehicle_radio", radio_type=RadioType.VEHICLE_MOUNTED)
        r.is_transmitting = True
        sim.add_radio(r)
        sim.tick(1000.0)
        assert sim.radios["v1"].battery == 1.0

    def test_tick_updates_jammed_status(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a", position=(0.0, 0.0))
        sim.add_radio(r)
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=50.0)
        sim.add_jammer(j)
        sim.tick(1.0)
        assert sim.radios["a"].is_jammed is True

    def test_tick_clears_jammed_when_jammer_removed(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a", position=(0.0, 0.0))
        sim.add_radio(r)
        j = Jammer(jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=50.0)
        sim.add_jammer(j)
        sim.tick(1.0)
        assert sim.radios["a"].is_jammed is True
        sim.remove_jammer("j1")
        sim.tick(1.0)
        assert sim.radios["a"].is_jammed is False


# ---------------------------------------------------------------------------
# CommsSimulator — get_comms_network
# ---------------------------------------------------------------------------

class TestCommsNetwork:
    def test_network_nodes(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(100.0, 0.0)))
        net = sim.get_comms_network()
        assert len(net["nodes"]) == 2
        ids = {n["radio_id"] for n in net["nodes"]}
        assert ids == {"a", "b"}

    def test_network_edges(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(500.0, 0.0)))
        net = sim.get_comms_network()
        assert len(net["edges"]) == 1
        edge = net["edges"][0]
        assert {edge["from"], edge["to"]} == {"a", "b"}

    def test_network_no_edge_out_of_range(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_radio(_make_radio("b", position=(5000.0, 0.0)))
        net = sim.get_comms_network()
        assert len(net["edges"]) == 0

    def test_network_channels(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        net = sim.get_comms_network()
        assert len(net["channels"]) == 1
        assert net["channels"][0]["channel_id"] == "ch1"

    def test_network_jammed_node(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(0.0, 0.0)))
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=50.0,
        ))
        net = sim.get_comms_network()
        node_a = [n for n in net["nodes"] if n["radio_id"] == "a"][0]
        assert node_a["jammed"] is True

    def test_empty_network(self, sim: CommsSimulator):
        net = sim.get_comms_network()
        assert net["nodes"] == []
        assert net["edges"] == []
        assert net["channels"] == []


# ---------------------------------------------------------------------------
# CommsSimulator — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_radio_visualization(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a"))
        viz = sim.to_three_js()
        assert len(viz["radios"]) == 1
        r = viz["radios"][0]
        assert r["id"] == "a"
        assert r["type"] == "handheld"
        assert r["color"] == "#00f0ff"  # not jammed, not transmitting

    def test_jammed_radio_color(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a", position=(0.0, 0.0))
        r.is_jammed = True
        sim.add_radio(r)
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=50.0,
        ))
        viz = sim.to_three_js()
        assert viz["radios"][0]["color"] == "#ff2a6d"

    def test_transmitting_radio_color(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        r = _make_radio("a")
        r.is_transmitting = True
        sim.add_radio(r)
        viz = sim.to_three_js()
        assert viz["radios"][0]["color"] == "#fcee0a"

    def test_jammer_visualization(self, sim: CommsSimulator):
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(100.0, 200.0), radius=500.0, power_w=50.0,
        ))
        viz = sim.to_three_js()
        assert len(viz["jammers"]) == 1
        j = viz["jammers"][0]
        assert j["id"] == "j1"
        assert j["radius"] == 500.0
        assert j["broadband"] is True

    def test_inactive_jammer_excluded(self, sim: CommsSimulator):
        sim.add_jammer(Jammer(
            jammer_id="j1", position=(0.0, 0.0), radius=100.0, power_w=10.0,
            is_active=False,
        ))
        viz = sim.to_three_js()
        assert len(viz["jammers"]) == 0

    def test_transmission_lines(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        a = _make_radio("a", position=(0.0, 0.0))
        a.is_transmitting = True
        sim.add_radio(a)
        sim.add_radio(_make_radio("b", position=(500.0, 0.0)))
        viz = sim.to_three_js()
        assert len(viz["transmissions"]) == 1
        line = viz["transmissions"][0]
        assert line["color"] == "#05ffa1"

    def test_intercept_indicators(self, sim: CommsSimulator, blue_channel: RadioChannel, red_channel: RadioChannel):
        sim.add_channel(blue_channel)
        sim.add_channel(red_channel)
        sim.add_radio(_make_radio("blue1", position=(0.0, 0.0), channel="ch_blue"))
        sim.add_radio(_make_radio("red1", position=(500.0, 0.0), channel="ch_red"))
        sim.transmit("blue1", "Interceptable message")
        viz = sim.to_three_js()
        assert len(viz["intercepts"]) == 1
        assert viz["intercepts"][0]["color"] == "#fcee0a"

    def test_satcom_range_capped(self, sim: CommsSimulator):
        ch = RadioChannel(channel_id="sat", frequency_mhz=1600.0, name="Sat")
        sim.add_channel(ch)
        sim.add_radio(_make_radio("s1", channel="sat", preset="satcom"))
        viz = sim.to_three_js()
        assert viz["radios"][0]["range"] == 100_000.0  # capped from inf

    def test_three_js_position_format(self, sim: CommsSimulator, squad_channel: RadioChannel):
        sim.add_channel(squad_channel)
        sim.add_radio(_make_radio("a", position=(10.0, 20.0)))
        viz = sim.to_three_js()
        pos = viz["radios"][0]["position"]
        assert pos == [10.0, 0.0, 20.0]  # x, y(up), z

    def test_empty_visualization(self, sim: CommsSimulator):
        viz = sim.to_three_js()
        assert viz["radios"] == []
        assert viz["jammers"] == []
        assert viz["transmissions"] == []
        assert viz["intercepts"] == []
