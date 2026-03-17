# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Communications and radio simulation for the Tritium sim engine.

Simulates radio communications, jamming, intercepts, and information warfare.
Models tactical radio networks with realistic propagation, battery drain,
frequency-selective jamming, and SIGINT intercept capabilities.

Usage::

    from tritium_lib.sim_engine.comms import CommsSimulator, RADIO_PRESETS

    sim = CommsSimulator()
    sim.add_channel(RadioChannel(channel_id="ch1", frequency_mhz=150.0, name="Squad Net"))
    sim.add_radio(Radio(
        radio_id="alpha-1", radio_type=RadioType.HANDHELD,
        position=(100.0, 200.0), current_channel="ch1",
        **RADIO_PRESETS["squad_radio"],
    ))
    report = sim.transmit("alpha-1", "Contact north!", msg_type="alert")
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RadioType(Enum):
    """Types of radios by form factor and capability."""
    HANDHELD = "handheld"
    MANPACK = "manpack"
    VEHICLE_MOUNTED = "vehicle_mounted"
    BASE_STATION = "base_station"
    SATELLITE = "satellite"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RadioChannel:
    """A radio frequency channel that radios can tune to."""
    channel_id: str
    frequency_mhz: float
    name: str
    encrypted: bool = False
    alliance: str = ""


@dataclass
class Radio:
    """A radio transceiver attached to a unit or position."""
    radio_id: str
    radio_type: RadioType
    position: Vec2
    transmit_power_w: float
    receive_sensitivity_dbm: float
    range_m: float
    current_channel: str
    is_transmitting: bool = False
    is_jammed: bool = False
    battery: float = 1.0  # 0.0 - 1.0


@dataclass
class RadioMessage:
    """A message transmitted over the radio network."""
    message_id: str
    sender_id: str
    channel_id: str
    content: str
    message_type: str  # voice, data, command, alert
    timestamp: float
    encrypted: bool
    position: Vec2  # sender position at time of transmission


@dataclass
class Jammer:
    """An electronic warfare jammer that disrupts radio communications."""
    jammer_id: str
    position: Vec2
    radius: float
    power_w: float
    frequencies: list[float] = field(default_factory=list)  # empty = broadband
    is_active: bool = True


# ---------------------------------------------------------------------------
# Radio presets
# ---------------------------------------------------------------------------

RADIO_PRESETS: dict[str, dict[str, Any]] = {
    "squad_radio": {
        "transmit_power_w": 5.0,
        "receive_sensitivity_dbm": -110.0,
        "range_m": 2_000.0,
    },
    "platoon_radio": {
        "transmit_power_w": 20.0,
        "receive_sensitivity_dbm": -115.0,
        "range_m": 10_000.0,
    },
    "vehicle_radio": {
        "transmit_power_w": 50.0,
        "receive_sensitivity_dbm": -118.0,
        "range_m": 30_000.0,
    },
    "base_station": {
        "transmit_power_w": 100.0,
        "receive_sensitivity_dbm": -120.0,
        "range_m": 50_000.0,
    },
    "satcom": {
        "transmit_power_w": 5.0,
        "receive_sensitivity_dbm": -125.0,
        "range_m": float("inf"),  # unlimited
    },
}


# ---------------------------------------------------------------------------
# Battery drain rates (fraction per second while transmitting)
# ---------------------------------------------------------------------------

_BATTERY_DRAIN_PER_SECOND: dict[RadioType, float] = {
    RadioType.HANDHELD: 0.0005,       # ~33 min continuous TX
    RadioType.MANPACK: 0.0003,        # ~55 min
    RadioType.VEHICLE_MOUNTED: 0.0,   # vehicle power
    RadioType.BASE_STATION: 0.0,      # mains power
    RadioType.SATELLITE: 0.0008,      # ~20 min
}


# ---------------------------------------------------------------------------
# CommsSimulator
# ---------------------------------------------------------------------------

class CommsSimulator:
    """Simulates tactical radio communications, jamming, and intercepts.

    Manages a network of radios on channels, calculates propagation,
    applies jamming effects, and tracks intercepted unencrypted traffic.
    """

    def __init__(self) -> None:
        self.radios: dict[str, Radio] = {}
        self.channels: dict[str, RadioChannel] = {}
        self.jammers: list[Jammer] = []
        self.message_log: list[RadioMessage] = []
        self.intercepted: list[RadioMessage] = []

    # -- mutators ----------------------------------------------------------

    def add_radio(self, radio: Radio) -> None:
        """Register a radio in the simulation."""
        self.radios[radio.radio_id] = radio

    def remove_radio(self, radio_id: str) -> None:
        """Remove a radio from the simulation."""
        self.radios.pop(radio_id, None)

    def add_channel(self, channel: RadioChannel) -> None:
        """Register a radio channel."""
        self.channels[channel.channel_id] = channel

    def remove_channel(self, channel_id: str) -> None:
        """Remove a channel."""
        self.channels.pop(channel_id, None)

    def add_jammer(self, jammer: Jammer) -> None:
        """Add a jammer to the simulation."""
        self.jammers.append(jammer)

    def remove_jammer(self, jammer_id: str) -> None:
        """Remove a jammer by ID."""
        self.jammers = [j for j in self.jammers if j.jammer_id != jammer_id]

    # -- queries -----------------------------------------------------------

    def is_jammed(self, radio: Radio) -> bool:
        """Check if a radio is being jammed by any active jammer.

        A radio is jammed if it is within a jammer's radius AND the jammer
        is either broadband (empty frequencies list) or covers the radio's
        current channel frequency.
        """
        channel = self.channels.get(radio.current_channel)
        if channel is None:
            return False

        for jammer in self.jammers:
            if not jammer.is_active:
                continue
            dist = distance(radio.position, jammer.position)
            if dist > jammer.radius:
                continue
            # Broadband jammer or frequency match
            if not jammer.frequencies:
                return True
            if channel.frequency_mhz in jammer.frequencies:
                return True
        return False

    def _in_range(self, sender: Radio, receiver: Radio) -> bool:
        """Check if receiver is within sender's effective range."""
        if sender.range_m == float("inf"):
            return True
        return distance(sender.position, receiver.position) <= sender.range_m

    def _on_same_channel(self, radio_a: Radio, radio_b: Radio) -> bool:
        """Check if two radios are tuned to the same channel."""
        return radio_a.current_channel == radio_b.current_channel

    def can_communicate(self, radio_a_id: str, radio_b_id: str) -> bool:
        """Check if two radios can communicate (range + frequency + no jamming).

        Both radios must be on the same channel, within range of each other,
        neither jammed, and both must have battery remaining.
        """
        radio_a = self.radios.get(radio_a_id)
        radio_b = self.radios.get(radio_b_id)
        if radio_a is None or radio_b is None:
            return False
        if radio_a.battery <= 0.0 or radio_b.battery <= 0.0:
            return False
        if not self._on_same_channel(radio_a, radio_b):
            return False
        if not self._in_range(radio_a, radio_b):
            return False
        if self.is_jammed(radio_a) or self.is_jammed(radio_b):
            return False
        return True

    # -- transmission ------------------------------------------------------

    def transmit(
        self,
        sender_id: str,
        content: str,
        msg_type: str = "voice",
    ) -> dict[str, Any]:
        """Transmit a message from a radio to all reachable radios on its channel.

        Returns a delivery report dict with keys:
            - ``success``: bool — whether the transmission was sent
            - ``sender_id``: str
            - ``channel_id``: str
            - ``message_id``: str (or empty on failure)
            - ``recipients``: list[str] — radio IDs that received the message
            - ``jammed``: bool — whether sender was jammed
            - ``intercepted``: bool — whether enemies intercepted the message
            - ``reason``: str — failure reason if not successful
        """
        sender = self.radios.get(sender_id)
        if sender is None:
            return {
                "success": False,
                "sender_id": sender_id,
                "channel_id": "",
                "message_id": "",
                "recipients": [],
                "jammed": False,
                "intercepted": False,
                "reason": "sender_not_found",
            }

        channel = self.channels.get(sender.current_channel)
        if channel is None:
            return {
                "success": False,
                "sender_id": sender_id,
                "channel_id": sender.current_channel,
                "message_id": "",
                "recipients": [],
                "jammed": False,
                "intercepted": False,
                "reason": "channel_not_found",
            }

        if sender.battery <= 0.0:
            return {
                "success": False,
                "sender_id": sender_id,
                "channel_id": sender.current_channel,
                "message_id": "",
                "recipients": [],
                "jammed": False,
                "intercepted": False,
                "reason": "no_battery",
            }

        # Check if sender is jammed
        if self.is_jammed(sender):
            sender.is_jammed = True
            return {
                "success": False,
                "sender_id": sender_id,
                "channel_id": sender.current_channel,
                "message_id": "",
                "recipients": [],
                "jammed": True,
                "intercepted": False,
                "reason": "sender_jammed",
            }

        sender.is_jammed = False

        # Build message
        msg = RadioMessage(
            message_id=uuid.uuid4().hex[:12],
            sender_id=sender_id,
            channel_id=sender.current_channel,
            content=content,
            message_type=msg_type,
            timestamp=time.time(),
            encrypted=channel.encrypted,
            position=sender.position,
        )

        # Find recipients: same channel, in range, not jammed
        recipients: list[str] = []
        for rid, radio in self.radios.items():
            if rid == sender_id:
                continue
            if radio.current_channel != sender.current_channel:
                continue
            if radio.battery <= 0.0:
                continue
            if not self._in_range(sender, radio):
                continue
            if self.is_jammed(radio):
                radio.is_jammed = True
                continue
            radio.is_jammed = False
            recipients.append(rid)

        sender.is_transmitting = True
        self.message_log.append(msg)

        # Check for interception: unencrypted messages on channels with an
        # alliance tag can be intercepted by radios of a different alliance
        # that are in range and on the same frequency.
        was_intercepted = False
        if not channel.encrypted and channel.alliance:
            for rid, radio in self.radios.items():
                if rid == sender_id:
                    continue
                if rid in recipients:
                    continue
                # Enemy radio listening on any channel at the same frequency
                enemy_channel = self.channels.get(radio.current_channel)
                if enemy_channel is None:
                    continue
                if enemy_channel.frequency_mhz != channel.frequency_mhz:
                    continue
                if not self._in_range(sender, radio):
                    continue
                if self.is_jammed(radio):
                    continue
                # Different alliance = intercept
                if enemy_channel.alliance and enemy_channel.alliance != channel.alliance:
                    self.intercepted.append(msg)
                    was_intercepted = True
                    break

        return {
            "success": True,
            "sender_id": sender_id,
            "channel_id": sender.current_channel,
            "message_id": msg.message_id,
            "recipients": recipients,
            "jammed": False,
            "intercepted": was_intercepted,
            "reason": "",
        }

    # -- network graph -----------------------------------------------------

    def get_comms_network(self) -> dict[str, Any]:
        """Return a graph of who can talk to who.

        Returns a dict with:
            - ``nodes``: list of {radio_id, position, channel, radio_type, battery, jammed}
            - ``edges``: list of {from, to, channel} for each communicating pair
            - ``channels``: list of {channel_id, name, frequency_mhz, encrypted}
        """
        nodes: list[dict[str, Any]] = []
        for rid, radio in self.radios.items():
            nodes.append({
                "radio_id": rid,
                "position": list(radio.position),
                "channel": radio.current_channel,
                "radio_type": radio.radio_type.value,
                "battery": round(radio.battery, 3),
                "jammed": self.is_jammed(radio),
            })

        edges: list[dict[str, Any]] = []
        radio_ids = list(self.radios.keys())
        for i, rid_a in enumerate(radio_ids):
            for rid_b in radio_ids[i + 1:]:
                if self.can_communicate(rid_a, rid_b):
                    edges.append({
                        "from": rid_a,
                        "to": rid_b,
                        "channel": self.radios[rid_a].current_channel,
                    })

        channel_list: list[dict[str, Any]] = []
        for cid, ch in self.channels.items():
            channel_list.append({
                "channel_id": cid,
                "name": ch.name,
                "frequency_mhz": ch.frequency_mhz,
                "encrypted": ch.encrypted,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "channels": channel_list,
        }

    # -- simulation tick ---------------------------------------------------

    def tick(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds.

        - Drains battery on transmitting radios.
        - Updates jammed status for all radios.
        - Stops transmission when battery dies.
        """
        for radio in self.radios.values():
            # Battery drain while transmitting
            if radio.is_transmitting and radio.battery > 0.0:
                drain = _BATTERY_DRAIN_PER_SECOND.get(radio.radio_type, 0.0003)
                radio.battery = max(0.0, radio.battery - drain * dt)
                if radio.battery <= 0.0:
                    radio.is_transmitting = False

            # Update jamming status
            radio.is_jammed = self.is_jammed(radio)

    # -- Three.js visualization -------------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export visualization data for Three.js rendering.

        Returns:
            - ``radios``: list of radio positions with range circles
            - ``jammers``: list of jammer positions with interference zones
            - ``transmissions``: active transmission lines between units
            - ``intercepts``: recent intercepted message indicators
        """
        radio_viz: list[dict[str, Any]] = []
        for rid, radio in self.radios.items():
            radio_viz.append({
                "id": rid,
                "position": [radio.position[0], 0.0, radio.position[1]],
                "range": radio.range_m if radio.range_m != float("inf") else 100_000.0,
                "type": radio.radio_type.value,
                "channel": radio.current_channel,
                "transmitting": radio.is_transmitting,
                "jammed": radio.is_jammed,
                "battery": round(radio.battery, 3),
                "color": "#ff2a6d" if radio.is_jammed else (
                    "#fcee0a" if radio.is_transmitting else "#00f0ff"
                ),
            })

        jammer_viz: list[dict[str, Any]] = []
        for jammer in self.jammers:
            if not jammer.is_active:
                continue
            jammer_viz.append({
                "id": jammer.jammer_id,
                "position": [jammer.position[0], 0.0, jammer.position[1]],
                "radius": jammer.radius,
                "power_w": jammer.power_w,
                "broadband": len(jammer.frequencies) == 0,
                "frequencies": jammer.frequencies,
                "color": "#ff2a6d",
            })

        # Active transmissions: lines between radios currently transmitting
        # and their reachable peers on the same channel
        tx_lines: list[dict[str, Any]] = []
        for rid, radio in self.radios.items():
            if not radio.is_transmitting:
                continue
            for peer_id, peer in self.radios.items():
                if peer_id == rid:
                    continue
                if self.can_communicate(rid, peer_id):
                    tx_lines.append({
                        "from": [radio.position[0], 0.5, radio.position[1]],
                        "to": [peer.position[0], 0.5, peer.position[1]],
                        "channel": radio.current_channel,
                        "color": "#05ffa1",
                    })

        # Recent intercepts (last 10)
        intercept_viz: list[dict[str, Any]] = []
        for msg in self.intercepted[-10:]:
            intercept_viz.append({
                "position": [msg.position[0], 1.0, msg.position[1]],
                "message_id": msg.message_id,
                "channel": msg.channel_id,
                "timestamp": msg.timestamp,
                "color": "#fcee0a",
            })

        return {
            "radios": radio_viz,
            "jammers": jammer_viz,
            "transmissions": tx_lines,
            "intercepts": intercept_viz,
        }
