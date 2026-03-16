# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Cyber warfare and electronic warfare engine for the Tritium sim engine.

Models offensive cyber and EW capabilities: jammers that degrade comms,
GPS spoofers that divert drones, drone hijacking, signals intercept,
decoy emitters, denial of service, and malware injection.

Each CyberAsset carries one or more CyberCapability. Launching an attack
creates a CyberEffect that persists in the world and affects units each
tick. The engine resolves which units are affected by range, alliance,
and line-of-sight checks.

Usage::

    from tritium_lib.sim_engine.cyber import (
        CyberWarfareEngine, CyberAsset, CyberCapability,
        CyberAttackType, CYBER_PRESETS,
    )

    engine = CyberWarfareEngine()
    asset = CyberAsset(
        asset_id="jammer-1", position=(100.0, 200.0), alliance="friendly",
        capabilities=[CyberCapability(
            capability_id="jam1", attack_type=CyberAttackType.JAMMING,
            power=0.8, range_m=200.0, duration=60.0, cooldown=10.0,
        )],
    )
    engine.deploy_asset(asset)
    effect = engine.launch_attack("jammer-1", "jam1", (150.0, 200.0))
    events = engine.tick(1.0, {"unit1": (160.0, 210.0)}, {"drone1": (170.0, 200.0)})
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CyberAttackType(Enum):
    """Types of cyber / electronic warfare attacks."""
    JAMMING = "jamming"                 # broadband or targeted RF jamming
    SPOOFING = "spoofing"               # false sensor contacts / identity spoofing
    DENIAL_OF_SERVICE = "denial_of_service"  # network / comms denial
    MALWARE = "malware"                 # inject malware into enemy systems
    INTERCEPT = "intercept"             # passive SIGINT — read enemy comms
    DECOY = "decoy"                     # emit false radar / sensor signatures
    GPS_SPOOFING = "gps_spoofing"       # broadcast false GPS signals
    DRONE_HIJACK = "drone_hijack"       # take control of enemy drone


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CyberCapability:
    """A single cyber/EW capability carried by an asset."""
    capability_id: str
    attack_type: CyberAttackType
    power: float                        # 0-1 effectiveness
    range_m: float                      # effective range in meters
    duration: float                     # seconds the effect lasts
    cooldown: float                     # seconds before reuse
    requires_los: bool = False          # line-of-sight required?
    # Runtime
    cooldown_remaining: float = 0.0     # time until ready

    @property
    def is_ready(self) -> bool:
        return self.cooldown_remaining <= 0.0


@dataclass
class CyberAsset:
    """A cyber/EW platform — vehicle, infantry team, or fixed site."""
    asset_id: str
    position: Vec2
    alliance: str
    capabilities: list[CyberCapability] = field(default_factory=list)
    active_attacks: list[dict] = field(default_factory=list)
    power_level: float = 1.0           # battery / fuel (0-1)
    detected: bool = False             # whether enemy has located this asset


@dataclass
class CyberEffect:
    """An active cyber/EW effect in the world."""
    effect_id: str
    attack_type: CyberAttackType
    position: Vec2
    radius: float
    intensity: float                   # 0-1 current strength
    remaining: float                   # seconds left
    source_id: str                     # asset that created this
    target_alliance: str               # which alliance is affected
    # GPS spoofing specifics
    fake_position: Vec2 | None = None  # where GPS-spoofed units think they are
    # Drone hijack specifics
    hijacked_drone_id: str | None = None
    # Intercept specifics
    intercepted_messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _make_cap(cid: str, atype: CyberAttackType, **kw: Any) -> CyberCapability:
    return CyberCapability(capability_id=cid, attack_type=atype, **kw)


CYBER_PRESETS: dict[str, dict[str, Any]] = {
    "jammer_truck": {
        "description": "Vehicle-mounted broadband jammer",
        "capabilities": [
            _make_cap("jam_main", CyberAttackType.JAMMING,
                      power=0.9, range_m=500.0, duration=120.0, cooldown=15.0),
        ],
    },
    "sigint_post": {
        "description": "Passive intercept station",
        "capabilities": [
            _make_cap("sigint_rx", CyberAttackType.INTERCEPT,
                      power=0.85, range_m=1000.0, duration=300.0, cooldown=5.0),
        ],
    },
    "gps_spoofer": {
        "description": "GPS denial / spoofing device",
        "capabilities": [
            _make_cap("gps_spoof", CyberAttackType.GPS_SPOOFING,
                      power=0.8, range_m=300.0, duration=90.0, cooldown=20.0),
            _make_cap("gps_jam", CyberAttackType.JAMMING,
                      power=0.6, range_m=200.0, duration=60.0, cooldown=10.0),
        ],
    },
    "drone_controller": {
        "description": "Drone hijack capability",
        "capabilities": [
            _make_cap("hijack", CyberAttackType.DRONE_HIJACK,
                      power=0.7, range_m=250.0, duration=45.0, cooldown=30.0,
                      requires_los=True),
        ],
    },
    "cyber_team": {
        "description": "Infantry unit with laptop, multiple capabilities",
        "capabilities": [
            _make_cap("ct_dos", CyberAttackType.DENIAL_OF_SERVICE,
                      power=0.6, range_m=150.0, duration=30.0, cooldown=20.0),
            _make_cap("ct_malware", CyberAttackType.MALWARE,
                      power=0.5, range_m=100.0, duration=60.0, cooldown=45.0),
            _make_cap("ct_spoof", CyberAttackType.SPOOFING,
                      power=0.65, range_m=200.0, duration=40.0, cooldown=15.0),
            _make_cap("ct_decoy", CyberAttackType.DECOY,
                      power=0.7, range_m=300.0, duration=50.0, cooldown=10.0),
            _make_cap("ct_intercept", CyberAttackType.INTERCEPT,
                      power=0.55, range_m=250.0, duration=120.0, cooldown=10.0),
        ],
    },
}


def create_asset_from_preset(
    preset_name: str,
    asset_id: str,
    position: Vec2,
    alliance: str,
) -> CyberAsset:
    """Create a CyberAsset from a named preset."""
    preset = CYBER_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name!r}")
    # Deep-copy capabilities so each asset has independent cooldown state
    caps = []
    for orig in preset["capabilities"]:
        caps.append(CyberCapability(
            capability_id=f"{asset_id}_{orig.capability_id}",
            attack_type=orig.attack_type,
            power=orig.power,
            range_m=orig.range_m,
            duration=orig.duration,
            cooldown=orig.cooldown,
            requires_los=orig.requires_los,
        ))
    return CyberAsset(
        asset_id=asset_id,
        position=position,
        alliance=alliance,
        capabilities=caps,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CyberWarfareEngine:
    """Cyber warfare and electronic warfare simulation engine.

    Manages cyber assets and the effects they create. Each ``tick()``
    updates effect durations, resolves impacts on units/drones, and
    returns a list of event dicts describing what happened.
    """

    def __init__(self, rng_seed: int | None = None) -> None:
        self.assets: dict[str, CyberAsset] = {}
        self.active_effects: list[CyberEffect] = []
        self._rng = random.Random(rng_seed)
        self._event_log: list[dict[str, Any]] = []

    # -- asset management ---------------------------------------------------

    def deploy_asset(self, asset: CyberAsset) -> None:
        """Deploy a cyber/EW asset."""
        self.assets[asset.asset_id] = asset

    def remove_asset(self, asset_id: str) -> None:
        """Remove a cyber/EW asset (destroyed or withdrawn)."""
        self.assets.pop(asset_id, None)
        # Remove effects sourced from this asset
        self.active_effects = [
            e for e in self.active_effects if e.source_id != asset_id
        ]

    # -- launching attacks --------------------------------------------------

    def launch_attack(
        self,
        asset_id: str,
        capability_id: str,
        target_pos: Vec2,
    ) -> CyberEffect | None:
        """Launch a cyber attack from an asset using a specific capability.

        Returns the created CyberEffect, or None if the capability is not
        ready, out of range, or the asset lacks power.
        """
        asset = self.assets.get(asset_id)
        if asset is None:
            return None

        # Find the capability
        cap: CyberCapability | None = None
        for c in asset.capabilities:
            if c.capability_id == capability_id:
                cap = c
                break
        if cap is None:
            return None

        # Check readiness
        if not cap.is_ready:
            return None
        if asset.power_level <= 0.0:
            return None

        # Range check
        dist = distance(asset.position, target_pos)
        if dist > cap.range_m:
            return None

        # Create the effect
        effect_id = f"cyber_{uuid.uuid4().hex[:8]}"
        intensity = cap.power * asset.power_level

        # GPS spoofing: generate a fake offset position
        fake_pos: Vec2 | None = None
        if cap.attack_type == CyberAttackType.GPS_SPOOFING:
            offset_angle = self._rng.uniform(0, 2 * math.pi)
            offset_dist = self._rng.uniform(50.0, 200.0) * intensity
            fake_pos = (
                target_pos[0] + math.cos(offset_angle) * offset_dist,
                target_pos[1] + math.sin(offset_angle) * offset_dist,
            )

        effect = CyberEffect(
            effect_id=effect_id,
            attack_type=cap.attack_type,
            position=target_pos,
            radius=cap.range_m * 0.5,  # effect radius is half weapon range
            intensity=intensity,
            remaining=cap.duration,
            source_id=asset_id,
            target_alliance=_opposing_alliance(asset.alliance),
            fake_position=fake_pos,
        )
        self.active_effects.append(effect)

        # Start cooldown
        cap.cooldown_remaining = cap.cooldown

        # Drain power
        drain = 0.05 * cap.power
        asset.power_level = max(0.0, asset.power_level - drain)

        # Record on asset
        asset.active_attacks.append({
            "effect_id": effect_id,
            "capability_id": capability_id,
            "target_pos": list(target_pos),
        })

        self._event_log.append({
            "type": "cyber_attack_launched",
            "asset_id": asset_id,
            "capability_id": capability_id,
            "attack_type": cap.attack_type.value,
            "effect_id": effect_id,
            "target_pos": list(target_pos),
            "intensity": round(intensity, 3),
        })

        return effect

    # -- tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, Vec2] | None = None,
        drone_positions: dict[str, Vec2] | None = None,
    ) -> list[dict]:
        """Advance the cyber warfare simulation by *dt* seconds.

        Resolves all active effects against the provided unit and drone
        positions. Returns a list of event dicts describing impacts.

        Args:
            dt: Time step in seconds.
            unit_positions: Mapping of unit_id -> position for ground units.
            drone_positions: Mapping of drone_id -> position for drones/UAVs.

        Returns:
            List of event dicts (jamming, interceptions, hijacks, etc.).
        """
        if unit_positions is None:
            unit_positions = {}
        if drone_positions is None:
            drone_positions = {}

        events: list[dict[str, Any]] = []

        # Tick cooldowns on all capabilities
        for asset in self.assets.values():
            for cap in asset.capabilities:
                if cap.cooldown_remaining > 0.0:
                    cap.cooldown_remaining = max(0.0, cap.cooldown_remaining - dt)

        # Process each active effect
        expired: list[str] = []
        for effect in self.active_effects:
            effect.remaining -= dt
            if effect.remaining <= 0.0:
                expired.append(effect.effect_id)
                events.append({
                    "type": "effect_expired",
                    "effect_id": effect.effect_id,
                    "attack_type": effect.attack_type.value,
                })
                continue

            # Decay intensity slightly over time
            ratio = effect.remaining / (effect.remaining + dt)
            effect.intensity *= (0.99 + 0.01 * ratio)

            # Resolve per attack type
            all_positions = {**unit_positions, **drone_positions}
            affected = self.get_affected_units(effect, all_positions)

            if effect.attack_type == CyberAttackType.JAMMING:
                for uid in affected:
                    events.append({
                        "type": "comms_degraded",
                        "unit_id": uid,
                        "source_effect": effect.effect_id,
                        "degradation": round(effect.intensity, 3),
                    })

            elif effect.attack_type == CyberAttackType.GPS_SPOOFING:
                drone_affected = [
                    d for d in affected if d in drone_positions
                ]
                for did in drone_affected:
                    events.append({
                        "type": "gps_spoofed",
                        "drone_id": did,
                        "source_effect": effect.effect_id,
                        "fake_position": list(effect.fake_position) if effect.fake_position else None,
                        "intensity": round(effect.intensity, 3),
                    })
                # Also affect ground units with GPS
                for uid in affected:
                    if uid not in drone_positions:
                        events.append({
                            "type": "gps_degraded",
                            "unit_id": uid,
                            "source_effect": effect.effect_id,
                            "intensity": round(effect.intensity, 3),
                        })

            elif effect.attack_type == CyberAttackType.DRONE_HIJACK:
                drone_affected = [
                    d for d in affected if d in drone_positions
                ]
                for did in drone_affected:
                    # Probabilistic hijack per tick
                    hijack_chance = effect.intensity * 0.02 * dt
                    if self._rng.random() < hijack_chance:
                        effect.hijacked_drone_id = did
                        events.append({
                            "type": "drone_hijacked",
                            "drone_id": did,
                            "source_effect": effect.effect_id,
                            "new_controller": effect.source_id,
                        })
                    else:
                        events.append({
                            "type": "drone_hijack_attempt",
                            "drone_id": did,
                            "source_effect": effect.effect_id,
                            "intensity": round(effect.intensity, 3),
                        })

            elif effect.attack_type == CyberAttackType.INTERCEPT:
                for uid in affected:
                    # Passive intercept — gather comms
                    if self._rng.random() < effect.intensity * 0.1 * dt:
                        msg = {
                            "intercepted_from": uid,
                            "timestamp": effect.remaining,
                            "confidence": round(effect.intensity, 2),
                        }
                        effect.intercepted_messages.append(msg)
                        events.append({
                            "type": "comms_intercepted",
                            "unit_id": uid,
                            "source_effect": effect.effect_id,
                            "confidence": round(effect.intensity, 2),
                        })

            elif effect.attack_type == CyberAttackType.DECOY:
                # Decoys persist as false contacts — no per-tick events needed
                # unless we want to report their ongoing presence
                if len(affected) > 0:
                    events.append({
                        "type": "decoy_active",
                        "effect_id": effect.effect_id,
                        "position": list(effect.position),
                        "observers": affected,
                    })

            elif effect.attack_type == CyberAttackType.DENIAL_OF_SERVICE:
                for uid in affected:
                    events.append({
                        "type": "network_denied",
                        "unit_id": uid,
                        "source_effect": effect.effect_id,
                        "severity": round(effect.intensity, 3),
                    })

            elif effect.attack_type == CyberAttackType.MALWARE:
                for uid in affected:
                    # Malware has a chance to "infect" per tick
                    if self._rng.random() < effect.intensity * 0.01 * dt:
                        events.append({
                            "type": "malware_infected",
                            "unit_id": uid,
                            "source_effect": effect.effect_id,
                        })
                    else:
                        events.append({
                            "type": "malware_attempt",
                            "unit_id": uid,
                            "source_effect": effect.effect_id,
                        })

            elif effect.attack_type == CyberAttackType.SPOOFING:
                for uid in affected:
                    events.append({
                        "type": "sensor_spoofed",
                        "unit_id": uid,
                        "source_effect": effect.effect_id,
                        "intensity": round(effect.intensity, 3),
                    })

        # Remove expired effects
        expired_set = set(expired)
        self.active_effects = [
            e for e in self.active_effects if e.effect_id not in expired_set
        ]

        # Clean up asset active_attacks references to expired effects
        for asset in self.assets.values():
            asset.active_attacks = [
                a for a in asset.active_attacks
                if a["effect_id"] not in expired_set
            ]

        return events

    # -- queries ------------------------------------------------------------

    def get_affected_units(
        self,
        effect: CyberEffect,
        unit_positions: dict[str, Vec2],
    ) -> list[str]:
        """Return IDs of units within the effect's radius."""
        affected: list[str] = []
        for uid, pos in unit_positions.items():
            if distance(pos, effect.position) <= effect.radius:
                affected.append(uid)
        return affected

    def is_jammed(self, position: Vec2) -> bool:
        """Check if a position is within any active jamming effect."""
        for effect in self.active_effects:
            if effect.attack_type != CyberAttackType.JAMMING:
                continue
            if distance(position, effect.position) <= effect.radius:
                return True
        return False

    def is_gps_spoofed(self, position: Vec2) -> tuple[bool, Vec2]:
        """Check if a position is affected by GPS spoofing.

        Returns (is_spoofed, fake_position). If not spoofed, fake_position
        equals the input position.
        """
        for effect in self.active_effects:
            if effect.attack_type != CyberAttackType.GPS_SPOOFING:
                continue
            if distance(position, effect.position) <= effect.radius:
                fake = effect.fake_position if effect.fake_position else position
                return (True, fake)
        return (False, position)

    def get_active_effects_at(self, position: Vec2) -> list[CyberEffect]:
        """Return all active effects covering a position."""
        return [
            e for e in self.active_effects
            if distance(position, e.position) <= e.radius
        ]

    def get_effects_by_type(self, attack_type: CyberAttackType) -> list[CyberEffect]:
        """Return all active effects of a given type."""
        return [e for e in self.active_effects if e.attack_type == attack_type]

    def get_intercepted_messages(self, asset_id: str) -> list[dict]:
        """Return all messages intercepted by effects sourced from an asset."""
        messages: list[dict] = []
        for effect in self.active_effects:
            if effect.source_id == asset_id and effect.attack_type == CyberAttackType.INTERCEPT:
                messages.extend(effect.intercepted_messages)
        return messages

    # -- visualization ------------------------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export cyber warfare state for Three.js visualization.

        Returns jamming zones, spoofing areas, attack lines, asset positions,
        and active effect overlays.
        """
        assets_out: list[dict[str, Any]] = []
        for asset in self.assets.values():
            caps_out = []
            for cap in asset.capabilities:
                caps_out.append({
                    "id": cap.capability_id,
                    "type": cap.attack_type.value,
                    "ready": cap.is_ready,
                    "cooldown_remaining": round(cap.cooldown_remaining, 1),
                })
            assets_out.append({
                "id": asset.asset_id,
                "position": [asset.position[0], 0.0, asset.position[1]],
                "alliance": asset.alliance,
                "power_level": round(asset.power_level, 3),
                "detected": asset.detected,
                "capabilities": caps_out,
                "color": _alliance_color(asset.alliance),
            })

        effects_out: list[dict[str, Any]] = []
        for effect in self.active_effects:
            effect_data: dict[str, Any] = {
                "id": effect.effect_id,
                "type": effect.attack_type.value,
                "position": [effect.position[0], 0.0, effect.position[1]],
                "radius": effect.radius,
                "intensity": round(effect.intensity, 3),
                "remaining": round(effect.remaining, 1),
                "source_id": effect.source_id,
                "color": _effect_color(effect.attack_type),
                "opacity": min(0.5, effect.intensity * 0.4),
            }
            if effect.fake_position:
                effect_data["fake_position"] = [
                    effect.fake_position[0], 0.0, effect.fake_position[1],
                ]
            if effect.hijacked_drone_id:
                effect_data["hijacked_drone"] = effect.hijacked_drone_id
            effects_out.append(effect_data)

        # Attack lines from asset to effect position
        attack_lines: list[dict[str, Any]] = []
        for asset in self.assets.values():
            for aa in asset.active_attacks:
                attack_lines.append({
                    "from": [asset.position[0], 0.0, asset.position[1]],
                    "to": [aa["target_pos"][0], 0.0, aa["target_pos"][1]],
                    "asset_id": asset.asset_id,
                    "effect_id": aa["effect_id"],
                    "color": "#00f0ff",
                })

        return {
            "assets": assets_out,
            "effects": effects_out,
            "attack_lines": attack_lines,
        }

    # -- internal helpers ---------------------------------------------------

    def drain_event_log(self) -> list[dict[str, Any]]:
        """Return and clear internal event log."""
        log = self._event_log.copy()
        self._event_log.clear()
        return log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opposing_alliance(alliance: str) -> str:
    """Return the opposing alliance string."""
    if alliance == "friendly":
        return "hostile"
    elif alliance == "hostile":
        return "friendly"
    return "unknown"


def _alliance_color(alliance: str) -> str:
    """Return a cyberpunk color for an alliance."""
    return {
        "friendly": "#05ffa1",
        "hostile": "#ff2a6d",
        "neutral": "#fcee0a",
    }.get(alliance, "#888888")


def _effect_color(attack_type: CyberAttackType) -> str:
    """Return a color for each attack type."""
    return {
        CyberAttackType.JAMMING: "#ff2a6d",
        CyberAttackType.SPOOFING: "#ff00ff",
        CyberAttackType.DENIAL_OF_SERVICE: "#ff8800",
        CyberAttackType.MALWARE: "#aa00ff",
        CyberAttackType.INTERCEPT: "#00f0ff",
        CyberAttackType.DECOY: "#fcee0a",
        CyberAttackType.GPS_SPOOFING: "#ff4444",
        CyberAttackType.DRONE_HIJACK: "#05ffa1",
    }.get(attack_type, "#888888")
