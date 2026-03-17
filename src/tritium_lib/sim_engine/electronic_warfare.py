# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Electronic warfare and cyber operations for the Tritium sim engine.

Simulates jammers that disrupt comms, cyber attacks on sensor systems,
EMP effects that disable electronics, and spoofing that creates false
radar contacts.

Usage::

    from tritium_lib.sim_engine.electronic_warfare import EWEngine, EWJammer, CyberAttack

    ew = EWEngine()
    ew.place_jammer(EWJammer(jammer_id="j1", position=(200.0, 200.0), radius=100.0))
    ew.launch_cyber_attack(CyberAttack(
        attack_id="c1", target_system="radar", target_id="sensor_1",
        duration=30.0, success_probability=0.7,
    ))
    result = ew.tick(0.1)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JammerType(Enum):
    """Types of electronic jammers."""
    BROADBAND = "broadband"        # jams all frequencies in radius
    TARGETED = "targeted"          # jams specific frequency bands
    GPS = "gps"                    # degrades GPS accuracy
    COMMUNICATIONS = "communications"  # disrupts radio comms only


class CyberAttackType(Enum):
    """Types of cyber operations."""
    SENSOR_BLIND = "sensor_blind"      # disable enemy sensors
    COMMS_INTERCEPT = "comms_intercept"  # intercept enemy comms
    DATA_CORRUPT = "data_corrupt"      # corrupt enemy intel data
    SPOOF_CONTACTS = "spoof_contacts"  # inject false contacts
    NETWORK_DENIAL = "network_denial"  # deny enemy network access


class EMPScale(Enum):
    """Scale of EMP events."""
    TACTICAL = "tactical"    # small radius, portable device
    THEATER = "theater"      # medium radius, vehicle-mounted
    STRATEGIC = "strategic"  # large radius, weapon-scale


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EWJammer:
    """An electronic jammer emplacement or vehicle."""
    jammer_id: str
    position: Vec2
    radius: float                   # meters
    jammer_type: JammerType = JammerType.BROADBAND
    power_w: float = 100.0
    is_active: bool = True
    alliance: str = "friendly"
    frequencies: list[float] = field(default_factory=list)  # MHz, for TARGETED
    battery: float = 1.0           # 0-1
    drain_rate: float = 0.001      # per second when active


@dataclass
class CyberAttack:
    """A cyber attack operation against enemy systems."""
    attack_id: str
    target_system: str              # "radar", "comms", "intel", "gps"
    target_id: str                  # specific target entity/system ID
    duration: float                 # seconds the attack lasts
    success_probability: float = 0.5  # chance of success (0-1)
    alliance: str = "friendly"
    # Runtime state
    is_active: bool = False
    time_remaining: float = 0.0
    succeeded: bool = False


@dataclass
class EMPEvent:
    """An electromagnetic pulse event."""
    emp_id: str
    position: Vec2
    radius: float
    scale: EMPScale = EMPScale.TACTICAL
    # Runtime
    is_active: bool = True
    time_remaining: float = 5.0    # seconds of effect
    intensity: float = 1.0         # decays over time


@dataclass
class SpoofContact:
    """A false radar/sensor contact injected by EW."""
    contact_id: str
    position: Vec2
    velocity: Vec2 = (0.0, 0.0)
    classification: str = "unknown"
    duration: float = 30.0
    time_remaining: float = 30.0
    alliance: str = "hostile"       # what alliance sees the spoof
    source_id: str = ""             # which EW system created it


@dataclass
class DisruptedSystem:
    """A system currently affected by EW operations."""
    system_id: str
    system_type: str                # "sensor", "comms", "gps", "network"
    disruption_type: str            # "jammed", "blinded", "spoofed", "emp"
    time_remaining: float
    severity: float = 1.0          # 0-1, how badly affected


# ---------------------------------------------------------------------------
# EMP presets
# ---------------------------------------------------------------------------

EMP_PRESETS: dict[str, dict[str, Any]] = {
    "tactical": {
        "radius": 50.0,
        "scale": EMPScale.TACTICAL,
        "duration": 5.0,
    },
    "theater": {
        "radius": 200.0,
        "scale": EMPScale.THEATER,
        "duration": 10.0,
    },
    "strategic": {
        "radius": 1000.0,
        "scale": EMPScale.STRATEGIC,
        "duration": 30.0,
    },
}


# ---------------------------------------------------------------------------
# EWEngine
# ---------------------------------------------------------------------------

class EWEngine:
    """Electronic warfare simulation engine.

    Manages jammers, cyber attacks, EMP events, and spoofed contacts.
    Each tick updates active operations, decays effects, and reports
    which systems are disrupted.
    """

    def __init__(self, rng_seed: int | None = None) -> None:
        self.jammers: dict[str, EWJammer] = {}
        self.cyber_attacks: list[CyberAttack] = []
        self.emp_events: list[EMPEvent] = []
        self.spoof_contacts: list[SpoofContact] = []
        self.disrupted_systems: dict[str, DisruptedSystem] = {}
        self._rng = random.Random(rng_seed)
        self._event_log: list[dict[str, Any]] = []

    # -- jammers ------------------------------------------------------------

    def place_jammer(self, jammer: EWJammer) -> None:
        """Deploy a jammer."""
        self.jammers[jammer.jammer_id] = jammer

    def remove_jammer(self, jammer_id: str) -> None:
        """Remove a jammer."""
        self.jammers.pop(jammer_id, None)

    def activate_jammer(self, jammer_id: str) -> bool:
        """Activate a jammer. Returns False if not found or no battery."""
        j = self.jammers.get(jammer_id)
        if j is None or j.battery <= 0.0:
            return False
        j.is_active = True
        return True

    def deactivate_jammer(self, jammer_id: str) -> bool:
        """Deactivate a jammer."""
        j = self.jammers.get(jammer_id)
        if j is None:
            return False
        j.is_active = False
        return True

    def is_position_jammed(self, position: Vec2) -> bool:
        """Check if a position is within any active jammer's radius."""
        for j in self.jammers.values():
            if not j.is_active or j.battery <= 0.0:
                continue
            if distance(position, j.position) <= j.radius:
                return True
        return False

    def get_jammers_affecting(self, position: Vec2) -> list[EWJammer]:
        """Return all active jammers affecting a given position."""
        result: list[EWJammer] = []
        for j in self.jammers.values():
            if not j.is_active or j.battery <= 0.0:
                continue
            if distance(position, j.position) <= j.radius:
                result.append(j)
        return result

    # -- cyber attacks ------------------------------------------------------

    def launch_cyber_attack(self, attack: CyberAttack) -> None:
        """Initiate a cyber attack."""
        attack.is_active = True
        attack.time_remaining = attack.duration
        # Resolve success on launch
        attack.succeeded = self._rng.random() < attack.success_probability
        self.cyber_attacks.append(attack)

        if attack.succeeded:
            # Create disruption on the target
            ds = DisruptedSystem(
                system_id=attack.target_id,
                system_type=attack.target_system,
                disruption_type="blinded" if attack.target_system == "radar" else "disrupted",
                time_remaining=attack.duration,
                severity=0.8,
            )
            self.disrupted_systems[attack.target_id] = ds
            self._event_log.append({
                "type": "cyber_attack_success",
                "attack_id": attack.attack_id,
                "target": attack.target_id,
                "system": attack.target_system,
            })
        else:
            self._event_log.append({
                "type": "cyber_attack_failed",
                "attack_id": attack.attack_id,
                "target": attack.target_id,
            })

    def is_system_disrupted(self, system_id: str) -> bool:
        """Check if a system is currently disrupted by EW."""
        ds = self.disrupted_systems.get(system_id)
        return ds is not None and ds.time_remaining > 0.0

    # -- EMP ----------------------------------------------------------------

    def detonate_emp(
        self,
        position: Vec2,
        preset: str = "tactical",
        emp_id: str | None = None,
    ) -> EMPEvent:
        """Detonate an EMP at a position using a preset."""
        params = EMP_PRESETS.get(preset, EMP_PRESETS["tactical"])
        if emp_id is None:
            emp_id = f"emp_{uuid.uuid4().hex[:8]}"
        emp = EMPEvent(
            emp_id=emp_id,
            position=position,
            radius=params["radius"],
            scale=params["scale"],
            time_remaining=params["duration"],
        )
        self.emp_events.append(emp)
        self._event_log.append({
            "type": "emp_detonation",
            "emp_id": emp_id,
            "position": list(position),
            "radius": emp.radius,
            "scale": emp.scale.value,
        })
        return emp

    def is_position_emp_affected(self, position: Vec2) -> bool:
        """Check if a position is within an active EMP effect."""
        for emp in self.emp_events:
            if not emp.is_active:
                continue
            if distance(position, emp.position) <= emp.radius:
                return True
        return False

    def get_emp_severity(self, position: Vec2) -> float:
        """Return the worst EMP severity affecting a position (0-1)."""
        worst = 0.0
        for emp in self.emp_events:
            if not emp.is_active:
                continue
            d = distance(position, emp.position)
            if d <= emp.radius:
                falloff = 1.0 - (d / emp.radius)
                severity = falloff * emp.intensity
                worst = max(worst, severity)
        return worst

    # -- spoofing -----------------------------------------------------------

    def create_spoof(
        self,
        position: Vec2,
        target_alliance: str = "hostile",
        classification: str = "unknown",
        duration: float = 30.0,
        velocity: Vec2 = (0.0, 0.0),
        source_id: str = "",
    ) -> SpoofContact:
        """Create a false contact visible to the target alliance's sensors."""
        sc = SpoofContact(
            contact_id=f"spoof_{uuid.uuid4().hex[:8]}",
            position=position,
            velocity=velocity,
            classification=classification,
            duration=duration,
            time_remaining=duration,
            alliance=target_alliance,
            source_id=source_id,
        )
        self.spoof_contacts.append(sc)
        self._event_log.append({
            "type": "spoof_created",
            "contact_id": sc.contact_id,
            "position": list(position),
            "target_alliance": target_alliance,
        })
        return sc

    def get_spoofs_for_alliance(self, alliance: str) -> list[SpoofContact]:
        """Return active spoof contacts visible to an alliance."""
        return [
            sc for sc in self.spoof_contacts
            if sc.alliance == alliance and sc.time_remaining > 0.0
        ]

    # -- queries ------------------------------------------------------------

    def get_disruption_summary(self) -> dict[str, Any]:
        """Summary of all active EW effects."""
        return {
            "active_jammers": sum(1 for j in self.jammers.values() if j.is_active),
            "total_jammers": len(self.jammers),
            "active_cyber_attacks": sum(1 for c in self.cyber_attacks if c.is_active),
            "active_emps": sum(1 for e in self.emp_events if e.is_active),
            "active_spoofs": sum(1 for s in self.spoof_contacts if s.time_remaining > 0),
            "disrupted_systems": len(self.disrupted_systems),
        }

    def drain_event_log(self) -> list[dict[str, Any]]:
        """Return and clear the event log."""
        log = self._event_log.copy()
        self._event_log.clear()
        return log

    # -- tick ---------------------------------------------------------------

    def tick(self, dt: float) -> dict[str, Any]:
        """Advance all EW systems by *dt* seconds.

        - Drains jammer batteries
        - Decays cyber attack durations
        - Decays EMP effects
        - Moves spoof contacts along their velocity
        - Cleans up expired effects

        Returns a dict with events and current disruption summary.
        """
        events: list[dict[str, Any]] = []

        # 1. Jammer battery drain
        for j in self.jammers.values():
            if j.is_active and j.battery > 0.0:
                j.battery = max(0.0, j.battery - j.drain_rate * dt)
                if j.battery <= 0.0:
                    j.is_active = False
                    events.append({
                        "type": "jammer_battery_dead",
                        "jammer_id": j.jammer_id,
                    })

        # 2. Cyber attacks tick
        for ca in self.cyber_attacks:
            if not ca.is_active:
                continue
            ca.time_remaining -= dt
            if ca.time_remaining <= 0.0:
                ca.is_active = False
                events.append({
                    "type": "cyber_attack_expired",
                    "attack_id": ca.attack_id,
                })

        # 3. Disrupted systems tick
        expired_systems: list[str] = []
        for sid, ds in self.disrupted_systems.items():
            ds.time_remaining -= dt
            if ds.time_remaining <= 0.0:
                expired_systems.append(sid)
                events.append({
                    "type": "system_restored",
                    "system_id": sid,
                    "system_type": ds.system_type,
                })
        for sid in expired_systems:
            del self.disrupted_systems[sid]

        # 4. EMP decay
        for emp in self.emp_events:
            if not emp.is_active:
                continue
            emp.time_remaining -= dt
            emp.intensity = max(0.0, emp.time_remaining / max(emp.time_remaining + dt, 0.01))
            if emp.time_remaining <= 0.0:
                emp.is_active = False
                events.append({
                    "type": "emp_expired",
                    "emp_id": emp.emp_id,
                })

        # 5. Spoof contact movement and decay
        for sc in self.spoof_contacts:
            if sc.time_remaining <= 0.0:
                continue
            sc.time_remaining -= dt
            # Move along velocity
            sc.position = (
                sc.position[0] + sc.velocity[0] * dt,
                sc.position[1] + sc.velocity[1] * dt,
            )

        # 6. Cleanup expired spoofs
        self.spoof_contacts = [
            sc for sc in self.spoof_contacts if sc.time_remaining > 0.0
        ]

        # Cleanup expired cyber attacks (keep recent for log)
        self.cyber_attacks = [
            ca for ca in self.cyber_attacks
            if ca.is_active or ca.time_remaining > -60.0
        ]

        # Cleanup expired EMPs
        self.emp_events = [e for e in self.emp_events if e.is_active]

        return {
            "events": events,
            "summary": self.get_disruption_summary(),
        }

    # -- Three.js visualization ---------------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export EW state for Three.js visualization.

        Returns jammer radii, disrupted zones, EMP effects, and false contacts.
        """
        jammers_out: list[dict[str, Any]] = []
        for j in self.jammers.values():
            jammers_out.append({
                "id": j.jammer_id,
                "position": [j.position[0], 0.0, j.position[1]],
                "radius": j.radius,
                "type": j.jammer_type.value,
                "active": j.is_active,
                "battery": round(j.battery, 3),
                "alliance": j.alliance,
                "color": "#ff2a6d" if j.is_active else "#333333",
                "opacity": 0.3 if j.is_active else 0.1,
            })

        emp_out: list[dict[str, Any]] = []
        for emp in self.emp_events:
            if not emp.is_active:
                continue
            emp_out.append({
                "id": emp.emp_id,
                "position": [emp.position[0], 0.0, emp.position[1]],
                "radius": emp.radius,
                "intensity": round(emp.intensity, 3),
                "scale": emp.scale.value,
                "color": "#fcee0a",
                "opacity": emp.intensity * 0.5,
            })

        spoofs_out: list[dict[str, Any]] = []
        for sc in self.spoof_contacts:
            spoofs_out.append({
                "id": sc.contact_id,
                "position": [sc.position[0], 0.0, sc.position[1]],
                "classification": sc.classification,
                "target_alliance": sc.alliance,
                "remaining": round(sc.time_remaining, 1),
                "color": "#ff00ff",  # magenta for spoof contacts
            })

        disrupted_out: list[dict[str, Any]] = []
        for sid, ds in self.disrupted_systems.items():
            disrupted_out.append({
                "id": sid,
                "type": ds.system_type,
                "disruption": ds.disruption_type,
                "severity": round(ds.severity, 2),
                "remaining": round(ds.time_remaining, 1),
                "color": "#ff8800",
            })

        return {
            "jammers": jammers_out,
            "emp_effects": emp_out,
            "spoof_contacts": spoofs_out,
            "disrupted_systems": disrupted_out,
        }
