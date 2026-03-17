# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Game audio event system for Three.js frontend.

Generates audio cue data and soundtrack state that the frontend consumes
via the Web Audio API. This module does NOT play audio — it produces
structured event data describing what sounds to trigger, where, and how.

Usage::

    engine = SoundtrackEngine()
    frame = engine.tick(sim_events, world_state)
    # frame["cues"]    — one-shot sound cues to play this tick
    # frame["music"]   — current music state (track, intensity, layers)
    # frame["ambient"] — persistent ambient loops
    # frame["stop"]    — list of cue_ids to stop
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AudioCategory(Enum):
    """Categories for audio cue prioritization and mixing."""
    COMBAT = "combat"
    AMBIENT = "ambient"
    UI = "ui"
    VOICE = "voice"
    MUSIC = "music"
    WEATHER = "weather"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioCue:
    """A single audio event to send to the frontend.

    Attributes:
        cue_id: Unique identifier for this cue instance.
        category: Audio mixing category.
        sound_name: Maps to a frontend audio file/buffer name.
        position: Optional (x, y) for 3D spatial positioning.
        volume: Gain multiplier, 0.0-1.0.
        pitch: Playback rate, 0.5-2.0.
        loop: Whether the sound should loop continuously.
        priority: 1 (highest) to 10 (lowest) for mixing decisions.
    """
    cue_id: str
    category: AudioCategory
    sound_name: str
    position: tuple[float, float] | None = None
    volume: float = 1.0
    pitch: float = 1.0
    loop: bool = False
    priority: int = 5

    def __post_init__(self) -> None:
        self.volume = max(0.0, min(1.0, self.volume))
        self.pitch = max(0.5, min(2.0, self.pitch))
        self.priority = max(1, min(10, self.priority))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a frontend-friendly dict."""
        d: dict[str, Any] = {
            "id": self.cue_id,
            "sound": self.sound_name,
            "vol": round(self.volume, 3),
            "pitch": round(self.pitch, 3),
            "loop": self.loop,
            "priority": self.priority,
            "category": self.category.value,
        }
        if self.position is not None:
            d["x"] = round(self.position[0], 2)
            d["y"] = round(self.position[1], 2)
        return d


@dataclass
class MusicState:
    """Describes the current music layer configuration.

    Attributes:
        current_track: Name of the active music track.
        intensity: Crossfade factor 0.0 (calm) to 1.0 (peak combat).
        layer_combat: Whether the combat music layer is active.
        layer_percussion: Whether the percussion layer is active.
    """
    current_track: str = "ambient"
    intensity: float = 0.0
    layer_combat: bool = False
    layer_percussion: bool = False

    def __post_init__(self) -> None:
        self.intensity = max(0.0, min(1.0, self.intensity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.current_track,
            "intensity": round(self.intensity, 3),
            "combat": self.layer_combat,
            "percussion": self.layer_percussion,
        }


# ---------------------------------------------------------------------------
# Sound mapping table — 30+ event-type to sound-name mappings
# ---------------------------------------------------------------------------

SOUND_MAP: dict[str, list[dict[str, Any]]] = {
    # Combat — gunfire
    "gunfire_rifle": [
        {"sound": "rifle_burst", "category": "combat", "priority": 2, "pitch_range": (0.9, 1.1)},
    ],
    "gunfire_sniper": [
        {"sound": "sniper_crack", "category": "combat", "priority": 1, "pitch_range": (0.95, 1.05)},
    ],
    "gunfire_mg": [
        {"sound": "mg_sustained", "category": "combat", "priority": 2, "pitch_range": (0.9, 1.0)},
    ],
    "gunfire_pistol": [
        {"sound": "pistol_shot", "category": "combat", "priority": 3, "pitch_range": (0.9, 1.1)},
    ],
    "gunfire_shotgun": [
        {"sound": "shotgun_blast", "category": "combat", "priority": 2, "pitch_range": (0.85, 1.0)},
    ],
    # Combat — explosions
    "explosion": [
        {"sound": "explosion_large", "category": "combat", "priority": 1, "pitch_range": (0.8, 1.0)},
    ],
    "explosion_small": [
        {"sound": "explosion_small", "category": "combat", "priority": 2, "pitch_range": (0.9, 1.1)},
    ],
    "grenade": [
        {"sound": "grenade_pop", "category": "combat", "priority": 2, "pitch_range": (0.9, 1.1)},
    ],
    # Combat — impacts
    "kill": [
        {"sound": "bullet_impact", "category": "combat", "priority": 2, "pitch_range": (0.9, 1.1)},
        {"sound": "body_fall", "category": "combat", "priority": 4, "pitch_range": (0.8, 1.2)},
    ],
    "hit": [
        {"sound": "bullet_impact", "category": "combat", "priority": 3, "pitch_range": (0.9, 1.1)},
    ],
    "ricochet": [
        {"sound": "ricochet_whine", "category": "combat", "priority": 4, "pitch_range": (0.8, 1.4)},
    ],
    "melee": [
        {"sound": "melee_strike", "category": "combat", "priority": 3, "pitch_range": (0.85, 1.15)},
    ],
    # Vehicles
    "vehicle": [
        {"sound": "engine_rumble", "category": "ambient", "priority": 5, "loop": True, "pitch_range": (0.7, 1.3)},
    ],
    "helicopter": [
        {"sound": "rotor_wash", "category": "ambient", "priority": 3, "loop": True, "pitch_range": (0.9, 1.1)},
    ],
    "vehicle_destroy": [
        {"sound": "vehicle_explosion", "category": "combat", "priority": 1, "pitch_range": (0.8, 1.0)},
        {"sound": "metal_crunch", "category": "combat", "priority": 3, "pitch_range": (0.7, 1.0)},
    ],
    "tire_screech": [
        {"sound": "tire_screech", "category": "ambient", "priority": 5, "pitch_range": (0.8, 1.2)},
    ],
    # Weather
    "rain": [
        {"sound": "rain_loop", "category": "weather", "priority": 7, "loop": True},
    ],
    "rain_heavy": [
        {"sound": "rain_heavy_loop", "category": "weather", "priority": 6, "loop": True},
    ],
    "wind": [
        {"sound": "wind_loop", "category": "weather", "priority": 8, "loop": True},
    ],
    "thunder": [
        {"sound": "thunder_crack", "category": "weather", "priority": 4, "pitch_range": (0.7, 1.0)},
    ],
    # Fire
    "fire": [
        {"sound": "fire_crackle", "category": "ambient", "priority": 5, "loop": True},
    ],
    "fire_large": [
        {"sound": "fire_roar", "category": "ambient", "priority": 4, "loop": True},
    ],
    # Crowd / voice
    "crowd_chant": [
        {"sound": "crowd_noise", "category": "voice", "priority": 5, "loop": True},
    ],
    "crowd_scream": [
        {"sound": "crowd_scream", "category": "voice", "priority": 3},
    ],
    # Comms
    "radio_message": [
        {"sound": "radio_beep", "category": "ui", "priority": 3},
        {"sound": "radio_static", "category": "ui", "priority": 6},
    ],
    "radio_jam": [
        {"sound": "radio_jam_noise", "category": "ui", "priority": 4, "loop": True},
    ],
    # UI
    "alert": [
        {"sound": "alert_tone", "category": "ui", "priority": 2},
    ],
    "notification": [
        {"sound": "notification_ping", "category": "ui", "priority": 5},
    ],
    "target_lock": [
        {"sound": "target_lock_beep", "category": "ui", "priority": 3},
    ],
    # Ambient environment
    "birds": [
        {"sound": "birds_ambient", "category": "ambient", "priority": 9, "loop": True},
    ],
    "insects": [
        {"sound": "insects_ambient", "category": "ambient", "priority": 9, "loop": True},
    ],
    "water": [
        {"sound": "water_flow", "category": "ambient", "priority": 8, "loop": True},
    ],
    # Structures
    "building_collapse": [
        {"sound": "building_collapse_rumble", "category": "combat", "priority": 1, "pitch_range": (0.7, 0.9)},
    ],
    "door_breach": [
        {"sound": "door_breach_bang", "category": "combat", "priority": 3, "pitch_range": (0.9, 1.1)},
    ],
    "glass_break": [
        {"sound": "glass_shatter", "category": "ambient", "priority": 4, "pitch_range": (0.8, 1.3)},
    ],
}


# ---------------------------------------------------------------------------
# SoundtrackEngine
# ---------------------------------------------------------------------------

class SoundtrackEngine:
    """Generates audio event frames for the Three.js frontend.

    Each call to :meth:`tick` examines simulation events and world state,
    producing a dict of audio cues, music state, and ambient loops that the
    frontend Web Audio API renderer consumes.
    """

    def __init__(self) -> None:
        self.cue_queue: list[AudioCue] = []
        self.music_state: MusicState = MusicState()
        self.ambient_sounds: list[AudioCue] = []
        self._active_loops: dict[str, AudioCue] = {}
        self._combat_timer: float = 0.0
        self._victory_cooldown: float = 0.0

    # -- public API --

    def tick(self, events: list[dict[str, Any]], world_state: dict[str, Any]) -> dict[str, Any]:
        """Process one simulation frame and return audio data for the frontend.

        Args:
            events: List of simulation events (each must have ``"type"`` key).
            world_state: Dict describing current world conditions. Recognized
                keys include ``"hostiles_count"``, ``"combat_active"``,
                ``"wave_cleared"``, ``"game_over"``, ``"game_won"``,
                ``"weather"``, ``"wind_speed"``, ``"rain_intensity"``,
                ``"time_of_day"``.

        Returns:
            Dict with ``"cues"``, ``"music"``, ``"ambient"``, and ``"stop"``
            keys ready for JSON serialization to the frontend.
        """
        self.cue_queue.clear()
        stop_ids: list[str] = []

        # Process sim events into audio cues
        new_cues = self.process_events(events, world_state)
        self.cue_queue.extend(new_cues)

        # Update music state from world conditions
        self.update_music(world_state)

        # Update ambient loops from world state
        old_ambient_ids = {c.cue_id for c in self.ambient_sounds}
        self._update_ambient(world_state)
        new_ambient_ids = {c.cue_id for c in self.ambient_sounds}

        # Determine which loops to stop
        for old_id in old_ambient_ids - new_ambient_ids:
            stop_ids.append(old_id)

        # Track loop cues from events (vehicles, fires, etc.)
        new_loop_ids: set[str] = set()
        for cue in self.cue_queue:
            if cue.loop:
                self._active_loops[cue.cue_id] = cue
                new_loop_ids.add(cue.cue_id)

        return {
            "cues": [c.to_dict() for c in self.cue_queue if not c.loop],
            "music": self.music_state.to_dict(),
            "ambient": [c.to_dict() for c in self.ambient_sounds],
            "stop": stop_ids,
        }

    def process_events(self, events: list[dict[str, Any]],
                       world_state: dict[str, Any]) -> list[AudioCue]:
        """Map simulation events to audio cues.

        Each event dict must have a ``"type"`` key. Optional keys:
        - ``"position"`` or ``"pos"``: ``(x, y)`` tuple
        - ``"radius"``: explosion radius (selects large vs small)
        - ``"weapon"``: weapon subtype (rifle, sniper, mg, pistol, shotgun)
        - ``"speed"``: vehicle speed for volume scaling
        - ``"intensity"``: intensity multiplier (rain, fire, crowd)
        - ``"size"``: crowd size for volume scaling

        Returns:
            List of AudioCue instances.
        """
        cues: list[AudioCue] = []

        for event in events:
            event_type = event.get("type", "")
            position = event.get("position") or event.get("pos")
            new_cues = self._event_to_cues(event_type, event, position)
            cues.extend(new_cues)

        return cues

    def update_music(self, world_state: dict[str, Any]) -> None:
        """Update the music state based on current world conditions.

        State machine:
        - No combat, no hostiles: ambient track, intensity 0.0
        - Hostiles detected (not in combat): tension track, rising intensity
        - Active combat: combat track, high intensity, percussion layer
        - Wave cleared: brief victory sting
        - Game over: victory or defeat track
        """
        hostiles = world_state.get("hostiles_count", 0)
        combat_active = world_state.get("combat_active", False)
        wave_cleared = world_state.get("wave_cleared", False)
        game_over = world_state.get("game_over", False)
        game_won = world_state.get("game_won", False)

        # Decrement cooldowns
        if self._victory_cooldown > 0:
            self._victory_cooldown -= 1.0

        # Game over takes highest priority
        if game_over:
            if game_won:
                self.music_state.current_track = "victory"
                self.music_state.intensity = 1.0
                self.music_state.layer_combat = False
                self.music_state.layer_percussion = False
            else:
                self.music_state.current_track = "defeat"
                self.music_state.intensity = 1.0
                self.music_state.layer_combat = False
                self.music_state.layer_percussion = False
            return

        # Wave cleared — brief victory sting
        if wave_cleared and self._victory_cooldown <= 0:
            self.music_state.current_track = "victory"
            self.music_state.intensity = 0.8
            self.music_state.layer_combat = False
            self.music_state.layer_percussion = False
            self._victory_cooldown = 5.0
            return

        # During victory cooldown, keep victory track
        if self._victory_cooldown > 0:
            return

        # Active combat
        if combat_active:
            self._combat_timer = 5.0
            self.music_state.current_track = "combat"
            self.music_state.intensity = min(1.0, 0.6 + hostiles * 0.05)
            self.music_state.layer_combat = True
            self.music_state.layer_percussion = hostiles > 3
            return

        # Combat just ended — fade out
        if self._combat_timer > 0:
            self._combat_timer -= 1.0
            self.music_state.intensity = max(0.0, self.music_state.intensity - 0.15)
            if self.music_state.intensity < 0.2:
                self.music_state.current_track = "tension"
                self.music_state.layer_combat = False
                self.music_state.layer_percussion = False
            return

        # Hostiles detected but no combat
        if hostiles > 0:
            self.music_state.current_track = "tension"
            self.music_state.intensity = min(0.6, 0.2 + hostiles * 0.05)
            self.music_state.layer_combat = False
            self.music_state.layer_percussion = False
            return

        # Peaceful — ambient
        self.music_state.current_track = "ambient"
        self.music_state.intensity = 0.0
        self.music_state.layer_combat = False
        self.music_state.layer_percussion = False

    # -- private helpers --

    def _event_to_cues(self, event_type: str, event: dict[str, Any],
                       position: tuple[float, float] | None) -> list[AudioCue]:
        """Convert a single event into one or more AudioCues."""
        cues: list[AudioCue] = []

        # Handle weapon subtypes for gunfire
        if event_type == "gunfire":
            weapon = event.get("weapon", "rifle")
            lookup_key = f"gunfire_{weapon}"
            if lookup_key not in SOUND_MAP:
                lookup_key = "gunfire_rifle"
            return self._make_cues_from_map(lookup_key, event, position)

        # Handle explosion size
        if event_type == "explosion":
            radius = event.get("radius", 10.0)
            if radius < 5.0:
                return self._make_cues_from_map("explosion_small", event, position)
            return self._make_cues_from_map("explosion", event, position)

        # Handle vehicle speed -> volume
        if event_type == "vehicle":
            speed = event.get("speed", 20.0)
            cues = self._make_cues_from_map("vehicle", event, position)
            for c in cues:
                c.volume = min(1.0, speed / 60.0)
                c.pitch = max(0.5, min(2.0, 0.7 + speed / 100.0))
            return cues

        # Handle rain intensity
        if event_type == "rain":
            intensity = event.get("intensity", 0.5)
            key = "rain_heavy" if intensity > 0.7 else "rain"
            cues = self._make_cues_from_map(key, event, position)
            for c in cues:
                c.volume = min(1.0, max(0.1, intensity))
            return cues

        # Handle wind speed
        if event_type == "wind":
            speed = event.get("speed", 10.0)
            cues = self._make_cues_from_map("wind", event, position)
            for c in cues:
                c.volume = min(1.0, speed / 40.0)
            return cues

        # Handle crowd size
        if event_type == "crowd_chant":
            size = event.get("size", 10)
            cues = self._make_cues_from_map("crowd_chant", event, position)
            for c in cues:
                c.volume = min(1.0, size / 50.0)
            return cues

        # Handle fire intensity
        if event_type == "fire":
            intensity = event.get("intensity", 0.5)
            key = "fire_large" if intensity > 0.7 else "fire"
            cues = self._make_cues_from_map(key, event, position)
            for c in cues:
                c.volume = min(1.0, max(0.2, intensity))
            return cues

        # Generic lookup
        if event_type in SOUND_MAP:
            return self._make_cues_from_map(event_type, event, position)

        return cues

    def _make_cues_from_map(self, key: str, event: dict[str, Any],
                            position: tuple[float, float] | None) -> list[AudioCue]:
        """Create AudioCue instances from SOUND_MAP entries."""
        cues: list[AudioCue] = []
        entries = SOUND_MAP.get(key, [])

        for entry in entries:
            pitch = 1.0
            pitch_range = entry.get("pitch_range")
            if pitch_range:
                import random
                pitch = random.uniform(pitch_range[0], pitch_range[1])

            cue = AudioCue(
                cue_id=f"{entry['sound']}_{uuid.uuid4().hex[:8]}",
                category=AudioCategory(entry["category"]),
                sound_name=entry["sound"],
                position=position,
                volume=event.get("volume", 1.0),
                pitch=pitch,
                loop=entry.get("loop", False),
                priority=entry.get("priority", 5),
            )
            cues.append(cue)

        return cues

    def _update_ambient(self, world_state: dict[str, Any]) -> None:
        """Update persistent ambient sound loops based on world state."""
        new_ambient: list[AudioCue] = []

        # Weather: rain
        rain = world_state.get("rain_intensity", 0.0)
        if rain > 0.05:
            sound = "rain_heavy_loop" if rain > 0.7 else "rain_loop"
            new_ambient.append(AudioCue(
                cue_id=f"ambient_rain",
                category=AudioCategory.WEATHER,
                sound_name=sound,
                volume=min(1.0, rain),
                loop=True,
                priority=7,
            ))

        # Weather: wind
        wind = world_state.get("wind_speed", 0.0)
        if wind > 2.0:
            new_ambient.append(AudioCue(
                cue_id=f"ambient_wind",
                category=AudioCategory.WEATHER,
                sound_name="wind_loop",
                volume=min(1.0, wind / 40.0),
                loop=True,
                priority=8,
            ))

        # Time-based ambient: birds during day, insects at night
        hour = world_state.get("time_of_day", 12.0)
        if 6.0 <= hour <= 18.0:
            new_ambient.append(AudioCue(
                cue_id="ambient_birds",
                category=AudioCategory.AMBIENT,
                sound_name="birds_ambient",
                volume=0.3,
                loop=True,
                priority=9,
            ))
        elif hour >= 20.0 or hour <= 4.0:
            new_ambient.append(AudioCue(
                cue_id="ambient_insects",
                category=AudioCategory.AMBIENT,
                sound_name="insects_ambient",
                volume=0.2,
                loop=True,
                priority=9,
            ))

        self.ambient_sounds = new_ambient
