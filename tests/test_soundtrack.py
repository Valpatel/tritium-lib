# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.soundtrack — game audio event system."""

import pytest

from tritium_lib.sim_engine.soundtrack import (
    AudioCategory,
    AudioCue,
    MusicState,
    SoundtrackEngine,
    SOUND_MAP,
)


# ---------------------------------------------------------------------------
# AudioCategory
# ---------------------------------------------------------------------------

class TestAudioCategory:
    def test_all_categories_exist(self):
        expected = {"COMBAT", "AMBIENT", "UI", "VOICE", "MUSIC", "WEATHER"}
        assert {c.name for c in AudioCategory} == expected

    def test_values(self):
        assert AudioCategory.COMBAT.value == "combat"
        assert AudioCategory.WEATHER.value == "weather"
        assert AudioCategory.UI.value == "ui"


# ---------------------------------------------------------------------------
# AudioCue
# ---------------------------------------------------------------------------

class TestAudioCue:
    def test_basic_construction(self):
        cue = AudioCue(
            cue_id="test_1",
            category=AudioCategory.COMBAT,
            sound_name="rifle_burst",
        )
        assert cue.cue_id == "test_1"
        assert cue.category == AudioCategory.COMBAT
        assert cue.sound_name == "rifle_burst"
        assert cue.position is None
        assert cue.volume == 1.0
        assert cue.pitch == 1.0
        assert cue.loop is False
        assert cue.priority == 5

    def test_with_position(self):
        cue = AudioCue(
            cue_id="pos_1",
            category=AudioCategory.AMBIENT,
            sound_name="engine_rumble",
            position=(50.0, 30.0),
        )
        assert cue.position == (50.0, 30.0)

    def test_volume_clamped_high(self):
        cue = AudioCue(cue_id="v", category=AudioCategory.UI, sound_name="s", volume=2.5)
        assert cue.volume == 1.0

    def test_volume_clamped_low(self):
        cue = AudioCue(cue_id="v", category=AudioCategory.UI, sound_name="s", volume=-0.5)
        assert cue.volume == 0.0

    def test_pitch_clamped_high(self):
        cue = AudioCue(cue_id="p", category=AudioCategory.UI, sound_name="s", pitch=5.0)
        assert cue.pitch == 2.0

    def test_pitch_clamped_low(self):
        cue = AudioCue(cue_id="p", category=AudioCategory.UI, sound_name="s", pitch=0.1)
        assert cue.pitch == 0.5

    def test_priority_clamped_high(self):
        cue = AudioCue(cue_id="p", category=AudioCategory.UI, sound_name="s", priority=20)
        assert cue.priority == 10

    def test_priority_clamped_low(self):
        cue = AudioCue(cue_id="p", category=AudioCategory.UI, sound_name="s", priority=0)
        assert cue.priority == 1

    def test_to_dict_no_position(self):
        cue = AudioCue(cue_id="d1", category=AudioCategory.COMBAT, sound_name="bang")
        d = cue.to_dict()
        assert d["id"] == "d1"
        assert d["sound"] == "bang"
        assert d["vol"] == 1.0
        assert d["pitch"] == 1.0
        assert d["loop"] is False
        assert d["category"] == "combat"
        assert "x" not in d
        assert "y" not in d

    def test_to_dict_with_position(self):
        cue = AudioCue(
            cue_id="d2", category=AudioCategory.WEATHER, sound_name="rain",
            position=(10.5, 20.3), volume=0.7,
        )
        d = cue.to_dict()
        assert d["x"] == 10.5
        assert d["y"] == 20.3
        assert d["vol"] == 0.7
        assert d["category"] == "weather"

    def test_loop_cue(self):
        cue = AudioCue(
            cue_id="loop1", category=AudioCategory.AMBIENT,
            sound_name="fire_crackle", loop=True,
        )
        assert cue.loop is True
        assert cue.to_dict()["loop"] is True


# ---------------------------------------------------------------------------
# MusicState
# ---------------------------------------------------------------------------

class TestMusicState:
    def test_defaults(self):
        ms = MusicState()
        assert ms.current_track == "ambient"
        assert ms.intensity == 0.0
        assert ms.layer_combat is False
        assert ms.layer_percussion is False

    def test_custom_values(self):
        ms = MusicState(current_track="combat", intensity=0.8,
                        layer_combat=True, layer_percussion=True)
        assert ms.current_track == "combat"
        assert ms.intensity == 0.8

    def test_intensity_clamped_high(self):
        ms = MusicState(intensity=5.0)
        assert ms.intensity == 1.0

    def test_intensity_clamped_low(self):
        ms = MusicState(intensity=-1.0)
        assert ms.intensity == 0.0

    def test_to_dict(self):
        ms = MusicState(current_track="tension", intensity=0.5,
                        layer_combat=True, layer_percussion=False)
        d = ms.to_dict()
        assert d == {
            "track": "tension",
            "intensity": 0.5,
            "combat": True,
            "percussion": False,
        }


# ---------------------------------------------------------------------------
# SOUND_MAP
# ---------------------------------------------------------------------------

class TestSoundMap:
    def test_at_least_30_entries(self):
        assert len(SOUND_MAP) >= 30

    def test_all_entries_have_sound_and_category(self):
        for key, entries in SOUND_MAP.items():
            assert isinstance(entries, list), f"{key} is not a list"
            for entry in entries:
                assert "sound" in entry, f"{key} missing 'sound'"
                assert "category" in entry, f"{key} missing 'category'"

    def test_kill_produces_two_sounds(self):
        assert len(SOUND_MAP["kill"]) == 2
        sounds = {e["sound"] for e in SOUND_MAP["kill"]}
        assert "bullet_impact" in sounds
        assert "body_fall" in sounds

    def test_vehicle_is_looping(self):
        for entry in SOUND_MAP["vehicle"]:
            assert entry.get("loop") is True

    def test_radio_message_sounds(self):
        sounds = {e["sound"] for e in SOUND_MAP["radio_message"]}
        assert "radio_beep" in sounds
        assert "radio_static" in sounds

    def test_weather_sounds_loop(self):
        for key in ("rain", "rain_heavy", "wind"):
            for entry in SOUND_MAP[key]:
                assert entry.get("loop") is True, f"{key} should loop"

    def test_categories_are_valid(self):
        valid = {c.value for c in AudioCategory}
        for key, entries in SOUND_MAP.items():
            for entry in entries:
                assert entry["category"] in valid, f"{key} has invalid category"


# ---------------------------------------------------------------------------
# SoundtrackEngine — process_events
# ---------------------------------------------------------------------------

class TestProcessEvents:
    def setup_method(self):
        self.engine = SoundtrackEngine()

    def test_empty_events(self):
        cues = self.engine.process_events([], {})
        assert cues == []

    def test_kill_event(self):
        events = [{"type": "kill", "position": (10, 20)}]
        cues = self.engine.process_events(events, {})
        assert len(cues) == 2
        sounds = {c.sound_name for c in cues}
        assert "bullet_impact" in sounds
        assert "body_fall" in sounds
        for c in cues:
            assert c.position == (10, 20)

    def test_gunfire_rifle(self):
        events = [{"type": "gunfire", "weapon": "rifle"}]
        cues = self.engine.process_events(events, {})
        assert len(cues) == 1
        assert cues[0].sound_name == "rifle_burst"

    def test_gunfire_sniper(self):
        events = [{"type": "gunfire", "weapon": "sniper"}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "sniper_crack"

    def test_gunfire_mg(self):
        events = [{"type": "gunfire", "weapon": "mg"}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "mg_sustained"

    def test_gunfire_unknown_weapon_defaults_to_rifle(self):
        events = [{"type": "gunfire", "weapon": "blaster"}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "rifle_burst"

    def test_explosion_large(self):
        events = [{"type": "explosion", "radius": 15.0, "position": (5, 5)}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "explosion_large"

    def test_explosion_small(self):
        events = [{"type": "explosion", "radius": 3.0}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "explosion_small"

    def test_vehicle_speed_volume(self):
        events = [{"type": "vehicle", "speed": 60.0}]
        cues = self.engine.process_events(events, {})
        assert len(cues) == 1
        assert cues[0].sound_name == "engine_rumble"
        assert cues[0].loop is True
        assert cues[0].volume == pytest.approx(1.0, abs=0.01)

    def test_vehicle_slow_low_volume(self):
        events = [{"type": "vehicle", "speed": 10.0}]
        cues = self.engine.process_events(events, {})
        assert cues[0].volume < 0.3

    def test_helicopter_loops(self):
        events = [{"type": "helicopter"}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "rotor_wash"
        assert cues[0].loop is True

    def test_rain_light(self):
        events = [{"type": "rain", "intensity": 0.3}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "rain_loop"
        assert cues[0].volume == pytest.approx(0.3, abs=0.01)

    def test_rain_heavy(self):
        events = [{"type": "rain", "intensity": 0.9}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "rain_heavy_loop"

    def test_wind_volume_scales_with_speed(self):
        events = [{"type": "wind", "speed": 20.0}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "wind_loop"
        assert cues[0].volume == pytest.approx(0.5, abs=0.01)

    def test_crowd_chant_volume(self):
        events = [{"type": "crowd_chant", "size": 25}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "crowd_noise"
        assert cues[0].volume == pytest.approx(0.5, abs=0.01)

    def test_fire_normal(self):
        events = [{"type": "fire", "intensity": 0.4}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "fire_crackle"

    def test_fire_large(self):
        events = [{"type": "fire", "intensity": 0.9}]
        cues = self.engine.process_events(events, {})
        assert cues[0].sound_name == "fire_roar"

    def test_radio_message(self):
        events = [{"type": "radio_message"}]
        cues = self.engine.process_events(events, {})
        sounds = {c.sound_name for c in cues}
        assert "radio_beep" in sounds
        assert "radio_static" in sounds

    def test_unknown_event_type_ignored(self):
        events = [{"type": "nonexistent_thing"}]
        cues = self.engine.process_events(events, {})
        assert cues == []

    def test_multiple_events(self):
        events = [
            {"type": "gunfire", "weapon": "rifle", "position": (1, 2)},
            {"type": "kill", "position": (3, 4)},
            {"type": "alert"},
        ]
        cues = self.engine.process_events(events, {})
        # 1 rifle + 2 kill + 1 alert = 4
        assert len(cues) == 4

    def test_position_from_pos_key(self):
        events = [{"type": "hit", "pos": (7, 8)}]
        cues = self.engine.process_events(events, {})
        assert cues[0].position == (7, 8)

    def test_cue_ids_are_unique(self):
        events = [{"type": "gunfire", "weapon": "rifle"} for _ in range(5)]
        cues = self.engine.process_events(events, {})
        ids = [c.cue_id for c in cues]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# SoundtrackEngine — update_music
# ---------------------------------------------------------------------------

class TestUpdateMusic:
    def setup_method(self):
        self.engine = SoundtrackEngine()

    def test_peaceful_ambient(self):
        self.engine.update_music({"hostiles_count": 0, "combat_active": False})
        assert self.engine.music_state.current_track == "ambient"
        assert self.engine.music_state.intensity == 0.0
        assert self.engine.music_state.layer_combat is False

    def test_hostiles_detected_tension(self):
        self.engine.update_music({"hostiles_count": 5, "combat_active": False})
        assert self.engine.music_state.current_track == "tension"
        assert self.engine.music_state.intensity > 0.0
        assert self.engine.music_state.layer_combat is False

    def test_active_combat(self):
        self.engine.update_music({"hostiles_count": 3, "combat_active": True})
        assert self.engine.music_state.current_track == "combat"
        assert self.engine.music_state.intensity >= 0.6
        assert self.engine.music_state.layer_combat is True

    def test_heavy_combat_percussion(self):
        self.engine.update_music({"hostiles_count": 10, "combat_active": True})
        assert self.engine.music_state.layer_percussion is True

    def test_light_combat_no_percussion(self):
        self.engine.update_music({"hostiles_count": 2, "combat_active": True})
        assert self.engine.music_state.layer_percussion is False

    def test_wave_cleared_victory(self):
        self.engine.update_music({"wave_cleared": True})
        assert self.engine.music_state.current_track == "victory"
        assert self.engine.music_state.intensity > 0.0

    def test_game_over_won(self):
        self.engine.update_music({"game_over": True, "game_won": True})
        assert self.engine.music_state.current_track == "victory"
        assert self.engine.music_state.intensity == 1.0

    def test_game_over_lost(self):
        self.engine.update_music({"game_over": True, "game_won": False})
        assert self.engine.music_state.current_track == "defeat"

    def test_combat_fade_out(self):
        # Enter combat
        self.engine.update_music({"hostiles_count": 5, "combat_active": True})
        assert self.engine.music_state.current_track == "combat"
        # End combat — should start fading
        self.engine.update_music({"hostiles_count": 0, "combat_active": False})
        # Combat timer still active, intensity decreasing
        assert self.engine.music_state.intensity < 1.0


# ---------------------------------------------------------------------------
# SoundtrackEngine — tick
# ---------------------------------------------------------------------------

class TestTick:
    def setup_method(self):
        self.engine = SoundtrackEngine()

    def test_tick_returns_all_keys(self):
        frame = self.engine.tick([], {})
        assert "cues" in frame
        assert "music" in frame
        assert "ambient" in frame
        assert "stop" in frame

    def test_tick_cues_are_dicts(self):
        events = [{"type": "alert"}]
        frame = self.engine.tick(events, {})
        for cue in frame["cues"]:
            assert isinstance(cue, dict)
            assert "sound" in cue

    def test_tick_music_is_dict(self):
        frame = self.engine.tick([], {"hostiles_count": 3, "combat_active": True})
        assert isinstance(frame["music"], dict)
        assert frame["music"]["track"] == "combat"

    def test_tick_ambient_rain(self):
        frame = self.engine.tick([], {"rain_intensity": 0.5})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "rain_loop" in ambient_sounds

    def test_tick_ambient_heavy_rain(self):
        frame = self.engine.tick([], {"rain_intensity": 0.9})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "rain_heavy_loop" in ambient_sounds

    def test_tick_ambient_wind(self):
        frame = self.engine.tick([], {"wind_speed": 15.0})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "wind_loop" in ambient_sounds

    def test_tick_ambient_birds_daytime(self):
        frame = self.engine.tick([], {"time_of_day": 12.0})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "birds_ambient" in ambient_sounds

    def test_tick_ambient_insects_nighttime(self):
        frame = self.engine.tick([], {"time_of_day": 23.0})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "insects_ambient" in ambient_sounds

    def test_tick_no_birds_at_night(self):
        frame = self.engine.tick([], {"time_of_day": 23.0})
        ambient_sounds = {a["sound"] for a in frame["ambient"]}
        assert "birds_ambient" not in ambient_sounds

    def test_tick_stop_list_on_ambient_change(self):
        # First tick with rain
        self.engine.tick([], {"rain_intensity": 0.5})
        # Second tick without rain — should signal stop
        frame = self.engine.tick([], {"rain_intensity": 0.0})
        assert "ambient_rain" in frame["stop"]

    def test_tick_full_combat_scenario(self):
        events = [
            {"type": "gunfire", "weapon": "rifle", "position": (10, 20)},
            {"type": "explosion", "radius": 12.0, "position": (15, 25)},
        ]
        world = {"hostiles_count": 5, "combat_active": True, "rain_intensity": 0.3}
        frame = self.engine.tick(events, world)
        # Should have combat cues
        assert len(frame["cues"]) >= 2
        # Music should be combat
        assert frame["music"]["track"] == "combat"
        assert frame["music"]["combat"] is True
        # Should have rain ambient
        assert any(a["sound"] == "rain_loop" for a in frame["ambient"])

    def test_tick_loop_cues_not_in_oneshot_list(self):
        events = [{"type": "vehicle", "speed": 30.0}]
        frame = self.engine.tick(events, {})
        # Loop cues should NOT appear in the one-shot "cues" list
        for cue in frame["cues"]:
            assert cue.get("loop") is not True

    def test_tick_empty_state(self):
        frame = self.engine.tick([], {})
        assert frame["cues"] == []
        assert frame["music"]["track"] == "ambient"
        assert frame["stop"] == []
