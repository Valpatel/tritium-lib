# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the sim_engine.commander module — battle narration and tactical advisor.

60+ tests covering NarrationEvent, BattleNarrator, TacticalAdvisor,
CommanderPersonality, and NarrationLog.
"""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.commander import (
    CATEGORIES,
    NarrationEvent,
    CommanderPersonality,
    BattleNarrator,
    TacticalAdvisor,
    NarrationLog,
    PERSONALITIES,
    VOICES,
    _grid_ref,
    _bearing_word,
    _dist_word,
    _unit_label,
)
from tritium_lib.sim_engine.ai.steering import Vec2


# ---------------------------------------------------------------------------
# Helpers — lightweight fake unit
# ---------------------------------------------------------------------------

class FakeUnit:
    def __init__(self, name: str = "Alpha-1", unit_id: str = "u001"):
        self.name = name
        self.unit_id = unit_id


# ---------------------------------------------------------------------------
# NarrationEvent
# ---------------------------------------------------------------------------

class TestNarrationEvent:
    def test_basic_creation(self):
        e = NarrationEvent(tick=1, time=0.5, category="combat", priority=2, text="Bang", voice="radio")
        assert e.tick == 1
        assert e.time == 0.5
        assert e.category == "combat"
        assert e.priority == 2
        assert e.text == "Bang"
        assert e.voice == "radio"

    def test_default_voice(self):
        e = NarrationEvent(tick=0, time=0.0, category="combat", priority=1, text="test")
        assert e.voice == "radio"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid category"):
            NarrationEvent(tick=0, time=0.0, category="nope", priority=1, text="x")

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Invalid priority"):
            NarrationEvent(tick=0, time=0.0, category="combat", priority=9, text="x")

    def test_invalid_voice_raises(self):
        with pytest.raises(ValueError, match="Invalid voice"):
            NarrationEvent(tick=0, time=0.0, category="combat", priority=1, text="x", voice="alien")

    def test_all_categories_valid(self):
        for cat in CATEGORIES:
            e = NarrationEvent(tick=0, time=0.0, category=cat, priority=1, text="ok")
            assert e.category == cat

    def test_all_voices_valid(self):
        for voice in VOICES:
            e = NarrationEvent(tick=0, time=0.0, category="combat", priority=1, text="ok", voice=voice)
            assert e.voice == voice

    def test_all_priorities_valid(self):
        for p in (1, 2, 3, 4):
            e = NarrationEvent(tick=0, time=0.0, category="combat", priority=p, text="ok")
            assert e.priority == p


# ---------------------------------------------------------------------------
# CommanderPersonality
# ---------------------------------------------------------------------------

class TestCommanderPersonality:
    def test_basic_creation(self):
        p = CommanderPersonality(name="Test", callsign="TST", style="professional")
        assert p.name == "Test"
        assert p.callsign == "TST"
        assert p.style == "professional"
        assert p.verbosity == 0.5
        assert p.humor == 0.0

    def test_verbosity_clamped_high(self):
        p = CommanderPersonality(name="X", callsign="X", style="aggressive", verbosity=2.0)
        assert p.verbosity == 1.0

    def test_verbosity_clamped_low(self):
        p = CommanderPersonality(name="X", callsign="X", style="aggressive", verbosity=-1.0)
        assert p.verbosity == 0.0

    def test_humor_clamped(self):
        p = CommanderPersonality(name="X", callsign="X", style="cautious", humor=5.0)
        assert p.humor == 1.0

    def test_preset_iron_hand(self):
        p = PERSONALITIES["iron_hand"]
        assert p.style == "professional"
        assert p.verbosity < 0.5
        assert p.humor == 0.0

    def test_preset_mad_dog(self):
        p = PERSONALITIES["mad_dog"]
        assert p.style == "aggressive"
        assert p.verbosity > 0.5

    def test_preset_ghost(self):
        p = PERSONALITIES["ghost"]
        assert p.style == "cautious"
        assert p.verbosity < 0.2

    def test_preset_showman(self):
        p = PERSONALITIES["showman"]
        assert p.style == "dramatic"
        assert p.verbosity == 1.0
        assert p.humor > 0.3

    def test_all_presets_exist(self):
        assert set(PERSONALITIES.keys()) == {"iron_hand", "mad_dog", "ghost", "showman"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_grid_ref(self):
        ref = _grid_ref((120.5, 85.3))
        assert ref == "120-085"

    def test_grid_ref_negative(self):
        ref = _grid_ref((-50.0, -10.0))
        assert ref == "050-010"

    def test_bearing_word_north(self):
        word = _bearing_word((0.0, 0.0), (0.0, 100.0))
        assert word == "north"

    def test_bearing_word_east(self):
        word = _bearing_word((0.0, 0.0), (100.0, 0.0))
        assert word == "east"

    def test_dist_word_danger_close(self):
        assert _dist_word(10) == "danger close"

    def test_dist_word_close(self):
        assert _dist_word(30) == "close range"

    def test_dist_word_medium(self):
        assert _dist_word(100) == "medium range"

    def test_dist_word_long(self):
        assert _dist_word(300) == "long range"

    def test_dist_word_extreme(self):
        assert _dist_word(500) == "extreme range"

    def test_unit_label_name(self):
        u = FakeUnit(name="Bravo-2")
        assert _unit_label(u) == "Bravo-2"

    def test_unit_label_id_fallback(self):
        class NoName:
            unit_id = "uid_42"
        assert _unit_label(NoName()) == "uid_42"

    def test_unit_label_string_fallback(self):
        assert _unit_label("raw_string") == "raw_string"


# ---------------------------------------------------------------------------
# BattleNarrator
# ---------------------------------------------------------------------------

class TestBattleNarrator:
    def setup_method(self):
        self.narrator = BattleNarrator(
            personality=PERSONALITIES["iron_hand"], tick=10, sim_time=5.0
        )
        self.killer = FakeUnit("Alpha-1")
        self.victim = FakeUnit("Hostile-3", "h003")

    def test_default_personality(self):
        n = BattleNarrator()
        assert n.personality.name == "Iron Hand"

    def test_set_tick(self):
        self.narrator.set_tick(20, 10.0)
        e = self.narrator.narrate_kill(self.killer, self.victim)
        assert e.tick == 20
        assert e.time == 10.0

    def test_narrate_kill_returns_event(self):
        e = self.narrator.narrate_kill(self.killer, self.victim)
        assert isinstance(e, NarrationEvent)
        assert e.category == "combat"
        assert e.priority == 2
        assert "Alpha-1" in e.text or "Hostile-3" in e.text

    def test_narrate_kill_contains_labels(self):
        e = self.narrator.narrate_kill(self.killer, self.victim)
        # At least one of the unit labels should appear
        assert "Alpha-1" in e.text or "Hostile-3" in e.text

    def test_narrate_kill_varied(self):
        """Multiple calls should not always produce the same text."""
        texts = set()
        for _ in range(50):
            e = self.narrator.narrate_kill(self.killer, self.victim)
            texts.add(e.text)
        assert len(texts) > 1, "Narration should vary"

    def test_narrate_engagement_hit(self):
        e = self.narrator.narrate_engagement(self.killer, self.victim, "hit")
        assert e.category == "combat"
        assert e.priority == 2

    def test_narrate_engagement_miss(self):
        e = self.narrator.narrate_engagement(self.killer, self.victim, "miss")
        assert e.priority == 1

    def test_narrate_engagement_suppression(self):
        e = self.narrator.narrate_engagement(self.killer, self.victim, "suppression")
        assert e.priority == 1
        assert isinstance(e.text, str) and len(e.text) > 0

    def test_narrate_engagement_unknown_result(self):
        e = self.narrator.narrate_engagement(self.killer, self.victim, "ricochet")
        assert "ricochet" in e.text

    def test_narrate_explosion_no_casualties(self):
        e = self.narrator.narrate_explosion((100.0, 200.0), 15.0, 0)
        assert e.category == "combat"
        assert e.priority == 2

    def test_narrate_explosion_one_casualty(self):
        e = self.narrator.narrate_explosion((100.0, 200.0), 15.0, 1)
        assert e.priority == 3

    def test_narrate_explosion_mass_casualties(self):
        e = self.narrator.narrate_explosion((100.0, 200.0), 25.0, 5)
        assert e.priority == 4
        assert "5" in e.text

    def test_narrate_movement_basic(self):
        e = self.narrator.narrate_movement(self.killer, (0.0, 0.0), (100.0, 0.0))
        assert e.category == "tactical"
        assert e.priority == 1

    def test_narrate_movement_retreat(self):
        e = self.narrator.narrate_movement(self.killer, (0.0, 0.0), (100.0, 0.0), "retreating")
        assert e.priority == 2

    def test_narrate_movement_advance(self):
        e = self.narrator.narrate_movement(self.killer, (0.0, 0.0), (100.0, 0.0), "advancing")
        assert e.category == "tactical"

    def test_narrate_wave_start(self):
        e = self.narrator.narrate_wave_start(3, 12)
        assert e.category == "combat"
        assert e.priority == 3
        assert e.voice == "commander"
        assert "3" in e.text
        assert "12" in e.text

    def test_narrate_wave_clear(self):
        e = self.narrator.narrate_wave_clear(3, 45.2)
        assert e.category == "combat"
        assert e.priority == 2
        assert e.voice == "commander"
        assert "3" in e.text

    def test_narrate_casualty(self):
        e = self.narrator.narrate_casualty(self.killer, "shrapnel to left leg")
        assert e.category == "medical"
        assert e.priority == 3
        assert e.voice == "medic"
        assert "shrapnel" in e.text.lower()

    def test_narrate_supply_low_moderate(self):
        e = self.narrator.narrate_supply_low("ammo", 0.3)
        assert e.category == "logistics"
        assert e.priority == 2
        assert "30%" in e.text

    def test_narrate_supply_low_critical(self):
        e = self.narrator.narrate_supply_low("fuel", 0.05)
        assert e.priority == 3

    def test_narrate_detection(self):
        e = self.narrator.narrate_detection("BLE", FakeUnit("Unknown-7"))
        assert e.category == "intel"
        assert e.priority == 2
        assert e.voice == "observer"
        assert "BLE" in e.text

    def test_narrate_weather_change(self):
        e = self.narrator.narrate_weather_change("clear", "heavy rain")
        assert e.category == "tactical"
        assert e.priority == 1
        assert "clear" in e.text
        assert "heavy rain" in e.text

    def test_narrate_achievement(self):
        e = self.narrator.narrate_achievement(self.killer, "First Blood")
        assert e.category == "combat"
        assert e.voice == "commander"
        assert "First Blood" in e.text

    def test_narrate_game_over(self):
        stats = {"total_kills": 25, "total_casualties": 3, "duration": 120.5}
        e = self.narrator.narrate_game_over("friendly", stats)
        assert e.category == "combat"
        assert e.priority == 4
        assert e.voice == "commander"
        assert "friendly" in e.text
        assert "25" in e.text

    def test_narrate_game_over_empty_stats(self):
        e = self.narrator.narrate_game_over("hostile", {})
        assert "hostile" in e.text

    # -- personality-specific tests --

    def test_aggressive_personality_kill(self):
        narrator = BattleNarrator(personality=PERSONALITIES["mad_dog"])
        texts = set()
        for _ in range(100):
            e = narrator.narrate_kill(self.killer, self.victim)
            texts.add(e.text)
        # Aggressive style adds extra templates
        assert len(texts) > 2

    def test_dramatic_personality_wave_start(self):
        narrator = BattleNarrator(personality=PERSONALITIES["showman"])
        texts = set()
        for _ in range(100):
            e = narrator.narrate_wave_start(1, 20)
            texts.add(e.text)
        assert len(texts) > 2

    def test_cautious_personality_detection(self):
        narrator = BattleNarrator(personality=PERSONALITIES["ghost"])
        e = narrator.narrate_detection("thermal", FakeUnit("Tango-1"))
        assert isinstance(e, NarrationEvent)


# ---------------------------------------------------------------------------
# TacticalAdvisor
# ---------------------------------------------------------------------------

class TestTacticalAdvisor:
    def setup_method(self):
        self.advisor = TacticalAdvisor(personality=PERSONALITIES["iron_hand"])

    def test_assess_outnumbered(self):
        recs = self.advisor.assess_situation({"friendly_count": 3, "hostile_count": 10})
        joined = " ".join(recs)
        assert "outnumbered" in joined.lower() or "Outnumbered" in joined

    def test_assess_overwhelming(self):
        recs = self.advisor.assess_situation({"friendly_count": 20, "hostile_count": 5})
        joined = " ".join(recs)
        assert "advantage" in joined.lower() or "press" in joined.lower()

    def test_assess_heavily_outnumbered(self):
        recs = self.advisor.assess_situation({"friendly_count": 2, "hostile_count": 10})
        joined = " ".join(recs)
        assert "CRITICAL" in joined

    def test_assess_high_casualties(self):
        recs = self.advisor.assess_situation({
            "friendly_count": 3,
            "hostile_count": 5,
            "friendly_casualties": 7,
        })
        joined = " ".join(recs)
        assert "casualties" in joined.lower() or "casualt" in joined.lower()

    def test_assess_low_ammo(self):
        recs = self.advisor.assess_situation({
            "friendly_count": 5,
            "hostile_count": 5,
            "ammo_level": 0.1,
        })
        joined = " ".join(recs)
        assert "ammo" in joined.lower() or "ammunition" in joined.lower()

    def test_assess_low_visibility(self):
        recs = self.advisor.assess_situation({
            "friendly_count": 5,
            "hostile_count": 5,
            "visibility": 0.2,
        })
        joined = " ".join(recs)
        assert "visibility" in joined.lower()

    def test_assess_area_clear(self):
        recs = self.advisor.assess_situation({"friendly_count": 5, "hostile_count": 0})
        joined = " ".join(recs)
        assert "clear" in joined.lower() or "secure" in joined.lower()

    def test_assess_empty_state(self):
        recs = self.advisor.assess_situation({})
        assert isinstance(recs, list)

    def test_recommend_action_low_health(self):
        rec = self.advisor.recommend_action("u1", {"health": 10})
        assert "medic" in rec.lower() or "fall back" in rec.lower()

    def test_recommend_action_suppressed_near_cover(self):
        rec = self.advisor.recommend_action("u1", {
            "suppressed": True, "in_cover": False, "nearest_cover_dist": 10.0
        })
        assert "cover" in rec.lower() or "sprint" in rec.lower()

    def test_recommend_action_suppressed_no_cover(self):
        rec = self.advisor.recommend_action("u1", {
            "suppressed": True, "in_cover": False, "nearest_cover_dist": 100.0
        })
        assert "stay low" in rec.lower() or "wait" in rec.lower()

    def test_recommend_action_no_ammo(self):
        rec = self.advisor.recommend_action("u1", {"ammo": 0})
        assert "winchester" in rec.lower() or "resupply" in rec.lower()

    def test_recommend_action_no_enemies(self):
        rec = self.advisor.recommend_action("u1", {"enemies_visible": 0})
        assert "hold" in rec.lower() or "scan" in rec.lower()

    def test_recommend_action_default_engage(self):
        rec = self.advisor.recommend_action("u1", {"enemies_visible": 1, "in_cover": True})
        assert "engage" in rec.lower()

    def test_recommend_action_cautious_multiple(self):
        advisor = TacticalAdvisor(personality=PERSONALITIES["ghost"])
        rec = advisor.recommend_action("u1", {"enemies_visible": 5, "in_cover": True})
        assert "hold" in rec.lower() or "support" in rec.lower() or "wait" in rec.lower()

    def test_threat_warning_empty(self):
        result = self.advisor.threat_warning([])
        assert "no active threats" in result.lower()

    def test_threat_warning_single(self):
        result = self.advisor.threat_warning([
            {"type": "infantry", "bearing": "north", "distance": 100}
        ])
        assert "infantry" in result
        assert "north" in result

    def test_threat_warning_multiple(self):
        result = self.advisor.threat_warning([
            {"type": "infantry", "bearing": "north", "distance": 100, "count": 3},
            {"type": "vehicle", "bearing": "east", "distance": 500},
        ])
        assert "3x" in result
        assert "vehicle" in result
        assert "IRON" in result  # callsign

    def test_sitrep_basic(self):
        report = self.advisor.sitrep({
            "friendly_count": 8,
            "hostile_count": 5,
            "friendly_casualties": 2,
            "current_wave": 3,
            "elapsed_time": 60.0,
            "ammo_level": 0.75,
        })
        assert "SITREP" in report
        assert "8" in report
        assert "5" in report
        assert "75%" in report

    def test_sitrep_empty_state(self):
        report = self.advisor.sitrep({})
        assert "SITREP" in report

    def test_sitrep_area_secure(self):
        report = self.advisor.sitrep({"friendly_count": 5, "hostile_count": 0})
        assert "secure" in report.lower() or "no enemy" in report.lower()

    def test_battle_summary(self):
        aar = {
            "winner": "friendly",
            "duration": 180.0,
            "waves_completed": 5,
            "total_kills": 30,
            "total_casualties": 2,
            "accuracy": 0.65,
        }
        report = self.advisor.battle_summary(aar)
        assert "AFTER-ACTION REPORT" in report
        assert "friendly" in report
        assert "30" in report
        assert "65.0%" in report

    def test_battle_summary_flawless(self):
        aar = {"winner": "friendly", "duration": 60.0, "waves_completed": 1,
               "total_kills": 10, "total_casualties": 0}
        report = self.advisor.battle_summary(aar)
        assert "flawless" in report.lower() or "no friendly" in report.lower()

    def test_battle_summary_pyrrhic(self):
        aar = {"winner": "friendly", "duration": 300.0, "waves_completed": 5,
               "total_kills": 10, "total_casualties": 8}
        report = self.advisor.battle_summary(aar)
        assert "pyrrhic" in report.lower() or "unacceptable" in report.lower()

    def test_battle_summary_empty(self):
        report = self.advisor.battle_summary({})
        assert "AFTER-ACTION REPORT" in report


# ---------------------------------------------------------------------------
# NarrationLog
# ---------------------------------------------------------------------------

class TestNarrationLog:
    def _make_event(self, tick: int = 0, category: str = "combat",
                    priority: int = 1, text: str = "test") -> NarrationEvent:
        return NarrationEvent(tick=tick, time=float(tick), category=category,
                              priority=priority, text=text)

    def test_empty_log(self):
        log = NarrationLog()
        assert len(log) == 0
        assert log.events == []

    def test_add_event(self):
        log = NarrationLog()
        log.add(self._make_event())
        assert len(log) == 1

    def test_events_returns_copy(self):
        log = NarrationLog()
        log.add(self._make_event())
        events = log.events
        events.clear()
        assert len(log) == 1

    def test_max_events_cap(self):
        log = NarrationLog(max_events=5)
        for i in range(10):
            log.add(self._make_event(tick=i, text=f"event_{i}"))
        assert len(log) == 5
        assert log.events[0].text == "event_5"

    def test_filter_by_priority(self):
        log = NarrationLog()
        log.add(self._make_event(priority=1, text="low"))
        log.add(self._make_event(priority=3, text="high"))
        log.add(self._make_event(priority=4, text="crit"))
        result = log.filter_by_priority(3)
        assert len(result) == 2
        assert all(e.priority >= 3 for e in result)

    def test_filter_by_category(self):
        log = NarrationLog()
        log.add(self._make_event(category="combat", text="fight"))
        log.add(self._make_event(category="medical", text="heal"))
        log.add(self._make_event(category="combat", text="boom"))
        result = log.filter_by_category("combat")
        assert len(result) == 2

    def test_recent(self):
        log = NarrationLog()
        for i in range(20):
            log.add(self._make_event(tick=i))
        result = log.recent(5)
        assert len(result) == 5
        assert result[0].tick == 15

    def test_recent_more_than_available(self):
        log = NarrationLog()
        log.add(self._make_event())
        result = log.recent(10)
        assert len(result) == 1

    def test_clear(self):
        log = NarrationLog()
        log.add(self._make_event())
        log.add(self._make_event())
        log.clear()
        assert len(log) == 0

    def test_to_three_js_structure(self):
        log = NarrationLog()
        result = log.to_three_js()
        assert "kill_feed" in result
        assert "alerts" in result
        assert "sitrep" in result

    def test_to_three_js_kill_feed(self):
        log = NarrationLog()
        for i in range(8):
            log.add(self._make_event(tick=i, category="combat", text=f"kill_{i}"))
        result = log.to_three_js()
        assert len(result["kill_feed"]) == 5  # max 5
        assert result["kill_feed"][-1]["text"] == "kill_7"

    def test_to_three_js_alerts(self):
        log = NarrationLog()
        log.add(self._make_event(priority=1, text="boring"))
        log.add(self._make_event(priority=3, text="urgent"))
        log.add(self._make_event(priority=4, text="critical"))
        result = log.to_three_js()
        assert len(result["alerts"]) == 2
        texts = [a["text"] for a in result["alerts"]]
        assert "urgent" in texts
        assert "critical" in texts

    def test_to_three_js_sitrep(self):
        log = NarrationLog()
        log.add(self._make_event(category="tactical", text="sitrep text here"))
        log.add(self._make_event(category="combat", text="boom"))
        result = log.to_three_js()
        assert result["sitrep"] == "sitrep text here"

    def test_to_three_js_sitrep_empty(self):
        log = NarrationLog()
        log.add(self._make_event(category="combat", text="only combat"))
        result = log.to_three_js()
        assert result["sitrep"] == ""

    def test_to_three_js_intel_as_sitrep(self):
        log = NarrationLog()
        log.add(self._make_event(category="intel", text="intel update"))
        result = log.to_three_js()
        assert result["sitrep"] == "intel update"


# ---------------------------------------------------------------------------
# Integration: narrator + log
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_narrator_to_log_flow(self):
        narrator = BattleNarrator(personality=PERSONALITIES["mad_dog"], tick=1, sim_time=0.5)
        log = NarrationLog()
        killer = FakeUnit("Alpha-1")
        victim = FakeUnit("Hostile-1")

        log.add(narrator.narrate_wave_start(1, 10))
        log.add(narrator.narrate_engagement(killer, victim, "hit"))
        log.add(narrator.narrate_kill(killer, victim))
        log.add(narrator.narrate_wave_clear(1, 30.0))

        assert len(log) == 4
        hud = log.to_three_js()
        assert len(hud["kill_feed"]) == 4  # all are combat
        assert len(hud["alerts"]) >= 1  # wave_start is priority 3

    def test_full_battle_scenario(self):
        narrator = BattleNarrator(personality=PERSONALITIES["showman"], tick=0, sim_time=0.0)
        advisor = TacticalAdvisor(personality=PERSONALITIES["showman"])
        log = NarrationLog()

        # Wave start
        log.add(narrator.narrate_wave_start(1, 8))
        narrator.set_tick(5, 2.5)

        # Detection
        target = FakeUnit("Hostile-1")
        log.add(narrator.narrate_detection("radar", target))

        # Engagement
        friendly = FakeUnit("Alpha-1")
        log.add(narrator.narrate_engagement(friendly, target, "hit"))
        narrator.set_tick(10, 5.0)

        # Kill
        log.add(narrator.narrate_kill(friendly, target))

        # Casualty
        log.add(narrator.narrate_casualty(FakeUnit("Alpha-2"), "gunshot wound to arm"))
        narrator.set_tick(15, 7.5)

        # Supply
        log.add(narrator.narrate_supply_low("ammo", 0.15))

        # Weather
        log.add(narrator.narrate_weather_change("clear", "fog"))

        # Wave clear
        log.add(narrator.narrate_wave_clear(1, 7.5))
        narrator.set_tick(20, 10.0)

        # Achievement
        log.add(narrator.narrate_achievement(friendly, "Sharpshooter"))

        # Game over
        stats = {"total_kills": 8, "total_casualties": 1, "duration": 10.0}
        log.add(narrator.narrate_game_over("friendly", stats))

        assert len(log) == 10

        # Advisor
        recs = advisor.assess_situation({
            "friendly_count": 4, "hostile_count": 0,
            "friendly_casualties": 1, "ammo_level": 0.15,
        })
        assert len(recs) > 0

        sitrep = advisor.sitrep({
            "friendly_count": 4, "hostile_count": 0,
            "friendly_casualties": 1, "current_wave": 1,
            "elapsed_time": 10.0, "ammo_level": 0.15,
        })
        assert "SITREP" in sitrep

        summary = advisor.battle_summary({
            "winner": "friendly", "duration": 10.0,
            "waves_completed": 1, "total_kills": 8,
            "total_casualties": 1, "accuracy": 0.8,
        })
        assert "AFTER-ACTION REPORT" in summary

        # Three.js export
        hud = log.to_three_js()
        assert len(hud["kill_feed"]) > 0
        assert isinstance(hud["alerts"], list)
        assert isinstance(hud["sitrep"], str)
