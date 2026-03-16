"""Tests for the sim engine scoring and after-action review system.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.scoring import (
    ACHIEVEMENTS,
    Achievement,
    ScoreCategory,
    ScoringEngine,
    TeamScorecard,
    UnitScorecard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ScoringEngine:
    """Fresh ScoringEngine with default achievements."""
    return ScoringEngine()


@pytest.fixture
def populated_engine() -> ScoringEngine:
    """Engine with two teams of 3 units each, some combat recorded."""
    eng = ScoringEngine()
    # Friendly team
    eng.register_unit("f1", "Alpha-1", "friendly")
    eng.register_unit("f2", "Alpha-2", "friendly")
    eng.register_unit("f3", "Alpha-3", "friendly")
    # Hostile team
    eng.register_unit("h1", "Bravo-1", "hostile")
    eng.register_unit("h2", "Bravo-2", "hostile")
    eng.register_unit("h3", "Bravo-3", "hostile")
    return eng


# ---------------------------------------------------------------------------
# ScoreCategory enum
# ---------------------------------------------------------------------------


class TestScoreCategory:
    def test_all_categories_present(self):
        names = {c.name for c in ScoreCategory}
        expected = {"KILLS", "ASSISTS", "OBJECTIVES", "SURVIVAL", "ACCURACY",
                    "TEAMWORK", "TACTICAL", "ECONOMY"}
        assert names == expected

    def test_values_are_lowercase(self):
        for c in ScoreCategory:
            assert c.value == c.name.lower()


# ---------------------------------------------------------------------------
# Achievement dataclass
# ---------------------------------------------------------------------------


class TestAchievement:
    def test_creation(self):
        a = Achievement("test", "Test", "A test", ScoreCategory.KILLS, 1.0, 100, "X")
        assert a.achievement_id == "test"
        assert a.points == 100
        assert a.category == ScoreCategory.KILLS

    def test_default_achievements_count(self):
        assert len(ACHIEVEMENTS) >= 20

    def test_default_achievements_unique_ids(self):
        ids = [a.achievement_id for a in ACHIEVEMENTS]
        assert len(ids) == len(set(ids)), "Duplicate achievement IDs found"

    def test_all_achievements_have_points(self):
        for a in ACHIEVEMENTS:
            assert a.points > 0, f"{a.achievement_id} has no points"

    def test_all_achievements_have_icon(self):
        for a in ACHIEVEMENTS:
            assert len(a.icon) > 0, f"{a.achievement_id} has no icon"


# ---------------------------------------------------------------------------
# UnitScorecard
# ---------------------------------------------------------------------------


class TestUnitScorecard:
    def test_defaults(self):
        card = UnitScorecard(unit_id="u1", name="Unit1", alliance="friendly")
        assert card.kills == 0
        assert card.deaths == 0
        assert card.damage_dealt == 0.0
        assert card.achievements == []
        assert card.supplies_used == {}

    def test_kd_ratio_zero_deaths(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f", kills=5)
        assert card.kd_ratio == 5.0

    def test_kd_ratio_normal(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f", kills=10, deaths=2)
        assert card.kd_ratio == 5.0

    def test_kd_ratio_no_kills(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f", kills=0, deaths=3)
        assert card.kd_ratio == 0.0

    def test_accuracy_no_shots(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f")
        assert card.accuracy == 0.0

    def test_accuracy_all_hit(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f",
                             shots_fired=10, shots_hit=10)
        assert card.accuracy == 1.0

    def test_accuracy_partial(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f",
                             shots_fired=20, shots_hit=15)
        assert card.accuracy == 0.75

    def test_score_basic(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f",
                             kills=5, assists=2, objectives_completed=1)
        # 5*100 + 2*50 + 1*200 = 800
        assert card.score >= 800

    def test_score_includes_accuracy_bonus(self):
        card = UnitScorecard(unit_id="u1", name="U", alliance="f",
                             shots_fired=20, shots_hit=18)
        score_with_accuracy = card.score
        card2 = UnitScorecard(unit_id="u2", name="U2", alliance="f",
                              shots_fired=0, shots_hit=0)
        assert score_with_accuracy > card2.score

    def test_to_dict(self):
        card = UnitScorecard(unit_id="u1", name="Unit1", alliance="friendly",
                             kills=3, deaths=1)
        d = card.to_dict()
        assert d["unit_id"] == "u1"
        assert d["kills"] == 3
        assert d["kd_ratio"] == 3.0
        assert "score" in d
        assert "_last_pos" not in d
        assert "_kill_times" not in d

    def test_supplies_used_independent(self):
        """Ensure default dict is not shared between instances."""
        c1 = UnitScorecard(unit_id="u1", name="A", alliance="f")
        c2 = UnitScorecard(unit_id="u2", name="B", alliance="f")
        c1.supplies_used["ammo"] = 10.0
        assert "ammo" not in c2.supplies_used


# ---------------------------------------------------------------------------
# TeamScorecard
# ---------------------------------------------------------------------------


class TestTeamScorecard:
    def test_mvp_empty(self):
        t = TeamScorecard(alliance="friendly")
        assert t.mvp is None

    def test_mvp_picks_highest_score(self):
        c1 = UnitScorecard(unit_id="u1", name="A", alliance="f", kills=5)
        c2 = UnitScorecard(unit_id="u2", name="B", alliance="f", kills=10)
        t = TeamScorecard(alliance="f", unit_scores=[c1, c2])
        assert t.mvp is c2

    def test_is_victorious(self):
        t = TeamScorecard(alliance="f", territory_controlled=0.6)
        assert t.is_victorious is True
        t2 = TeamScorecard(alliance="h", territory_controlled=0.3)
        assert t2.is_victorious is False

    def test_to_dict(self):
        c1 = UnitScorecard(unit_id="u1", name="A", alliance="f", kills=5)
        t = TeamScorecard(alliance="f", total_kills=5, unit_scores=[c1])
        d = t.to_dict()
        assert d["alliance"] == "f"
        assert d["total_kills"] == 5
        assert d["mvp"] == "A"
        assert d["unit_count"] == 1


# ---------------------------------------------------------------------------
# ScoringEngine — registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_unit(self, engine: ScoringEngine):
        card = engine.register_unit("u1", "Alpha", "friendly")
        assert card.unit_id == "u1"
        assert "u1" in engine.unit_scores
        assert "friendly" in engine.team_scores

    def test_register_creates_team(self, engine: ScoringEngine):
        engine.register_unit("u1", "A", "red")
        engine.register_unit("u2", "B", "blue")
        assert len(engine.team_scores) == 2

    def test_register_same_team(self, engine: ScoringEngine):
        engine.register_unit("u1", "A", "red")
        engine.register_unit("u2", "B", "red")
        assert len(engine.team_scores) == 1
        assert len(engine.team_scores["red"].unit_scores) == 2


# ---------------------------------------------------------------------------
# ScoringEngine — kill recording
# ---------------------------------------------------------------------------


class TestRecordKill:
    def test_kill_increments(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        assert eng.unit_scores["f1"].kills == 1
        assert eng.unit_scores["h1"].deaths == 1

    def test_kill_updates_team(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        assert eng.team_scores["friendly"].total_kills == 1
        assert eng.team_scores["hostile"].total_deaths == 1

    def test_first_blood_event(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        fb_events = [e for e in eng.timeline if e.get("event") == "first_blood"]
        assert len(fb_events) == 1
        assert fb_events[0]["unit"] == "f1"

    def test_only_one_first_blood(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        eng.record_kill("f2", "h2")
        fb = [e for e in eng.timeline if e.get("event") == "first_blood"]
        assert len(fb) == 1

    def test_kill_unknown_units_no_crash(self, engine: ScoringEngine):
        """Recording kills with unknown IDs should not raise."""
        engine.record_kill("ghost1", "ghost2")

    def test_multiple_kills_same_killer(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        eng.record_kill("f1", "h2")
        eng.record_kill("f1", "h3")
        assert eng.unit_scores["f1"].kills == 3


# ---------------------------------------------------------------------------
# ScoringEngine — damage
# ---------------------------------------------------------------------------


class TestRecordDamage:
    def test_damage_dealt_and_taken(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_damage("f1", "h1", 50.0)
        assert eng.unit_scores["f1"].damage_dealt == 50.0
        assert eng.unit_scores["h1"].damage_taken == 50.0

    def test_cumulative_damage(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_damage("f1", "h1", 30.0)
        eng.record_damage("f1", "h1", 20.0)
        assert eng.unit_scores["f1"].damage_dealt == 50.0

    def test_damage_unknown_no_crash(self, engine: ScoringEngine):
        engine.record_damage("x", "y", 100.0)


# ---------------------------------------------------------------------------
# ScoringEngine — shots
# ---------------------------------------------------------------------------


class TestRecordShot:
    def test_shot_hit(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_shot("f1", True)
        assert eng.unit_scores["f1"].shots_fired == 1
        assert eng.unit_scores["f1"].shots_hit == 1

    def test_shot_miss(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_shot("f1", False)
        assert eng.unit_scores["f1"].shots_fired == 1
        assert eng.unit_scores["f1"].shots_hit == 0

    def test_accuracy_after_shots(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(8):
            eng.record_shot("f1", True)
        for _ in range(2):
            eng.record_shot("f1", False)
        assert eng.unit_scores["f1"].accuracy == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ScoringEngine — healing
# ---------------------------------------------------------------------------


class TestRecordHealing:
    def test_healing(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_healing("f3", "f1", 75.0)
        assert eng.unit_scores["f3"].healing_done == 75.0

    def test_healing_timeline_event(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_healing("f3", "f1", 50.0)
        events = [e for e in eng.timeline if e["event"] == "healing"]
        assert len(events) == 1
        assert events[0]["amount"] == 50.0


# ---------------------------------------------------------------------------
# ScoringEngine — objectives
# ---------------------------------------------------------------------------


class TestRecordObjective:
    def test_objective(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_objective("f1", "capture_point_a")
        assert eng.unit_scores["f1"].objectives_completed == 1
        assert eng.team_scores["friendly"].objectives_completed == 1

    def test_objective_timeline(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_objective("f1", "defuse_bomb")
        events = [e for e in eng.timeline if e["event"] == "objective"]
        assert events[0]["objective"] == "defuse_bomb"


# ---------------------------------------------------------------------------
# ScoringEngine — other events
# ---------------------------------------------------------------------------


class TestOtherEvents:
    def test_record_revive(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_revive("f3", "f1")
        assert eng.unit_scores["f3"].allies_revived == 1

    def test_record_vehicle_destroyed(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_vehicle_destroyed("f1")
        assert eng.unit_scores["f1"].vehicles_destroyed == 1

    def test_record_structure_destroyed(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_structure_destroyed("f1")
        assert eng.unit_scores["f1"].structures_destroyed == 1

    def test_record_generic_event(self, engine: ScoringEngine):
        engine.record_event("explosion", {"radius": 10})
        assert len(engine.timeline) == 1
        assert engine.timeline[0]["event"] == "explosion"
        assert engine.timeline[0]["radius"] == 10

    def test_record_supply_use(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_supply_use("f1", "ammo", 20.0)
        eng.record_supply_use("f1", "ammo", 10.0)
        assert eng.unit_scores["f1"].supplies_used["ammo"] == 30.0

    def test_record_detection(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_detection("f1")
        assert eng.unit_scores["f1"]._detected is True


# ---------------------------------------------------------------------------
# ScoringEngine — tick
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_advances_time(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(1.0, alive_units={"f1", "f2", "f3"})
        assert eng._sim_time == pytest.approx(1.0)

    def test_tick_increments_time_alive(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(2.0, alive_units={"f1"})
        assert eng.unit_scores["f1"].time_alive == pytest.approx(2.0)
        # Dead units don't accumulate
        assert eng.unit_scores["h1"].time_alive == pytest.approx(0.0)

    def test_tick_tracks_distance(self, populated_engine: ScoringEngine):
        eng = populated_engine
        # Set initial position
        eng.tick(0.1, alive_units={"f1"}, unit_positions={"f1": (0.0, 0.0)})
        eng.tick(0.1, alive_units={"f1"}, unit_positions={"f1": (3.0, 4.0)})
        assert eng.unit_scores["f1"].distance_moved == pytest.approx(5.0)

    def test_tick_ignores_jitter(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(0.1, unit_positions={"f1": (0.0, 0.0)})
        eng.tick(0.1, unit_positions={"f1": (0.001, 0.001)})
        assert eng.unit_scores["f1"].distance_moved == pytest.approx(0.0)

    def test_tick_defaults_all_alive(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(1.0)
        # All units should get time_alive since alive_units defaults to all
        for card in eng.unit_scores.values():
            assert card.time_alive == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ScoringEngine — achievements
# ---------------------------------------------------------------------------


class TestAchievements:
    def test_first_blood(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        # record_kill auto-checks achievements, so first_blood is already awarded
        assert "first_blood" in eng.unit_scores["f1"].achievements

    def test_sharpshooter(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(20):
            eng.record_shot("f1", True)
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "sharpshooter" in ids

    def test_sharpshooter_needs_min_shots(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(10):
            eng.record_shot("f1", True)
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "sharpshooter" not in ids

    def test_team_player(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].assists = 5
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "team_player" in ids

    def test_medic_achievement(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_healing("f3", "f1", 500.0)
        earned = eng.check_achievements("f3")
        ids = [a.achievement_id for a in earned]
        assert "medic" in ids

    def test_architect(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(3):
            eng.record_structure_destroyed("f1")
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "architect" in ids

    def test_convoy_killer(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(3):
            eng.record_vehicle_destroyed("f1")
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "convoy_killer" in ids

    def test_lone_wolf(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for i in range(10):
            eng.record_kill("f1", f"h{i % 3 + 1}")
        # record_kill auto-checks achievements after each kill
        assert "lone_wolf" in eng.unit_scores["f1"].achievements

    def test_lone_wolf_not_with_assists(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].kills = 10
        eng.unit_scores["f1"].assists = 1
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "lone_wolf" not in ids

    def test_objective_master(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for i in range(3):
            eng.record_objective("f1", f"obj_{i}")
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "objective_master" in ids

    def test_field_surgeon(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for _ in range(3):
            eng.record_revive("f3", "f1")
        earned = eng.check_achievements("f3")
        ids = [a.achievement_id for a in earned]
        assert "field_surgeon" in ids

    def test_marathon(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].distance_moved = 5001.0
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "marathon" in ids

    def test_ghost_undetected(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")  # needs at least one kill
        # record_kill auto-checks achievements; ghost should be awarded
        assert "ghost" in eng.unit_scores["f1"].achievements

    def test_ghost_fails_if_detected(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        eng.record_detection("f1")
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "ghost" not in ids

    def test_double_kill(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng._sim_time = 10.0
        eng.record_kill("f1", "h1")
        eng._sim_time = 13.0
        eng.record_kill("f1", "h2")
        # record_kill auto-checks achievements
        assert "double_kill" in eng.unit_scores["f1"].achievements

    def test_triple_kill(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng._sim_time = 10.0
        eng.record_kill("f1", "h1")
        eng._sim_time = 15.0
        eng.record_kill("f1", "h2")
        eng._sim_time = 19.0
        eng.record_kill("f1", "h3")
        # record_kill auto-checks achievements
        assert "triple_kill" in eng.unit_scores["f1"].achievements

    def test_no_duplicate_achievements(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].assists = 5
        eng.check_achievements("f1")
        earned2 = eng.check_achievements("f1")
        # Second check should return nothing new
        assert len(earned2) == 0

    def test_achievement_added_to_timeline(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].assists = 5
        eng.check_achievements("f1")
        ach_events = [e for e in eng.timeline if e.get("event") == "achievement"]
        assert len(ach_events) >= 1
        assert ach_events[0]["achievement"] == "team_player"

    def test_check_unknown_unit(self, engine: ScoringEngine):
        assert engine.check_achievements("nonexistent") == []

    def test_centurion(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].kills = 100
        eng.unit_scores["f1"]._kill_times = [float(i) for i in range(100)]
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "centurion" in ids

    def test_tank_buster(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].vehicles_destroyed = 5
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "tank_buster" in ids

    def test_support_mvp(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f3"].healing_done = 1000.0
        eng.unit_scores["f3"].allies_revived = 5
        earned = eng.check_achievements("f3")
        ids = [a.achievement_id for a in earned]
        assert "support_mvp" in ids

    def test_iron_will(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"]._low_health_time = 31.0
        earned = eng.check_achievements("f1")
        ids = [a.achievement_id for a in earned]
        assert "iron_will" in ids


# ---------------------------------------------------------------------------
# ScoringEngine — rampage multi-kill window
# ---------------------------------------------------------------------------


class TestMultiKillWindow:
    def test_rampage_within_window(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for i in range(5):
            eng._sim_time = float(i * 5)  # 0, 5, 10, 15, 20 — all within 30s
            eng.record_kill("f1", f"h{i % 3 + 1}")
        # record_kill auto-checks achievements after each kill
        assert "rampage" in eng.unit_scores["f1"].achievements

    def test_rampage_outside_window(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for i in range(5):
            eng._sim_time = float(i * 10)  # 0, 10, 20, 30, 40 — spans 40s
            eng.record_kill("f1", f"h{i % 3 + 1}")
        # record_kill auto-checks achievements after each kill
        assert "rampage" not in eng.unit_scores["f1"].achievements


# ---------------------------------------------------------------------------
# ScoringEngine — deferred achievements
# ---------------------------------------------------------------------------


class TestDeferredAchievements:
    def test_untouchable_at_aar(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(10.0, alive_units={"f1"})
        # f1 took no damage
        aar = eng.generate_aar(winner_alliance="friendly")
        f1_achs = [a for a in aar["achievements"] if a["unit"] == "f1"]
        ach_ids = [a["achievement"] for a in f1_achs]
        assert "untouchable" in ach_ids

    def test_untouchable_not_if_damaged(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(10.0, alive_units={"f1"})
        eng.record_damage("h1", "f1", 1.0)
        aar = eng.generate_aar(winner_alliance="friendly")
        f1_achs = [a for a in aar["achievements"] if a["unit"] == "f1"]
        ach_ids = [a["achievement"] for a in f1_achs]
        assert "untouchable" not in ach_ids

    def test_flawless_victory(self, populated_engine: ScoringEngine):
        eng = populated_engine
        # No friendly deaths
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar(winner_alliance="friendly")
        friendly_achs = [a for a in aar["achievements"]
                         if a["unit"] in ("f1", "f2", "f3")]
        ach_ids = [a["achievement"] for a in friendly_achs]
        assert "flawless_victory" in ach_ids

    def test_flawless_fails_with_death(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("h1", "f2")  # friendly death
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar(winner_alliance="friendly")
        friendly_achs = [a for a in aar["achievements"]
                         if a["unit"] in ("f1", "f2", "f3")]
        ach_ids = [a["achievement"] for a in friendly_achs]
        assert "flawless_victory" not in ach_ids

    def test_pacifist(self, populated_engine: ScoringEngine):
        eng = populated_engine
        # f3 gets no kills but is on winning team
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar(winner_alliance="friendly")
        f3_achs = [a for a in aar["achievements"] if a["unit"] == "f3"]
        ach_ids = [a["achievement"] for a in f3_achs]
        assert "pacifist" in ach_ids


# ---------------------------------------------------------------------------
# ScoringEngine — leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def test_leaderboard_sorted_by_score(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        eng.record_kill("f1", "h2")
        eng.record_kill("f2", "h3")
        lb = eng.get_leaderboard()
        assert lb[0]["unit_id"] == "f1"
        assert lb[0]["kills"] == 2

    def test_leaderboard_by_kills(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f2"].kills = 10
        eng.unit_scores["f1"].kills = 3
        lb = eng.get_leaderboard(ScoreCategory.KILLS)
        assert lb[0]["unit_id"] == "f2"

    def test_leaderboard_by_accuracy(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].shots_fired = 10
        eng.unit_scores["f1"].shots_hit = 9
        eng.unit_scores["f2"].shots_fired = 10
        eng.unit_scores["f2"].shots_hit = 5
        lb = eng.get_leaderboard(ScoreCategory.ACCURACY)
        assert lb[0]["unit_id"] == "f1"

    def test_leaderboard_by_objectives(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["h1"].objectives_completed = 5
        lb = eng.get_leaderboard(ScoreCategory.OBJECTIVES)
        assert lb[0]["unit_id"] == "h1"

    def test_leaderboard_by_survival(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f3"].time_alive = 999.0
        lb = eng.get_leaderboard(ScoreCategory.SURVIVAL)
        assert lb[0]["unit_id"] == "f3"

    def test_leaderboard_by_teamwork(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f3"].healing_done = 500.0
        eng.unit_scores["f3"].allies_revived = 3
        lb = eng.get_leaderboard(ScoreCategory.TEAMWORK)
        assert lb[0]["unit_id"] == "f3"

    def test_leaderboard_by_tactical(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].structures_destroyed = 5
        lb = eng.get_leaderboard(ScoreCategory.TACTICAL)
        assert lb[0]["unit_id"] == "f1"

    def test_leaderboard_empty(self, engine: ScoringEngine):
        lb = engine.get_leaderboard()
        assert lb == []


# ---------------------------------------------------------------------------
# ScoringEngine — AAR
# ---------------------------------------------------------------------------


class TestGenerateAAR:
    def test_aar_structure(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(60.0)
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar(winner_alliance="friendly")
        assert "summary" in aar
        assert "timeline" in aar
        assert "teams" in aar
        assert "leaderboard" in aar
        assert "achievements" in aar
        assert "heatmaps" in aar

    def test_aar_summary(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(120.0)
        eng.record_kill("f1", "h1")
        eng.record_kill("f2", "h2")
        aar = eng.generate_aar(winner_alliance="friendly")
        s = aar["summary"]
        assert s["duration"] == pytest.approx(120.0)
        assert s["winner"] == "friendly"
        assert s["total_kills"] == 2
        assert s["units_registered"] == 6

    def test_aar_teams(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar()
        assert len(aar["teams"]) == 2

    def test_aar_leaderboard_present(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar()
        assert len(aar["leaderboard"]) == 6

    def test_aar_heatmaps(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.tick(0.1, unit_positions={"f1": (10.0, 20.0)})
        eng.record_kill("f1", "h1")
        aar = eng.generate_aar()
        assert "kills" in aar["heatmaps"]
        assert "deaths" in aar["heatmaps"]
        assert "movement" in aar["heatmaps"]

    def test_aar_no_winner(self, populated_engine: ScoringEngine):
        eng = populated_engine
        aar = eng.generate_aar()
        assert aar["summary"]["winner"] is None


# ---------------------------------------------------------------------------
# ScoringEngine — Three.js overlay
# ---------------------------------------------------------------------------


class TestToThreeJS:
    def test_three_js_structure(self, populated_engine: ScoringEngine):
        eng = populated_engine
        data = eng.to_three_js()
        assert "sim_time" in data
        assert "scoreboard" in data
        assert "kill_feed" in data
        assert "achievement_popups" in data
        assert "team_summary" in data

    def test_kill_feed(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        eng.record_kill("f2", "h2")
        data = eng.to_three_js()
        assert len(data["kill_feed"]) == 2
        assert data["kill_feed"][0]["killer"] == "Alpha-1"
        assert data["kill_feed"][0]["victim"] == "Bravo-1"

    def test_kill_feed_max_10(self, populated_engine: ScoringEngine):
        eng = populated_engine
        for i in range(15):
            eng.record_kill("f1", "h1")
        data = eng.to_three_js()
        assert len(data["kill_feed"]) == 10

    def test_achievement_popups(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.unit_scores["f1"].assists = 5
        eng.check_achievements("f1")
        data = eng.to_three_js()
        assert len(data["achievement_popups"]) >= 1
        assert data["achievement_popups"][0]["achievement"] == "Team Player"

    def test_team_summary(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        data = eng.to_three_js()
        assert "friendly" in data["team_summary"]
        assert data["team_summary"]["friendly"]["total_kills"] == 1

    def test_first_blood_in_kill_feed(self, populated_engine: ScoringEngine):
        eng = populated_engine
        eng.record_kill("f1", "h1")
        data = eng.to_three_js()
        assert data["kill_feed"][0]["first_blood"] is True
