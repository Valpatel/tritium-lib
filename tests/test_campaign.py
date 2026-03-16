# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the campaign and mission chain system.

60+ tests covering MissionType, MissionBriefing, MissionResult,
PersistentState, Campaign class, presets, save/load, Three.js output,
grading, veteran units, unlocks, resources, and reputation.
"""

import copy
import math
import pytest

from tritium_lib.sim_engine.campaign import (
    MissionType,
    MissionBriefing,
    MissionResult,
    PersistentState,
    Campaign,
    CAMPAIGNS,
    compute_grade,
    _grade_to_numeric,
    _numeric_to_grade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_briefing(
    mission_id: str = "test_01",
    name: str = "Test Mission",
    mission_type: MissionType = MissionType.ASSAULT,
    difficulty: int = 2,
    **kwargs,
) -> MissionBriefing:
    """Factory for test MissionBriefing objects."""
    defaults = {
        "description": "A test mission.",
        "objectives": [{"type": "eliminate_all", "target": 1.0,
                        "description": "Clear all"}],
        "available_units": ["infantry", "infantry", "sniper"],
        "max_units": 3,
        "time_limit": 300.0,
        "environment": {"weather": "clear", "time_of_day": "day",
                        "terrain_type": "urban"},
        "enemy_composition": [{"template": "infantry", "count": 5}],
        "rewards": {"xp": 200, "unlock_units": ["scout"],
                    "unlock_weapons": ["flashbang"]},
    }
    defaults.update(kwargs)
    return MissionBriefing(
        mission_id=mission_id,
        name=name,
        mission_type=mission_type,
        difficulty=difficulty,
        **defaults,
    )


def _make_result(
    mission_id: str = "test_01",
    success: bool = True,
    **kwargs,
) -> MissionResult:
    """Factory for test MissionResult objects."""
    defaults = {
        "time_taken": 120.0,
        "casualties_friendly": 1,
        "casualties_enemy": 5,
        "objectives_completed": 1,
        "objectives_total": 1,
        "score": 800,
        "grade": "A",
        "mvp": "infantry_01",
        "achievements": ["first_blood"],
        "xp_earned": 250,
    }
    defaults.update(kwargs)
    return MissionResult(mission_id=mission_id, success=success, **defaults)


def _make_campaign(num_missions: int = 3) -> Campaign:
    """Create a simple campaign with N missions."""
    missions = [
        _make_briefing(
            mission_id=f"m_{i+1:02d}",
            name=f"Mission {i+1}",
            difficulty=min(i + 1, 5),
        )
        for i in range(num_missions)
    ]
    return Campaign(
        campaign_id="test_campaign",
        name="Test Campaign",
        missions=missions,
        description="A test campaign.",
        initial_reputation={"allies": 50.0, "enemies": -30.0},
    )


# ---------------------------------------------------------------------------
# MissionType enum tests
# ---------------------------------------------------------------------------


class TestMissionType:
    def test_all_mission_types_exist(self):
        expected = {
            "ASSAULT", "DEFENSE", "ESCORT", "RECON", "RESCUE",
            "DEMOLITION", "STEALTH", "PATROL", "AMBUSH", "SIEGE",
        }
        actual = {m.name for m in MissionType}
        assert actual == expected

    def test_mission_type_values(self):
        assert MissionType.ASSAULT.value == "assault"
        assert MissionType.SIEGE.value == "siege"
        assert MissionType.STEALTH.value == "stealth"

    def test_mission_type_from_value(self):
        assert MissionType("patrol") == MissionType.PATROL
        assert MissionType("recon") == MissionType.RECON

    def test_mission_type_count(self):
        assert len(MissionType) == 10


# ---------------------------------------------------------------------------
# MissionBriefing tests
# ---------------------------------------------------------------------------


class TestMissionBriefing:
    def test_basic_creation(self):
        b = _make_briefing()
        assert b.mission_id == "test_01"
        assert b.name == "Test Mission"
        assert b.mission_type == MissionType.ASSAULT
        assert b.difficulty == 2

    def test_to_dict(self):
        b = _make_briefing()
        d = b.to_dict()
        assert d["mission_id"] == "test_01"
        assert d["mission_type"] == "assault"
        assert d["difficulty"] == 2
        assert isinstance(d["objectives"], list)
        assert isinstance(d["environment"], dict)

    def test_default_values(self):
        b = MissionBriefing(
            mission_id="bare",
            name="Bare",
            description="",
            mission_type=MissionType.RECON,
            difficulty=1,
        )
        assert b.available_units == []
        assert b.max_units == 10
        assert b.time_limit is None
        assert b.enemy_composition == []
        assert b.rewards == {}

    def test_difficulty_range(self):
        for diff in range(1, 6):
            b = _make_briefing(difficulty=diff)
            assert b.difficulty == diff

    def test_objectives_structure(self):
        b = _make_briefing()
        assert len(b.objectives) == 1
        assert b.objectives[0]["type"] == "eliminate_all"

    def test_time_limit_none(self):
        b = _make_briefing(time_limit=None)
        assert b.time_limit is None

    def test_environment_fields(self):
        b = _make_briefing()
        assert b.environment["weather"] == "clear"
        assert b.environment["time_of_day"] == "day"
        assert b.environment["terrain_type"] == "urban"


# ---------------------------------------------------------------------------
# MissionResult tests
# ---------------------------------------------------------------------------


class TestMissionResult:
    def test_basic_creation(self):
        r = _make_result()
        assert r.mission_id == "test_01"
        assert r.success is True
        assert r.grade == "A"

    def test_to_dict(self):
        r = _make_result()
        d = r.to_dict()
        assert d["success"] is True
        assert d["grade"] == "A"
        assert isinstance(d["achievements"], list)
        assert d["time_taken"] == 120.0

    def test_failed_result(self):
        r = _make_result(success=False, grade="F", score=50)
        assert r.success is False
        assert r.grade == "F"

    def test_all_grades(self):
        for grade in ["S", "A", "B", "C", "D", "F"]:
            r = _make_result(grade=grade)
            assert r.grade == grade

    def test_default_values(self):
        r = MissionResult(mission_id="x", success=True, time_taken=10.0)
        assert r.casualties_friendly == 0
        assert r.casualties_enemy == 0
        assert r.mvp is None
        assert r.achievements == []
        assert r.xp_earned == 0


# ---------------------------------------------------------------------------
# PersistentState tests
# ---------------------------------------------------------------------------


class TestPersistentState:
    def test_basic_creation(self):
        s = PersistentState(campaign_id="test")
        assert s.campaign_id == "test"
        assert s.current_mission == 0
        assert s.total_xp == 0

    def test_to_dict(self):
        s = PersistentState(campaign_id="test")
        s.total_xp = 500
        s.unlocked_units = ["sniper"]
        d = s.to_dict()
        assert d["campaign_id"] == "test"
        assert d["total_xp"] == 500
        assert "sniper" in d["unlocked_units"]

    def test_from_dict_roundtrip(self):
        s = PersistentState(campaign_id="test")
        s.total_xp = 1000
        s.current_mission = 3
        s.unlocked_weapons = ["rpg", "c4"]
        s.reputation = {"allies": 75.0}
        s.resources = {"ammo_stockpile": 60.0}
        d = s.to_dict()
        s2 = PersistentState.from_dict(d)
        assert s2.campaign_id == "test"
        assert s2.total_xp == 1000
        assert s2.current_mission == 3
        assert s2.unlocked_weapons == ["rpg", "c4"]
        assert s2.reputation == {"allies": 75.0}
        assert s2.resources == {"ammo_stockpile": 60.0}

    def test_from_dict_with_completed_missions(self):
        r = _make_result()
        s = PersistentState(campaign_id="test")
        s.completed_missions = [r]
        d = s.to_dict()
        s2 = PersistentState.from_dict(d)
        assert len(s2.completed_missions) == 1
        assert s2.completed_missions[0].mission_id == "test_01"
        assert s2.completed_missions[0].success is True

    def test_veteran_units(self):
        s = PersistentState(campaign_id="test")
        s.veteran_units.append({
            "unit_id": "vet_01", "name": "Sgt. Rock",
            "template": "infantry", "xp": 100,
            "kills": 10, "missions_survived": 2,
        })
        d = s.to_dict()
        assert len(d["veteran_units"]) == 1
        assert d["veteran_units"][0]["name"] == "Sgt. Rock"


# ---------------------------------------------------------------------------
# Grade computation tests
# ---------------------------------------------------------------------------


class TestGrading:
    def test_compute_grade_s(self):
        assert compute_grade(950, 1000) == "S"

    def test_compute_grade_a(self):
        assert compute_grade(870, 1000) == "A"

    def test_compute_grade_b(self):
        assert compute_grade(750, 1000) == "B"

    def test_compute_grade_c(self):
        assert compute_grade(550, 1000) == "C"

    def test_compute_grade_d(self):
        assert compute_grade(350, 1000) == "D"

    def test_compute_grade_f(self):
        assert compute_grade(100, 1000) == "F"

    def test_compute_grade_zero_max(self):
        assert compute_grade(500, 0) == "C"

    def test_grade_to_numeric(self):
        assert _grade_to_numeric("S") == 5.0
        assert _grade_to_numeric("A") == 4.0
        assert _grade_to_numeric("F") == 0.0

    def test_numeric_to_grade(self):
        assert _numeric_to_grade(4.8) == "S"
        assert _numeric_to_grade(3.7) == "A"
        assert _numeric_to_grade(2.6) == "B"
        assert _numeric_to_grade(1.8) == "C"
        assert _numeric_to_grade(0.6) == "D"
        assert _numeric_to_grade(0.3) == "F"

    def test_grade_roundtrip(self):
        for grade in ["S", "A", "B", "C", "D", "F"]:
            n = _grade_to_numeric(grade)
            assert _numeric_to_grade(n) == grade


# ---------------------------------------------------------------------------
# Campaign class tests
# ---------------------------------------------------------------------------


class TestCampaign:
    def test_creation(self):
        c = _make_campaign()
        assert c.campaign_id == "test_campaign"
        assert c.name == "Test Campaign"
        assert len(c.missions) == 3

    def test_current_mission(self):
        c = _make_campaign()
        m = c.current_mission()
        assert m.mission_id == "m_01"

    def test_current_mission_advances(self):
        c = _make_campaign()
        result = _make_result(mission_id="m_01")
        c.complete_mission(result)
        m = c.current_mission()
        assert m.mission_id == "m_02"

    def test_current_mission_raises_when_complete(self):
        c = _make_campaign(num_missions=1)
        result = _make_result(mission_id="m_01")
        c.complete_mission(result)
        with pytest.raises(IndexError):
            c.current_mission()

    def test_is_complete(self):
        c = _make_campaign(num_missions=2)
        assert c.is_complete() is False
        c.complete_mission(_make_result(mission_id="m_01"))
        assert c.is_complete() is False
        c.complete_mission(_make_result(mission_id="m_02"))
        assert c.is_complete() is True

    def test_start_mission_returns_config(self):
        c = _make_campaign()
        config = c.start_mission()
        assert config["name"] == "Mission 1"
        assert "waves" in config
        assert "objectives" in config
        assert "friendly_units" in config
        assert config["mission_type"] == "assault"
        assert config["difficulty"] == 1

    def test_start_mission_deducts_resources(self):
        c = _make_campaign()
        initial_ammo = c.state.resources["ammo_stockpile"]
        c.start_mission()
        assert c.state.resources["ammo_stockpile"] < initial_ammo

    def test_start_mission_time_limit(self):
        c = _make_campaign()
        config = c.start_mission()
        # time_limit=300.0, tick_rate=10 -> max_ticks=3000
        assert config["max_ticks"] == 3000

    def test_start_mission_no_time_limit(self):
        c = _make_campaign()
        c.missions[0].time_limit = None
        config = c.start_mission()
        assert config["max_ticks"] == 6000

    def test_complete_mission_xp(self):
        c = _make_campaign()
        result = _make_result(xp_earned=300)
        c.complete_mission(result)
        # xp_earned + reward xp (200 from briefing)
        assert c.state.total_xp == 300 + 200

    def test_complete_mission_unlocks_units(self):
        c = _make_campaign()
        result = _make_result()
        c.complete_mission(result)
        assert "scout" in c.state.unlocked_units

    def test_complete_mission_unlocks_weapons(self):
        c = _make_campaign()
        result = _make_result()
        c.complete_mission(result)
        assert "flashbang" in c.state.unlocked_weapons

    def test_complete_mission_no_duplicate_unlocks(self):
        c = _make_campaign()
        c.state.unlocked_units = ["scout"]
        result = _make_result()
        c.complete_mission(result)
        assert c.state.unlocked_units.count("scout") == 1

    def test_complete_mission_veteran_units(self):
        c = _make_campaign()
        result = _make_result(casualties_friendly=0)
        c.complete_mission(result)
        assert len(c.state.veteran_units) > 0
        assert c.state.veteran_units[0]["template"] == "infantry"

    def test_complete_mission_failed_no_bonus_xp(self):
        c = _make_campaign()
        result = _make_result(success=False, xp_earned=100)
        c.complete_mission(result)
        # Only xp_earned, no bonus from rewards
        assert c.state.total_xp == 100

    def test_complete_mission_failed_no_veterans(self):
        c = _make_campaign()
        result = _make_result(success=False, xp_earned=0)
        c.complete_mission(result)
        assert len(c.state.veteran_units) == 0

    def test_reputation_increases_on_success(self):
        c = _make_campaign()
        initial_rep = dict(c.state.reputation)
        result = _make_result()
        c.complete_mission(result)
        for faction in initial_rep:
            assert c.state.reputation[faction] > initial_rep[faction]

    def test_reputation_decreases_on_failure(self):
        c = _make_campaign()
        initial_rep = dict(c.state.reputation)
        result = _make_result(success=False)
        c.complete_mission(result)
        for faction in initial_rep:
            assert c.state.reputation[faction] < initial_rep[faction]

    def test_medical_supplies_recover_on_success(self):
        c = _make_campaign()
        c.state.resources["medical_supplies"] = 50.0
        result = _make_result(score=500)
        c.complete_mission(result)
        assert c.state.resources["medical_supplies"] > 50.0

    def test_overall_grade_no_missions(self):
        c = _make_campaign()
        assert c.overall_grade() == "C"

    def test_overall_grade_all_s(self):
        c = _make_campaign(num_missions=3)
        for i in range(3):
            c.complete_mission(_make_result(
                mission_id=f"m_{i+1:02d}", grade="S",
            ))
        assert c.overall_grade() == "S"

    def test_overall_grade_mixed(self):
        c = _make_campaign(num_missions=2)
        c.complete_mission(_make_result(mission_id="m_01", grade="A"))
        c.complete_mission(_make_result(mission_id="m_02", grade="C"))
        grade = c.overall_grade()
        assert grade == "B"  # avg(4.0, 2.0) = 3.0 -> B

    def test_resource_costs_scale_with_difficulty(self):
        c = _make_campaign()
        c.missions[0].difficulty = 1
        c.start_mission()
        ammo_after_easy = c.state.resources["ammo_stockpile"]

        c2 = _make_campaign()
        c2.missions[0].difficulty = 5
        c2.start_mission()
        ammo_after_hard = c2.state.resources["ammo_stockpile"]

        assert ammo_after_easy > ammo_after_hard


# ---------------------------------------------------------------------------
# Save/Load tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_returns_dict(self):
        c = _make_campaign()
        data = c.save()
        assert isinstance(data, dict)
        assert data["campaign_id"] == "test_campaign"
        assert len(data["missions"]) == 3
        assert "state" in data

    def test_load_restores_state(self):
        c = _make_campaign()
        c.complete_mission(_make_result(mission_id="m_01"))
        c.state.unlocked_weapons = ["rpg"]
        saved = c.save()

        c2 = _make_campaign()
        c2.load(saved)
        assert c2.state.current_mission == 1
        assert len(c2.state.completed_missions) == 1
        assert "rpg" in c2.state.unlocked_weapons

    def test_save_load_roundtrip(self):
        c = _make_campaign()
        for i in range(2):
            c.complete_mission(_make_result(
                mission_id=f"m_{i+1:02d}", xp_earned=100 * (i + 1),
            ))
        saved = c.save()

        c2 = _make_campaign()
        c2.load(saved)
        assert c2.state.current_mission == c.state.current_mission
        assert c2.state.total_xp == c.state.total_xp
        assert len(c2.state.completed_missions) == 2

    def test_load_restores_missions(self):
        c = _make_campaign()
        saved = c.save()
        c2 = Campaign("empty", "Empty", [])
        c2.load(saved)
        assert len(c2.missions) == 3
        assert c2.missions[0].mission_type == MissionType.ASSAULT


# ---------------------------------------------------------------------------
# Three.js output tests
# ---------------------------------------------------------------------------


class TestThreeJs:
    def test_to_three_js_structure(self):
        c = _make_campaign()
        tj = c.to_three_js()
        assert tj["campaign_id"] == "test_campaign"
        assert tj["name"] == "Test Campaign"
        assert isinstance(tj["missions"], list)
        assert isinstance(tj["edges"], list)
        assert "progress" in tj
        assert "total_xp" in tj
        assert "overall_grade" in tj
        assert "is_complete" in tj

    def test_to_three_js_mission_status(self):
        c = _make_campaign()
        tj = c.to_three_js()
        assert tj["missions"][0]["status"] == "current"
        assert tj["missions"][1]["status"] == "locked"

    def test_to_three_js_after_completion(self):
        c = _make_campaign()
        c.complete_mission(_make_result(mission_id="m_01", grade="A"))
        tj = c.to_three_js()
        assert tj["missions"][0]["status"] == "completed"
        assert tj["missions"][0]["grade"] == "A"
        assert tj["missions"][1]["status"] == "current"

    def test_to_three_js_failed_mission(self):
        c = _make_campaign()
        c.complete_mission(_make_result(
            mission_id="m_01", success=False, grade="F",
        ))
        tj = c.to_three_js()
        assert tj["missions"][0]["status"] == "failed"

    def test_to_three_js_edges(self):
        c = _make_campaign()
        tj = c.to_three_js()
        assert len(tj["edges"]) == 2
        assert tj["edges"][0]["from"] == "m_01"
        assert tj["edges"][0]["to"] == "m_02"
        assert tj["edges"][0]["unlocked"] is False

    def test_to_three_js_progress(self):
        c = _make_campaign(num_missions=4)
        c.complete_mission(_make_result(mission_id="m_01"))
        c.complete_mission(_make_result(mission_id="m_02"))
        tj = c.to_three_js()
        assert tj["progress"] == 0.5

    def test_to_three_js_positions_are_numeric(self):
        c = _make_campaign()
        tj = c.to_three_js()
        for m in tj["missions"]:
            assert isinstance(m["position"]["x"], float)
            assert isinstance(m["position"]["y"], float)

    def test_to_three_js_resources(self):
        c = _make_campaign()
        tj = c.to_three_js()
        assert "resources" in tj
        assert "ammo_stockpile" in tj["resources"]


# ---------------------------------------------------------------------------
# Campaign presets tests
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_exist(self):
        expected = {"tutorial", "urban_warfare", "insurgency",
                    "naval_campaign", "air_superiority"}
        assert set(CAMPAIGNS.keys()) == expected

    def test_tutorial_has_3_missions(self):
        assert len(CAMPAIGNS["tutorial"]["missions"]) == 3

    def test_urban_warfare_has_7_missions(self):
        assert len(CAMPAIGNS["urban_warfare"]["missions"]) == 7

    def test_insurgency_has_10_missions(self):
        assert len(CAMPAIGNS["insurgency"]["missions"]) == 10

    def test_naval_has_5_missions(self):
        assert len(CAMPAIGNS["naval_campaign"]["missions"]) == 5

    def test_air_has_5_missions(self):
        assert len(CAMPAIGNS["air_superiority"]["missions"]) == 5

    def test_from_preset_tutorial(self):
        c = Campaign.from_preset("tutorial")
        assert c.name == "Basic Training"
        assert len(c.missions) == 3
        assert c.state.campaign_id == "tutorial"

    def test_from_preset_unknown_raises(self):
        with pytest.raises(KeyError):
            Campaign.from_preset("nonexistent")

    def test_all_presets_have_unique_mission_ids(self):
        for name, preset in CAMPAIGNS.items():
            ids = [m.mission_id for m in preset["missions"]]
            assert len(ids) == len(set(ids)), (
                f"Duplicate mission IDs in {name}: {ids}"
            )

    def test_all_missions_have_valid_types(self):
        for name, preset in CAMPAIGNS.items():
            for m in preset["missions"]:
                assert isinstance(m.mission_type, MissionType), (
                    f"Invalid type in {name}/{m.mission_id}"
                )

    def test_all_missions_have_valid_difficulty(self):
        for name, preset in CAMPAIGNS.items():
            for m in preset["missions"]:
                assert 1 <= m.difficulty <= 5, (
                    f"Bad difficulty {m.difficulty} in {name}/{m.mission_id}"
                )

    def test_difficulty_progression_in_campaigns(self):
        """Each campaign's difficulty should generally not decrease."""
        for name, preset in CAMPAIGNS.items():
            missions = preset["missions"]
            if len(missions) < 2:
                continue
            # First mission should be <= last mission difficulty
            assert missions[0].difficulty <= missions[-1].difficulty, (
                f"{name}: first mission harder than last"
            )

    def test_all_presets_can_be_played(self):
        """Verify each preset can be started and a mission can begin."""
        for name in CAMPAIGNS:
            c = Campaign.from_preset(name)
            config = c.start_mission()
            assert config["name"] == c.missions[0].name
            assert len(config["waves"]) > 0

    def test_preset_rewards_have_xp(self):
        """Every preset mission should have XP rewards."""
        for name, preset in CAMPAIGNS.items():
            for m in preset["missions"]:
                assert m.rewards.get("xp", 0) > 0, (
                    f"No XP reward in {name}/{m.mission_id}"
                )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_campaign_with_no_missions(self):
        c = Campaign("empty", "Empty", [])
        assert c.is_complete() is True
        assert c.overall_grade() == "C"

    def test_three_js_empty_campaign(self):
        c = Campaign("empty", "Empty", [])
        tj = c.to_three_js()
        assert tj["missions"] == []
        assert tj["edges"] == []
        assert tj["progress"] == 0.0
        assert tj["is_complete"] is True

    def test_save_load_empty_campaign(self):
        c = Campaign("empty", "Empty", [])
        saved = c.save()
        c2 = Campaign("x", "X", [])
        c2.load(saved)
        assert c2.campaign_id == "empty"
        assert len(c2.missions) == 0

    def test_resources_dont_go_negative(self):
        c = _make_campaign()
        c.state.resources["ammo_stockpile"] = 1.0
        c.missions[0].difficulty = 5  # high cost
        c.start_mission()
        assert c.state.resources["ammo_stockpile"] >= 0.0

    def test_resources_dont_exceed_cap(self):
        c = _make_campaign()
        c.state.resources["medical_supplies"] = 99.0
        result = _make_result(score=10000)
        c.complete_mission(result)
        assert c.state.resources["medical_supplies"] <= 100.0

    def test_reputation_clamped_positive(self):
        c = _make_campaign()
        c.state.reputation = {"allies": 98.0}
        result = _make_result()
        c.complete_mission(result)
        assert c.state.reputation["allies"] <= 100.0

    def test_reputation_clamped_negative(self):
        c = _make_campaign()
        c.state.reputation = {"enemies": -98.0}
        result = _make_result(success=False)
        c.complete_mission(result)
        assert c.state.reputation["enemies"] >= -100.0

    def test_single_mission_campaign(self):
        c = _make_campaign(num_missions=1)
        assert c.is_complete() is False
        c.complete_mission(_make_result(mission_id="m_01"))
        assert c.is_complete() is True

    def test_complete_all_missions_sequentially(self):
        c = _make_campaign(num_missions=5)
        for i in range(5):
            assert c.is_complete() is False
            c.complete_mission(_make_result(mission_id=f"m_{i+1:02d}"))
        assert c.is_complete() is True
        assert len(c.state.completed_missions) == 5
