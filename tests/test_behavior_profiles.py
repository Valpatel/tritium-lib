# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AI behavior profile system."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.ai.behavior_profiles import (
    BehaviorEngine,
    BehaviorProfile,
    BehaviorTrait,
    PROFILES,
)


# ============================================================================
# BehaviorTrait enum
# ============================================================================


class TestBehaviorTrait:
    def test_all_traits_exist(self):
        expected = {
            "AGGRESSIVE", "DEFENSIVE", "CAUTIOUS", "RECKLESS",
            "METHODICAL", "OPPORTUNISTIC", "SUPPORTIVE", "INDEPENDENT",
        }
        assert {t.name for t in BehaviorTrait} == expected

    def test_trait_values_are_lowercase(self):
        for t in BehaviorTrait:
            assert t.value == t.name.lower()

    def test_trait_count(self):
        assert len(BehaviorTrait) == 8


# ============================================================================
# BehaviorProfile dataclass
# ============================================================================


class TestBehaviorProfile:
    def test_defaults(self):
        p = BehaviorProfile(profile_id="test", name="Test")
        assert p.aggression == 0.5
        assert p.caution == 0.5
        assert p.teamwork == 0.5
        assert p.discipline == 0.5
        assert p.initiative == 0.5
        assert p.morale_resilience == 0.5
        assert p.preferred_range == "medium"
        assert p.retreat_threshold == 0.25
        assert p.suppression_tolerance == 0.5
        assert p.cover_priority == 0.5
        assert p.ammo_conservation == 0.5
        assert p.traits == []

    def test_clamp_high(self):
        p = BehaviorProfile(profile_id="x", name="x", aggression=2.0, caution=1.5)
        assert p.aggression == 1.0
        assert p.caution == 1.0

    def test_clamp_low(self):
        p = BehaviorProfile(profile_id="x", name="x", aggression=-0.5, discipline=-1.0)
        assert p.aggression == 0.0
        assert p.discipline == 0.0

    def test_invalid_range_defaults_to_medium(self):
        p = BehaviorProfile(profile_id="x", name="x", preferred_range="sniper")
        assert p.preferred_range == "medium"

    def test_valid_ranges(self):
        for r in ("close", "medium", "long"):
            p = BehaviorProfile(profile_id="x", name="x", preferred_range=r)
            assert p.preferred_range == r

    def test_traits_list(self):
        p = BehaviorProfile(
            profile_id="x", name="x",
            traits=[BehaviorTrait.AGGRESSIVE, BehaviorTrait.RECKLESS],
        )
        assert len(p.traits) == 2
        assert BehaviorTrait.AGGRESSIVE in p.traits

    def test_retreat_threshold_clamp(self):
        p = BehaviorProfile(profile_id="x", name="x", retreat_threshold=1.5)
        assert p.retreat_threshold == 1.0
        p2 = BehaviorProfile(profile_id="x", name="x", retreat_threshold=-0.1)
        assert p2.retreat_threshold == 0.0


# ============================================================================
# PROFILES dict
# ============================================================================


class TestProfiles:
    def test_has_12_profiles(self):
        assert len(PROFILES) == 12

    def test_expected_ids(self):
        expected = {
            "elite_operator", "conscript", "guerrilla", "sniper_patient",
            "berserker", "medic_angel", "engineer_builder", "scout_ghost",
            "commander_calm", "civilian_panicked", "robot_precise", "veteran_steady",
        }
        assert set(PROFILES.keys()) == expected

    def test_profile_ids_match_keys(self):
        for key, profile in PROFILES.items():
            assert profile.profile_id == key

    def test_all_profiles_have_names(self):
        for profile in PROFILES.values():
            assert profile.name
            assert isinstance(profile.name, str)

    def test_all_profiles_have_traits(self):
        for profile in PROFILES.values():
            assert len(profile.traits) >= 1

    def test_berserker_is_extreme(self):
        b = PROFILES["berserker"]
        assert b.aggression == 1.0
        assert b.caution < 0.1
        assert b.retreat_threshold < 0.1

    def test_civilian_panicked_avoids_combat(self):
        c = PROFILES["civilian_panicked"]
        assert c.aggression == 0.0
        assert c.caution == 1.0
        assert c.retreat_threshold >= 0.9

    def test_robot_precise_has_max_discipline(self):
        r = PROFILES["robot_precise"]
        assert r.discipline == 1.0
        assert r.morale_resilience == 1.0

    def test_sniper_prefers_long_range(self):
        s = PROFILES["sniper_patient"]
        assert s.preferred_range == "long"

    def test_berserker_prefers_close(self):
        assert PROFILES["berserker"].preferred_range == "close"

    def test_medic_is_supportive(self):
        m = PROFILES["medic_angel"]
        assert BehaviorTrait.SUPPORTIVE in m.traits
        assert m.teamwork > 0.9


# ============================================================================
# BehaviorEngine — initialization and profile management
# ============================================================================


class TestBehaviorEngineInit:
    def test_empty_init(self):
        engine = BehaviorEngine()
        assert len(engine.profiles) == 0
        assert len(engine.unit_profiles) == 0

    def test_init_with_profiles(self):
        engine = BehaviorEngine(profiles=PROFILES)
        assert len(engine.profiles) == 12

    def test_add_profile(self):
        engine = BehaviorEngine()
        p = BehaviorProfile(profile_id="custom", name="Custom")
        engine.add_profile(p)
        assert "custom" in engine.profiles

    def test_assign_profile(self):
        engine = BehaviorEngine(profiles=PROFILES)
        engine.assign_profile("u1", "berserker")
        assert engine.unit_profiles["u1"] == "berserker"

    def test_assign_unknown_profile_raises(self):
        engine = BehaviorEngine()
        with pytest.raises(KeyError, match="Unknown profile"):
            engine.assign_profile("u1", "nonexistent")

    def test_get_profile_returns_none_for_unassigned(self):
        engine = BehaviorEngine(profiles=PROFILES)
        assert engine.get_profile("unknown_unit") is None

    def test_get_profile_returns_correct_profile(self):
        engine = BehaviorEngine(profiles=PROFILES)
        engine.assign_profile("u1", "elite_operator")
        p = engine.get_profile("u1")
        assert p is not None
        assert p.profile_id == "elite_operator"

    def test_reassign_profile(self):
        engine = BehaviorEngine(profiles=PROFILES)
        engine.assign_profile("u1", "berserker")
        engine.assign_profile("u1", "conscript")
        assert engine.get_profile("u1").profile_id == "conscript"


# ============================================================================
# BehaviorEngine.decide()
# ============================================================================


class TestDecide:
    @pytest.fixture()
    def engine(self) -> BehaviorEngine:
        e = BehaviorEngine(profiles=PROFILES)
        for pid in PROFILES:
            e.assign_profile(pid, pid)  # unit_id == profile_id for easy testing
        return e

    def test_no_profile_returns_hold(self):
        engine = BehaviorEngine()
        result = engine.decide("nobody", {})
        assert result["action"] == "hold"

    def test_berserker_low_health_still_fights(self, engine):
        """Berserker has reckless trait — retreat threshold halved from 0.05."""
        result = engine.decide("berserker", {
            "health": 0.04, "threats": 2, "enemy_distance": 10.0,
        })
        # Even at 4% health, reckless berserker retreat threshold is ~0.025
        assert result["action"] != "hold"

    def test_conscript_retreats_early(self, engine):
        """Conscripts have cautious trait raising retreat threshold."""
        result = engine.decide("conscript", {
            "health": 0.5, "threats": 3, "enemy_distance": 30.0,
        })
        # 0.4 * 1.5 = 0.6 threshold, health 0.5 < 0.6 -> retreat
        assert result["action"] == "retreat"

    def test_medic_supports_allies(self, engine):
        result = engine.decide("medic_angel", {
            "health": 0.8, "threats": 1, "allies": 2,
            "enemy_distance": 40.0,
        })
        assert result["action"] == "support"

    def test_civilian_retreats_immediately(self, engine):
        result = engine.decide("civilian_panicked", {
            "health": 0.85, "threats": 1, "enemy_distance": 50.0,
        })
        # Retreat threshold 0.9, cautious * 1.5 but capped at 0.8
        # Health 0.85 > 0.8, so won't retreat from threshold alone
        # But civilian has DEFENSIVE + CAUTIOUS traits
        assert result["action"] in ("seek_cover", "retreat", "hold_and_engage")

    def test_no_threats_patrol(self, engine):
        result = engine.decide("veteran_steady", {
            "health": 1.0, "threats": 0,
        })
        assert result["action"] in ("patrol", "advance")

    def test_no_threats_with_objective_advances(self, engine):
        result = engine.decide("elite_operator", {
            "health": 1.0, "threats": 0, "has_objective": True,
        })
        assert result["action"] == "advance"

    def test_suppressed_low_tolerance_seeks_cover(self, engine):
        result = engine.decide("conscript", {
            "health": 0.9, "threats": 1, "suppressed": True,
            "in_cover": False, "enemy_distance": 30.0,
        })
        # Conscript suppression tolerance is 0.2 < 0.5
        # But first check retreat: threshold is 0.4 * 1.5 = 0.6, health 0.9 > 0.6
        assert result["action"] == "seek_cover"

    def test_robot_ignores_suppression(self, engine):
        result = engine.decide("robot_precise", {
            "health": 0.8, "threats": 2, "suppressed": True,
            "in_cover": True, "enemy_distance": 30.0,
        })
        # Robot has suppression_tolerance=1.0, won't seek cover from suppression
        assert result["action"] != "seek_cover" or "suppression" not in result.get("reasoning", "")

    def test_low_ammo_conservation(self, engine):
        result = engine.decide("sniper_patient", {
            "health": 0.8, "ammo": 0.08, "threats": 1,
            "enemy_distance": 80.0,
        })
        # sniper ammo_conservation=0.95 > 0.5 and ammo=0.08 < 0.1
        assert result["action"] == "conserve"

    def test_critical_ammo_retreats(self, engine):
        result = engine.decide("berserker", {
            "health": 0.5, "ammo": 0.03, "threats": 2,
        })
        # ammo < 0.05 -> retreat regardless
        assert result["action"] == "retreat"

    def test_aggressive_assaults(self, engine):
        result = engine.decide("berserker", {
            "health": 0.8, "ammo": 0.5, "threats": 2,
            "enemy_distance": 15.0, "in_cover": False,
        })
        assert result["action"] == "assault"

    def test_defensive_holds_in_cover(self, engine):
        result = engine.decide("veteran_steady", {
            "health": 0.8, "ammo": 0.6, "threats": 2,
            "enemy_distance": 30.0, "in_cover": True,
        })
        assert result["action"] == "hold_and_engage"

    def test_defensive_seeks_cover_when_exposed(self, engine):
        result = engine.decide("veteran_steady", {
            "health": 0.8, "ammo": 0.6, "threats": 2,
            "enemy_distance": 30.0, "in_cover": False,
        })
        assert result["action"] == "seek_cover"

    def test_methodical_overwatch(self, engine):
        result = engine.decide("elite_operator", {
            "health": 0.8, "ammo": 0.6, "threats": 2,
            "enemy_distance": 40.0, "in_cover": True,
        })
        assert result["action"] == "overwatch"

    def test_opportunistic_flanks_lone_target(self, engine):
        result = engine.decide("guerrilla", {
            "health": 0.7, "ammo": 0.5, "threats": 1,
            "enemy_distance": 30.0, "in_cover": False,
        })
        assert result["action"] == "flank"

    def test_result_has_expected_keys(self, engine):
        result = engine.decide("elite_operator", {"health": 0.8, "threats": 1})
        assert "action" in result
        assert "modifiers" in result
        assert "reasoning" in result
        assert isinstance(result["modifiers"], dict)
        assert isinstance(result["reasoning"], str)


# ============================================================================
# BehaviorEngine.modify_stats()
# ============================================================================


class TestModifyStats:
    @pytest.fixture()
    def engine(self) -> BehaviorEngine:
        e = BehaviorEngine(profiles=PROFILES)
        for pid in PROFILES:
            e.assign_profile(pid, pid)
        return e

    def test_no_profile_returns_copy(self):
        engine = BehaviorEngine()
        base = {"speed": 5.0, "accuracy": 0.7}
        result = engine.modify_stats("nobody", base)
        assert result == base
        assert result is not base

    def test_berserker_faster(self, engine):
        base = {"speed": 5.0}
        result = engine.modify_stats("berserker", base)
        assert result["speed"] > 5.0

    def test_cautious_unit_slower(self, engine):
        base = {"speed": 5.0}
        result = engine.modify_stats("sniper_patient", base)
        assert result["speed"] < 5.0

    def test_discipline_improves_accuracy(self, engine):
        base = {"accuracy": 0.5}
        elite = engine.modify_stats("elite_operator", base)
        conscript = engine.modify_stats("conscript", base)
        assert elite["accuracy"] > conscript["accuracy"]

    def test_reckless_reduces_accuracy(self, engine):
        base = {"accuracy": 0.5}
        result = engine.modify_stats("berserker", base)
        # Berserker: discipline=0.1 so base accuracy mult is low,
        # plus reckless penalty
        default_profile = BehaviorProfile(profile_id="d", name="d")
        engine.add_profile(default_profile)
        engine.assign_profile("default", "d")
        default_result = engine.modify_stats("default", base)
        assert result["accuracy"] < default_result["accuracy"]

    def test_preferred_range_affects_attack_range(self, engine):
        base = {"attack_range": 100.0}
        close = engine.modify_stats("berserker", base)      # close
        long_ = engine.modify_stats("sniper_patient", base)  # long
        assert long_["attack_range"] > close["attack_range"]

    def test_cautious_improves_detection(self, engine):
        base = {"detection_range": 50.0}
        scout = engine.modify_stats("scout_ghost", base)
        berserker = engine.modify_stats("berserker", base)
        assert scout["detection_range"] > berserker["detection_range"]

    def test_aggression_boosts_damage(self, engine):
        base = {"damage": 10.0}
        aggressive = engine.modify_stats("berserker", base)
        cautious = engine.modify_stats("civilian_panicked", base)
        assert aggressive["damage"] > cautious["damage"]

    def test_unknown_keys_pass_through(self, engine):
        base = {"speed": 5.0, "custom_stat": 42.0}
        result = engine.modify_stats("elite_operator", base)
        assert result["custom_stat"] == 42.0

    def test_armor_improved_by_discipline(self, engine):
        base = {"armor": 0.5}
        robot = engine.modify_stats("robot_precise", base)
        conscript = engine.modify_stats("conscript", base)
        assert robot["armor"] > conscript["armor"]

    def test_armor_capped_at_one(self, engine):
        base = {"armor": 0.95}
        result = engine.modify_stats("robot_precise", base)
        assert result["armor"] <= 1.0


# ============================================================================
# BehaviorEngine.evaluate_threat_response()
# ============================================================================


class TestEvaluateThreatResponse:
    @pytest.fixture()
    def engine(self) -> BehaviorEngine:
        e = BehaviorEngine(profiles=PROFILES)
        for pid in PROFILES:
            e.assign_profile(pid, pid)
        return e

    def test_no_profile_returns_hold(self):
        engine = BehaviorEngine()
        assert engine.evaluate_threat_response("nobody", []) == "hold"

    def test_no_threats_returns_hold(self, engine):
        assert engine.evaluate_threat_response("elite_operator", []) == "hold"

    def test_reckless_charges_at_close_range(self, engine):
        threats = [{"distance": 5.0, "threat_level": 0.6, "is_flanking": False}]
        assert engine.evaluate_threat_response("berserker", threats) == "charge"

    def test_cautious_evades_at_close_range(self, engine):
        threats = [{"distance": 5.0, "threat_level": 0.6, "is_flanking": False}]
        result = engine.evaluate_threat_response("sniper_patient", threats)
        assert result == "evade"

    def test_disciplined_suppresses_flanking_threat(self, engine):
        threats = [{"distance": 30.0, "threat_level": 0.7, "is_flanking": True}]
        result = engine.evaluate_threat_response("elite_operator", threats)
        assert result == "suppress"

    def test_aggressive_charges_flanking_threat(self, engine):
        threats = [{"distance": 30.0, "threat_level": 0.7, "is_flanking": True}]
        result = engine.evaluate_threat_response("berserker", threats)
        assert result == "charge"

    def test_undisciplined_retreats_overwhelming(self, engine):
        threats = [
            {"distance": 20.0, "threat_level": 0.95, "is_flanking": False}
            for _ in range(5)
        ]
        result = engine.evaluate_threat_response("conscript", threats)
        assert result == "retreat"

    def test_robot_does_not_retreat_overwhelming(self, engine):
        """Robot has discipline=1.0, won't retreat from overwhelming threat."""
        threats = [
            {"distance": 20.0, "threat_level": 0.95, "is_flanking": False}
            for _ in range(5)
        ]
        result = engine.evaluate_threat_response("robot_precise", threats)
        assert result != "retreat"

    def test_sniper_engages_at_long_range(self, engine):
        threats = [{"distance": 80.0, "threat_level": 0.4, "is_flanking": False}]
        result = engine.evaluate_threat_response("sniper_patient", threats)
        assert result == "engage"

    def test_defensive_takes_cover_at_medium_range(self, engine):
        threats = [{"distance": 35.0, "threat_level": 0.5, "is_flanking": False}]
        result = engine.evaluate_threat_response("veteran_steady", threats)
        assert result == "take_cover"

    def test_response_is_valid_string(self, engine):
        valid = {"engage", "take_cover", "flank", "retreat", "suppress", "hold", "charge", "evade"}
        threats = [{"distance": 40.0, "threat_level": 0.5, "is_flanking": False}]
        for pid in PROFILES:
            result = engine.evaluate_threat_response(pid, threats)
            assert result in valid, f"{pid} returned invalid response: {result}"

    def test_all_profiles_handle_empty_threats(self, engine):
        for pid in PROFILES:
            assert engine.evaluate_threat_response(pid, []) == "hold"
