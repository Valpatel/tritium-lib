# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unit tests for the TacticsEngine tactical decision-making system.

Tests the full decision tree: threat assessment, situation evaluation,
action selection, personality modifiers, and squad coordination.
"""

import math

import pytest

from tritium_lib.sim_engine.ai.tactics import (
    TacticsEngine,
    TacticalAction,
    TacticalSituation,
    ThreatAssessment,
    AIPersonality,
    PERSONALITY_PRESETS,
)


# ===================================================================
# Threat Assessment
# ===================================================================


class TestThreatAssessment:
    def test_empty_enemies(self):
        engine = TacticsEngine()
        result = engine.assess_threats((0.0, 0.0), [], dt=0.1)
        assert result == []

    def test_single_enemy_close(self):
        engine = TacticsEngine()
        enemies = [{"id": "e1", "pos": (5.0, 0.0), "damage": 5.0}]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        assert len(result) == 1
        assert result[0].threat_id == "e1"
        assert result[0].distance == pytest.approx(5.0)
        assert result[0].threat_level > 0.0

    def test_closer_enemy_higher_threat(self):
        engine = TacticsEngine()
        enemies = [
            {"id": "near", "pos": (10.0, 0.0), "damage": 5.0},
            {"id": "far", "pos": (80.0, 0.0), "damage": 5.0},
        ]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        near = next(t for t in result if t.threat_id == "near")
        far = next(t for t in result if t.threat_id == "far")
        assert near.threat_level > far.threat_level

    def test_higher_damage_higher_threat(self):
        engine = TacticsEngine()
        enemies = [
            {"id": "weak", "pos": (20.0, 0.0), "damage": 1.0},
            {"id": "strong", "pos": (20.0, 0.0), "damage": 9.0},
        ]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        weak = next(t for t in result if t.threat_id == "weak")
        strong = next(t for t in result if t.threat_id == "strong")
        assert strong.threat_level > weak.threat_level

    def test_sorted_by_threat_level(self):
        engine = TacticsEngine()
        enemies = [
            {"id": "e1", "pos": (80.0, 0.0), "damage": 1.0},
            {"id": "e2", "pos": (10.0, 0.0), "damage": 8.0},
            {"id": "e3", "pos": (40.0, 0.0), "damage": 3.0},
        ]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        levels = [t.threat_level for t in result]
        assert levels == sorted(levels, reverse=True)

    def test_suppressing_flag(self):
        engine = TacticsEngine()
        enemies = [{"id": "e1", "pos": (20.0, 0.0), "damage": 5.0, "suppressing": True}]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        assert result[0].is_suppressing is True

    def test_threat_level_capped_at_one(self):
        engine = TacticsEngine()
        enemies = [{"id": "e1", "pos": (1.0, 0.0), "damage": 100.0}]
        result = engine.assess_threats((0.0, 0.0), enemies, dt=0.1)
        assert result[0].threat_level <= 1.0


# ===================================================================
# Situation Evaluation
# ===================================================================


class TestSituationEvaluation:
    def _make_threats(self, positions):
        return [
            ThreatAssessment(
                threat_id=f"t{i}",
                position=pos,
                distance=math.hypot(pos[0], pos[1]),
                threat_level=0.5,
                is_flanking=False,
                is_suppressing=False,
                last_seen=0.0,
                estimated_health=1.0,
            )
            for i, pos in enumerate(positions)
        ]

    def test_no_threats_no_cover(self):
        engine = TacticsEngine()
        unit = {"pos": (0.0, 0.0), "health": 1.0, "ammo": 1.0, "morale": 1.0}
        situation = engine.evaluate_situation(unit, [], [], [])
        assert situation.threats == []
        assert situation.allies_nearby == 0
        assert not situation.in_cover

    def test_allies_counted_within_range(self):
        engine = TacticsEngine()
        unit = {"pos": (0.0, 0.0), "health": 1.0, "ammo": 1.0, "morale": 1.0}
        allies = [
            {"pos": (10.0, 0.0)},  # within 30m
            {"pos": (20.0, 0.0)},  # within 30m
            {"pos": (50.0, 0.0)},  # outside 30m
        ]
        situation = engine.evaluate_situation(unit, [], allies, [])
        assert situation.allies_nearby == 2

    def test_unit_health_propagated(self):
        engine = TacticsEngine()
        unit = {"pos": (0.0, 0.0), "health": 0.3, "ammo": 0.5, "morale": 0.6}
        situation = engine.evaluate_situation(unit, [], [], [])
        assert situation.unit_health == pytest.approx(0.3)
        assert situation.unit_ammo == pytest.approx(0.5)
        assert situation.unit_morale == pytest.approx(0.6)


# ===================================================================
# Action Decision
# ===================================================================


class TestActionDecision:
    def _make_situation(self, **overrides):
        defaults = {
            "unit_pos": (0.0, 0.0),
            "unit_health": 1.0,
            "unit_ammo": 1.0,
            "unit_morale": 1.0,
            "threats": [],
            "allies_nearby": 3,
            "in_cover": False,
            "cover_positions": [(10.0, 0.0), (-10.0, 0.0)],
            "has_los_to_threats": [],
            "squad_order": None,
        }
        defaults.update(overrides)
        return TacticalSituation(**defaults)

    def _make_threat(self, distance=20.0, flanking=False, suppressing=False):
        return ThreatAssessment(
            threat_id="t1",
            position=(distance, 0.0),
            distance=distance,
            threat_level=0.5,
            is_flanking=flanking,
            is_suppressing=suppressing,
            last_seen=0.0,
            estimated_health=1.0,
        )

    def test_critical_health_retreats(self):
        """Units with critical health should retreat."""
        engine = TacticsEngine(AIPersonality(discipline=0.5))
        threat = self._make_threat()
        situation = self._make_situation(
            unit_health=0.1,
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "retreat"
        assert "health" in action.reasoning.lower() or "Health" in action.reasoning

    def test_out_of_ammo_retreats(self):
        """Units with no ammo should retreat to resupply."""
        engine = TacticsEngine()
        threat = self._make_threat()
        situation = self._make_situation(
            unit_ammo=0.05,
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "retreat"
        assert "ammo" in action.reasoning.lower() or "Ammo" in action.reasoning

    def test_broken_morale_retreats(self):
        """Units with broken morale should retreat."""
        engine = TacticsEngine(AIPersonality(discipline=0.5))
        threat = self._make_threat()
        situation = self._make_situation(
            unit_morale=0.1,
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "retreat"
        assert "morale" in action.reasoning.lower() or "Morale" in action.reasoning

    def test_no_threats_advance_or_hold(self):
        """Without threats, units should advance or hold."""
        engine = TacticsEngine()
        situation = self._make_situation(threats=[])
        action = engine.decide_action(situation)
        assert action.action_type in ("advance", "hold")

    def test_aggressive_personality_advances(self):
        """Aggressive units advance when no threats."""
        engine = TacticsEngine(AIPersonality(aggression=0.9))
        situation = self._make_situation(threats=[])
        action = engine.decide_action(situation)
        assert action.action_type == "advance"

    def test_cautious_personality_holds(self):
        """Cautious units hold when no threats."""
        engine = TacticsEngine(AIPersonality(aggression=0.2))
        situation = self._make_situation(threats=[])
        action = engine.decide_action(situation)
        assert action.action_type == "hold"

    def test_in_cover_with_los_engages(self):
        """Units in cover with LOS should engage."""
        engine = TacticsEngine()
        threat = self._make_threat()
        situation = self._make_situation(
            in_cover=True,
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "engage"

    def test_flanking_threat_triggers_relocate(self):
        """A flanking threat should cause relocation."""
        engine = TacticsEngine()
        threat = self._make_threat(flanking=True)
        situation = self._make_situation(
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "relocate"

    def test_heavily_outnumbered_retreats_or_suppresses(self):
        """When outnumbered 3:1, unit should suppress or retreat."""
        engine = TacticsEngine()
        threats = [
            self._make_threat(distance=20.0 + i * 5)
            for i in range(4)
        ]
        situation = self._make_situation(
            threats=threats,
            allies_nearby=0,  # alone
            has_los_to_threats=[True] * 4,
        )
        action = engine.decide_action(situation)
        assert action.action_type in ("suppress", "retreat")

    def test_clustered_enemies_grenade(self):
        """Clustered enemies should trigger a grenade throw."""
        engine = TacticsEngine()
        # All enemies at nearly the same spot, not flanking, not suppressing
        threats = []
        for i in range(3):
            threats.append(ThreatAssessment(
                threat_id=f"t{i}",
                position=(30.0 + i, 0.0 + i),  # within 10m of each other
                distance=30.0,
                threat_level=0.5,
                is_flanking=False,
                is_suppressing=False,
                last_seen=0.0,
                estimated_health=1.0,
            ))
        situation = self._make_situation(
            threats=threats,
            allies_nearby=3,  # not outnumbered
            in_cover=False,
            cover_positions=[],  # no cover to seek
            has_los_to_threats=[True, True, True],
        )
        action = engine.decide_action(situation)
        assert action.action_type == "throw_grenade"

    def test_disciplined_unit_resists_morale_break(self):
        """A highly disciplined unit should not retreat at normal morale levels."""
        engine = TacticsEngine(AIPersonality(discipline=0.95))
        threat = self._make_threat()
        # Morale at 0.25 — would trigger retreat for undisciplined units
        # but for discipline=0.95, threshold = 0.3 * (2.0 - 0.95) = 0.315
        # So 0.25 < 0.315, this unit WILL retreat (correct behavior)
        # Let's test with morale=0.35 which should NOT trigger retreat
        situation = self._make_situation(
            unit_morale=0.35,
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.action_type != "retreat", (
            f"Disciplined unit at 35% morale should not retreat, got {action.action_type}"
        )

    def test_action_has_reasoning(self):
        """Every action should include non-empty reasoning."""
        engine = TacticsEngine()
        threat = self._make_threat()
        situation = self._make_situation(
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert action.reasoning
        assert len(action.reasoning) > 5

    def test_action_priority_range(self):
        """Action priority should be in [0, 1]."""
        engine = TacticsEngine()
        threat = self._make_threat()
        situation = self._make_situation(
            threats=[threat],
            has_los_to_threats=[True],
        )
        action = engine.decide_action(situation)
        assert 0.0 <= action.priority <= 1.0


# ===================================================================
# Personality Presets
# ===================================================================


class TestPersonalityPresets:
    def test_all_presets_exist(self):
        expected = {"veteran", "recruit", "berserker", "sniper", "medic", "leader"}
        assert set(PERSONALITY_PRESETS.keys()) == expected

    def test_preset_aggression_ranges(self):
        for name, p in PERSONALITY_PRESETS.items():
            assert 0.0 <= p.aggression <= 1.0, f"{name}.aggression out of range"
            assert 0.0 <= p.discipline <= 1.0, f"{name}.discipline out of range"
            assert 0.0 <= p.teamwork <= 1.0, f"{name}.teamwork out of range"

    def test_berserker_is_most_aggressive(self):
        berserker = PERSONALITY_PRESETS["berserker"]
        for name, p in PERSONALITY_PRESETS.items():
            if name != "berserker":
                assert berserker.aggression >= p.aggression

    def test_medic_is_least_aggressive(self):
        medic = PERSONALITY_PRESETS["medic"]
        for name, p in PERSONALITY_PRESETS.items():
            if name != "medic":
                assert medic.aggression <= p.aggression


# ===================================================================
# Squad Coordination
# ===================================================================


class TestSquadCoordination:
    def test_empty_squad(self):
        engine = TacticsEngine()
        result = engine.decide_squad_action({"units": []})
        assert result == []

    def test_no_threats_all_advance(self):
        engine = TacticsEngine()
        squad_sit = {
            "units": [
                {"id": "u1", "pos": (0.0, 0.0), "health": 1.0, "ammo": 1.0,
                 "morale": 1.0, "role": "infantry"},
                {"id": "u2", "pos": (5.0, 0.0), "health": 1.0, "ammo": 1.0,
                 "morale": 1.0, "role": "infantry"},
            ],
            "threats": [],
            "cover_positions": [],
            "objective": (50.0, 50.0),
        }
        result = engine.decide_squad_action(squad_sit)
        assert len(result) == 2
        for uid, action in result:
            assert action.action_type == "advance"

    def test_medic_heals_wounded(self):
        engine = TacticsEngine()
        squad_sit = {
            "units": [
                {"id": "m1", "pos": (0.0, 0.0), "health": 1.0, "ammo": 1.0,
                 "morale": 1.0, "role": "medic"},
                {"id": "u1", "pos": (5.0, 0.0), "health": 0.3, "ammo": 1.0,
                 "morale": 1.0, "role": "infantry"},
            ],
            "threats": [],
            "cover_positions": [],
            "objective": (50.0, 50.0),
        }
        result = engine.decide_squad_action(squad_sit)
        medic_action = next(a for uid, a in result if uid == "m1")
        assert medic_action.action_type == "heal_ally"

    def test_fire_teams_with_threats(self):
        """With threats, squad should split into suppressors and flankers."""
        engine = TacticsEngine()
        threat = ThreatAssessment(
            threat_id="e1", position=(50.0, 0.0), distance=50.0,
            threat_level=0.7, is_flanking=False, is_suppressing=False,
            last_seen=0.0, estimated_health=1.0,
        )
        squad_sit = {
            "units": [
                {"id": f"u{i}", "pos": (float(i * 3), 0.0), "health": 1.0,
                 "ammo": 1.0, "morale": 1.0, "role": "infantry"}
                for i in range(4)
            ],
            "threats": [threat],
            "cover_positions": [(20.0, 10.0)],
            "objective": (100.0, 0.0),
        }
        result = engine.decide_squad_action(squad_sit)
        action_types = {a.action_type for _, a in result}
        # Should have both suppress and flank actions
        assert "suppress" in action_types, "Squad should have suppressors"
        assert "flank" in action_types, "Squad should have flankers"


# ===================================================================
# Environment + Tactics interaction (unit tests)
# ===================================================================


class TestEnvironmentModifiers:
    """Test that environment modifiers produce valid combat-affecting values."""

    def test_clear_day_full_visibility(self):
        from tritium_lib.sim_engine.environment import Environment, TimeOfDay, Weather
        env = Environment(time=TimeOfDay(12.0))  # noon, clear
        assert env.visibility() > 0.8

    def test_night_reduced_visibility(self):
        from tritium_lib.sim_engine.environment import Environment, TimeOfDay
        env = Environment(time=TimeOfDay(0.0))  # midnight
        assert env.visibility() < 0.5

    def test_fog_reduces_visibility(self):
        from tritium_lib.sim_engine.environment import (
            Environment, TimeOfDay, WeatherSimulator, Weather, WeatherState,
        )
        ws = WeatherSimulator(initial=Weather.FOG)
        ws.state.intensity = 0.8
        env = Environment(time=TimeOfDay(12.0), weather=ws)
        assert env.visibility() < 0.3

    def test_storm_reduces_accuracy(self):
        from tritium_lib.sim_engine.environment import (
            Environment, TimeOfDay, WeatherSimulator, Weather,
        )
        ws = WeatherSimulator(initial=Weather.STORM)
        ws.state.intensity = 0.9
        ws.state.wind_speed = 20.0
        env = Environment(time=TimeOfDay(12.0), weather=ws)
        assert env.accuracy_modifier() < 0.7

    def test_snow_slows_movement(self):
        from tritium_lib.sim_engine.environment import (
            Environment, TimeOfDay, WeatherSimulator, Weather,
        )
        ws = WeatherSimulator(initial=Weather.SNOW)
        ws.state.intensity = 0.8
        env = Environment(time=TimeOfDay(12.0), weather=ws)
        assert env.movement_speed_modifier() < 0.7

    def test_describe_returns_string(self):
        from tritium_lib.sim_engine.environment import Environment
        env = Environment()
        desc = env.describe()
        assert isinstance(desc, str)
        assert len(desc) > 5

    def test_snapshot_has_all_keys(self):
        from tritium_lib.sim_engine.environment import Environment
        env = Environment()
        snap = env.snapshot()
        required = {
            "hour", "is_day", "is_night", "light_level", "sun_angle",
            "weather", "intensity", "wind_speed", "wind_direction",
            "temperature", "humidity", "visibility",
            "movement_modifier", "accuracy_modifier", "detection_range_modifier",
        }
        assert required.issubset(set(snap.keys())), (
            f"Missing keys: {required - set(snap.keys())}"
        )
