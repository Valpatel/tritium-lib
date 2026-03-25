# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.rules — configurable IF-THEN automation rules engine."""

from __future__ import annotations

import json
import time

import pytest

from tritium_lib.rules import (
    Action,
    ActionType,
    AndCondition,
    Condition,
    ConditionBase,
    NotCondition,
    OrCondition,
    Rule,
    RuleEngine,
    RuleResult,
    RuleSet,
    condition_from_dict,
    register_condition,
)


# ---------------------------------------------------------------------------
# Helpers — reusable state fixtures
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    """Create a base tracking state dict for testing."""
    state = {
        "timestamp": time.time(),
        "targets": {
            "t1": {
                "zone_id": "alpha",
                "zone_ids": ["alpha", "perimeter"],
                "threat_level": 0.8,
                "alliance": "hostile",
                "first_seen": time.time() - 600,
                "zone_entered_at": time.time() - 600,
            },
            "t2": {
                "zone_id": "beta",
                "zone_ids": ["beta"],
                "threat_level": 0.2,
                "alliance": "friendly",
                "first_seen": time.time() - 30,
                "zone_entered_at": time.time() - 30,
            },
        },
        "zones": {
            "alpha": {"target_count": 3, "zone_type": "restricted"},
            "beta": {"target_count": 1, "zone_type": "public"},
        },
        "sensors": {
            "cam_north": {"status": "online", "last_seen": time.time()},
            "cam_south": {"status": "offline", "last_seen": time.time() - 600},
        },
    }
    state.update(overrides)
    return state


# ===========================================================================
# Test Condition
# ===========================================================================

class TestCondition:
    """Tests for the Condition class and built-in evaluators."""

    def test_target_enters_zone_true(self):
        state = _base_state()
        cond = Condition("target_enters_zone", zone_id="alpha")
        assert cond.evaluate(state) is True

    def test_target_enters_zone_false(self):
        state = _base_state()
        cond = Condition("target_enters_zone", zone_id="gamma")
        assert cond.evaluate(state) is False

    def test_target_enters_zone_via_zone_ids(self):
        state = _base_state()
        cond = Condition("target_enters_zone", zone_id="perimeter")
        assert cond.evaluate(state) is True

    def test_threat_level_above_true(self):
        state = _base_state()
        cond = Condition("threat_level_above", level=0.5)
        assert cond.evaluate(state) is True

    def test_threat_level_above_false(self):
        state = _base_state()
        cond = Condition("threat_level_above", level=0.9)
        assert cond.evaluate(state) is False

    def test_target_count_in_zone_exceeds_true(self):
        state = _base_state()
        # t1 is in alpha, only 1 target in zone
        cond = Condition("target_count_in_zone_exceeds", zone_id="alpha", count=0)
        assert cond.evaluate(state) is True

    def test_target_count_in_zone_exceeds_false(self):
        state = _base_state()
        cond = Condition("target_count_in_zone_exceeds", zone_id="alpha", count=5)
        assert cond.evaluate(state) is False

    def test_target_dwell_exceeds_true(self):
        state = _base_state()
        # t1 entered alpha 600s ago, check for > 5 minutes
        cond = Condition("target_dwell_exceeds", zone_id="alpha", minutes=5)
        assert cond.evaluate(state) is True

    def test_target_dwell_exceeds_false(self):
        state = _base_state()
        # t2 entered beta 30s ago, check for > 5 minutes
        cond = Condition("target_dwell_exceeds", zone_id="beta", minutes=5)
        assert cond.evaluate(state) is False

    def test_sensor_offline_by_status(self):
        state = _base_state()
        cond = Condition("sensor_offline", sensor_id="cam_south", minutes=1)
        assert cond.evaluate(state) is True

    def test_sensor_offline_by_last_seen(self):
        state = _base_state()
        # cam_south last seen 600s ago
        cond = Condition("sensor_offline", sensor_id="cam_south", minutes=5)
        assert cond.evaluate(state) is True

    def test_sensor_offline_false(self):
        state = _base_state()
        cond = Condition("sensor_offline", sensor_id="cam_north", minutes=5)
        assert cond.evaluate(state) is False

    def test_sensor_offline_unknown_sensor(self):
        state = _base_state()
        cond = Condition("sensor_offline", sensor_id="doesnt_exist", minutes=1)
        assert cond.evaluate(state) is False

    def test_target_alliance_is_true(self):
        state = _base_state()
        cond = Condition("target_alliance_is", alliance="hostile")
        assert cond.evaluate(state) is True

    def test_target_alliance_is_false(self):
        state = _base_state()
        cond = Condition("target_alliance_is", alliance="neutral")
        assert cond.evaluate(state) is False

    def test_field_compare_eq(self):
        state = _base_state()
        cond = Condition("field_compare", field_path="zones.alpha.zone_type", operator="eq", value="restricted")
        assert cond.evaluate(state) is True

    def test_field_compare_gt(self):
        state = _base_state()
        cond = Condition("field_compare", field_path="zones.alpha.target_count", operator="gt", value=2)
        assert cond.evaluate(state) is True

    def test_field_compare_lt(self):
        state = _base_state()
        cond = Condition("field_compare", field_path="zones.beta.target_count", operator="lt", value=5)
        assert cond.evaluate(state) is True

    def test_field_compare_contains(self):
        state = _base_state()
        cond = Condition("field_compare", field_path="zones.alpha.zone_type", operator="contains", value="restrict")
        assert cond.evaluate(state) is True

    def test_field_compare_missing_path(self):
        state = _base_state()
        cond = Condition("field_compare", field_path="zones.gamma.target_count", operator="eq", value=0)
        assert cond.evaluate(state) is False

    def test_unknown_condition_returns_false(self):
        state = _base_state()
        cond = Condition("nonexistent_condition", foo="bar")
        assert cond.evaluate(state) is False

    def test_condition_serialization(self):
        cond = Condition("target_enters_zone", zone_id="alpha")
        d = cond.to_dict()
        assert d["type"] == "condition"
        assert d["name"] == "target_enters_zone"
        assert d["params"]["zone_id"] == "alpha"
        restored = Condition.from_dict(d)
        assert restored.name == cond.name
        assert restored.params == cond.params


# ===========================================================================
# Test Composite Conditions
# ===========================================================================

class TestCompositeConditions:
    """Tests for AND, OR, NOT condition combinators."""

    def test_and_condition_both_true(self):
        state = _base_state()
        cond = AndCondition([
            Condition("target_enters_zone", zone_id="alpha"),
            Condition("threat_level_above", level=0.5),
        ])
        assert cond.evaluate(state) is True

    def test_and_condition_one_false(self):
        state = _base_state()
        cond = AndCondition([
            Condition("target_enters_zone", zone_id="alpha"),
            Condition("target_enters_zone", zone_id="gamma"),
        ])
        assert cond.evaluate(state) is False

    def test_or_condition_one_true(self):
        state = _base_state()
        cond = OrCondition([
            Condition("target_enters_zone", zone_id="gamma"),
            Condition("threat_level_above", level=0.5),
        ])
        assert cond.evaluate(state) is True

    def test_or_condition_both_false(self):
        state = _base_state()
        cond = OrCondition([
            Condition("target_enters_zone", zone_id="gamma"),
            Condition("threat_level_above", level=0.99),
        ])
        assert cond.evaluate(state) is False

    def test_not_condition_inverts(self):
        state = _base_state()
        cond = NotCondition(Condition("target_enters_zone", zone_id="gamma"))
        assert cond.evaluate(state) is True  # gamma doesn't exist, so NOT(False) = True

    def test_not_condition_inverts_true(self):
        state = _base_state()
        cond = NotCondition(Condition("target_enters_zone", zone_id="alpha"))
        assert cond.evaluate(state) is False  # alpha exists, so NOT(True) = False

    def test_operator_overloads_and(self):
        c1 = Condition("target_enters_zone", zone_id="alpha")
        c2 = Condition("threat_level_above", level=0.5)
        combined = c1 & c2
        assert isinstance(combined, AndCondition)
        state = _base_state()
        assert combined.evaluate(state) is True

    def test_operator_overloads_or(self):
        c1 = Condition("target_enters_zone", zone_id="gamma")
        c2 = Condition("threat_level_above", level=0.5)
        combined = c1 | c2
        assert isinstance(combined, OrCondition)
        state = _base_state()
        assert combined.evaluate(state) is True

    def test_operator_overloads_not(self):
        c1 = Condition("target_enters_zone", zone_id="gamma")
        negated = ~c1
        assert isinstance(negated, NotCondition)
        state = _base_state()
        assert negated.evaluate(state) is True

    def test_nested_composite(self):
        """Test deeply nested AND(OR(...), NOT(...))."""
        state = _base_state()
        cond = AndCondition([
            OrCondition([
                Condition("target_enters_zone", zone_id="gamma"),
                Condition("target_enters_zone", zone_id="alpha"),
            ]),
            NotCondition(
                Condition("target_alliance_is", alliance="neutral")
            ),
        ])
        assert cond.evaluate(state) is True

    def test_composite_serialization_roundtrip(self):
        cond = AndCondition([
            OrCondition([
                Condition("target_enters_zone", zone_id="alpha"),
                Condition("threat_level_above", level=0.5),
            ]),
            NotCondition(Condition("sensor_offline", sensor_id="cam_north", minutes=1)),
        ])
        d = cond.to_dict()
        restored = condition_from_dict(d)
        state = _base_state()
        assert restored.evaluate(state) == cond.evaluate(state)


# ===========================================================================
# Test Action
# ===========================================================================

class TestAction:
    """Tests for the Action class."""

    def test_action_creation(self):
        action = Action(ActionType.SEND_ALERT, message="Breach detected!")
        assert action.action_type == ActionType.SEND_ALERT
        assert action.params["message"] == "Breach detected!"

    def test_action_serialization(self):
        action = Action(ActionType.DISPATCH_UNIT, unit_id="drone_1", zone_id="alpha")
        d = action.to_dict()
        assert d["action_type"] == "dispatch_unit"
        assert d["params"]["unit_id"] == "drone_1"
        restored = Action.from_dict(d)
        assert restored.action_type == ActionType.DISPATCH_UNIT
        assert restored.params == action.params

    def test_action_equality(self):
        a1 = Action(ActionType.LOG, message="test")
        a2 = Action(ActionType.LOG, message="test")
        assert a1 == a2

    def test_action_default(self):
        action = Action()
        assert action.action_type == ActionType.LOG
        assert action.params == {}


# ===========================================================================
# Test Rule
# ===========================================================================

class TestRule:
    """Tests for the Rule class and fluent API."""

    def test_rule_basic_match(self):
        rule = Rule("r1", condition=Condition("target_enters_zone", zone_id="alpha"))
        state = _base_state()
        assert rule.matches(state) is True

    def test_rule_no_condition_always_matches(self):
        rule = Rule("r1")
        state = _base_state()
        assert rule.matches(state) is True

    def test_rule_disabled_never_matches(self):
        rule = Rule("r1", enabled=False)
        state = _base_state()
        assert rule.matches(state) is False

    def test_rule_fluent_api(self):
        rule = (
            Rule("perimeter_breach", name="Perimeter Breach")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Breach!"))
            .then(Action(ActionType.START_RECORDING, sensor_id="cam_north"))
            .with_priority(10)
            .with_cooldown(30)
            .with_tags(["perimeter", "security"])
            .with_max_fires(100)
            .with_description("Fires when targets enter perimeter zone")
        )
        assert rule.name == "Perimeter Breach"
        assert rule.priority == 10
        assert rule.cooldown_seconds == 30
        assert len(rule.actions) == 2
        assert rule.tags == ["perimeter", "security"]
        assert rule.max_fires == 100
        assert rule.description == "Fires when targets enter perimeter zone"

    def test_rule_fluent_when_combines_with_and(self):
        rule = (
            Rule("r1")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .when(Condition("threat_level_above", level=0.5))
        )
        assert isinstance(rule.condition, AndCondition)
        state = _base_state()
        assert rule.matches(state) is True

    def test_rule_cooldown(self):
        rule = Rule("r1", cooldown_seconds=60)
        rule.record_firing(now=time.time())
        assert rule.is_cooled_down() is False
        # Simulate past firing
        rule.last_fired_at = time.time() - 120
        assert rule.is_cooled_down() is True

    def test_rule_max_fires(self):
        rule = Rule(
            "r1",
            max_fires=2,
            condition=Condition("target_enters_zone", zone_id="alpha"),
        )
        state = _base_state()
        assert rule.matches(state) is True
        rule.record_firing()
        assert rule.matches(state) is True
        rule.record_firing()
        assert rule.matches(state) is False  # max_fires reached

    def test_rule_chaining_setup(self):
        rule = Rule("r1").chains("r2", "r3")
        assert rule.chains_to == ["r2", "r3"]

    def test_rule_serialization_roundtrip(self):
        rule = (
            Rule("test_rule", name="Test Rule")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Alert!"))
            .with_priority(5)
            .with_cooldown(30)
            .with_tags(["test"])
            .chains("r2")
            .with_max_fires(10)
            .with_description("A test rule")
        )
        d = rule.to_dict()
        restored = Rule.from_dict(d)
        assert restored.rule_id == "test_rule"
        assert restored.name == "Test Rule"
        assert restored.priority == 5
        assert restored.cooldown_seconds == 30
        assert restored.tags == ["test"]
        assert restored.chains_to == ["r2"]
        assert restored.max_fires == 10
        assert restored.description == "A test rule"
        assert len(restored.actions) == 1

    def test_rule_record_firing_increments(self):
        rule = Rule("r1")
        assert rule.fire_count == 0
        rule.record_firing()
        assert rule.fire_count == 1
        rule.record_firing()
        assert rule.fire_count == 2
        assert rule.last_fired_at > 0


# ===========================================================================
# Test RuleSet
# ===========================================================================

class TestRuleSet:
    """Tests for the RuleSet class."""

    def test_ruleset_add_and_get(self):
        rs = RuleSet("defense", name="Perimeter Defense")
        r1 = Rule("r1", priority=10)
        r2 = Rule("r2", priority=5)
        rs.add_rule(r1)
        rs.add_rule(r2)
        assert rs.count() == 2
        assert rs.get_rule("r1") is r1
        assert rs.get_rule("nonexistent") is None

    def test_ruleset_remove(self):
        rs = RuleSet("defense")
        rs.add_rule(Rule("r1"))
        assert rs.remove_rule("r1") is True
        assert rs.remove_rule("r1") is False
        assert rs.count() == 0

    def test_ruleset_priority_ordering(self):
        rs = RuleSet("defense")
        rs.add_rule(Rule("low", priority=1))
        rs.add_rule(Rule("high", priority=10))
        rs.add_rule(Rule("mid", priority=5))
        rules = rs.get_rules()
        assert rules[0].rule_id == "high"
        assert rules[1].rule_id == "mid"
        assert rules[2].rule_id == "low"

    def test_ruleset_serialization_roundtrip(self):
        rs = RuleSet("defense", name="Defense", description="Perimeter rules", tags=["security"])
        rs.add_rule(
            Rule("r1", name="Rule 1", priority=5)
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Alert!"))
        )
        d = rs.to_dict()
        restored = RuleSet.from_dict(d)
        assert restored.ruleset_id == "defense"
        assert restored.name == "Defense"
        assert restored.description == "Perimeter rules"
        assert restored.tags == ["security"]
        assert restored.count() == 1
        assert restored.get_rule("r1") is not None


# ===========================================================================
# Test RuleEngine
# ===========================================================================

class TestRuleEngine:
    """Tests for the RuleEngine class."""

    def test_engine_evaluate_fires_matching(self):
        engine = RuleEngine()
        engine.add_rule(
            Rule("r1")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Zone alert"))
        )
        state = _base_state()
        fired = engine.evaluate(state)
        assert len(fired) == 1
        assert fired[0].rule_id == "r1"

    def test_engine_evaluate_no_match(self):
        engine = RuleEngine()
        engine.add_rule(
            Rule("r1")
            .when(Condition("target_enters_zone", zone_id="gamma"))
            .then(Action(ActionType.SEND_ALERT, message="Zone alert"))
        )
        state = _base_state()
        fired = engine.evaluate(state)
        assert len(fired) == 0

    def test_engine_priority_ordering(self):
        engine = RuleEngine()
        fired_order = []

        def handler(action, state):
            fired_order.append(action.params.get("order"))

        engine.register_action_handler(ActionType.LOG, handler)
        engine.add_rule(Rule("low", priority=1).then(Action(ActionType.LOG, order="low")))
        engine.add_rule(Rule("high", priority=10).then(Action(ActionType.LOG, order="high")))
        engine.add_rule(Rule("mid", priority=5).then(Action(ActionType.LOG, order="mid")))

        engine.evaluate(_base_state())
        assert fired_order == ["high", "mid", "low"]

    def test_engine_action_handler(self):
        engine = RuleEngine()
        captured = []

        def alert_handler(action, state):
            captured.append(action.params.get("message"))

        engine.register_action_handler(ActionType.SEND_ALERT, alert_handler)
        engine.add_rule(
            Rule("r1").then(Action(ActionType.SEND_ALERT, message="Hello"))
        )
        engine.evaluate(_base_state())
        assert captured == ["Hello"]

    def test_engine_ruleset_evaluation(self):
        engine = RuleEngine()
        rs = RuleSet("defense", name="Defense")
        rs.add_rule(
            Rule("r1")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Zone alert"))
        )
        engine.add_ruleset(rs)
        fired = engine.evaluate(_base_state())
        assert len(fired) == 1
        assert fired[0].ruleset_id == "defense"

    def test_engine_disabled_ruleset_skipped(self):
        engine = RuleEngine()
        rs = RuleSet("defense", enabled=False)
        rs.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.add_ruleset(rs)
        fired = engine.evaluate(_base_state())
        assert len(fired) == 0

    def test_engine_rule_chaining(self):
        engine = RuleEngine()
        engine.add_rule(
            Rule("trigger")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.LOG, message="Trigger"))
            .chains("response")
        )
        engine.add_rule(
            Rule("response")
            .then(Action(ActionType.SEND_ALERT, message="Response"))
        )
        fired = engine.evaluate(_base_state())
        rule_ids = [r.rule_id for r in fired]
        assert "trigger" in rule_ids
        assert "response" in rule_ids

    def test_engine_chain_depth_limit(self):
        """Circular chain should be capped by max_chain_depth."""
        engine = RuleEngine(max_chain_depth=3)
        # Create a circular chain: a -> b -> a (infinite loop).
        # Without depth limiting this would recurse forever.
        engine.add_rule(
            Rule("a", max_fires=100)
            .chains("b")
            .then(Action(ActionType.LOG, tag="a"))
        )
        engine.add_rule(
            Rule("b", max_fires=100)
            .chains("a")
            .then(Action(ActionType.LOG, tag="b"))
        )

        fired = engine.evaluate(_base_state())
        rule_ids = [r.rule_id for r in fired]
        assert "a" in rule_ids
        assert "b" in rule_ids
        # Depth limit prevents infinite recursion. With max_chain_depth=3:
        # depth 0: a fires, b fires (top-level)
        # depth 1: a chains to b, b chains to a
        # depth 2: b chains to a, a chains to b
        # depth 3: a chains to b, b chains to a
        # depth 4: STOPPED
        # Total firings should be bounded, not infinite.
        assert len(fired) < 50  # generous bound; without limit would be infinite

    def test_engine_history(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.evaluate(_base_state())
        history = engine.get_history()
        assert len(history) == 1
        assert history[0].rule_id == "r1"

    def test_engine_history_filtering(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.add_rule(Rule("r2").then(Action(ActionType.LOG)))
        engine.evaluate(_base_state())
        assert len(engine.get_history(rule_id="r1")) == 1
        assert len(engine.get_history(rule_id="r2")) == 1
        assert len(engine.get_history(rule_id="r3")) == 0

    def test_engine_stats(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.evaluate(_base_state())
        stats = engine.get_stats()
        assert stats["total_standalone_rules"] == 1
        assert stats["total_evaluations"] == 1
        assert stats["total_rules_fired"] == 1
        assert stats["total_actions_executed"] == 1

    def test_engine_clear_history(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.evaluate(_base_state())
        count = engine.clear_history()
        assert count == 1
        assert len(engine.get_history()) == 0

    def test_engine_reset(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1"))
        engine.add_ruleset(RuleSet("rs1"))
        engine.evaluate(_base_state())
        engine.reset()
        assert engine.get_stats()["total_standalone_rules"] == 0
        assert engine.get_stats()["total_rulesets"] == 0
        assert engine.get_stats()["total_evaluations"] == 0

    def test_engine_reset_counters(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        engine.evaluate(_base_state())
        engine.reset_counters()
        stats = engine.get_stats()
        assert stats["total_evaluations"] == 0
        assert stats["total_rules_fired"] == 0

    def test_engine_enable_disable_rule(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1"))
        assert engine.disable_rule("r1") is True
        assert engine.get_rule("r1").enabled is False
        assert engine.enable_rule("r1") is True
        assert engine.get_rule("r1").enabled is True
        assert engine.disable_rule("nonexistent") is False

    def test_engine_enable_disable_ruleset(self):
        engine = RuleEngine()
        engine.add_ruleset(RuleSet("rs1"))
        assert engine.disable_ruleset("rs1") is True
        assert engine.get_ruleset("rs1").enabled is False
        assert engine.enable_ruleset("rs1") is True
        assert engine.get_ruleset("rs1").enabled is True
        assert engine.disable_ruleset("nonexistent") is False

    def test_engine_remove_rule(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1"))
        assert engine.remove_rule("r1") is True
        assert engine.remove_rule("r1") is False
        assert engine.get_rule("r1") is None

    def test_engine_remove_ruleset(self):
        engine = RuleEngine()
        engine.add_ruleset(RuleSet("rs1"))
        assert engine.remove_ruleset("rs1") is True
        assert engine.remove_ruleset("rs1") is False

    def test_engine_get_rule_from_ruleset(self):
        engine = RuleEngine()
        rs = RuleSet("rs1")
        rs.add_rule(Rule("r1"))
        engine.add_ruleset(rs)
        assert engine.get_rule("r1") is not None

    def test_engine_json_export_import(self):
        engine = RuleEngine()
        engine.add_rule(
            Rule("r1", name="Test Rule")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Alert!"))
            .with_priority(5)
        )
        rs = RuleSet("defense", name="Defense")
        rs.add_rule(
            Rule("r2", priority=3).then(Action(ActionType.LOG))
        )
        engine.add_ruleset(rs)

        exported = engine.export_json()
        data = json.loads(exported)
        assert len(data["rules"]) == 1
        assert len(data["rulesets"]) == 1

        engine2 = RuleEngine()
        counts = engine2.import_json(exported)
        assert counts["rules"] == 1
        assert counts["rulesets"] == 1
        assert engine2.get_rule("r1") is not None
        assert engine2.get_ruleset("defense") is not None

    def test_engine_multiple_actions_per_rule(self):
        engine = RuleEngine()
        captured = []

        def handler(action, state):
            captured.append(action.action_type.value)

        engine.register_action_handler(ActionType.SEND_ALERT, handler)
        engine.register_action_handler(ActionType.START_RECORDING, handler)
        engine.add_rule(
            Rule("r1")
            .then(Action(ActionType.SEND_ALERT, message="Alert"))
            .then(Action(ActionType.START_RECORDING, sensor_id="cam"))
        )
        engine.evaluate(_base_state())
        assert "send_alert" in captured
        assert "start_recording" in captured

    def test_engine_cooldown_suppression(self):
        engine = RuleEngine()
        rule = (
            Rule("r1", cooldown_seconds=60)
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.LOG))
        )
        engine.add_rule(rule)

        state = _base_state()
        fired1 = engine.evaluate(state)
        assert len(fired1) == 1

        fired2 = engine.evaluate(state)
        assert len(fired2) == 0  # cooldown suppression

    def test_engine_timestamp_auto_injected(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1").then(Action(ActionType.LOG)))
        # State without timestamp
        fired = engine.evaluate({"targets": {}, "zones": {}, "sensors": {}})
        assert len(fired) == 1
        assert fired[0].timestamp > 0


# ===========================================================================
# Test Custom Condition Registration
# ===========================================================================

class TestCustomConditions:
    """Tests for registering custom condition evaluators."""

    def test_register_custom_condition(self):
        def custom_eval(state, params):
            return state.get("custom_flag") == params.get("expected")

        register_condition("custom_flag_check", custom_eval)
        cond = Condition("custom_flag_check", expected=True)
        assert cond.evaluate({"custom_flag": True}) is True
        assert cond.evaluate({"custom_flag": False}) is False

    def test_field_compare_regex(self):
        state = {"data": {"name": "target_alpha_001"}}
        cond = Condition("field_compare", field_path="data.name", operator="regex", value=r"alpha_\d+")
        assert cond.evaluate(state) is True

    def test_field_compare_in(self):
        state = {"data": {"status": "active"}}
        cond = Condition("field_compare", field_path="data.status", operator="in", value=["active", "pending"])
        assert cond.evaluate(state) is True

    def test_field_compare_neq(self):
        state = {"data": {"count": 5}}
        cond = Condition("field_compare", field_path="data.count", operator="neq", value=3)
        assert cond.evaluate(state) is True


# ===========================================================================
# Test End-to-End Scenario
# ===========================================================================

class TestEndToEndScenario:
    """Integration-style tests for realistic rule configurations."""

    def test_perimeter_defense_scenario(self):
        """Perimeter defense ruleset with multiple rules."""
        engine = RuleEngine()
        alerts = []
        dispatches = []

        engine.register_action_handler(
            ActionType.SEND_ALERT,
            lambda a, s: alerts.append(a.params.get("message")),
        )
        engine.register_action_handler(
            ActionType.DISPATCH_UNIT,
            lambda a, s: dispatches.append(a.params.get("unit_id")),
        )

        rs = RuleSet("perimeter_defense", name="Perimeter Defense")
        rs.add_rule(
            Rule("detect_intrusion", priority=10)
            .when(
                Condition("target_enters_zone", zone_id="alpha")
                & Condition("target_alliance_is", alliance="hostile")
            )
            .then(Action(ActionType.SEND_ALERT, message="Hostile in perimeter!"))
            .then(Action(ActionType.DISPATCH_UNIT, unit_id="drone_1"))
        )
        rs.add_rule(
            Rule("high_threat", priority=5)
            .when(Condition("threat_level_above", level=0.7))
            .then(Action(ActionType.SEND_ALERT, message="High threat detected"))
        )
        engine.add_ruleset(rs)

        fired = engine.evaluate(_base_state())
        assert len(fired) == 2
        assert "Hostile in perimeter!" in alerts
        assert "High threat detected" in alerts
        assert "drone_1" in dispatches

    def test_vip_protection_scenario(self):
        """VIP protection with sensor monitoring and crowd control."""
        engine = RuleEngine()

        rs = RuleSet("vip_protection", name="VIP Protection")
        rs.add_rule(
            Rule("crowd_alert", priority=10)
            .when(Condition("target_count_in_zone_exceeds", zone_id="alpha", count=0))
            .then(Action(ActionType.SEND_ALERT, message="Crowding in VIP zone"))
        )
        rs.add_rule(
            Rule("sensor_check", priority=8)
            .when(Condition("sensor_offline", sensor_id="cam_south", minutes=5))
            .then(Action(ActionType.SEND_ALERT, message="Security camera down"))
            .then(Action(ActionType.DISPATCH_UNIT, unit_id="tech_1"))
        )
        engine.add_ruleset(rs)

        fired = engine.evaluate(_base_state())
        assert len(fired) == 2
        rule_ids = {r.rule_id for r in fired}
        assert "crowd_alert" in rule_ids
        assert "sensor_check" in rule_ids
