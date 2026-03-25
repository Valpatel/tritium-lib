# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.rules — configurable IF-THEN automation rules engine.

A general-purpose rules engine where users define IF-THEN rules over
tracking data.  Rules are composed of boolean conditions (with AND/OR/NOT
combinators) and actions to execute when conditions are satisfied.  Rules
are JSON-serializable, support priority ordering, and can be grouped into
named RuleSets.

This module is distinct from ``tritium_lib.alerting`` — alerting generates
notifications, while the rules engine executes arbitrary actions (dispatch
unit, start recording, generate report, etc.).

Quick start::

    from tritium_lib.rules import (
        Rule, RuleEngine, RuleSet,
        Condition, AndCondition, OrCondition, NotCondition,
        Action, ActionType,
    )

    # DSL-like rule building
    rule = (
        Rule("perimeter_breach")
        .when(Condition("target_enters_zone", zone_id="perimeter_alpha"))
        .then(Action(ActionType.SEND_ALERT, message="Perimeter breached!"))
        .then(Action(ActionType.START_RECORDING, sensor_id="cam_north"))
        .with_priority(10)
    )

    engine = RuleEngine()
    engine.add_rule(rule)

    # Evaluate against current state
    fired = engine.evaluate({
        "targets": {"t1": {"zone_id": "perimeter_alpha", "threat_level": 0.8}},
        "zones": {"perimeter_alpha": {"target_count": 3}},
        "sensors": {"cam_north": {"status": "online", "last_seen": 1711000000}},
    })

Architecture
------------
- **Condition** — a single boolean predicate over state data
- **AndCondition** / **OrCondition** / **NotCondition** — boolean combinators
- **Action** — what to do when a rule fires (send_alert, dispatch_unit, etc.)
- **ActionType** — enumeration of built-in action types
- **Rule** — IF condition THEN action(s), with priority and metadata
- **RuleSet** — named collection of rules (e.g. "perimeter_defense")
- **RuleEngine** — evaluates rules against current state, tracks history
- **RuleResult** — record of a rule firing with context
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("tritium.rules")


# ---------------------------------------------------------------------------
# ActionType — enumeration of built-in action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Built-in action types the rules engine can trigger."""
    SEND_ALERT = "send_alert"
    DISPATCH_UNIT = "dispatch_unit"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    GENERATE_REPORT = "generate_report"
    PUBLISH_EVENT = "publish_event"
    LOG = "log"
    ESCALATE = "escalate"
    SET_THREAT_LEVEL = "set_threat_level"
    LOCK_ZONE = "lock_zone"
    UNLOCK_ZONE = "unlock_zone"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Action — what to do when a rule fires
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """An action to execute when a rule fires.

    Attributes
    ----------
    action_type:
        The type of action to perform.
    params:
        Action-specific parameters (message, sensor_id, zone_id, etc.).
    """
    action_type: ActionType = ActionType.LOG
    params: dict[str, Any] = field(default_factory=dict)

    def __init__(self, action_type: ActionType = ActionType.LOG, **kwargs: Any) -> None:
        self.action_type = action_type
        self.params = dict(kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "action_type": self.action_type.value,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Deserialize from a plain dict."""
        at = data.get("action_type", "log")
        if isinstance(at, str):
            at = ActionType(at)
        action = cls.__new__(cls)
        action.action_type = at
        action.params = dict(data.get("params", {}))
        return action

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.action_type == other.action_type and self.params == other.params

    def __repr__(self) -> str:
        return f"Action({self.action_type.value!r}, {self.params})"


# ---------------------------------------------------------------------------
# Condition — boolean predicates over state
# ---------------------------------------------------------------------------

class ConditionBase:
    """Abstract base for all condition types.

    Subclasses must implement ``evaluate(state)`` and ``to_dict()``.
    """

    def evaluate(self, state: dict[str, Any]) -> bool:
        """Evaluate this condition against the given state.

        Parameters
        ----------
        state:
            A dictionary with keys like ``"targets"``, ``"zones"``,
            ``"sensors"`` containing the current tracking state.

        Returns True if the condition is satisfied.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        raise NotImplementedError

    def __and__(self, other: ConditionBase) -> AndCondition:
        return AndCondition([self, other])

    def __or__(self, other: ConditionBase) -> OrCondition:
        return OrCondition([self, other])

    def __invert__(self) -> NotCondition:
        return NotCondition(self)


# ---------------------------------------------------------------------------
# Built-in condition types
# ---------------------------------------------------------------------------

class Condition(ConditionBase):
    """A single boolean predicate identified by name with parameters.

    Built-in condition names:

    - ``target_enters_zone(zone_id)`` — any target is in the zone
    - ``threat_level_above(level)`` — any target's threat exceeds level
    - ``target_count_in_zone_exceeds(zone_id, count)`` — too many targets
    - ``target_dwell_exceeds(zone_id, minutes)`` — target stayed too long
    - ``sensor_offline(sensor_id, minutes)`` — sensor hasn't reported
    - ``target_alliance_is(alliance)`` — any target with given alliance
    - ``field_compare(field_path, operator, value)`` — generic field comparison

    Parameters
    ----------
    name:
        The condition name (must match a registered evaluator).
    **kwargs:
        Condition-specific parameters.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.params = dict(kwargs)

    def evaluate(self, state: dict[str, Any]) -> bool:
        """Evaluate using the built-in evaluator registry."""
        evaluator = _CONDITION_EVALUATORS.get(self.name)
        if evaluator is None:
            logger.warning("Unknown condition name: %s", self.name)
            return False
        return evaluator(state, self.params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "condition",
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(name=data["name"], **data.get("params", {}))

    def __repr__(self) -> str:
        return f"Condition({self.name!r}, {self.params})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self.name == other.name and self.params == other.params


class AndCondition(ConditionBase):
    """Logical AND — all child conditions must be true."""

    def __init__(self, conditions: list[ConditionBase]) -> None:
        self.conditions = list(conditions)

    def evaluate(self, state: dict[str, Any]) -> bool:
        return all(c.evaluate(state) for c in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "and",
            "conditions": [c.to_dict() for c in self.conditions],
        }

    def __repr__(self) -> str:
        return f"AndCondition({self.conditions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AndCondition):
            return NotImplemented
        return self.conditions == other.conditions


class OrCondition(ConditionBase):
    """Logical OR — at least one child condition must be true."""

    def __init__(self, conditions: list[ConditionBase]) -> None:
        self.conditions = list(conditions)

    def evaluate(self, state: dict[str, Any]) -> bool:
        return any(c.evaluate(state) for c in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "or",
            "conditions": [c.to_dict() for c in self.conditions],
        }

    def __repr__(self) -> str:
        return f"OrCondition({self.conditions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrCondition):
            return NotImplemented
        return self.conditions == other.conditions


class NotCondition(ConditionBase):
    """Logical NOT — inverts a single child condition."""

    def __init__(self, condition: ConditionBase) -> None:
        self.condition = condition

    def evaluate(self, state: dict[str, Any]) -> bool:
        return not self.condition.evaluate(state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "not",
            "condition": self.condition.to_dict(),
        }

    def __repr__(self) -> str:
        return f"NotCondition({self.condition})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NotCondition):
            return NotImplemented
        return self.condition == other.condition


# ---------------------------------------------------------------------------
# condition_from_dict — deserialize any condition from JSON
# ---------------------------------------------------------------------------

def condition_from_dict(data: dict[str, Any]) -> ConditionBase:
    """Deserialize a condition tree from a JSON-compatible dict.

    Handles ``"condition"``, ``"and"``, ``"or"``, and ``"not"`` types.
    """
    ctype = data.get("type", "condition")
    if ctype == "and":
        return AndCondition([condition_from_dict(c) for c in data["conditions"]])
    elif ctype == "or":
        return OrCondition([condition_from_dict(c) for c in data["conditions"]])
    elif ctype == "not":
        return NotCondition(condition_from_dict(data["condition"]))
    else:
        return Condition.from_dict(data)


# ---------------------------------------------------------------------------
# Built-in condition evaluators
# ---------------------------------------------------------------------------

def _eval_target_enters_zone(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if any target is currently in the specified zone."""
    zone_id = params.get("zone_id", "")
    targets = state.get("targets", {})
    for target in targets.values():
        if isinstance(target, dict):
            if target.get("zone_id") == zone_id:
                return True
            # Also check zone_ids list
            if zone_id in target.get("zone_ids", []):
                return True
    return False


def _eval_threat_level_above(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if any target's threat level exceeds the threshold."""
    level = float(params.get("level", 0.5))
    targets = state.get("targets", {})
    for target in targets.values():
        if isinstance(target, dict):
            threat = float(target.get("threat_level", target.get("threat_score", 0.0)))
            if threat > level:
                return True
    return False


def _eval_target_count_in_zone_exceeds(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if the number of targets in a zone exceeds a threshold."""
    zone_id = params.get("zone_id", "")
    count_threshold = int(params.get("count", 0))
    targets = state.get("targets", {})
    zone_count = 0
    for target in targets.values():
        if isinstance(target, dict):
            if target.get("zone_id") == zone_id:
                zone_count += 1
            elif zone_id in target.get("zone_ids", []):
                zone_count += 1
    return zone_count > count_threshold


def _eval_target_dwell_exceeds(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if any target has been in a zone longer than the threshold."""
    zone_id = params.get("zone_id", "")
    minutes = float(params.get("minutes", 5))
    threshold_seconds = minutes * 60.0
    now = state.get("timestamp", time.time())
    targets = state.get("targets", {})
    for target in targets.values():
        if isinstance(target, dict):
            in_zone = (
                target.get("zone_id") == zone_id
                or zone_id in target.get("zone_ids", [])
            )
            if in_zone:
                entered_at = target.get("zone_entered_at", target.get("first_seen", now))
                dwell = now - entered_at
                if dwell > threshold_seconds:
                    return True
    return False


def _eval_sensor_offline(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if a sensor has been offline for more than the threshold."""
    sensor_id = params.get("sensor_id", "")
    minutes = float(params.get("minutes", 5))
    threshold_seconds = minutes * 60.0
    now = state.get("timestamp", time.time())
    sensors = state.get("sensors", {})
    sensor = sensors.get(sensor_id, {})
    if not sensor:
        return False
    status = sensor.get("status", "unknown")
    if status == "offline":
        return True
    last_seen = sensor.get("last_seen", now)
    elapsed = now - last_seen
    return elapsed > threshold_seconds


def _eval_target_alliance_is(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check if any target has the specified alliance."""
    alliance = params.get("alliance", "")
    targets = state.get("targets", {})
    for target in targets.values():
        if isinstance(target, dict):
            if target.get("alliance") == alliance:
                return True
    return False


def _eval_field_compare(state: dict[str, Any], params: dict[str, Any]) -> bool:
    """Generic field comparison using dot-path navigation.

    Parameters:
        field_path: dot-separated path into state (e.g. "zones.alpha.target_count")
        operator: eq, neq, gt, lt, gte, lte, contains, in
        value: expected value
    """
    field_path = params.get("field_path", "")
    operator = params.get("operator", "eq")
    expected = params.get("value")

    # Navigate state by dot path
    parts = field_path.split(".")
    current: Any = state
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return False
        if current is None:
            return False

    actual = current

    if operator == "eq":
        return actual == expected
    elif operator == "neq":
        return actual != expected
    elif operator == "gt":
        return _to_float(actual) > _to_float(expected)
    elif operator == "lt":
        return _to_float(actual) < _to_float(expected)
    elif operator == "gte":
        return _to_float(actual) >= _to_float(expected)
    elif operator == "lte":
        return _to_float(actual) <= _to_float(expected)
    elif operator == "contains":
        return expected in actual if hasattr(actual, "__contains__") else False
    elif operator == "in":
        return actual in expected if hasattr(expected, "__contains__") else False
    elif operator == "regex":
        try:
            return bool(re.search(str(expected), str(actual)))
        except re.error:
            return False
    return False


def _to_float(v: Any) -> float:
    """Safely convert a value to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# Registry of built-in condition evaluators
_CONDITION_EVALUATORS: dict[str, Callable[[dict, dict], bool]] = {
    "target_enters_zone": _eval_target_enters_zone,
    "threat_level_above": _eval_threat_level_above,
    "target_count_in_zone_exceeds": _eval_target_count_in_zone_exceeds,
    "target_dwell_exceeds": _eval_target_dwell_exceeds,
    "sensor_offline": _eval_sensor_offline,
    "target_alliance_is": _eval_target_alliance_is,
    "field_compare": _eval_field_compare,
}


def register_condition(name: str, evaluator: Callable[[dict, dict], bool]) -> None:
    """Register a custom condition evaluator.

    Parameters
    ----------
    name:
        The condition name (used in ``Condition(name, ...)``.
    evaluator:
        A callable(state, params) -> bool.
    """
    _CONDITION_EVALUATORS[name] = evaluator


# ---------------------------------------------------------------------------
# RuleResult — record of a rule firing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleResult:
    """Immutable record of a rule that fired during evaluation.

    Attributes
    ----------
    result_id:
        Unique identifier for this result.
    rule_id:
        The rule that fired.
    rule_name:
        Human-readable rule name.
    actions:
        The actions that were triggered.
    timestamp:
        When the rule fired.
    state_snapshot:
        Relevant portion of the state at firing time.
    ruleset_id:
        Which ruleset this rule belongs to (empty if standalone).
    """
    result_id: str
    rule_id: str
    rule_name: str
    actions: list[dict[str, Any]]
    timestamp: float
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    ruleset_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "actions": list(self.actions),
            "timestamp": self.timestamp,
            "state_snapshot": dict(self.state_snapshot),
            "ruleset_id": self.ruleset_id,
        }


# ---------------------------------------------------------------------------
# Rule — IF condition THEN action(s)
# ---------------------------------------------------------------------------

class Rule:
    """An IF-THEN automation rule with priority, cooldown, and metadata.

    Supports a fluent/DSL-like API for building rules::

        rule = (
            Rule("my_rule", name="My Rule")
            .when(Condition("target_enters_zone", zone_id="alpha"))
            .then(Action(ActionType.SEND_ALERT, message="Breach!"))
            .with_priority(10)
            .with_cooldown(30)
            .with_tags(["perimeter", "security"])
        )

    Attributes
    ----------
    rule_id:
        Unique identifier.
    name:
        Human-readable name.
    condition:
        The condition tree (ConditionBase). None means always true.
    actions:
        List of actions to execute when the rule fires.
    priority:
        Higher priority rules are evaluated first (default 0).
    cooldown_seconds:
        Minimum seconds between firings (0 = no cooldown).
    enabled:
        Whether the rule is active.
    tags:
        Arbitrary string tags for filtering/grouping.
    chains_to:
        Optional list of rule IDs to evaluate after this rule fires.
    max_fires:
        Maximum number of times this rule can fire (0 = unlimited).
    description:
        Human-readable description.
    """

    def __init__(
        self,
        rule_id: str,
        *,
        name: str = "",
        condition: ConditionBase | None = None,
        actions: list[Action] | None = None,
        priority: int = 0,
        cooldown_seconds: int = 0,
        enabled: bool = True,
        tags: list[str] | None = None,
        chains_to: list[str] | None = None,
        max_fires: int = 0,
        description: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.name = name or rule_id
        self.condition = condition
        self.actions = list(actions) if actions else []
        self.priority = priority
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.tags = list(tags) if tags else []
        self.chains_to = list(chains_to) if chains_to else []
        self.max_fires = max_fires
        self.description = description

        # Runtime state
        self.fire_count: int = 0
        self.last_fired_at: float = 0.0

    # -- Fluent API --------------------------------------------------------

    def when(self, condition: ConditionBase) -> Rule:
        """Set the rule's condition (fluent API).

        If a condition is already set, combines with AND.
        """
        if self.condition is None:
            self.condition = condition
        else:
            self.condition = AndCondition([self.condition, condition])
        return self

    def then(self, action: Action) -> Rule:
        """Add an action to execute when the rule fires (fluent API)."""
        self.actions.append(action)
        return self

    def with_priority(self, priority: int) -> Rule:
        """Set the rule's priority (fluent API). Higher = evaluated first."""
        self.priority = priority
        return self

    def with_cooldown(self, seconds: int) -> Rule:
        """Set the cooldown period in seconds (fluent API)."""
        self.cooldown_seconds = seconds
        return self

    def with_tags(self, tags: list[str]) -> Rule:
        """Set the rule's tags (fluent API)."""
        self.tags = list(tags)
        return self

    def chains(self, *rule_ids: str) -> Rule:
        """Set rule IDs to evaluate after this rule fires (fluent API)."""
        self.chains_to = list(rule_ids)
        return self

    def with_max_fires(self, n: int) -> Rule:
        """Set the maximum number of times this rule can fire (fluent API)."""
        self.max_fires = n
        return self

    def with_description(self, desc: str) -> Rule:
        """Set a human-readable description (fluent API)."""
        self.description = desc
        return self

    # -- Evaluation --------------------------------------------------------

    def matches(self, state: dict[str, Any]) -> bool:
        """Check if this rule should fire for the given state.

        Returns True if:
        - Rule is enabled
        - Max fires not exceeded (if set)
        - Cooldown has elapsed
        - Condition evaluates to True (or no condition set)
        """
        if not self.enabled:
            return False
        if self.max_fires > 0 and self.fire_count >= self.max_fires:
            return False
        if not self.is_cooled_down():
            return False
        if self.condition is None:
            return True
        return self.condition.evaluate(state)

    def is_cooled_down(self, now: float | None = None) -> bool:
        """Check if cooldown has elapsed since last firing."""
        if self.cooldown_seconds <= 0:
            return True
        if self.last_fired_at <= 0:
            return True
        now = now if now is not None else time.time()
        elapsed = now - self.last_fired_at
        return elapsed >= self.cooldown_seconds

    def record_firing(self, now: float | None = None) -> None:
        """Record that this rule has fired."""
        self.fire_count += 1
        self.last_fired_at = now if now is not None else time.time()

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition.to_dict() if self.condition else None,
            "actions": [a.to_dict() for a in self.actions],
            "priority": self.priority,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "chains_to": list(self.chains_to),
            "max_fires": self.max_fires,
            "description": self.description,
            "fire_count": self.fire_count,
            "last_fired_at": self.last_fired_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        """Deserialize from a JSON-compatible dict."""
        cond_data = data.get("condition")
        condition = condition_from_dict(cond_data) if cond_data else None

        actions = [Action.from_dict(a) for a in data.get("actions", [])]

        rule = cls(
            rule_id=data["rule_id"],
            name=data.get("name", ""),
            condition=condition,
            actions=actions,
            priority=data.get("priority", 0),
            cooldown_seconds=data.get("cooldown_seconds", 0),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
            chains_to=data.get("chains_to", []),
            max_fires=data.get("max_fires", 0),
            description=data.get("description", ""),
        )
        rule.fire_count = data.get("fire_count", 0)
        rule.last_fired_at = data.get("last_fired_at", 0.0)
        return rule

    def __repr__(self) -> str:
        return f"Rule({self.rule_id!r}, priority={self.priority}, enabled={self.enabled})"


# ---------------------------------------------------------------------------
# RuleSet — named collection of rules
# ---------------------------------------------------------------------------

class RuleSet:
    """A named, ordered collection of rules.

    RuleSets allow organizing rules into logical groups (e.g.,
    "perimeter_defense", "vip_protection") and can be enabled/disabled
    as a unit.

    Attributes
    ----------
    ruleset_id:
        Unique identifier.
    name:
        Human-readable name.
    description:
        Human-readable description.
    rules:
        Ordered list of rules (sorted by priority on evaluation).
    enabled:
        Whether this ruleset is active.
    tags:
        Arbitrary string tags.
    """

    def __init__(
        self,
        ruleset_id: str,
        *,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        tags: list[str] | None = None,
    ) -> None:
        self.ruleset_id = ruleset_id
        self.name = name or ruleset_id
        self.description = description
        self.enabled = enabled
        self.tags = list(tags) if tags else []
        self._rules: dict[str, Rule] = {}

    def add_rule(self, rule: Rule) -> Rule:
        """Add a rule to this set. Returns the rule."""
        self._rules[rule.rule_id] = rule
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found."""
        return self._rules.pop(rule_id, None) is not None

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_rules(self) -> list[Rule]:
        """Return all rules sorted by priority (highest first)."""
        return sorted(self._rules.values(), key=lambda r: r.priority, reverse=True)

    def count(self) -> int:
        """Return the number of rules in this set."""
        return len(self._rules)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "ruleset_id": self.ruleset_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "rules": [r.to_dict() for r in self.get_rules()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleSet:
        """Deserialize from a JSON-compatible dict."""
        rs = cls(
            ruleset_id=data["ruleset_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
        )
        for rd in data.get("rules", []):
            rs.add_rule(Rule.from_dict(rd))
        return rs

    def __repr__(self) -> str:
        return (
            f"RuleSet({self.ruleset_id!r}, rules={len(self._rules)}, "
            f"enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# RuleEngine — evaluates rules against current state
# ---------------------------------------------------------------------------

class RuleEngine:
    """Evaluate rules against current tracking state and execute actions.

    Thread-safe. All public methods acquire the internal lock.

    Parameters
    ----------
    action_handlers:
        Optional dict mapping ActionType -> callable(Action, state).
        Called when an action of that type is triggered.
    max_history:
        Maximum number of RuleResults to retain.
    enable_chaining:
        If True, after a rule fires its ``chains_to`` rules are
        immediately evaluated (depth-limited to prevent infinite loops).
    max_chain_depth:
        Maximum depth for rule chaining (default 5).
    """

    def __init__(
        self,
        *,
        action_handlers: dict[ActionType, Callable[[Action, dict], None]] | None = None,
        max_history: int = 5000,
        enable_chaining: bool = True,
        max_chain_depth: int = 5,
    ) -> None:
        self._lock = threading.Lock()
        self._rules: dict[str, Rule] = {}
        self._rulesets: dict[str, RuleSet] = {}
        self._action_handlers: dict[ActionType, Callable[[Action, dict], None]] = (
            dict(action_handlers) if action_handlers else {}
        )
        self._max_history = max_history
        self._enable_chaining = enable_chaining
        self._max_chain_depth = max_chain_depth
        self._history: list[RuleResult] = []

        # Counters
        self._total_evaluations = 0
        self._total_rules_fired = 0
        self._total_actions_executed = 0
        self._total_cooldown_suppressed = 0

    # -- Rule management ---------------------------------------------------

    def add_rule(self, rule: Rule) -> Rule:
        """Add or update a standalone rule. Returns the rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule
        logger.info("Rule added: %s (%s)", rule.name, rule.rule_id)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a standalone rule by ID. Returns True if found."""
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a rule by ID (from standalone or any ruleset)."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                return rule
            for rs in self._rulesets.values():
                rule = rs.get_rule(rule_id)
                if rule:
                    return rule
        return None

    def get_rules(self) -> list[Rule]:
        """Return all standalone rules sorted by priority (highest first)."""
        with self._lock:
            return sorted(
                self._rules.values(), key=lambda r: r.priority, reverse=True
            )

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule. Returns True if found."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule. Returns True if found."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    # -- RuleSet management ------------------------------------------------

    def add_ruleset(self, ruleset: RuleSet) -> RuleSet:
        """Add or update a ruleset. Returns the ruleset."""
        with self._lock:
            self._rulesets[ruleset.ruleset_id] = ruleset
        logger.info("RuleSet added: %s (%s)", ruleset.name, ruleset.ruleset_id)
        return ruleset

    def remove_ruleset(self, ruleset_id: str) -> bool:
        """Remove a ruleset by ID. Returns True if found."""
        with self._lock:
            return self._rulesets.pop(ruleset_id, None) is not None

    def get_ruleset(self, ruleset_id: str) -> RuleSet | None:
        """Get a ruleset by ID."""
        with self._lock:
            return self._rulesets.get(ruleset_id)

    def get_rulesets(self) -> list[RuleSet]:
        """Return all rulesets."""
        with self._lock:
            return list(self._rulesets.values())

    def enable_ruleset(self, ruleset_id: str) -> bool:
        """Enable a ruleset. Returns True if found."""
        with self._lock:
            rs = self._rulesets.get(ruleset_id)
            if rs:
                rs.enabled = True
                return True
        return False

    def disable_ruleset(self, ruleset_id: str) -> bool:
        """Disable a ruleset. Returns True if found."""
        with self._lock:
            rs = self._rulesets.get(ruleset_id)
            if rs:
                rs.enabled = False
                return True
        return False

    # -- Action handlers ---------------------------------------------------

    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable[[Action, dict], None],
    ) -> None:
        """Register a handler for a specific action type."""
        with self._lock:
            self._action_handlers[action_type] = handler

    # -- Evaluation --------------------------------------------------------

    def evaluate(self, state: dict[str, Any]) -> list[RuleResult]:
        """Evaluate all rules against the given state.

        Rules are evaluated in priority order (highest first). For each
        matching rule, actions are executed and a RuleResult is recorded.
        If chaining is enabled, chained rules are evaluated recursively.

        Parameters
        ----------
        state:
            The current tracking state. Expected keys:
            - ``targets``: dict of target_id -> target data
            - ``zones``: dict of zone_id -> zone data
            - ``sensors``: dict of sensor_id -> sensor data
            - ``timestamp``: current time (defaults to time.time())

        Returns
        -------
        list[RuleResult]:
            All rules that fired during this evaluation.
        """
        if "timestamp" not in state:
            state = dict(state, timestamp=time.time())

        with self._lock:
            self._total_evaluations += 1

            # Collect all rules from standalone + rulesets
            all_rules: list[tuple[Rule, str]] = []
            for rule in self._rules.values():
                all_rules.append((rule, ""))
            for rs in self._rulesets.values():
                if not rs.enabled:
                    continue
                for rule in rs.get_rules():
                    all_rules.append((rule, rs.ruleset_id))

            # Sort by priority (highest first)
            all_rules.sort(key=lambda t: t[0].priority, reverse=True)

        fired: list[RuleResult] = []
        self._evaluate_rules(all_rules, state, fired, depth=0)
        return fired

    def _evaluate_rules(
        self,
        rules: list[tuple[Rule, str]],
        state: dict[str, Any],
        fired: list[RuleResult],
        depth: int,
    ) -> None:
        """Internal recursive rule evaluation with chain support."""
        if depth > self._max_chain_depth:
            logger.warning("Max chain depth %d exceeded", self._max_chain_depth)
            return

        for rule, ruleset_id in rules:
            if rule.matches(state):
                now = state.get("timestamp", time.time())
                rule.record_firing(now)

                result = RuleResult(
                    result_id=uuid.uuid4().hex[:12],
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    actions=[a.to_dict() for a in rule.actions],
                    timestamp=now,
                    ruleset_id=ruleset_id,
                )
                fired.append(result)

                with self._lock:
                    self._total_rules_fired += 1
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]

                # Execute actions
                for action in rule.actions:
                    self._execute_action(action, state)

                # Rule chaining
                if self._enable_chaining and rule.chains_to:
                    chained: list[tuple[Rule, str]] = []
                    for chain_id in rule.chains_to:
                        chain_rule = self.get_rule(chain_id)
                        if chain_rule is not None:
                            chained.append((chain_rule, ruleset_id))
                    if chained:
                        self._evaluate_rules(chained, state, fired, depth + 1)
            else:
                # Check if it was a cooldown suppression
                if rule.enabled and rule.condition is not None:
                    if rule.condition.evaluate(state) and not rule.is_cooled_down():
                        with self._lock:
                            self._total_cooldown_suppressed += 1

    def _execute_action(self, action: Action, state: dict[str, Any]) -> None:
        """Execute a single action."""
        with self._lock:
            self._total_actions_executed += 1

        handler = self._action_handlers.get(action.action_type)
        if handler is not None:
            try:
                handler(action, state)
            except Exception:
                logger.debug(
                    "Action handler failed for %s",
                    action.action_type.value,
                    exc_info=True,
                )
        else:
            logger.info(
                "Rule action: %s %s",
                action.action_type.value,
                action.params,
            )

    # -- History & stats ---------------------------------------------------

    def get_history(
        self,
        limit: int = 100,
        *,
        rule_id: str = "",
        ruleset_id: str = "",
        since: float = 0.0,
    ) -> list[RuleResult]:
        """Retrieve firing history with optional filtering.

        Returns results newest-first.
        """
        with self._lock:
            records = list(self._history)

        if rule_id:
            records = [r for r in records if r.rule_id == rule_id]
        if ruleset_id:
            records = [r for r in records if r.ruleset_id == ruleset_id]
        if since > 0:
            records = [r for r in records if r.timestamp >= since]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Return engine-wide statistics."""
        with self._lock:
            return {
                "total_standalone_rules": len(self._rules),
                "total_rulesets": len(self._rulesets),
                "total_rules_in_rulesets": sum(
                    rs.count() for rs in self._rulesets.values()
                ),
                "total_evaluations": self._total_evaluations,
                "total_rules_fired": self._total_rules_fired,
                "total_actions_executed": self._total_actions_executed,
                "total_cooldown_suppressed": self._total_cooldown_suppressed,
                "history_size": len(self._history),
                "max_history": self._max_history,
                "chaining_enabled": self._enable_chaining,
                "max_chain_depth": self._max_chain_depth,
                "action_handlers": [h.value for h in self._action_handlers.keys()],
            }

    def clear_history(self) -> int:
        """Clear firing history. Returns count removed."""
        with self._lock:
            count = len(self._history)
            self._history.clear()
            return count

    def reset(self) -> None:
        """Reset all state: rules, rulesets, history, counters."""
        with self._lock:
            self._rules.clear()
            self._rulesets.clear()
            self._history.clear()
            self._total_evaluations = 0
            self._total_rules_fired = 0
            self._total_actions_executed = 0
            self._total_cooldown_suppressed = 0

    def reset_counters(self) -> None:
        """Reset statistics counters without clearing rules or history."""
        with self._lock:
            self._total_evaluations = 0
            self._total_rules_fired = 0
            self._total_actions_executed = 0
            self._total_cooldown_suppressed = 0
            for rule in self._rules.values():
                rule.fire_count = 0
                rule.last_fired_at = 0.0
            for rs in self._rulesets.values():
                for rule in rs.get_rules():
                    rule.fire_count = 0
                    rule.last_fired_at = 0.0

    # -- JSON import/export ------------------------------------------------

    def export_json(self) -> str:
        """Export all rules and rulesets as a JSON string."""
        with self._lock:
            data = {
                "rules": [r.to_dict() for r in self._rules.values()],
                "rulesets": [rs.to_dict() for rs in self._rulesets.values()],
            }
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> dict[str, int]:
        """Import rules and rulesets from a JSON string.

        Returns a dict with counts: {"rules": N, "rulesets": M}.
        """
        data = json.loads(json_str)
        rules_count = 0
        rulesets_count = 0

        for rd in data.get("rules", []):
            rule = Rule.from_dict(rd)
            self.add_rule(rule)
            rules_count += 1

        for rsd in data.get("rulesets", []):
            rs = RuleSet.from_dict(rsd)
            self.add_ruleset(rs)
            rulesets_count += 1

        return {"rules": rules_count, "rulesets": rulesets_count}

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"RuleEngine(rules={len(self._rules)}, "
                f"rulesets={len(self._rulesets)}, "
                f"history={len(self._history)})"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Action",
    "ActionType",
    "AndCondition",
    "Condition",
    "ConditionBase",
    "NotCondition",
    "OrCondition",
    "Rule",
    "RuleEngine",
    "RuleResult",
    "RuleSet",
    "condition_from_dict",
    "register_condition",
]
