# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Lightweight behavior tree for simulation unit decision-making.

Decides WHAT a unit does (flee, patrol, engage, wander).  The steering
module handles HOW it moves.  Mirrors the state concepts from
tritium-sc's UnitBehaviors FSM but uses composable tree nodes instead
of hard-coded if/else chains.

Usage::

    tree = make_patrol_tree()
    ctx = {"unit": unit, "threats": [...], "dt": 0.1}
    status = tree.tick(ctx)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class Status(Enum):
    """Result of ticking a behavior tree node."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------

class Node(ABC):
    """Abstract base for all behavior tree nodes."""

    @abstractmethod
    def tick(self, context: dict) -> Status:
        """Evaluate this node.  *context* carries per-tick state."""
        ...

    def reset(self) -> None:
        """Optional hook to clear internal running state."""


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

class Sequence(Node):
    """Run children left-to-right.  Fail on first FAILURE, succeed when all
    succeed.  A RUNNING child pauses the sequence until it resolves."""

    def __init__(self, children: list[Node]) -> None:
        self.children = children
        self._running_idx = 0

    def tick(self, context: dict) -> Status:
        for i in range(self._running_idx, len(self.children)):
            status = self.children[i].tick(context)
            if status == Status.FAILURE:
                self._running_idx = 0
                return Status.FAILURE
            if status == Status.RUNNING:
                self._running_idx = i
                return Status.RUNNING
        self._running_idx = 0
        return Status.SUCCESS

    def reset(self) -> None:
        self._running_idx = 0
        for c in self.children:
            c.reset()


class Selector(Node):
    """Run children left-to-right.  Succeed on first SUCCESS.  Fail only if
    every child fails.  A RUNNING child pauses the selector."""

    def __init__(self, children: list[Node]) -> None:
        self.children = children
        self._running_idx = 0

    def tick(self, context: dict) -> Status:
        for i in range(self._running_idx, len(self.children)):
            status = self.children[i].tick(context)
            if status == Status.SUCCESS:
                self._running_idx = 0
                return Status.SUCCESS
            if status == Status.RUNNING:
                self._running_idx = i
                return Status.RUNNING
        self._running_idx = 0
        return Status.FAILURE

    def reset(self) -> None:
        self._running_idx = 0
        for c in self.children:
            c.reset()


class Parallel(Node):
    """Tick all children every frame.  Succeed if at least *threshold*
    children succeed.  Fail if enough children have failed that the
    threshold can no longer be met."""

    def __init__(self, children: list[Node], threshold: int | None = None) -> None:
        self.children = children
        self.threshold = threshold if threshold is not None else len(children)

    def tick(self, context: dict) -> Status:
        successes = 0
        failures = 0
        for child in self.children:
            status = child.tick(context)
            if status == Status.SUCCESS:
                successes += 1
            elif status == Status.FAILURE:
                failures += 1
        if successes >= self.threshold:
            return Status.SUCCESS
        if failures > len(self.children) - self.threshold:
            return Status.FAILURE
        return Status.RUNNING

    def reset(self) -> None:
        for c in self.children:
            c.reset()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

class Inverter(Node):
    """Flip SUCCESS <-> FAILURE.  RUNNING passes through."""

    def __init__(self, child: Node) -> None:
        self.child = child

    def tick(self, context: dict) -> Status:
        status = self.child.tick(context)
        if status == Status.SUCCESS:
            return Status.FAILURE
        if status == Status.FAILURE:
            return Status.SUCCESS
        return Status.RUNNING

    def reset(self) -> None:
        self.child.reset()


class Repeater(Node):
    """Repeat *child* up to *count* times (0 = infinite).  Stops early on
    FAILURE.  Returns RUNNING while repeating, SUCCESS when done."""

    def __init__(self, child: Node, count: int = 0) -> None:
        self.child = child
        self.count = count
        self._done = 0

    def tick(self, context: dict) -> Status:
        status = self.child.tick(context)
        if status == Status.FAILURE:
            self._done = 0
            return Status.FAILURE
        if status == Status.SUCCESS:
            self._done += 1
            if self.count > 0 and self._done >= self.count:
                self._done = 0
                return Status.SUCCESS
        return Status.RUNNING

    def reset(self) -> None:
        self._done = 0
        self.child.reset()


class Cooldown(Node):
    """Prevent *child* from running again for *seconds* after it last
    returned SUCCESS or FAILURE.  Returns FAILURE while cooling down."""

    def __init__(self, child: Node, seconds: float) -> None:
        self.child = child
        self.seconds = seconds
        self._last_done: float = -float("inf")

    def tick(self, context: dict) -> Status:
        now = context.get("time", time.monotonic())
        if now - self._last_done < self.seconds:
            return Status.FAILURE
        status = self.child.tick(context)
        if status in (Status.SUCCESS, Status.FAILURE):
            self._last_done = now
        return status

    def reset(self) -> None:
        self._last_done = -float("inf")
        self.child.reset()


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class Action(Node):
    """Run a callable.  The callable receives *context* and must return a
    :class:`Status`.  If it returns ``None`` it is treated as SUCCESS."""

    def __init__(self, fn: Callable[[dict], Status | None], name: str = "") -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "action")

    def tick(self, context: dict) -> Status:
        result = self.fn(context)
        if result is None:
            return Status.SUCCESS
        return result

    def __repr__(self) -> str:
        return f"Action({self.name!r})"


class Condition(Node):
    """Check a predicate.  Returns SUCCESS if truthy, FAILURE otherwise."""

    def __init__(self, predicate: Callable[[dict], bool], name: str = "") -> None:
        self.predicate = predicate
        self.name = name or getattr(predicate, "__name__", "condition")

    def tick(self, context: dict) -> Status:
        return Status.SUCCESS if self.predicate(context) else Status.FAILURE

    def __repr__(self) -> str:
        return f"Condition({self.name!r})"


# ---------------------------------------------------------------------------
# Shared predicates and actions used by pre-built trees
# ---------------------------------------------------------------------------

def _has_threat(ctx: dict) -> bool:
    """True when a threat is within detection range."""
    return bool(ctx.get("threats"))


def _is_healthy(ctx: dict) -> bool:
    """True when unit health is above retreat threshold."""
    health = ctx.get("health", 1.0)
    return health > ctx.get("retreat_threshold", 0.3)


def _is_at_destination(ctx: dict) -> bool:
    """True when unit has reached its current waypoint."""
    return bool(ctx.get("at_destination"))


def _has_waypoints(ctx: dict) -> bool:
    """True when unit has patrol waypoints remaining."""
    return bool(ctx.get("waypoints"))


def _threat_in_range(ctx: dict) -> bool:
    """True when nearest threat is within weapon range."""
    return bool(ctx.get("threat_in_range"))


def _is_calm(ctx: dict) -> bool:
    """True when no threats detected for cooldown period (unit calms down)."""
    return not ctx.get("threats") and not ctx.get("recently_threatened")


def _set_state(state: str) -> Callable[[dict], Status | None]:
    """Return an action that sets ``ctx['state']`` and succeeds."""
    def _action(ctx: dict) -> Status:
        ctx["state"] = state
        return Status.SUCCESS
    _action.__name__ = f"set_{state}"
    return _action


def _set_decision(decision: str) -> Callable[[dict], Status | None]:
    """Return an action that sets ``ctx['decision']`` and succeeds."""
    def _action(ctx: dict) -> Status:
        ctx["decision"] = decision
        return Status.SUCCESS
    _action.__name__ = f"decide_{decision}"
    return _action


# ---------------------------------------------------------------------------
# Pre-built behavior trees for simulation unit archetypes
# ---------------------------------------------------------------------------

def make_civilian_tree() -> Node:
    """Civilian: wander -> notice threat -> flee -> hide -> calm down -> resume.

    Priority (selector):
      1. If threatened and unhealthy -> hide
      2. If threatened -> flee
      3. If calmed down -> resume wandering
      4. Default -> wander
    """
    return Selector([
        # Branch 1: low health + threat -> hide
        Sequence([
            Condition(_has_threat, "threat_nearby"),
            Inverter(Condition(_is_healthy, "is_healthy")),
            Action(_set_decision("hide")),
        ]),
        # Branch 2: threat -> flee
        Sequence([
            Condition(_has_threat, "threat_nearby"),
            Action(_set_decision("flee")),
        ]),
        # Branch 3: calm -> resume normal
        Sequence([
            Condition(_is_calm, "is_calm"),
            Action(_set_decision("wander")),
        ]),
        # Branch 4: default wander
        Action(_set_decision("wander")),
    ])


def make_patrol_tree() -> Node:
    """Patrol unit: follow route -> detect intruder -> pursue -> engage -> return.

    Priority (selector):
      1. Threat in weapon range -> engage
      2. Threat detected -> pursue
      3. Has waypoints -> patrol
      4. Default -> idle
    """
    return Selector([
        # Branch 1: engage
        Sequence([
            Condition(_has_threat, "threat_detected"),
            Condition(_threat_in_range, "threat_in_range"),
            Action(_set_decision("engage")),
        ]),
        # Branch 2: pursue
        Sequence([
            Condition(_has_threat, "threat_detected"),
            Action(_set_decision("pursue")),
        ]),
        # Branch 3: patrol route
        Sequence([
            Condition(_has_waypoints, "has_waypoints"),
            Action(_set_decision("patrol")),
        ]),
        # Branch 4: idle
        Action(_set_decision("idle")),
    ])


def make_vehicle_tree() -> Node:
    """Vehicle: pick destination -> drive route -> park -> wait -> repeat.

    Priority (selector):
      1. At destination -> park
      2. Has waypoints -> drive
      3. Default -> pick new destination
    """
    return Selector([
        # Branch 1: arrived -> park
        Sequence([
            Condition(_is_at_destination, "at_destination"),
            Action(_set_decision("park")),
        ]),
        # Branch 2: driving
        Sequence([
            Condition(_has_waypoints, "has_waypoints"),
            Action(_set_decision("drive")),
        ]),
        # Branch 3: pick destination
        Action(_set_decision("pick_destination")),
    ])


def make_friendly_tree() -> Node:
    """Friendly: engage threats, seek cover, approach detected enemies, patrol.

    Similar to hostile tree but with slightly more defensive posture — seeks
    cover first when a threat is in range, then engages.  Will proactively
    approach detected threats rather than idling.

    Priority (selector):
      1. Badly hurt -> retreat
      2. Threat in range -> seek cover (cooldown), then engage
      3. Threat detected -> approach
      4. Has waypoints -> patrol
      5. Default -> idle
    """
    return Selector([
        # Branch 1: retreat when hurt
        Sequence([
            Inverter(Condition(_is_healthy, "is_healthy")),
            Action(_set_decision("retreat")),
        ]),
        # Branch 2: seek cover when threat in range (cooldown-gated)
        Sequence([
            Condition(_threat_in_range, "threat_in_range"),
            Cooldown(
                Action(_set_decision("seek_cover")),
                seconds=6.0,
            ),
        ]),
        # Branch 2b: engage when cover-seek is on cooldown
        Sequence([
            Condition(_threat_in_range, "threat_in_range"),
            Action(_set_decision("engage")),
        ]),
        # Branch 3: approach detected threat
        Sequence([
            Condition(_has_threat, "has_threat"),
            Action(_set_decision("approach")),
        ]),
        # Branch 4: patrol route
        Sequence([
            Condition(_has_waypoints, "has_waypoints"),
            Action(_set_decision("patrol")),
        ]),
        # Branch 5: idle
        Action(_set_decision("idle")),
    ])


def make_hostile_tree() -> Node:
    """Hostile: approach target -> find cover -> engage -> retreat when hurt -> regroup.

    Priority (selector):
      1. Badly hurt -> retreat
      2. Threat in range -> engage (with cover-seek cooldown)
      3. Threat detected -> approach
      4. Default -> regroup
    """
    return Selector([
        # Branch 1: retreat when hurt
        Sequence([
            Inverter(Condition(_is_healthy, "is_healthy")),
            Action(_set_decision("retreat")),
        ]),
        # Branch 2: engage from cover
        Sequence([
            Condition(_threat_in_range, "threat_in_range"),
            Cooldown(
                Action(_set_decision("seek_cover")),
                seconds=5.0,
            ),
        ]),
        # Branch 2b: engage without cover seek (cooldown active)
        Sequence([
            Condition(_threat_in_range, "threat_in_range"),
            Action(_set_decision("engage")),
        ]),
        # Branch 3: approach threat
        Sequence([
            Condition(_has_threat, "has_threat"),
            Action(_set_decision("approach")),
        ]),
        # Branch 4: regroup
        Action(_set_decision("regroup")),
    ])
