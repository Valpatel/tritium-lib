# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Patrol game mode — perimeter-breach defeat hook (breadth roadmap #8).

`/api/game/modes` advertised `patrol` ("Units must patrol routes and respond
to intrusions") but it collapsed to battle — a lying manifest. Patrol is a
genuinely distinct dynamic vs the other modes: a friendly SECURE ZONE that
hostiles must not BREACH. The lose condition is a single intruder reaching the
protected point (intrusion-response doctrine) — distinct from battle's
all-friendlies-eliminated, drone_swarm's infrastructure ATTRITION, and escort's
moving-protectee loss. Victory is clearing all intrusion waves WITHOUT a breach.

This covers the mode's defeat contract at the GameMode level; the engine
computes the breach geometry (it owns hostile positions) and calls the hook.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from tritium_lib.sim_engine.game.game_mode import GameMode

pytestmark = pytest.mark.unit


@dataclass
class _FakeTarget:
    target_id: str
    alliance: str = "friendly"
    is_combatant: bool = True
    status: str = "active"


class _Bus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []
    def publish(self, name, data=None):
        self.events.append((name, data or {}))
    def subscribe(self, *a, **k):
        return None
    def game_overs(self):
        return [d for n, d in self.events if n == "game_over"]


class _Engine:
    def __init__(self):
        self._targets: dict[str, _FakeTarget] = {}
        self._map_bounds = 100.0
        self.hazard_manager = MagicMock()
        self.stats_tracker = MagicMock()
        for i in range(3):  # 3 friendly combatants so defeat doesn't auto-trigger
            self._targets[f"f{i}"] = _FakeTarget(f"f{i}")
    def get_targets(self):
        return list(self._targets.values())
    def add_target(self, t):
        self._targets[t.target_id] = t
    def set_map_bounds(self, b):
        self._map_bounds = b


class _Combat:
    def reset_streaks(self): pass
    def clear(self): pass


def _patrol(state="active"):
    bus = _Bus()
    gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
    gm.game_mode_type = "patrol"
    gm.state = state
    return gm, bus


class TestPatrolWinLose:
    def test_new_patrol_state_fields_exist(self):
        gm, _ = _patrol(state="setup")
        assert gm.protected_point is None
        assert gm.breach_radius > 0
        assert gm.perimeter_breached is False

    def test_breach_is_defeat(self):
        gm, bus = _patrol()
        gm.on_perimeter_breached()
        assert gm.state == "defeat"
        assert gm.perimeter_breached is True
        gos = bus.game_overs()
        assert gos and gos[-1]["result"] == "defeat"
        assert gos[-1]["reason"] == "perimeter_breached"

    def test_breach_hook_is_noop_in_other_modes(self):
        # battle mode must NOT lose on the perimeter-breach hook.
        bus = _Bus()
        gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
        gm.game_mode_type = "battle"
        gm.state = "active"
        gm.on_perimeter_breached()
        assert gm.state == "active"
        assert bus.game_overs() == []

    def test_no_double_game_over_after_breach(self):
        gm, bus = _patrol()
        gm.on_perimeter_breached()
        gm.on_perimeter_breached()  # second intruder — already lost
        assert gm.state == "defeat"
        assert len(bus.game_overs()) == 1

    def test_breach_ignored_once_game_over(self):
        # a breach after victory must not flip the result.
        gm, bus = _patrol(state="victory")
        gm.on_perimeter_breached()
        assert gm.state == "victory"
        assert gm.perimeter_breached is True  # recorded, but no defeat
        assert bus.game_overs() == []

    def test_to_dict_exposes_patrol_state(self):
        gm, _ = _patrol()
        gm.protected_point = (10.0, -20.0)
        gm.breach_radius = 25.0
        st = gm.get_state()
        assert st["game_mode_type"] == "patrol"
        assert st["protected_point"] == [10.0, -20.0]
        assert st["breach_radius"] == 25.0
        assert st["perimeter_breached"] is False

    def test_reset_clears_patrol_state(self):
        gm, _ = _patrol()
        gm.protected_point = (10.0, -20.0)
        gm.perimeter_breached = True
        gm.reset()
        assert gm.protected_point is None
        assert gm.perimeter_breached is False
        assert gm.game_mode_type == "battle"
