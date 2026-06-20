# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Defense mode — hold the strongpoint against a sustained siege.

Defense is the last advertised mode that previously COLLAPSED to battle
(roadmap #8 "stop over-promising modes").  Its distinct objective: a fixed
strongpoint with an integrity (infrastructure) pool that decays under siege;
the mission is LOST when the strongpoint is overrun (integrity 0) — distinct
from battle (lose only on all-friendlies-eliminated), patrol (single breach =
instant loss), and drone_swarm (aerial swarm + planes/bombers).  Win = clear
all waves with the strongpoint intact.

These tests pin the lib GameMode half: the strongpoint-overrun defeat fires
for defense (and ONLY defense + drone_swarm), with its own reason, and the
state dict exposes the integrity pool so the HUD can render it.
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
        for i in range(3):  # friendly combatants so battle-defeat doesn't auto-trigger
            self._targets[f"f{i}"] = _FakeTarget(f"f{i}")

    def get_targets(self):
        return list(self._targets.values())

    def add_target(self, t):
        self._targets[t.target_id] = t

    def set_map_bounds(self, b):
        self._map_bounds = b


class _Combat:
    def reset_streaks(self):
        pass

    def clear(self):
        pass


def _defense(state="active", integrity=1000.0):
    bus = _Bus()
    gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
    gm.game_mode_type = "defense"
    gm.infrastructure_max = 1000.0
    gm.infrastructure_health = integrity
    gm.state = state
    return gm, bus


class TestDefenseStrongpoint:
    def test_partial_siege_damage_is_not_defeat(self):
        gm, bus = _defense(integrity=1000.0)
        gm.on_infrastructure_damaged(250.0)
        assert gm.infrastructure_health == 750.0
        assert gm.state == "active"
        assert bus.game_overs() == []

    def test_strongpoint_overrun_is_defeat(self):
        gm, bus = _defense(integrity=40.0)
        gm.on_infrastructure_damaged(40.0)
        assert gm.infrastructure_health == 0.0
        assert gm.state == "defeat"
        gos = bus.game_overs()
        assert gos and gos[-1]["result"] == "defeat"
        # Defense gets its OWN reason, distinct from drone_swarm.
        assert gos[-1]["reason"] == "strongpoint_overrun"

    def test_overrun_ignored_in_battle_mode(self):
        # battle has no strongpoint — the infra hook must not lose the game.
        bus = _Bus()
        gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
        gm.game_mode_type = "battle"
        gm.infrastructure_health = 10.0
        gm.state = "active"
        gm.on_infrastructure_damaged(50.0)
        assert gm.state == "active"
        assert bus.game_overs() == []

    def test_no_double_game_over_after_overrun(self):
        gm, bus = _defense(integrity=5.0)
        gm.on_infrastructure_damaged(50.0)
        gm.on_infrastructure_damaged(50.0)
        assert len(bus.game_overs()) == 1

    def test_state_exposes_strongpoint_integrity(self):
        gm, _ = _defense(integrity=620.0)
        st = gm.get_state()
        assert st["game_mode_type"] == "defense"
        assert st["infrastructure_health"] == 620.0
        assert st["infrastructure_max"] == 1000.0
