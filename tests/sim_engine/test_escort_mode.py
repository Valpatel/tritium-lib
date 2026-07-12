# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Escort game mode — win/lose hooks (breadth roadmap #8).

`/api/game/modes` advertised `escort` ("Escort VIP from A to B") but it
collapsed to battle — a lying manifest. Escort is a genuinely NEW dynamic
vs the three static-defense modes: a MOVING protectee whose ARRIVAL is the
victory and whose LOSS is the defeat. This covers the mode's win/lose
contract at the GameMode level (the engine spawns + moves the VIP).
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


def _escort(state="active"):
    bus = _Bus()
    gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
    gm.game_mode_type = "escort"
    gm.state = state
    return gm, bus


class TestEscortWinLose:
    def test_new_escort_state_fields_exist(self):
        gm, _ = _escort(state="setup")
        assert gm.protectee_id is None
        assert gm.escort_destination is None
        assert gm.protectee_arrived is False
        assert gm.protectee_lost is False

    def test_arrival_is_victory(self):
        gm, bus = _escort()
        gm.on_protectee_arrived()
        assert gm.state == "victory"
        assert gm.protectee_arrived is True
        gos = bus.game_overs()
        assert gos and gos[-1]["result"] == "victory"
        assert gos[-1]["reason"] == "protectee_reached_destination"

    def test_loss_is_defeat(self):
        gm, bus = _escort()
        gm.on_protectee_lost()
        assert gm.state == "defeat"
        assert gm.protectee_lost is True
        gos = bus.game_overs()
        assert gos and gos[-1]["result"] == "defeat"
        assert gos[-1]["reason"] == "protectee_lost"

    def test_hooks_are_noops_in_other_modes(self):
        # battle mode must NOT win/lose on protectee hooks.
        bus = _Bus()
        gm = GameMode(event_bus=bus, engine=_Engine(), combat_system=_Combat())
        gm.game_mode_type = "battle"
        gm.state = "active"
        gm.on_protectee_arrived()
        gm.on_protectee_lost()
        assert gm.state == "active"
        assert bus.game_overs() == []

    def test_no_double_game_over_after_arrival(self):
        gm, bus = _escort()
        gm.on_protectee_arrived()
        gm.on_protectee_lost()   # too late — already won
        assert gm.state == "victory"
        assert len(bus.game_overs()) == 1

    def test_to_dict_exposes_escort_state(self):
        gm, _ = _escort()
        gm.protectee_id = "vip_1"
        gm.escort_destination = (100.0, 50.0)
        st = gm.get_state()
        assert st["game_mode_type"] == "escort"
        assert st["protectee_id"] == "vip_1"
        assert st["escort_destination"] == [100.0, 50.0]
        assert st["protectee_arrived"] is False
        assert st["protectee_lost"] is False

    def test_reset_clears_escort_state(self):
        gm, _ = _escort()
        gm.protectee_id = "vip_1"
        gm.escort_destination = (100.0, 50.0)
        gm.protectee_arrived = True
        gm.reset()
        assert gm.protectee_id is None
        assert gm.escort_destination is None
        assert gm.protectee_arrived is False
        assert gm.protectee_lost is False
