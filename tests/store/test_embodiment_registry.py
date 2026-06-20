# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for EmbodimentRegistry — the shared slot/occupant registry.

This registry was extracted out of the SC web router (app.routers.embodiments)
into tritium_lib.store so the simulation engine and MQTT bridge import the SAME
state from the LOWEST layer instead of reaching up into the web router (a
layering inversion). These tests pin the pure state operations + the JSON
persistence contract: the durable slot inventory + the per-Graphling leaderboard
survive a restart, but occupancy does NOT (a Graphling re-checks-in after a
reboot).
"""

import json

import pytest

from tritium_lib.store.embodiment_registry import EmbodimentRegistry


@pytest.fixture
def reg():
    return EmbodimentRegistry()


class TestSlotLifecycle:
    def test_register_is_idempotent_and_starts_as_stand_in(self, reg):
        a = reg.register("u1", kind="stand-in", label="Unit 1", capabilities=["move", "fire"])
        b = reg.register("u1")  # same id -> same record, not a reset
        assert a is b
        assert a["occupant"] == "stand-in"
        assert a["graphling_id"] is None
        assert a["capabilities"] == ["move", "fire"]

    def test_occupancy_and_pending_action_flow(self, reg):
        reg.register("u1")
        assert not reg.is_occupied("u1")
        assert reg.occupied_ids() == []
        # A checkin is a router concern; simulate the occupant taking the slot.
        reg.embodiments["u1"]["occupant"] = "graphling"
        reg.embodiments["u1"]["graphling_id"] = "g-orion"
        assert reg.is_occupied("u1")
        assert reg.occupied_ids() == ["u1"]
        # perception in, action out
        reg.set_perception("u1", {"self": {"x": 1.0}})
        assert reg.get_perception("u1")["self"]["x"] == 1.0
        reg.embodiments["u1"]["pending_action"] = {"action": {"type": "move_to"}}
        assert reg.pop_pending_action("u1")["action"]["type"] == "move_to"
        assert reg.pop_pending_action("u1") is None  # consumed exactly once

    def test_pop_pending_action_declines_for_stand_in(self, reg):
        reg.register("u1")
        reg.embodiments["u1"]["pending_action"] = {"action": {"type": "fire"}}
        # stand-in slot: the engine must NOT consume an action for it
        assert reg.pop_pending_action("u1") is None

    def test_deregister_silent(self, reg):
        reg.register("u1")
        assert reg.deregister_silent("u1") is True
        assert reg.deregister_silent("u1") is False  # already gone, no raise


class TestLeaderboardStats:
    def test_record_kill_and_score(self, reg):
        reg.record_kill("g-orion", points=100)
        reg.record_kill("g-orion", points=100)
        reg.record_score("g-orion", 50)
        s = reg.stats["g-orion"]
        assert s["kills"] == 2
        assert s["score"] == 250

    def test_empty_graphling_id_is_ignored(self, reg):
        reg.record_kill("", points=100)
        assert reg.stats == {}


class TestPersistence:
    def test_save_load_roundtrip(self, reg, tmp_path):
        p = tmp_path / "emb.json"
        reg.configure_persistence(p)
        reg.register("mqtt_rover-1", kind="stand-in", label="Rover", capabilities=["move"])
        reg.embodiments["mqtt_rover-1"]["occupant"] = "graphling"
        reg.embodiments["mqtt_rover-1"]["graphling_id"] = "g-orion"
        reg.record_kill("g-orion", points=100)  # triggers a save
        reg.save()
        assert p.exists()
        data = json.loads(p.read_text())
        assert any(s["embodiment_id"] == "mqtt_rover-1" for s in data["slots"])
        assert data["stats"]["g-orion"]["kills"] == 1

    def test_occupancy_does_not_survive_restart_but_leaderboard_does(self, reg, tmp_path):
        p = tmp_path / "emb.json"
        reg.configure_persistence(p)
        reg.register("mqtt_rover-1")
        reg.embodiments["mqtt_rover-1"]["occupant"] = "graphling"
        reg.embodiments["mqtt_rover-1"]["graphling_id"] = "g-orion"
        reg.record_kill("g-orion", points=100)
        reg.save()
        # Fresh process: a brand-new registry loads the same file.
        fresh = EmbodimentRegistry()
        fresh.configure_persistence(p)
        slot = fresh.embodiments["mqtt_rover-1"]
        assert slot["occupant"] == "stand-in", "occupancy must NOT persist across restart"
        assert slot["graphling_id"] is None
        assert slot["perception"] is None
        assert fresh.stats["g-orion"]["kills"] == 1, "leaderboard MUST persist across restart"

    def test_unconfigured_save_is_a_noop(self, reg):
        # No persistence path configured -> save() must not raise or write.
        reg.register("u1")
        reg.save()  # should be a silent no-op
