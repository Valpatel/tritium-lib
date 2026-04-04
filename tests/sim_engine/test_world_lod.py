# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.lod — level of detail system."""

import pytest
from unittest.mock import MagicMock

from tritium_lib.sim_engine.world.lod import (
    LODSystem,
    LODTier,
    ViewportState,
    TIER_TICK_DIVISOR,
    TIER_IDLE_THRESHOLD,
    TIER_TELEMETRY_DIVISOR,
)


def _make_target(tid, pos, alliance="neutral", is_combatant=False):
    t = MagicMock()
    t.target_id = tid
    t.position = pos
    t.alliance = alliance
    t.is_combatant = is_combatant
    t.status = "active"
    return t


class TestLODTier:
    def test_ordering(self):
        assert LODTier.FULL < LODTier.MEDIUM < LODTier.LOW

    def test_values(self):
        assert LODTier.FULL == 0
        assert LODTier.MEDIUM == 1
        assert LODTier.LOW == 2


class TestViewportState:
    def test_defaults(self):
        v = ViewportState()
        assert v.center_x == 0.0
        assert v.center_y == 0.0
        assert v.radius == 150.0
        assert v._set is False


class TestLODSystem:
    def test_no_viewport_returns_full(self):
        lod = LODSystem()
        t = _make_target("t1", (500.0, 500.0))
        tier = lod.compute_tier(t)
        assert tier == LODTier.FULL

    def test_within_viewport_is_full(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        t = _make_target("t1", (50.0, 50.0))
        tier = lod.compute_tier(t)
        assert tier == LODTier.FULL

    def test_nearby_offscreen_is_medium(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        # Beyond 1.2x radius but within 3x = MEDIUM
        t = _make_target("t1", (200.0, 0.0))
        tier = lod.compute_tier(t)
        assert tier == LODTier.MEDIUM

    def test_far_away_is_low(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        # Beyond 3x radius = LOW
        t = _make_target("t1", (500.0, 0.0))
        tier = lod.compute_tier(t)
        assert tier == LODTier.LOW

    def test_combatant_never_lower_than_medium(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        t = _make_target("t1", (500.0, 0.0), alliance="hostile", is_combatant=True)
        tier = lod.compute_tier(t)
        assert tier == LODTier.MEDIUM

    def test_neutral_combatant_can_be_low(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        t = _make_target("t1", (500.0, 0.0), alliance="neutral", is_combatant=True)
        tier = lod.compute_tier(t)
        assert tier == LODTier.LOW

    def test_compute_tiers_batch(self):
        lod = LODSystem()
        lod.update_viewport(0.0, 0.0, radius=100.0)
        targets = {
            "close": _make_target("close", (10.0, 10.0)),
            "mid": _make_target("mid", (200.0, 0.0)),
            "far": _make_target("far", (500.0, 0.0)),
        }
        tiers = lod.compute_tiers(targets)
        assert tiers["close"] == LODTier.FULL
        assert tiers["mid"] == LODTier.MEDIUM
        assert tiers["far"] == LODTier.LOW

    def test_should_tick(self):
        lod = LODSystem()
        lod._tiers["full"] = LODTier.FULL
        lod._tiers["medium"] = LODTier.MEDIUM
        lod._tiers["low"] = LODTier.LOW

        # FULL ticks every frame
        assert lod.should_tick("full", 0) is True
        assert lod.should_tick("full", 1) is True

        # MEDIUM ticks every 3rd frame
        assert lod.should_tick("medium", 0) is True
        assert lod.should_tick("medium", 1) is False
        assert lod.should_tick("medium", 3) is True

        # LOW ticks every 10th frame
        assert lod.should_tick("low", 0) is True
        assert lod.should_tick("low", 5) is False
        assert lod.should_tick("low", 10) is True

    def test_should_run_behaviors(self):
        lod = LODSystem()
        lod._tiers["low"] = LODTier.LOW
        lod._tiers["full"] = LODTier.FULL

        # LOW never runs behaviors
        assert lod.should_run_behaviors("low", 0) is False
        assert lod.should_run_behaviors("low", 10) is False

        # FULL always runs
        assert lod.should_run_behaviors("full", 0) is True

    def test_update_viewport_from_zoom(self):
        lod = LODSystem()
        lod.update_viewport(100.0, 200.0, zoom=18.0)
        v = lod.viewport
        assert v.center_x == 100.0
        assert v.center_y == 200.0
        assert v.zoom == 18.0
        # At zoom 18: 300 * 2^(16-18) = 300 * 0.25 = 75
        assert abs(v.radius - 75.0) < 1.0

    def test_has_viewport(self):
        lod = LODSystem()
        assert lod.has_viewport is False
        lod.update_viewport(0.0, 0.0)
        assert lod.has_viewport is True

    def test_get_stats(self):
        lod = LODSystem()
        lod._tiers = {
            "a": LODTier.FULL,
            "b": LODTier.FULL,
            "c": LODTier.MEDIUM,
            "d": LODTier.LOW,
        }
        stats = lod.get_stats()
        assert stats["FULL"] == 2
        assert stats["MEDIUM"] == 1
        assert stats["LOW"] == 1

    def test_remove_unit(self):
        lod = LODSystem()
        lod._tiers["t1"] = LODTier.FULL
        lod.remove_unit("t1")
        assert lod.get_tier("t1") == LODTier.FULL  # default

    def test_reset(self):
        lod = LODSystem()
        lod._tiers = {"a": LODTier.LOW, "b": LODTier.MEDIUM}
        lod.reset()
        assert len(lod._tiers) == 0
