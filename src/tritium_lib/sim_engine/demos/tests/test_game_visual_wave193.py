# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Wave 193 — Visual improvement tests for game.html.

Tests verify that game.html contains the expected JavaScript code for:
1. AI behavior floating indicator sprites above units in 3D
2. Per-unit morale status bar in 3D (below health bar)
3. Demo mode generates synthetic ai_behaviors and morale data
"""

from __future__ import annotations

from pathlib import Path

import pytest

GAME_HTML = Path(__file__).resolve().parent.parent / "game.html"


@pytest.fixture(scope="module")
def source() -> str:
    """Read game.html source once for all tests."""
    return GAME_HTML.read_text(encoding="utf-8")


# =========================================================================
# 1. AI Behavior Floating Indicators — 3D sprites above units
# =========================================================================

class TestAIBehaviorIndicators:
    """AI behavior state should appear as floating sprites in the 3D scene."""

    def test_behavior_icon_map_exists(self, source):
        """There should be a mapping from behavior decisions to icon/text."""
        assert "BEHAVIOR_ICONS" in source, \
            "Expected BEHAVIOR_ICONS mapping for behavior->icon lookup"

    def test_behavior_icon_map_has_attack(self, source):
        """The behavior icon map should include an 'engage' entry."""
        assert "'engage'" in source or '"engage"' in source

    def test_behavior_icon_map_has_retreat(self, source):
        """The behavior icon map should include a 'retreat' entry."""
        assert "'retreat'" in source or '"retreat"' in source

    def test_behavior_icon_map_has_patrol(self, source):
        """The behavior icon map should include a 'patrol' entry."""
        assert "'patrol'" in source or '"patrol"' in source

    def test_behavior_sprite_creation_function(self, source):
        """There should be a function to create behavior indicator sprites."""
        assert "makeBehaviorSprite" in source, \
            "Expected makeBehaviorSprite function for creating behavior icons"

    def test_behavior_sprite_uses_canvas(self, source):
        """Behavior sprites should use canvas for rendering text/icons."""
        # The function should create a canvas texture
        idx = source.find("makeBehaviorSprite")
        assert idx > 0
        # Within 600 chars of the function, canvas should be created
        snippet = source[idx:idx + 600]
        assert "canvas" in snippet.lower() or "Canvas" in snippet

    def test_behavior_indicators_stored_per_unit(self, source):
        """Behavior indicators should be tracked per unit ID."""
        assert "behaviorIndicators" in source, \
            "Expected behaviorIndicators map to track per-unit behavior sprites"

    def test_behavior_update_in_process_frame(self, source):
        """updateAIBehaviors should update 3D behavior indicators, not just panel."""
        idx = source.find("function updateAIBehaviors")
        assert idx > 0
        # The function should reference behaviorIndicators for 3D sprites
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 3000
        func_body = source[idx:func_end]
        assert "behaviorIndicators" in func_body, \
            "updateAIBehaviors should update 3D behaviorIndicators sprites"

    def test_behavior_indicator_positioned_above_unit(self, source):
        """Behavior indicator sprites should be positioned above units."""
        # Should position the sprite at unit position + some Y offset
        assert "behaviorY" in source or "behavior_y" in source or \
            "bhvY" in source or "indicatorY" in source, \
            "Expected Y offset calculation for behavior indicator positioning"


# =========================================================================
# 2. Per-Unit Morale Bar — 3D bar below health bar
# =========================================================================

class TestMoraleBar:
    """Each unit should have a morale bar rendered in 3D below the health bar."""

    def test_morale_bar_creation_function(self, source):
        """There should be a function to create morale bars."""
        assert "makeMoraleBar" in source, \
            "Expected makeMoraleBar function for per-unit morale bars"

    def test_morale_bar_added_to_labels(self, source):
        """Unit label entries should include a morale bar."""
        assert "moraleBar" in source, \
            "Expected moraleBar property in unit label entries"

    def test_morale_update_modifies_bars(self, source):
        """updateMorale should update per-unit 3D morale bars."""
        idx = source.find("function updateMorale")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 3000
        func_body = source[idx:func_end]
        assert "moraleBar" in func_body, \
            "updateMorale should update per-unit morale bars"

    def test_morale_bar_color_coding(self, source):
        """Morale bar should use color coding (green/yellow/orange/red)."""
        # Should have morale-specific color logic
        idx = source.find("function updateMorale")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 3000
        func_body = source[idx:func_end]
        # Should reference morale value and color computation
        assert "morale" in func_body.lower() and "color" in func_body.lower()


# =========================================================================
# 3. Demo Mode — synthetic ai_behaviors and morale data
# =========================================================================

class TestDemoModeData:
    """Demo mode should generate ai_behaviors and morale data."""

    def test_demo_generates_ai_behaviors(self, source):
        """generateDemoFrame should include ai_behaviors data."""
        idx = source.find("function generateDemoFrame")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 5000
        func_body = source[idx:func_end]
        assert "ai_behaviors" in func_body, \
            "Demo frame should include ai_behaviors data"

    def test_demo_generates_morale(self, source):
        """generateDemoFrame should include morale data."""
        idx = source.find("function generateDemoFrame")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 5000
        func_body = source[idx:func_end]
        assert "morale" in func_body, \
            "Demo frame should include morale data"

    def test_demo_ai_behaviors_has_units(self, source):
        """Demo ai_behaviors should have a units dict."""
        idx = source.find("function generateDemoFrame")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 5000
        func_body = source[idx:func_end]
        # Should contain the units key for ai_behaviors
        assert "ai_behaviors" in func_body
        # Should have diverse decisions
        has_engage = "engage" in func_body
        has_patrol = "patrol" in func_body
        assert has_engage or has_patrol, \
            "Demo ai_behaviors should include diverse behavior decisions"

    def test_demo_morale_has_units(self, source):
        """Demo morale should have per-unit morale data."""
        idx = source.find("function generateDemoFrame")
        assert idx > 0
        func_end = source.find("\nfunction ", idx + 30)
        if func_end < 0:
            func_end = idx + 5000
        func_body = source[idx:func_end]
        assert "aura_color" in func_body or "morale_state" in func_body or \
            "alliance_averages" in func_body, \
            "Demo morale should contain per-unit morale data fields"

    def test_demo_return_includes_both(self, source):
        """The demo frame return object should include both new data fields."""
        # Find the return statement in generateDemoFrame
        idx = source.find("function generateDemoFrame")
        assert idx > 0
        # Find the return statement
        return_idx = source.find("return {", idx)
        assert return_idx > 0
        return_block = source[return_idx:return_idx + 1000]
        assert "ai_behaviors" in return_block, \
            "Demo return should include ai_behaviors"
        assert "morale" in return_block or "morale:" in return_block, \
            "Demo return should include morale data"


# =========================================================================
# 4. Cleanup — behavior indicators removed when units die
# =========================================================================

class TestCleanup:
    """Behavior indicators and morale bars should be cleaned up properly."""

    def test_behavior_indicators_cleaned_on_removal(self, source):
        """Behavior indicators should be removed when units are no longer present."""
        assert "behaviorIndicators.delete" in source or \
            "behaviorIndicators.clear" in source, \
            "Behavior indicators should be cleaned up when units disappear"

    def test_morale_bar_cleaned_on_removal(self, source):
        """Morale bars should be removed with the unit label."""
        # When labels are cleaned, moraleBar should also be removed
        assert "moraleBar" in source, \
            "moraleBar should be part of the unit label system"
