"""
Tests for city3d.html AI Decision Panel — unit click-to-inspect visualization.
Source-string tests that verify the HTML file contains required code patterns.

Created by Matthew Valancy
Copyright 2026 Valpatel Software LLC
Licensed under AGPL-3.0
"""
import os
import pytest

CITY3D_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "city3d.html"
)


@pytest.fixture(scope="module")
def source():
    with open(CITY3D_PATH, "r") as f:
        return f.read()


# =========================================================================
# 1. CSS PANEL STYLING
# =========================================================================

class TestAIPanelCSS:
    def test_panel_id_exists(self, source):
        assert "#ai-panel" in source, "Missing #ai-panel CSS rule"

    def test_panel_visible_class(self, source):
        assert "#ai-panel.visible" in source, "Missing .visible display toggle"

    def test_panel_cyberpunk_border(self, source):
        assert "#00f0ff" in source, "Panel should use cyan border for cyberpunk theme"

    def test_ai_bar_class(self, source):
        assert ".ai-bar" in source, "Missing .ai-bar for morale/health bars"

    def test_ai_bar_fill_class(self, source):
        assert ".ai-bar-fill" in source, "Missing .ai-bar-fill for bar fill"

    def test_ai_action_class(self, source):
        assert ".ai-action" in source, "Missing .ai-action CSS for current action"

    def test_ai_reason_class(self, source):
        assert ".ai-reason" in source, "Missing .ai-reason CSS for decision reasoning"


# =========================================================================
# 2. HTML PANEL ELEMENT
# =========================================================================

class TestAIPanelHTML:
    def test_panel_div_exists(self, source):
        assert 'id="ai-panel"' in source, "Missing ai-panel div in HTML"


# =========================================================================
# 3. CLICK DETECTION & RAYCASTING
# =========================================================================

class TestClickDetection:
    def test_raycaster_created(self, source):
        assert "aiRaycaster" in source or "new THREE.Raycaster" in source, \
            "Missing raycaster for click detection"

    def test_click_event_listener(self, source):
        assert "addEventListener('click'" in source, "Missing click event listener"

    def test_plane_intersection(self, source):
        assert "intersectPlane" in source, "Must raycast to ground plane"

    def test_nearest_unit_search(self, source):
        assert "bestDist" in source or "nearDist" in source, \
            "Must search for nearest unit to click point"

    def test_search_both_sides(self, source):
        # Must search both protestors and police
        assert "protestors" in source and "police" in source


# =========================================================================
# 4. UNIT SELECTION STATE
# =========================================================================

class TestSelectionState:
    def test_selected_unit_variable(self, source):
        assert "selectedUnit" in source, "Missing selectedUnit state variable"

    def test_selected_side_variable(self, source):
        assert "selectedSide" in source, "Missing selectedSide (police/protestor)"

    def test_deselect_function(self, source):
        assert "deselectUnit" in source, "Missing deselectUnit function"

    def test_escape_key_deselect(self, source):
        assert "'Escape'" in source, "Missing Escape key handler for deselect"


# =========================================================================
# 5. AI STATE DISPLAY
# =========================================================================

class TestAIStateDisplay:
    def test_action_display(self, source):
        assert "getUnitAction" in source, "Missing getUnitAction function"

    def test_reasoning_display(self, source):
        assert "getUnitReasoning" in source, "Missing getUnitReasoning function"

    def test_personality_display(self, source):
        assert "getPersonality" in source, "Missing getPersonality function"

    def test_morale_bar(self, source):
        assert "morale" in source and "ai-bar" in source, "Must show morale bar"

    def test_health_bar(self, source):
        assert "health" in source and "ai-bar" in source, "Must show health bar"

    def test_threat_display(self, source):
        assert "threatStr" in source or "nearDist" in source, \
            "Must show nearest threat distance"

    def test_bar_color_function(self, source):
        assert "barColor" in source, "Missing barColor function for bar coloring"


# =========================================================================
# 6. AI DECISION REASONING STRINGS
# =========================================================================

class TestDecisionReasoning:
    def test_high_morale_reasoning(self, source):
        assert "High morale" in source, "Missing high morale reasoning text"

    def test_low_morale_reasoning(self, source):
        assert "Low morale" in source, "Missing low morale reasoning text"

    def test_fleeing_reasoning(self, source):
        assert "Morale broken" in source or "FLEE" in source, \
            "Missing fleeing reasoning text"

    def test_hold_line_reasoning(self, source):
        assert "HOLD LINE" in source, "Missing hold line reasoning text"

    def test_stunned_reasoning(self, source):
        assert "Stunned" in source or "STUNNED" in source, \
            "Missing stunned reasoning text"


# =========================================================================
# 7. PERSONALITY TYPES
# =========================================================================

class TestPersonalityTypes:
    def test_aggressive_type(self, source):
        assert "AGGRESSIVE" in source, "Missing AGGRESSIVE personality type"

    def test_disciplined_type(self, source):
        assert "DISCIPLINED" in source, "Missing DISCIPLINED personality type"

    def test_cautious_type(self, source):
        assert "CAUTIOUS" in source, "Missing CAUTIOUS personality type"


# =========================================================================
# 8. SELECTION RING
# =========================================================================

class TestSelectionRing:
    def test_white_ring_on_select(self, source):
        assert "0xffffff" in source, "Selection ring should turn white"

    def test_ring_color_restore(self, source):
        assert "ringColor" in source, "Must restore original ring color on deselect"


# =========================================================================
# 9. TARGET LINE
# =========================================================================

class TestTargetLine:
    def test_target_line_geometry(self, source):
        assert "targetLineGeo" in source or "targetLine" in source, \
            "Missing target destination line"

    def test_target_line_visibility(self, source):
        assert "targetLine.visible" in source, "Target line must toggle visibility"


# =========================================================================
# 10. HUD INTEGRATION
# =========================================================================

class TestHUDIntegration:
    def test_update_ai_panel_called(self, source):
        assert "updateAIPanel()" in source, "updateAIPanel must be called in HUD loop"

    def test_controls_hint_click(self, source):
        assert "Inspect" in source, "Controls bar should mention click to inspect"

    def test_controls_hint_esc(self, source):
        assert "Deselect" in source, "Controls bar should mention ESC to deselect"

    def test_deselect_on_riot_clear(self, source):
        # When riot is toggled off, selected unit should be deselected
        assert "deselectUnit" in source, "Must deselect when riot clears"
