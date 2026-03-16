"""
Tests for city3d.html multi-phase campaign progression.
Source-string tests that verify the HTML file contains required code patterns.

Demonstrates campaign.py and scenario.py from sim_engine.

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
# 1. CAMPAIGN STATE (demonstrates campaign.py)
# =========================================================================

class TestCampaignState:
    def test_campaign_phase_variable(self, source):
        assert "campaignPhase" in source, "Missing campaignPhase state variable"

    def test_campaign_phase_starts_at_zero(self, source):
        assert "campaignPhase = 0" in source, "campaignPhase should start at 0 (not started)"

    def test_campaign_timer_variable(self, source):
        assert "campaignTimer" in source, "Missing campaignTimer for phase timing"

    def test_campaign_definitions(self, source):
        assert "CAMPAIGN_PHASES" in source, "Missing CAMPAIGN_PHASES definitions array"


# =========================================================================
# 2. PHASE DEFINITIONS (demonstrates scenario.py)
# =========================================================================

class TestPhaseDefinitions:
    def test_phase_1_morning_protest(self, source):
        assert "Morning Protest" in source, "Missing Phase 1 name"

    def test_phase_2_riot_response(self, source):
        assert "Riot Response" in source, "Missing Phase 2 name"

    def test_phase_3_aftermath(self, source):
        assert "Aftermath" in source, "Missing Phase 3 name"

    def test_campaign_title(self, source):
        assert "City Crisis" in source, "Missing campaign title"

    def test_phase_objectives(self, source):
        assert "Monitor situation" in source, "Missing Phase 1 objective"
        assert "Contain riot" in source, "Missing Phase 2 objective"


# =========================================================================
# 3. PHASE TRANSITION LOGIC
# =========================================================================

class TestPhaseTransitions:
    def test_phase_1_auto_transition(self, source):
        assert "campaignTimer" in source and "60" in source, \
            "Phase 1 should auto-complete after 60 seconds"

    def test_phase_2_to_3_condition(self, source):
        assert "campaignPhase === 3" in source or "campaignPhase = 3" in source, \
            "Must transition to phase 3"

    def test_skip_phase_keypress(self, source):
        assert "KeyP" in source, "Missing P key for phase skip"


# =========================================================================
# 4. HUD DISPLAY
# =========================================================================

class TestCampaignHUD:
    def test_campaign_phase_display_element(self, source):
        assert "campaign-phase" in source, "Missing campaign-phase HUD element"

    def test_phase_indicator_colors(self, source):
        assert "campaign-phase" in source, "Campaign phase display needed"


# =========================================================================
# 5. END-OF-CAMPAIGN STATS OVERLAY
# =========================================================================

class TestCampaignOverlay:
    def test_campaign_overlay_element(self, source):
        assert "campaign-overlay" in source, "Missing campaign stats overlay div"

    def test_mission_complete_text(self, source):
        assert "MISSION COMPLETE" in source, "Missing MISSION COMPLETE text"

    def test_mission_failed_text(self, source):
        assert "MISSION FAILED" in source, "Missing MISSION FAILED text"

    def test_overlay_shows_arrests(self, source):
        assert "arrestCount" in source and "campaign-overlay" in source, \
            "Overlay should show arrest stats"

    def test_overlay_shows_budget(self, source):
        assert "policeBudget" in source and "campaign-overlay" in source, \
            "Overlay should show budget stats"
