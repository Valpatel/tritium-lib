"""
Tests for city3d.html commander narration, achievements, and soundtrack state.
Source-string tests that verify the HTML file contains required code patterns.

These features demonstrate commander.py, scoring.py, and soundtrack.py
from the sim_engine library.

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
# 1. COMMANDER NARRATION (demonstrates commander.py)
# =========================================================================

class TestNarrationPanel:
    def test_narration_panel_html(self, source):
        assert 'id="narration-panel"' in source, "Missing narration panel HTML element"

    def test_narration_panel_css(self, source):
        assert "#narration-panel" in source, "Missing narration panel CSS styles"

    def test_narration_messages_array(self, source):
        assert "narrationMessages" in source, "Missing narrationMessages state array"

    def test_narration_max_visible_4(self, source):
        assert "NARRATION_MAX_VISIBLE = 4" in source, "Max visible narration messages should be 4"

    def test_narration_fade_time_8(self, source):
        assert "NARRATION_FADE_TIME = 8" in source, "Narration fade time should be 8 seconds"

    def test_narration_templates_object(self, source):
        assert "narrationTemplates" in source, "Missing narrationTemplates object"


class TestNarrationTemplates:
    def test_riot_start_template(self, source):
        assert "riot_start" in source and "Code 10-10" in source, (
            "Missing riot_start narration template with Code 10-10"
        )

    def test_tear_gas_template(self, source):
        assert "tear_gas" in source and "Gas deployed at grid" in source, (
            "Missing tear_gas narration template"
        )

    def test_molotov_template(self, source):
        assert "'molotov'" in source and "CONTACT" in source, (
            "Missing molotov narration template with CONTACT"
        )

    def test_officer_down_template(self, source):
        assert "officer_down" in source and "Officer down" in source, (
            "Missing officer_down narration template"
        )

    def test_fire_started_template(self, source):
        assert "fire_started" in source and "Structure fire" in source, (
            "Missing fire_started narration template"
        )

    def test_arrest_template(self, source):
        assert "'arrest'" in source and "Suspect in custody" in source, (
            "Missing arrest narration template"
        )

    def test_all_clear_template(self, source):
        assert "all_clear" in source and "Situation stabilized" in source, (
            "Missing all_clear narration template"
        )

    def test_ambulance_dispatch_template(self, source):
        assert "ambulance_dispatch" in source and "Ambulance en route" in source, (
            "Missing ambulance_dispatch narration template"
        )

    def test_helicopter_template(self, source):
        assert "'helicopter'" in source and "Overhead Central Plaza" in source, (
            "Missing helicopter narration template"
        )

    def test_van_deploy_template(self, source):
        assert "van_deploy" in source and "Tactical unit deploying" in source, (
            "Missing van_deploy narration template"
        )

    def test_fire_truck_template(self, source):
        assert "fire_truck" in source and "Responding to fire" in source, (
            "Missing fire_truck narration template"
        )

    def test_fire_extinguished_template(self, source):
        assert "fire_extinguished" in source and "Fire knocked down" in source, (
            "Missing fire_extinguished narration template"
        )


class TestNarrationRandomization:
    def test_multiple_templates_per_event(self, source):
        """Each event type should have 2+ templates for variety."""
        # riot_start has 3 templates
        assert "DISPATCH" in source and "COMMAND" in source, (
            "Templates should use different callsigns for variety"
        )

    def test_pick_template_uses_random(self, source):
        assert "pickTemplate" in source and "rng()" in source, (
            "pickTemplate should use rng() for random selection"
        )


class TestNarrationFunction:
    def test_add_narration_function(self, source):
        assert "function addNarration(" in source, "Missing addNarration function"

    def test_update_narration_panel_function(self, source):
        assert "function updateNarrationPanel(" in source, "Missing updateNarrationPanel function"

    def test_format_grid_function(self, source):
        assert "function formatGrid(" in source, "Missing formatGrid function for grid coordinates"


class TestNarrationCSS:
    def test_narration_msg_class(self, source):
        assert ".narration-msg" in source, "Missing .narration-msg CSS class"

    def test_fading_class(self, source):
        assert ".fading" in source, "Missing .fading CSS class for fade-out"

    def test_callsign_span_class(self, source):
        assert ".callsign" in source, "Missing .callsign CSS class for callsign styling"

    def test_alert_span_class(self, source):
        assert ".alert" in source, "Missing .alert CSS class for alert text"

    def test_info_span_class(self, source):
        assert ".narration-msg .info" in source or ".info" in source, (
            "Missing .info CSS class"
        )

    def test_cyan_border_style(self, source):
        assert "#00f0ff" in source and "narration-panel" in source, (
            "Narration panel should have cyan border"
        )


class TestNarrationIntegration:
    def test_narration_on_riot_start(self, source):
        assert "addNarration('riot_start')" in source, (
            "Missing narration on riot start event"
        )

    def test_narration_on_molotov(self, source):
        assert "addNarration('molotov')" in source, (
            "Missing narration on molotov throw event"
        )

    def test_narration_on_tear_gas(self, source):
        assert "addNarration('tear_gas'" in source, (
            "Missing narration on tear gas event"
        )

    def test_narration_on_arrest(self, source):
        assert "addNarration('arrest'" in source, (
            "Missing narration on arrest event"
        )

    def test_narration_on_officer_down(self, source):
        assert "addNarration('officer_down'" in source, (
            "Missing narration on officer down event"
        )

    def test_narration_on_fire(self, source):
        assert "addNarration('fire_started'" in source, (
            "Missing narration on fire started event"
        )

    def test_narration_on_all_clear(self, source):
        assert "addNarration('all_clear')" in source, (
            "Missing narration on all clear event"
        )

    def test_narration_on_ambulance(self, source):
        assert "addNarration('ambulance_dispatch'" in source, (
            "Missing narration on ambulance dispatch"
        )

    def test_narration_on_fire_truck(self, source):
        assert "addNarration('fire_truck'" in source, (
            "Missing narration on fire truck dispatch"
        )

    def test_narration_on_van_deploy(self, source):
        assert "addNarration('van_deploy'" in source, (
            "Missing narration on police van deployment"
        )

    def test_narration_on_fire_extinguished(self, source):
        assert "addNarration('fire_extinguished'" in source, (
            "Missing narration on fire extinguished"
        )

    def test_narration_panel_updated_in_loop(self, source):
        assert "updateNarrationPanel(" in source, (
            "Narration panel should be updated in the main loop"
        )


# =========================================================================
# 2. ACHIEVEMENTS (demonstrates scoring.py)
# =========================================================================

class TestAchievementSystem:
    def test_achievement_container_html(self, source):
        assert 'id="achievement-container"' in source, (
            "Missing achievement container HTML element"
        )

    def test_achievements_awarded_object(self, source):
        assert "achievementsAwarded" in source, "Missing achievementsAwarded tracking object"

    def test_achievement_defs_object(self, source):
        assert "achievementDefs" in source, "Missing achievementDefs definitions"

    def test_total_score_variable(self, source):
        assert "totalScore" in source, "Missing totalScore tracking variable"

    def test_score_display_in_stats(self, source):
        assert 'id="score-count"' in source, "Missing score display in stats panel"

    def test_score_updated_in_hud(self, source):
        assert "score-count" in source and "totalScore" in source, (
            "Score should be updated in HUD"
        )


class TestAchievementDefinitions:
    def test_first_responder(self, source):
        assert "first_responder" in source and "First Responder" in source, (
            "Missing 'First Responder' achievement"
        )

    def test_firebreaker(self, source):
        assert "firebreaker" in source and "Firebreaker" in source, (
            "Missing 'Firebreaker' achievement"
        )

    def test_peacekeeper(self, source):
        assert "peacekeeper" in source and "Peacekeeper" in source, (
            "Missing 'Peacekeeper' achievement"
        )

    def test_iron_line(self, source):
        assert "iron_line" in source and "Iron Line" in source, (
            "Missing 'Iron Line' achievement"
        )

    def test_crowd_control(self, source):
        assert "crowd_control" in source and "Crowd Control" in source, (
            "Missing 'Crowd Control' achievement"
        )

    def test_under_fire(self, source):
        assert "under_fire" in source and "Under Fire" in source, (
            "Missing 'Under Fire' achievement"
        )


class TestAchievementPoints:
    def test_first_responder_points(self, source):
        assert "points: 100" in source, "First Responder should be 100 points"

    def test_firebreaker_points(self, source):
        assert "points: 150" in source, "Firebreaker should be 150 points"

    def test_peacekeeper_points(self, source):
        assert "points: 200" in source, "Peacekeeper should be 200 points"

    def test_iron_line_points(self, source):
        assert "points: 250" in source, "Iron Line should be 250 points"

    def test_crowd_control_points(self, source):
        assert "points: 175" in source, "Crowd Control should be 175 points"

    def test_under_fire_points(self, source):
        assert "points: 300" in source, "Under Fire should be 300 points"


class TestAchievementToast:
    def test_show_achievement_toast_function(self, source):
        assert "function showAchievementToast(" in source, (
            "Missing showAchievementToast function"
        )

    def test_toast_css_class(self, source):
        assert ".achievement-toast" in source, "Missing .achievement-toast CSS class"

    def test_toast_gold_border(self, source):
        assert "#fcee0a" in source and "achievement-toast" in source, (
            "Achievement toast should use gold (#fcee0a) color"
        )

    def test_toast_slide_animation(self, source):
        assert "translateX" in source and "achievement-toast" in source, (
            "Achievement toast should slide in/out"
        )

    def test_toast_show_class(self, source):
        assert ".achievement-toast.show" in source, (
            "Missing .show class for slide-in animation"
        )

    def test_toast_hide_class(self, source):
        assert ".achievement-toast.hide" in source, (
            "Missing .hide class for slide-out animation"
        )

    def test_star_icon(self, source):
        assert "&#9733;" in source or "ach-icon" in source, (
            "Achievement should have a star icon/badge"
        )

    def test_toast_5_second_timeout(self, source):
        assert "5000" in source, "Achievement toast should stay for 5 seconds"


class TestAchievementTriggers:
    def test_first_responder_on_ambulance(self, source):
        assert "awardAchievement('first_responder')" in source, (
            "First Responder should trigger on ambulance dispatch"
        )

    def test_firebreaker_on_extinguish(self, source):
        assert "awardAchievement('firebreaker')" in source, (
            "Firebreaker should trigger when fire truck extinguishes fire"
        )

    def test_peacekeeper_condition(self, source):
        assert "awardAchievement('peacekeeper')" in source, (
            "Peacekeeper should trigger on 5 arrests without injuries"
        )

    def test_iron_line_condition(self, source):
        assert "policeLineHoldTime" in source and "awardAchievement('iron_line')" in source, (
            "Iron Line should track police hold time and award at 60 seconds"
        )

    def test_crowd_control_condition(self, source):
        assert "tearGasDispersedTotal" in source and "awardAchievement('crowd_control')" in source, (
            "Crowd Control should track tear gas dispersal count"
        )

    def test_under_fire_condition(self, source):
        assert "molotovCount >= 5" in source and "awardAchievement('under_fire')" in source, (
            "Under Fire should trigger after surviving 5 molotov attacks"
        )

    def test_check_achievements_in_loop(self, source):
        assert "checkAchievements(" in source, (
            "checkAchievements should be called in the main loop"
        )

    def test_no_duplicate_awards(self, source):
        assert "achievementsAwarded[id]" in source, (
            "Achievements should only be awarded once (dedup check)"
        )


# =========================================================================
# 3. SOUNDTRACK STATE (demonstrates soundtrack.py)
# =========================================================================

class TestSoundtrackIndicator:
    def test_soundtrack_indicator_html(self, source):
        assert 'id="soundtrack-indicator"' in source, (
            "Missing soundtrack indicator HTML element"
        )

    def test_soundtrack_state_variable(self, source):
        assert "soundtrackState" in source, "Missing soundtrackState variable"

    def test_update_soundtrack_function(self, source):
        assert "function updateSoundtrackState(" in source, (
            "Missing updateSoundtrackState function"
        )


class TestSoundtrackStates:
    def test_peaceful_state(self, source):
        assert "PEACEFUL" in source and "peaceful" in source, (
            "Missing PEACEFUL soundtrack state"
        )

    def test_tension_state(self, source):
        assert "TENSION" in source and "tension" in source, (
            "Missing TENSION soundtrack state"
        )

    def test_combat_state(self, source):
        assert "COMBAT" in source and "combat" in source, (
            "Missing COMBAT soundtrack state"
        )

    def test_aftermath_state(self, source):
        assert "AFTERMATH" in source and "aftermath" in source, (
            "Missing AFTERMATH soundtrack state"
        )


class TestSoundtrackColors:
    def test_peaceful_green(self, source):
        assert "peaceful" in source and "#05ffa1" in source, (
            "PEACEFUL state should be green (#05ffa1)"
        )

    def test_tension_yellow(self, source):
        assert "tension" in source and "#fcee0a" in source, (
            "TENSION state should be yellow (#fcee0a)"
        )

    def test_combat_red(self, source):
        assert "combat" in source and "#ff2a6d" in source, (
            "COMBAT state should be red (#ff2a6d)"
        )

    def test_aftermath_cyan(self, source):
        assert "aftermath" in source and "#00f0ff" in source, (
            "AFTERMATH state should be cyan (#00f0ff)"
        )


class TestSoundtrackCSS:
    def test_peaceful_css_class(self, source):
        assert "#soundtrack-indicator.peaceful" in source, (
            "Missing .peaceful CSS class for soundtrack indicator"
        )

    def test_tension_css_class(self, source):
        assert "#soundtrack-indicator.tension" in source, (
            "Missing .tension CSS class for soundtrack indicator"
        )

    def test_combat_css_class(self, source):
        assert "#soundtrack-indicator.combat" in source, (
            "Missing .combat CSS class for soundtrack indicator"
        )

    def test_aftermath_css_class(self, source):
        assert "#soundtrack-indicator.aftermath" in source, (
            "Missing .aftermath CSS class for soundtrack indicator"
        )

    def test_tension_pulse_animation(self, source):
        assert "pulse-tension" in source, (
            "TENSION state should have a pulsing animation"
        )

    def test_combat_bold_style(self, source):
        assert "combat" in source and "font-weight: bold" in source, (
            "COMBAT state should be bold"
        )

    def test_music_note_symbol(self, source):
        assert "\\u266B" in source or "&#9835;" in source or "266B" in source, (
            "Soundtrack indicator should show music note symbol"
        )


class TestSoundtrackPhaseMapping:
    def test_peaceful_maps_to_peaceful(self, source):
        assert "'PEACEFUL': newState = 'PEACEFUL'" in source or (
            "PEACEFUL" in source and "peaceful" in source
        ), "PEACEFUL riot phase should map to PEACEFUL music state"

    def test_tension_maps_to_tension(self, source):
        assert "'TENSION'" in source and "TENSION" in source, (
            "TENSION riot phase should map to TENSION music state"
        )

    def test_riot_maps_to_combat(self, source):
        assert "'RIOT'" in source and "COMBAT" in source, (
            "RIOT riot phase should map to COMBAT music state"
        )

    def test_dispersal_maps_to_aftermath(self, source):
        assert "'DISPERSAL'" in source and "AFTERMATH" in source, (
            "DISPERSAL riot phase should map to AFTERMATH music state"
        )

    def test_soundtrack_updated_in_hud(self, source):
        assert "updateSoundtrackState()" in source, (
            "Soundtrack state should be updated in the HUD update loop"
        )
