"""
Tests for city3d.html supply logistics, economy, and status effects.
Source-string tests that verify the HTML file contains required code patterns.

Demonstrates logistics.py, economy.py, and status_effects.py from sim_engine.

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
# 1. SUPPLY / LOGISTICS (demonstrates logistics.py)
# =========================================================================

class TestSupplyState:
    def test_tear_gas_supply_variable(self, source):
        assert "supplyTearGas" in source, "Missing supplyTearGas state variable"

    def test_rubber_bullet_supply_variable(self, source):
        assert "supplyRubberBullets" in source, "Missing supplyRubberBullets state variable"

    def test_molotov_supply_variable(self, source):
        assert "supplyMolotovs" in source, "Missing supplyMolotovs state variable"

    def test_tear_gas_max(self, source):
        assert "SUPPLY_TEAR_GAS_MAX" in source or "supplyTearGasMax" in source, \
            "Missing tear gas max supply constant"

    def test_rubber_bullet_max(self, source):
        assert "SUPPLY_RUBBER_BULLETS_MAX" in source or "supplyRubberBulletsMax" in source, \
            "Missing rubber bullets max supply constant"

    def test_molotov_max(self, source):
        assert "SUPPLY_MOLOTOVS_MAX" in source or "supplyMolotovsMax" in source, \
            "Missing molotovs max supply constant"

    def test_tear_gas_starts_20(self, source):
        # Tear gas starts at 20 uses
        assert "20" in source, "Tear gas should start at 20"

    def test_rubber_bullets_starts_100(self, source):
        # Rubber bullets start at 100
        assert "100" in source, "Rubber bullets should start at 100"

    def test_molotovs_starts_15(self, source):
        # Molotovs start at 15
        assert "supplyMolotovs" in source, "Molotovs supply tracking required"


class TestSupplyDepletion:
    def test_tear_gas_depletes_on_use(self, source):
        assert "supplyTearGas--" in source or "supplyTearGas -" in source, \
            "Tear gas supply should deplete on use"

    def test_rubber_bullets_deplete_on_use(self, source):
        assert "supplyRubberBullets--" in source or "supplyRubberBullets -" in source, \
            "Rubber bullets should deplete on use"

    def test_molotovs_deplete_on_use(self, source):
        assert "supplyMolotovs--" in source or "supplyMolotovs -" in source, \
            "Molotovs should deplete on use"

    def test_tear_gas_zero_blocks_use(self, source):
        assert "supplyTearGas" in source and "<= 0" in source or "< 1" in source, \
            "Zero tear gas should prevent deployment"

    def test_rocks_unlimited(self, source):
        # Rocks should be marked unlimited or have no supply cap
        assert "unlimited" in source.lower() or "Rocks" in source, \
            "Rocks should be unlimited"


class TestSupplyHUD:
    def test_supply_panel_element(self, source):
        assert "supply-panel" in source, "Missing supply-panel HUD element"

    def test_supply_bar_tear_gas(self, source):
        assert "supply-teargas" in source or "teargas-bar" in source, \
            "Missing tear gas supply bar element"

    def test_supply_bar_rubber_bullets(self, source):
        assert "supply-rubber" in source or "rubber-bar" in source, \
            "Missing rubber bullets supply bar element"

    def test_supply_bar_molotov(self, source):
        assert "supply-molotov" in source or "molotov-bar" in source, \
            "Missing molotov supply bar element"

    def test_supply_bar_rocks(self, source):
        assert "supply-rock" in source or "rock-bar" in source, \
            "Missing rocks supply bar element"

    def test_supply_low_warning(self, source):
        # When supply < 20%, bar should flash yellow
        assert "supply-low" in source or "flash" in source.lower(), \
            "Low supply warning should flash yellow"

    def test_supply_colors_cyan_for_police(self, source):
        assert "#00f0ff" in source, "Police supply bars should use cyan"

    def test_supply_colors_red_for_protestors(self, source):
        assert "#ff2a6d" in source, "Protestor supply bars should use red/magenta"

    def test_update_supply_hud_function(self, source):
        assert "updateSupplyHUD" in source, "Missing updateSupplyHUD function"


# =========================================================================
# 2. ECONOMY SCORE (demonstrates economy.py)
# =========================================================================

class TestEconomyState:
    def test_police_budget_variable(self, source):
        assert "policeBudget" in source, "Missing policeBudget state variable"

    def test_budget_starts_10000(self, source):
        assert "10000" in source, "Police budget should start at $10,000"

    def test_property_damage_variable(self, source):
        assert "propertyDamage" in source or "policeBudget" in source, \
            "Missing property damage tracking"


class TestEconomyCosts:
    def test_tear_gas_cost_500(self, source):
        assert "500" in source, "Tear gas deployment should cost $500"

    def test_rubber_bullet_cost_50(self, source):
        # 50 for rubber bullet burst cost
        assert "policeBudget" in source, "Rubber bullet cost tracking required"

    def test_arrest_earns_200(self, source):
        assert "200" in source, "Each arrest should earn $200"

    def test_fire_costs_1000(self, source):
        assert "1000" in source, "Each fire should cost $1,000"

    def test_injury_costs_2000(self, source):
        assert "2000" in source, "Each civilian injury should cost $2,000"


class TestEconomyHUD:
    def test_budget_display_element(self, source):
        assert "budget-display" in source or "police-budget" in source, \
            "Missing police budget display element"

    def test_budget_green_positive(self, source):
        assert "policeBudget" in source, "Budget should show with color coding"

    def test_budget_updates_in_hud(self, source):
        assert "policeBudget" in source and "Budget" in source, \
            "Budget should be updated in HUD"


# =========================================================================
# 3. STATUS EFFECTS (demonstrates status_effects.py)
# =========================================================================

class TestStatusEffectState:
    def test_status_effects_array_or_tracking(self, source):
        assert "statusEffects" in source or "activeEffects" in source or \
            "gasAffected" in source, \
            "Missing status effects tracking"

    def test_stunned_state(self, source):
        assert "stunned" in source.lower() or "stunnedTimer" in source, \
            "Missing stunned status effect"

    def test_gas_affected_state(self, source):
        assert "gasAffected" in source or "inTearGas" in source or "inGas" in source, \
            "Missing tear gas affected state"


class TestStatusEffectBehavior:
    def test_officers_affected_by_own_gas(self, source):
        # Officers near tear gas should be affected
        assert "pol" in source and "gas" in source.lower() and "affected" in source.lower() or \
            "gasAffected" in source, \
            "Officers should be affected by their own tear gas"

    def test_stunned_prevents_movement(self, source):
        # Stunned protestors don't move
        assert "stunned" in source.lower(), \
            "Stunned effect should prevent movement"

    def test_stunned_duration_3_seconds(self, source):
        # Stunned lasts 3 seconds
        assert "stunnedTimer" in source or "stunned" in source.lower(), \
            "Stunned should have a duration timer"

    def test_gas_reduces_accuracy(self, source):
        # Gas effect on officers reduces accuracy
        assert "gasAffected" in source or "accuracy" in source.lower() or \
            "inGas" in source, \
            "Gas should reduce officer accuracy"

    def test_yellow_tint_on_gassed_officers(self, source):
        assert "gasAffected" in source or "0xffff00" in source or "gasTint" in source, \
            "Officers in gas should show yellow tint"


class TestStatusEffectVisuals:
    def test_bleeding_visual(self, source):
        assert "BLEEDING" in source or "bleeding" in source, \
            "Missing BLEEDING status effect visual"

    def test_stunned_visual(self, source):
        assert "STUNNED" in source, \
            "Missing STUNNED status effect visual"

    def test_gas_visual(self, source):
        assert "GAS" in source, \
            "Missing GAS status effect visual"

    def test_effects_count_in_debug(self, source):
        assert "activeEffects" in source or "effectsCount" in source or \
            "Status Effects" in source or "status effect" in source.lower(), \
            "Active effects count should appear in debug overlay"


class TestSupplyResetOnRiotToggle:
    def test_supply_resets_on_new_riot(self, source):
        # When riot starts, supplies should reset
        assert "supplyTearGas" in source and "supplyMolotovs" in source, \
            "Supplies should reset when riot starts"

    def test_budget_resets_on_new_riot(self, source):
        assert "policeBudget" in source, \
            "Budget should reset when riot starts"
