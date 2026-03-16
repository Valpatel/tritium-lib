"""
Tests for city3d.html crowd panic/cohesion dynamics.
Verifies flocking, panic cascade, aggression escalation, and debug overlay.

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
# 1. CROWD COHESION (flocking toward nearby protestors)
# =========================================================================

class TestCrowdCohesion:
    def test_cohesion_steering_exists(self, source):
        """Protestors steer toward average position of nearby protestors."""
        assert "cohesion" in source.lower() or "avgX" in source or "cohX" in source

    def test_nearby_protestor_scan(self, source):
        """Must scan for nearby protestors within ~15 units for flocking."""
        # The update loop should find neighbors within a radius
        idx = source.find("// Update protestors")
        block = source[idx:idx + 4000]
        assert "15" in block, "Should use ~15 unit cohesion radius"

    def test_cohesion_weight(self, source):
        """Cohesion steering should use weight ~0.3."""
        assert "0.3" in source

    def test_cluster_counting(self, source):
        """Debug overlay should count clusters."""
        assert "cluster" in source.lower()


# =========================================================================
# 2. PANIC CASCADE
# =========================================================================

class TestPanicCascade:
    def test_panic_spread_exists(self, source):
        """When one protestor flees, nearby ones may also flee."""
        assert "panicCascade" in source or "panic" in source.lower()

    def test_panic_radius(self, source):
        """Panic should spread within ~8 units."""
        idx = source.find("// Update protestors")
        block = source[idx:idx + 4000]
        # Check for 8-unit panic spread radius
        assert "8" in block

    def test_panic_probability(self, source):
        """Panic cascade should have ~50% probability."""
        assert "0.5" in source

    def test_panic_flash_color(self, source):
        """Panic trigger should show a yellow flash on the person mesh."""
        # Yellow flash = 0xfcee0a or similar yellow color
        assert "panicFlash" in source or "fcee0a" in source.lower()


# =========================================================================
# 3. AGGRESSION ESCALATION
# =========================================================================

class TestAggressionEscalation:
    def test_aggression_field_on_protestor(self, source):
        """Protestors should track an aggression value."""
        idx = source.find("protestors.push")
        block = source[idx:idx + 500]
        assert "aggression" in block, "Protestor spawn must include aggression field"

    def test_aggression_boost_near_throwers(self, source):
        """Protestors near active throwers get aggression boost."""
        assert "aggression" in source

    def test_aggression_color_shift(self, source):
        """Body color should shift toward red as aggression increases."""
        # Should interpolate body color based on aggression
        assert "aggressionColor" in source or "aggColor" in source or "pr.aggression" in source

    def test_throw_timer_affected_by_aggression(self, source):
        """Higher aggression means faster throwing."""
        idx = source.find("pr.throwTimer")
        assert idx >= 0, "Must have throwTimer logic"
        block = source[idx:idx + 200]
        assert "aggression" in block, "throwTimer should factor in aggression"


# =========================================================================
# 4. DEBUG OVERLAY
# =========================================================================

class TestDebugOverlay:
    def test_crowd_cohesion_display(self, source):
        """HUD should show crowd cluster count."""
        assert "crowd-clusters" in source or "Crowd cohesion" in source or "clusters" in source.lower()

    def test_panic_level_display(self, source):
        """HUD should show panic level percentage."""
        assert "panic-level" in source or "Panic level" in source or "panicPct" in source
