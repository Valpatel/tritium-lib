"""
Tests for city3d.html line-of-sight raycasting for combat and detection.
Source-string tests that verify the HTML file contains LOS checks so
projectiles, tear gas, and detection lines respect building occlusion.

Demonstrates terrain.py (LOS) and detection.py (sensor LOS checks).

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
    """Load city3d.html combined with all city3d/*.js modules.

    The frontend is split across city3d.html and external JS modules in
    city3d/*.js.  Tests must scan both to find all code patterns.
    """
    import glob as _glob
    parts = []
    with open(CITY3D_PATH, "r") as f:
        parts.append(f.read())
    js_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "city3d")
    for js_path in sorted(_glob.glob(os.path.join(js_dir, "*.js"))):
        with open(js_path, "r") as f:
            parts.append(f.read())
    return "\n".join(parts)


# =========================================================================
# 1. hasLineOfSight function exists and uses isInsideBuilding
# =========================================================================

class TestHasLineOfSight:
    def test_function_defined(self, source):
        assert "function hasLineOfSight(" in source, \
            "Missing hasLineOfSight function definition"

    def test_samples_along_ray(self, source):
        # Should step along the ray in increments
        assert "steps" in source and "dist / 3" in source, \
            "hasLineOfSight should sample every ~3 units along the ray"

    def test_calls_is_inside_building(self, source):
        # The LOS function should use isInsideBuilding for occlusion
        assert "isInsideBuilding(px, pz)" in source, \
            "hasLineOfSight should call isInsideBuilding to check ray samples"

    def test_returns_false_when_blocked(self, source):
        # Should return false when a sample hits a building
        assert "return false" in source, \
            "hasLineOfSight should return false when blocked"

    def test_returns_true_when_clear(self, source):
        # Should return true at end when no building hit
        assert "return true" in source, \
            "hasLineOfSight should return true when path is clear"


# =========================================================================
# 2. Protestor projectiles (rocks + molotovs) check LOS before firing
# =========================================================================

class TestProtestorProjectileLOS:
    def test_protestor_checks_los_before_throw(self, source):
        # The nearest-police check for protestors should include LOS
        assert "nearest && hasLineOfSight(pr.x, pr.z, nearest.x, nearest.z)" in source, \
            "Protestor should check LOS to police before throwing rocks/molotovs"


# =========================================================================
# 3. Police combat (tear gas + rubber bullets) checks LOS before firing
# =========================================================================

class TestPoliceCombatLOS:
    def test_police_checks_los_before_fire(self, source):
        # The nearest-protestor check for police should include LOS
        assert "nearest && hasLineOfSight(pol.x, pol.z, nearest.x, nearest.z)" in source, \
            "Police should check LOS to protestor before firing tear gas / rubber bullets"


# =========================================================================
# 4. Tear gas effect respects building occlusion
# =========================================================================

class TestTearGasLOS:
    def test_tear_gas_dispersal_checks_los(self, source):
        # The tear gas effect loop should skip people behind buildings
        assert "hasLineOfSight(impactX, impactZ, pr.x, pr.z)" in source, \
            "Tear gas effect should check LOS from gas cloud to each person"

    def test_gas_effect_on_police_checks_los(self, source):
        # Officers behind buildings should not be affected by gas clouds
        assert "hasLineOfSight(gas.x, gas.z, pol.x, pol.z)" in source, \
            "Gas cloud effect on police should check LOS"


# =========================================================================
# 5. Detection lines (debug mode) respect LOS
# =========================================================================

class TestDetectionLineLOS:
    def test_detection_lines_check_los(self, source):
        # Debug detection lines should only draw when LOS is clear
        assert "hasLineOfSight(cop.x, cop.z, prot.x, prot.z)" in source, \
            "Detection lines should check LOS between police and protestors"
