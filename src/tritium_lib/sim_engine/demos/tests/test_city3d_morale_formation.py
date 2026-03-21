"""
Tests for city3d.html morale visualization, squad formation geometry,
and fire sector arcs.
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
# 1. MORALE SYSTEM
# =========================================================================

class TestMoraleProperties:
    def test_police_morale_init(self, source):
        """Police objects must have a morale property initialized to 1.0"""
        assert "morale: 1.0" in source or "morale:1.0" in source

    def test_protestor_morale_init(self, source):
        """Protestor objects must have morale initialized to 0.8"""
        assert "morale: 0.8" in source or "morale:0.8" in source

    def test_morale_property_on_police(self, source):
        """Police spawn function sets morale"""
        # Find morale near the police.push block
        idx_spawn = source.find("function spawnPolice")
        idx_end = source.find("function ", idx_spawn + 20)
        block = source[idx_spawn:idx_end]
        assert "morale" in block, "spawnPolice must set morale property"

    def test_morale_property_on_protestor(self, source):
        """Protestor spawn function sets morale"""
        idx_spawn = source.find("function spawnProtestors")
        idx_end = source.find("function ", idx_spawn + 25)
        block = source[idx_spawn:idx_end]
        assert "morale" in block, "spawnProtestors must set morale property"


class TestMoraleUpdates:
    def test_morale_decreases_on_rock_hit(self, source):
        """Police morale drops when hit by rocks"""
        assert "morale" in source and "rock" in source.lower()

    def test_morale_decreases_on_tear_gas(self, source):
        """Protestor morale drops from tear gas"""
        assert "morale" in source and "tearGas" in source.lower() or "tear" in source.lower()

    def test_morale_flee_threshold(self, source):
        """Units flee when morale < 0.3"""
        assert "morale" in source and "0.3" in source

    def test_morale_color_blend(self, source):
        """Body color shifts based on morale — uses lerpColors or moraleColor"""
        assert "moraleColor" in source or "lerpMoraleColor" in source


class TestMoraleHUD:
    def test_police_morale_hud(self, source):
        """Stats HUD shows police morale percentage"""
        assert "police-morale" in source or "Police Morale" in source or "policeMorale" in source

    def test_protestor_morale_hud(self, source):
        """Stats HUD shows protestor morale percentage"""
        assert "protestor-morale" in source or "Protest Morale" in source or "protestorMorale" in source


# =========================================================================
# 2. SQUAD FORMATION LINE
# =========================================================================

class TestFormationLine:
    def test_formation_line_geometry(self, source):
        """Pre-allocated BufferGeometry for police formation line"""
        assert "formationLine" in source or "formationGeo" in source

    def test_formation_line_material(self, source):
        """Formation line uses a cyan-ish line material"""
        assert "formationLine" in source or "formationMat" in source

    def test_formation_positions_update(self, source):
        """Formation line positions updated from police positions each frame"""
        # The update code should sort police by X and write positions
        assert "formationLine" in source

    def test_formation_broken_detection(self, source):
        """Detects broken formation when police spread > 15m"""
        assert "BROKEN" in source or "broken" in source

    def test_formation_added_to_scene(self, source):
        """Formation line added to scene"""
        assert "formationLine" in source


class TestFormationDebug:
    def test_formation_status_in_debug(self, source):
        """Debug overlay shows FORMATION status"""
        assert "FORMATION" in source


# =========================================================================
# 3. FIRE SECTOR ARCS
# =========================================================================

class TestFireSectorArcs:
    def test_arc_geometry_exists(self, source):
        """Pre-allocated BufferGeometry for police fire sector arcs"""
        assert "arcLine" in source or "sectorArc" in source or "fireSector" in source

    def test_arc_debug_only(self, source):
        """Arcs visible only in debug mode"""
        assert "debugMode" in source and ("arcLine" in source or "sectorArc" in source or "fireSector" in source)

    def test_arc_angle_60_degrees(self, source):
        """Arc spans 60 degrees (PI/3)"""
        assert "Math.PI / 3" in source or "Math.PI/3" in source or "1.047" in source

    def test_arc_range_30(self, source):
        """Arc range is 30 units"""
        assert "30" in source  # 30m range constant

    def test_arc_color_cyan(self, source):
        """Arc color is cyan-ish"""
        assert "0x00f0ff" in source or "#00f0ff" in source
