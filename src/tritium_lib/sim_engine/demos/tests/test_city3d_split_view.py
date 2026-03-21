"""
Tests for city3d.html split-view multiplayer indicator overlay.
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


class TestSplitViewState:
    def test_split_view_mode_variable(self, source):
        assert "splitViewMode" in source, "Missing splitViewMode boolean"

    def test_digit2_key_handler(self, source):
        assert "'Digit2'" in source, "Missing Digit2 key handler"

    def test_toggle_active_class(self, source):
        assert "split-view" in source, "Missing split-view element reference"


class TestSplitViewHTML:
    def test_split_view_div_exists(self, source):
        assert 'id="split-view"' in source, "Missing split-view container div"

    def test_police_cmd_label(self, source):
        assert "POLICE CMD" in source, "Missing POLICE CMD label"

    def test_protest_cmd_label(self, source):
        assert "PROTEST CMD" in source, "Missing PROTEST CMD label"

    def test_sv_police_panel(self, source):
        assert 'id="sv-police"' in source, "Missing sv-police panel"

    def test_sv_protest_panel(self, source):
        assert 'id="sv-protest"' in source, "Missing sv-protest panel"

    def test_vertical_divider_line(self, source):
        assert "sv-line" in source, "Missing vertical divider line element"


class TestSplitViewCSS:
    def test_split_view_hidden_by_default(self, source):
        assert "display:none" in source and "split-view" in source, \
            "Split view must be hidden by default"

    def test_split_view_active_display(self, source):
        assert "#split-view.active" in source, \
            "Missing .active CSS rule for split-view"

    def test_cyan_divider_color(self, source):
        assert "sv-line" in source and "#00f0ff" in source, \
            "Divider line should use cyan color"


class TestSplitViewHUD:
    def test_police_stats_in_update(self, source):
        assert "POLICE FORCES" in source, \
            "Missing police stats in split view update"

    def test_protest_stats_in_update(self, source):
        assert "PROTEST FORCES" in source, \
            "Missing protest stats in split view update"

    def test_controls_hint(self, source):
        assert "Split View" in source, \
            "Missing Split View in controls help bar"
