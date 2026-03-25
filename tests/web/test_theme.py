# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.web.theme — cyberpunk theme system."""

from tritium_lib.web.theme import TritiumTheme


class TestTritiumTheme:
    def setup_method(self):
        self.theme = TritiumTheme()

    def test_design_tokens(self):
        assert self.theme.BG == "#0a0a0a"
        assert self.theme.ACCENT == "#00ffd0"
        assert self.theme.DANGER == "#ff3366"
        assert self.theme.WARNING == "#ffaa00"

    def test_css_output(self):
        css = self.theme.css()
        assert isinstance(css, str)
        assert len(css) > 1000
        assert "#0a0a0a" in css
        assert "#00ffd0" in css
        assert "monospace" in css
        # Has responsive media queries
        assert "@media" in css

    def test_nav_html(self):
        pages = [("/", "Home"), ("/map", "Map"), ("/fleet", "Fleet")]
        html = self.theme.nav_html(pages, active="/map")
        assert '<div class="nav">' in html
        assert "Home" in html
        assert "Map" in html
        assert 'class="active"' in html

    def test_nav_html_no_active(self):
        pages = [("/", "Home")]
        html = self.theme.nav_html(pages)
        assert "active" not in html

    def test_card(self):
        html = self.theme.card("Test Card", "<p>Body content</p>")
        assert "card" in html
        assert "Test Card" in html
        assert "Body content" in html

    def test_card_no_title(self):
        html = self.theme.card("", "<p>Body only</p>")
        assert "<h3>" not in html
        assert "Body only" in html

    def test_card_css_class(self):
        html = self.theme.card("Title", "Body", css_class="highlight")
        assert "highlight" in html

    def test_table(self):
        headers = ["Name", "Status", "Score"]
        rows = [["Alice", "Online", "100"], ["Bob", "Offline", "50"]]
        html = self.theme.table(headers, rows)
        assert "<table>" in html
        assert "<th>" in html
        assert "Alice" in html
        assert "Bob" in html

    def test_badge(self):
        html = self.theme.badge("Online", "online")
        assert "badge" in html
        assert "Online" in html

    def test_badge_variants(self):
        for variant in ("online", "offline", "error", "updating"):
            html = self.theme.badge("Test", variant)
            assert variant in html

    def test_progress_bar(self):
        html = self.theme.progress_bar(75.0)
        assert "bar-bg" in html
        assert "bar-fill" in html
        assert "75%" in html

    def test_progress_bar_clamping(self):
        html_over = self.theme.progress_bar(150.0)
        assert "100%" in html_over
        html_under = self.theme.progress_bar(-10.0)
        assert "0%" in html_under

    def test_rssi_bar(self):
        html = self.theme.rssi_bar(-50)
        assert "rssi-bars" in html
        assert "-50dBm" in html

    def test_rssi_bar_weak_signal(self):
        html = self.theme.rssi_bar(-95)
        assert "rssi-bars" in html

    def test_html_escaping(self):
        """Ensure XSS-dangerous characters are escaped."""
        html = self.theme.card("<script>alert('xss')</script>", "body")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
