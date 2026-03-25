# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for web template system — page generation, escaping, layout."""

import pytest

from tritium_lib.web.templates import (
    admin_page,
    commissioning_page,
    full_page,
    mobile_page,
)
from tritium_lib.web.theme import TritiumTheme


# ── full_page ───────────────────────────────────────────────────────

class TestFullPage:
    """Tests for full_page() template."""

    def test_returns_html(self):
        html = full_page("Test Page", "<p>Hello</p>")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_title_in_head(self):
        html = full_page("My Title", "")
        assert "My Title" in html
        assert "Tritium" in html

    def test_body_content_included(self):
        html = full_page("T", "<div class='test'>Content</div>")
        assert "<div class='test'>Content</div>" in html

    def test_css_included(self):
        html = full_page("T", "")
        assert "<style>" in html
        assert "Tritium Theme" in html

    def test_meta_viewport(self):
        html = full_page("T", "")
        assert "viewport" in html

    def test_charset(self):
        html = full_page("T", "")
        assert "utf-8" in html

    def test_html_escaping_in_title(self):
        html = full_page("<script>alert('xss')</script>", "")
        assert "<script>" not in html.split("<style>")[0]
        assert "&lt;script&gt;" in html

    def test_custom_theme(self):
        theme = TritiumTheme()
        html = full_page("T", "", theme=theme)
        assert html is not None
        assert len(html) > 100

    def test_default_theme_when_none(self):
        html = full_page("T", "", theme=None)
        assert "Tritium Theme" in html


# ── admin_page ──────────────────────────────────────────────────────

class TestAdminPage:
    """Tests for admin_page() with sidebar navigation."""

    def test_has_sidebar(self):
        html = admin_page(
            "Admin",
            nav_pages=[("/dashboard", "Dashboard"), ("/devices", "Devices")],
            body_html="<p>Content</p>",
        )
        assert "sidebar" in html
        assert "Dashboard" in html
        assert "Devices" in html

    def test_active_link_highlighted(self):
        html = admin_page(
            "Admin",
            nav_pages=[("/a", "A"), ("/b", "B")],
            body_html="",
            active_url="/a",
        )
        assert 'class="active"' in html

    def test_body_content(self):
        html = admin_page("T", [], "<p>Main content</p>")
        assert "<p>Main content</p>" in html

    def test_title_escaped(self):
        html = admin_page("<bad>", [], "")
        assert "&lt;bad&gt;" in html

    def test_nav_urls_escaped(self):
        html = admin_page("T", [('/a"b', "Link")], "")
        # URL should be escaped
        assert "Link" in html

    def test_tritium_branding(self):
        html = admin_page("T", [], "")
        assert "TRITIUM" in html


# ── mobile_page ─────────────────────────────────────────────────────

class TestMobilePage:
    """Tests for mobile_page() mobile-optimized layout."""

    def test_returns_html(self):
        html = mobile_page("Mobile", "<p>Content</p>")
        assert "<!DOCTYPE html>" in html

    def test_mobile_meta_tags(self):
        html = mobile_page("Mobile", "")
        assert "maximum-scale=1" in html
        assert "user-scalable=no" in html

    def test_apple_web_app_capable(self):
        html = mobile_page("Mobile", "")
        assert "apple-mobile-web-app-capable" in html

    def test_mobile_web_app_capable(self):
        html = mobile_page("Mobile", "")
        assert "mobile-web-app-capable" in html

    def test_body_content(self):
        html = mobile_page("T", "<div>Mobile content</div>")
        assert "<div>Mobile content</div>" in html


# ── commissioning_page ──────────────────────────────────────────────

class TestCommissioningPage:
    """Tests for commissioning_page() WiFi setup page."""

    def test_returns_html(self):
        html = commissioning_page()
        assert "<!DOCTYPE html>" in html

    def test_has_wifi_form(self):
        html = commissioning_page()
        assert "wifi-form" in html
        assert "ssid" in html
        assert "password" in html

    def test_has_registration_form(self):
        html = commissioning_page()
        assert "register-form" in html
        assert "device_name" in html
        assert "server_url" in html

    def test_has_scan_button(self):
        html = commissioning_page()
        assert "scanNetworks" in html

    def test_has_javascript(self):
        html = commissioning_page()
        assert "<script>" in html


# ── TritiumTheme components ─────────────────────────────────────────

class TestTritiumThemeComponents:
    """Tests for theme HTML component generators."""

    def setup_method(self):
        self.theme = TritiumTheme()

    def test_nav_html(self):
        html = self.theme.nav_html([("/a", "A"), ("/b", "B")])
        assert 'class="nav"' in html
        assert "A" in html
        assert "B" in html

    def test_nav_active_link(self):
        html = self.theme.nav_html([("/a", "A"), ("/b", "B")], active="/a")
        assert 'class="active"' in html

    def test_card(self):
        html = self.theme.card("Title", "<p>Body</p>")
        assert "card" in html
        assert "Title" in html
        assert "<p>Body</p>" in html

    def test_card_extra_class(self):
        html = self.theme.card("T", "", css_class="highlight")
        assert "highlight" in html

    def test_card_no_title(self):
        html = self.theme.card("", "<p>B</p>")
        assert "<h3>" not in html

    def test_table(self):
        html = self.theme.table(["Name", "Value"], [["A", "1"], ["B", "2"]])
        assert "<table>" in html
        assert "Name" in html
        assert "Value" in html
        assert "A" in html

    def test_badge(self):
        html = self.theme.badge("ONLINE", "online")
        assert "badge" in html
        assert "ONLINE" in html

    def test_progress_bar(self):
        html = self.theme.progress_bar(75.0)
        assert "bar-bg" in html
        assert "75%" in html

    def test_progress_bar_clamped(self):
        html_low = self.theme.progress_bar(-10)
        assert "0%" in html_low
        html_high = self.theme.progress_bar(200)
        assert "100%" in html_high

    def test_rssi_bar(self):
        html = self.theme.rssi_bar(-65)
        assert "rssi-bars" in html
        assert "-65dBm" in html

    def test_rssi_bar_strong(self):
        html = self.theme.rssi_bar(-30)
        assert "rssi-bars" in html

    def test_rssi_bar_weak(self):
        html = self.theme.rssi_bar(-95)
        assert "rssi-bars" in html

    def test_css_returns_string(self):
        css = self.theme.css()
        assert isinstance(css, str)
        assert "Tritium Theme" in css
        assert len(css) > 500


class TestTritiumThemeDesignTokens:
    """Verify design tokens are the correct cyberpunk values."""

    def test_accent_color(self):
        assert TritiumTheme.ACCENT == "#00ffd0"

    def test_danger_color(self):
        assert TritiumTheme.DANGER == "#ff3366"

    def test_bg_color(self):
        assert TritiumTheme.BG == "#0a0a0a"

    def test_font_monospace(self):
        assert "monospace" in TritiumTheme.FONT
