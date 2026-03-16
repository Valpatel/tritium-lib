# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.web.sanitize module."""

from tritium_lib.web.sanitize import html_escape, json_safe, sanitize_filename


class TestHtmlEscape:
    def test_basic_tags(self):
        assert html_escape("<script>") == "&lt;script&gt;"

    def test_quotes(self):
        assert html_escape('"hello"') == "&quot;hello&quot;"

    def test_ampersand(self):
        assert html_escape("a & b") == "a &amp; b"

    def test_none(self):
        assert html_escape(None) == ""

    def test_empty(self):
        assert html_escape("") == ""

    def test_safe_text(self):
        assert html_escape("hello world") == "hello world"

    def test_single_quote(self):
        result = html_escape("it's")
        assert "&" in result or "'" not in result or result == "it&#x27;s"

    def test_mixed(self):
        result = html_escape('<a href="x">test</a>')
        assert "<" not in result
        assert ">" not in result


class TestJsonSafe:
    def test_none(self):
        assert json_safe(None) == ""

    def test_normal_string(self):
        assert json_safe("hello") == "hello"

    def test_quotes(self):
        assert json_safe('say "hi"') == 'say \\"hi\\"'

    def test_newlines(self):
        assert json_safe("line1\nline2") == "line1\\nline2"

    def test_tabs(self):
        assert json_safe("a\tb") == "a\\tb"

    def test_backslash(self):
        assert json_safe("a\\b") == "a\\\\b"

    def test_control_chars_removed(self):
        result = json_safe("hello\x01world")
        assert "\x01" not in result

    def test_truncation(self):
        long_str = "a" * 20000
        result = json_safe(long_str, max_length=100)
        assert len(result) <= 100


class TestSanitizeFilename:
    def test_none(self):
        assert sanitize_filename(None) == "unnamed"

    def test_empty(self):
        assert sanitize_filename("") == "unnamed"

    def test_normal(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_path_traversal(self):
        result = sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_null_bytes(self):
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_special_chars(self):
        result = sanitize_filename('file<>:"|?*.txt')
        assert "<" not in result
        assert ">" not in result

    def test_leading_dots(self):
        result = sanitize_filename("...hidden")
        assert not result.startswith(".")

    def test_backslash(self):
        result = sanitize_filename("path\\to\\file")
        assert "\\" not in result

    def test_max_length(self):
        result = sanitize_filename("a" * 300, max_length=50)
        assert len(result) <= 50

    def test_preserves_extension_on_truncation(self):
        result = sanitize_filename("a" * 300 + ".pdf", max_length=50)
        assert result.endswith(".pdf")

    def test_spaces_to_underscores(self):
        result = sanitize_filename("my  file   name.txt")
        assert "  " not in result
