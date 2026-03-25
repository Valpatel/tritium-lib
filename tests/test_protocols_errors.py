# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.protocols.errors — shared parse error type."""

import pytest

from tritium_lib.protocols.errors import ParseError


class TestParseError:
    def test_basic(self):
        err = ParseError("NMEA", "Empty sentence")
        assert err.protocol == "NMEA"
        assert err.reason == "Empty sentence"
        assert err.raw_data is None
        assert "[NMEA]" in str(err)
        assert "Empty sentence" in str(err)

    def test_with_raw_data(self):
        err = ParseError("WiFi", "Frame too short", b"\x00\x01")
        assert err.raw_data == b"\x00\x01"

    def test_is_exception(self):
        with pytest.raises(ParseError):
            raise ParseError("ADS-B", "Invalid CRC")

    def test_str_format(self):
        err = ParseError("BLE", "Truncated packet")
        assert str(err) == "[BLE] Truncated packet"

    def test_catch_as_exception(self):
        try:
            raise ParseError("CoT", "Invalid XML")
        except Exception as e:
            assert isinstance(e, ParseError)
            assert e.protocol == "CoT"
