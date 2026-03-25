# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.protocols.wifi_probe — WiFi probe request parser."""

import pytest

from tritium_lib.protocols.errors import ParseError
from tritium_lib.protocols.wifi_probe import (
    WiFiHTCapabilities,
    WiFiProbeParser,
    WiFiProbeRequest,
    WiFiRate,
)


class TestWiFiRate:
    def test_basic(self):
        r = WiFiRate(rate_mbps=54.0, is_basic=True)
        assert r.rate_mbps == 54.0
        assert r.is_basic is True


class TestWiFiHTCapabilities:
    def test_defaults(self):
        ht = WiFiHTCapabilities()
        assert ht.channel_width_40mhz is False
        assert ht.short_gi_20mhz is False
        assert ht.max_amsdu_length == 3839


class TestWiFiProbeRequest:
    def test_defaults(self):
        pr = WiFiProbeRequest()
        assert pr.source_mac == ""
        assert pr.destination_mac == "ff:ff:ff:ff:ff:ff"
        assert pr.is_broadcast_probe is True
        assert pr.ssid == ""

    def test_all_rates_mbps(self):
        pr = WiFiProbeRequest(
            supported_rates=[WiFiRate(1.0), WiFiRate(2.0)],
            extended_rates=[WiFiRate(11.0)],
        )
        assert pr.all_rates_mbps == [1.0, 2.0, 11.0]

    def test_max_rate_mbps(self):
        pr = WiFiProbeRequest(
            supported_rates=[WiFiRate(1.0), WiFiRate(54.0)],
        )
        assert pr.max_rate_mbps == 54.0

    def test_max_rate_empty(self):
        pr = WiFiProbeRequest()
        assert pr.max_rate_mbps == 0.0

    def test_oui(self):
        pr = WiFiProbeRequest(source_mac="AA:BB:CC:DD:EE:FF")
        assert pr.oui == "AA:BB:CC"

    def test_oui_empty(self):
        pr = WiFiProbeRequest(source_mac="")
        assert pr.oui == ""

    def test_is_randomized_mac_false(self):
        pr = WiFiProbeRequest(source_mac="00:11:22:33:44:55")
        assert pr.is_randomized_mac is False

    def test_is_randomized_mac_true(self):
        # 0x02 bit set in first octet = locally administered
        pr = WiFiProbeRequest(source_mac="02:11:22:33:44:55")
        assert pr.is_randomized_mac is True

    def test_is_randomized_mac_empty(self):
        pr = WiFiProbeRequest(source_mac="")
        assert pr.is_randomized_mac is False


class TestWiFiProbeParserFromFields:
    def test_basic(self):
        parser = WiFiProbeParser()
        result = parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            ssid="MyNetwork",
        )
        assert result.source_mac == "aa:bb:cc:dd:ee:ff"
        assert result.ssid == "MyNetwork"
        assert result.is_broadcast_probe is False

    def test_broadcast_probe(self):
        parser = WiFiProbeParser()
        result = parser.from_fields(source_mac="AA:BB:CC:DD:EE:FF", ssid="")
        assert result.is_broadcast_probe is True

    def test_with_rates_tag(self):
        parser = WiFiProbeParser()
        # Tag 1 (Supported Rates): 1 Mbps (0x82 = 1.0 basic), 2 Mbps (0x84 = 2.0 basic)
        result = parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            tags={1: bytes([0x82, 0x84])},
        )
        assert len(result.supported_rates) == 2
        assert result.supported_rates[0].rate_mbps == 1.0
        assert result.supported_rates[0].is_basic is True

    def test_with_ht_tag(self):
        parser = WiFiProbeParser()
        # HT capabilities with 40MHz support (bit 1 set)
        cap_bytes = (0x02).to_bytes(2, "little")
        result = parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            tags={45: cap_bytes + b"\x00" * 24},
        )
        assert result.has_ht is True
        assert result.ht_capabilities is not None
        assert result.ht_capabilities.channel_width_40mhz is True

    def test_with_vendor_ie(self):
        parser = WiFiProbeParser()
        # Microsoft OUI
        vendor_data = bytes([0x00, 0x50, 0xF2, 0x01, 0x00])
        result = parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            tags={221: vendor_data},
        )
        assert len(result.vendor_ies) == 1
        assert result.vendor_ies[0]["vendor"] == "Microsoft"


class TestWiFiProbeParserParse:
    def test_parse_too_short(self):
        parser = WiFiProbeParser()
        with pytest.raises(ParseError, match="too short"):
            parser.parse(b"\x40\x00" + b"\x00" * 10)

    def test_parse_not_probe_request(self):
        parser = WiFiProbeParser()
        # Frame control with wrong type/subtype
        frame = b"\x00\x00" + b"\x00" * 22
        with pytest.raises(ParseError, match="Not a probe request"):
            parser.parse(frame)

    def test_parse_invalid_type(self):
        parser = WiFiProbeParser()
        with pytest.raises(ParseError):
            parser.parse(12345)

    def test_parse_hex_string(self):
        parser = WiFiProbeParser()
        # Build a valid probe request frame (type=0, subtype=4)
        # FC: 0x0040 (probe request), duration, DA(6), SA(6), BSSID(6), Seq(2)
        fc = b"\x40\x00"
        duration = b"\x00\x00"
        da = b"\xff\xff\xff\xff\xff\xff"
        sa = b"\xaa\xbb\xcc\xdd\xee\xff"
        bssid = b"\xff\xff\xff\xff\xff\xff"
        seq = b"\x00\x00"
        frame = fc + duration + da + sa + bssid + seq
        hex_str = frame.hex()
        result = parser.parse(hex_str)
        assert isinstance(result, WiFiProbeRequest)
        assert result.source_mac == "aa:bb:cc:dd:ee:ff"

    def test_parse_bad_hex(self):
        parser = WiFiProbeParser()
        with pytest.raises(ParseError, match="Invalid hex"):
            parser.parse("ZZZZ")


class TestWiFiProbeParserNormalizeMac:
    def test_normalize(self):
        result = WiFiProbeParser._normalize_mac(b"\xaa\xbb\xcc\xdd\xee\xff")
        assert result == "aa:bb:cc:dd:ee:ff"

    def test_short(self):
        result = WiFiProbeParser._normalize_mac(b"\xaa\xbb")
        assert result == ""


class TestWiFiProbeParserRates:
    def test_parse_rates(self):
        rates = WiFiProbeParser._parse_rates(bytes([0x82, 0x84, 0x8B, 0x96]))
        assert len(rates) == 4
        assert rates[0].rate_mbps == 1.0
        assert rates[0].is_basic is True

    def test_parse_ht_empty(self):
        ht = WiFiProbeParser._parse_ht_capabilities(b"")
        assert ht.channel_width_40mhz is False

    def test_parse_vendor_ie_short(self):
        v = WiFiProbeParser._parse_vendor_ie(b"\x00")
        assert v["oui"] == ""
