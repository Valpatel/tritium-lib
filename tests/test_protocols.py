# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.protocols — radio protocol parsers.

25+ tests per parser (BLE, WiFi, AIS, ADS-B, Meshtastic, NMEA).
"""

import math
import struct

import pytest

from tritium_lib.protocols import (
    BLEAdvertParser,
    WiFiProbeParser,
    AISParser,
    ADSBParser,
    MeshtasticParser,
    NMEAParser,
    ParseError,
)
from tritium_lib.protocols.ble_advert import (
    BLEAdvertisement,
    BLEFlags,
    ManufacturerData,
    ServiceData,
    COMPANY_IDS,
)
from tritium_lib.protocols.wifi_probe import (
    WiFiProbeRequest,
    WiFiRate,
    WiFiHTCapabilities,
)
from tritium_lib.protocols.ais import (
    AISPositionReport,
    AISStaticData,
    AISSentence,
)
from tritium_lib.protocols.adsb import (
    ADSBMessage,
    ADSBIdentification,
    ADSBAirbornePosition,
    ADSBVelocity,
    ADSBSquawk,
    ADSBAltitudeReply,
    ADSBEmergency,
    ADSBSurfacePosition,
)
from tritium_lib.protocols.meshtastic import (
    MeshtasticPacket,
    MeshtasticPosition,
    MeshtasticNodeInfo,
    MeshtasticTelemetry,
    PORTNUM_MAP,
    HW_MODEL_MAP,
)
from tritium_lib.protocols.nmea import (
    NMEAGGA,
    NMEARMC,
    NMEAGSA,
    NMEAVTG,
    NMEAGLL,
    NMEAGSV,
    NMEAPosition,
    NMEATime,
    NMEADate,
    NMEASatellite,
)


# =============================================================================
# ParseError tests
# =============================================================================


class TestParseError:
    def test_basic_error(self):
        err = ParseError("BLE", "bad data")
        assert err.protocol == "BLE"
        assert err.reason == "bad data"
        assert "[BLE]" in str(err)

    def test_error_with_raw_data(self):
        err = ParseError("AIS", "too short", b"\x00\x01")
        assert err.raw_data == b"\x00\x01"

    def test_error_inherits_exception(self):
        with pytest.raises(ParseError):
            raise ParseError("NMEA", "test")

    def test_error_message_format(self):
        err = ParseError("WiFi", "bad frame")
        assert str(err) == "[WiFi] bad frame"


# =============================================================================
# BLE Advertisement Parser (25+ tests)
# =============================================================================


class TestBLEAdvertParser:
    """Tests for BLEAdvertParser."""

    def setup_method(self):
        self.parser = BLEAdvertParser()

    # -- Flags -----------------------------------------------------------------

    def test_flags_general_discoverable(self):
        # AD: length=2, type=0x01, flags=0x06 (general disc + BR/EDR not supported)
        advert = self.parser.parse(b"\x02\x01\x06")
        assert advert.flags is not None
        assert advert.flags.le_general_discoverable is True
        assert advert.flags.br_edr_not_supported is True
        assert advert.flags.le_limited_discoverable is False

    def test_flags_limited_discoverable(self):
        advert = self.parser.parse(b"\x02\x01\x01")
        assert advert.flags.le_limited_discoverable is True
        assert advert.flags.le_general_discoverable is False

    def test_flags_all_bits(self):
        advert = self.parser.parse(b"\x02\x01\x1f")
        assert advert.flags.raw == 0x1F
        assert advert.flags.le_limited_discoverable is True
        assert advert.flags.le_general_discoverable is True
        assert advert.flags.br_edr_not_supported is True
        assert advert.flags.le_br_edr_controller is True
        assert advert.flags.le_br_edr_host is True

    # -- Service UUIDs --------------------------------------------------------

    def test_16bit_service_uuids(self):
        # type=0x03 (complete 16-bit UUIDs), UUID 0x180F (battery), 0x180A (device info)
        advert = self.parser.parse(b"\x02\x01\x06\x05\x03\x0f\x18\x0a\x18")
        assert "180f" in advert.service_uuids_16
        assert "180a" in advert.service_uuids_16
        assert len(advert.service_uuids_16) == 2

    def test_incomplete_16bit_uuids(self):
        # type=0x02 (incomplete list)
        advert = self.parser.parse(b"\x03\x02\x0f\x18")
        assert "180f" in advert.service_uuids_16

    def test_32bit_service_uuids(self):
        # type=0x05, 32-bit UUID 0x12345678
        data = b"\x05\x05\x78\x56\x34\x12"
        advert = self.parser.parse(data)
        assert "12345678" in advert.service_uuids_32

    def test_128bit_service_uuids(self):
        # type=0x07, 128-bit UUID (16 bytes, little-endian)
        uuid_le = bytes(range(16))  # 00 01 02 ... 0F
        data = b"\x11\x07" + uuid_le
        advert = self.parser.parse(data)
        assert len(advert.service_uuids_128) == 1
        # Verify it's a properly formatted UUID string
        assert "-" in advert.service_uuids_128[0]

    def test_all_service_uuids_property(self):
        advert = self.parser.parse(b"\x03\x03\x0f\x18")
        assert len(advert.all_service_uuids) >= 1

    # -- Local Name -----------------------------------------------------------

    def test_complete_local_name(self):
        name = b"iPhone"
        data = bytes([len(name) + 1, 0x09]) + name
        advert = self.parser.parse(data)
        assert advert.local_name == "iPhone"

    def test_shortened_local_name(self):
        name = b"iPho"
        data = bytes([len(name) + 1, 0x08]) + name
        advert = self.parser.parse(data)
        assert advert.shortened_name == "iPho"

    def test_display_name_prefers_complete(self):
        data = b"\x05\x08iPho\x07\x09iPhone"
        advert = self.parser.parse(data)
        assert advert.display_name == "iPhone"

    def test_display_name_falls_back_to_shortened(self):
        data = b"\x05\x08iPho"
        advert = self.parser.parse(data)
        assert advert.display_name == "iPho"

    # -- TX Power Level -------------------------------------------------------

    def test_tx_power_positive(self):
        advert = self.parser.parse(b"\x02\x0a\x04")
        assert advert.tx_power == 4

    def test_tx_power_negative(self):
        # -10 dBm = 0xF6 (unsigned) -> -10 signed
        advert = self.parser.parse(b"\x02\x0a\xf6")
        assert advert.tx_power == -10

    # -- Manufacturer Data ----------------------------------------------------

    def test_apple_manufacturer_data(self):
        # Company ID 0x004C (Apple), then some iBeacon data
        data = b"\x05\xff\x4c\x00\x01\x02"
        advert = self.parser.parse(data)
        assert len(advert.manufacturer_data) == 1
        assert advert.manufacturer_data[0].company_id == 0x004C
        assert advert.manufacturer_data[0].company_name == "Apple"
        assert advert.manufacturer_data[0].data == b"\x01\x02"

    def test_microsoft_manufacturer_data(self):
        data = b"\x04\xff\x06\x00\xAB"
        advert = self.parser.parse(data)
        assert advert.manufacturer_data[0].company_id == 0x0006
        assert advert.manufacturer_data[0].company_name == "Microsoft"

    def test_unknown_manufacturer(self):
        data = b"\x04\xff\xfe\xfe\x00"
        advert = self.parser.parse(data)
        assert advert.manufacturer_data[0].company_name == "Unknown"

    # -- Service Data ---------------------------------------------------------

    def test_service_data_16bit(self):
        # type=0x16, UUID=0x180F, data=0x64 (battery 100%)
        data = b"\x04\x16\x0f\x18\x64"
        advert = self.parser.parse(data)
        assert len(advert.service_data) == 1
        assert advert.service_data[0].uuid == "180f"
        assert advert.service_data[0].data == b"\x64"

    # -- Hex input -------------------------------------------------------------

    def test_parse_hex_string(self):
        advert = self.parser.parse("0201060303 0f18")
        assert advert.flags is not None
        assert "180f" in advert.service_uuids_16

    def test_parse_hex_with_0x_prefix(self):
        advert = self.parser.parse("0x02010603030f18")
        assert advert.flags is not None

    def test_parse_hex_convenience(self):
        advert = self.parser.parse_hex("020106")
        assert advert.flags.le_general_discoverable is True

    # -- Error handling -------------------------------------------------------

    def test_empty_data_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse(b"")

    def test_invalid_hex_raises(self):
        with pytest.raises(ParseError, match="Invalid hex"):
            self.parser.parse("ZZZZ")

    def test_invalid_type_raises(self):
        with pytest.raises(ParseError):
            self.parser.parse(12345)

    def test_truncated_ad_struct(self):
        # Length says 5 bytes follow, but only 2 present — should not crash
        advert = self.parser.parse(b"\x05\x01\x06")
        # Should still return a valid object, just without parsed flags
        assert isinstance(advert, BLEAdvertisement)

    def test_zero_length_padding(self):
        # Zero-length AD structures should be skipped
        advert = self.parser.parse(b"\x00\x00\x02\x01\x06")
        assert advert.flags is not None

    # -- Properties -----------------------------------------------------------

    def test_is_connectable(self):
        advert = self.parser.parse(b"\x02\x01\x06")
        assert advert.is_connectable is True

    def test_not_connectable_no_flags(self):
        advert = self.parser.parse(b"\x03\x03\x0f\x18")
        assert advert.is_connectable is False

    def test_raw_structures_populated(self):
        advert = self.parser.parse(b"\x02\x01\x06\x03\x03\x0f\x18")
        assert len(advert.raw_structures) == 2
        assert advert.raw_structures[0]["type"] == 0x01
        assert advert.raw_structures[1]["type"] == 0x03

    # -- Complex advertisement ------------------------------------------------

    def test_complex_advertisement(self):
        """Test a realistic multi-structure BLE advertisement."""
        data = bytearray()
        # Flags
        data += b"\x02\x01\x06"
        # Complete Local Name: "Tritium"
        name = b"Tritium"
        data += bytes([len(name) + 1, 0x09]) + name
        # 16-bit Service UUID: 0x180F
        data += b"\x03\x03\x0f\x18"
        # TX Power: -4 dBm
        data += b"\x02\x0a\xfc"
        # Manufacturer Data: Espressif (0x0131), 2 bytes
        data += b"\x05\xff\x31\x01\xAA\xBB"

        advert = self.parser.parse(bytes(data))
        assert advert.flags.le_general_discoverable is True
        assert advert.local_name == "Tritium"
        assert "180f" in advert.service_uuids_16
        assert advert.tx_power == -4
        assert advert.manufacturer_data[0].company_id == 0x0131
        assert advert.manufacturer_data[0].company_name == "Espressif"


# =============================================================================
# WiFi Probe Request Parser (25+ tests)
# =============================================================================


class TestWiFiProbeParser:
    """Tests for WiFiProbeParser."""

    def setup_method(self):
        self.parser = WiFiProbeParser()

    def _build_probe_frame(
        self,
        src_mac: bytes = b"\xAA\xBB\xCC\xDD\xEE\xFF",
        dst_mac: bytes = b"\xFF\xFF\xFF\xFF\xFF\xFF",
        bssid: bytes = b"\xFF\xFF\xFF\xFF\xFF\xFF",
        seq: int = 0,
        tagged_params: bytes = b"",
    ) -> bytes:
        """Build a minimal probe request frame."""
        # Frame control: probe request (type=0, subtype=4)
        fc = b"\x40\x00"
        # Duration
        dur = b"\x00\x00"
        # Sequence control
        seq_ctrl = struct.pack("<H", (seq << 4) & 0xFFF0)
        return fc + dur + dst_mac + src_mac + bssid + seq_ctrl + tagged_params

    def _build_ssid_tag(self, ssid: str) -> bytes:
        """Build an SSID tagged parameter."""
        encoded = ssid.encode("utf-8")
        return bytes([0, len(encoded)]) + encoded

    def _build_rates_tag(self, rates: list[int]) -> bytes:
        """Build supported rates tagged parameter."""
        return bytes([1, len(rates)] + rates)

    # -- Basic parsing --------------------------------------------------------

    def test_minimal_probe_request(self):
        frame = self._build_probe_frame()
        probe = self.parser.parse(frame)
        assert isinstance(probe, WiFiProbeRequest)
        assert probe.source_mac == "aa:bb:cc:dd:ee:ff"
        assert probe.destination_mac == "ff:ff:ff:ff:ff:ff"

    def test_broadcast_probe(self):
        frame = self._build_probe_frame(tagged_params=b"\x00\x00")
        probe = self.parser.parse(frame)
        assert probe.is_broadcast_probe is True
        assert probe.ssid == ""

    def test_directed_probe(self):
        ssid_tag = self._build_ssid_tag("MyNetwork")
        frame = self._build_probe_frame(tagged_params=ssid_tag)
        probe = self.parser.parse(frame)
        assert probe.ssid == "MyNetwork"
        assert probe.is_broadcast_probe is False

    def test_sequence_number(self):
        frame = self._build_probe_frame(seq=1234)
        probe = self.parser.parse(frame)
        assert probe.sequence_number == 1234

    # -- Rates ----------------------------------------------------------------

    def test_supported_rates(self):
        rates_tag = self._build_rates_tag([0x82, 0x84, 0x8B, 0x96])
        frame = self._build_probe_frame(tagged_params=rates_tag)
        probe = self.parser.parse(frame)
        assert len(probe.supported_rates) == 4
        assert probe.supported_rates[0].is_basic is True
        assert probe.supported_rates[0].rate_mbps == 1.0

    def test_extended_rates(self):
        # Tag 50 = extended rates
        ext_rates = bytes([50, 4, 0x0C, 0x12, 0x18, 0x24])
        frame = self._build_probe_frame(tagged_params=ext_rates)
        probe = self.parser.parse(frame)
        assert len(probe.extended_rates) == 4

    def test_max_rate(self):
        rates = self._build_rates_tag([0x82, 0x84, 0x8B, 0x96, 0x24, 0x30])
        frame = self._build_probe_frame(tagged_params=rates)
        probe = self.parser.parse(frame)
        assert probe.max_rate_mbps == 24.0

    def test_all_rates_combined(self):
        rates = self._build_rates_tag([0x82, 0x84])
        ext = bytes([50, 2, 0x0C, 0x12])
        frame = self._build_probe_frame(tagged_params=rates + ext)
        probe = self.parser.parse(frame)
        assert len(probe.all_rates_mbps) == 4

    # -- HT Capabilities -----------------------------------------------------

    def test_ht_capabilities(self):
        # Tag 45, length 26 (minimal HT cap info)
        ht_cap = 0x0062  # 40 MHz + short GI 20 + short GI 40
        ht_data = struct.pack("<H", ht_cap) + b"\x00" * 24
        ht_tag = bytes([45, len(ht_data)]) + ht_data
        frame = self._build_probe_frame(tagged_params=ht_tag)
        probe = self.parser.parse(frame)
        assert probe.has_ht is True
        assert probe.ht_capabilities.channel_width_40mhz is True
        assert probe.ht_capabilities.short_gi_20mhz is True
        assert probe.ht_capabilities.short_gi_40mhz is True

    # -- Vendor IEs -----------------------------------------------------------

    def test_vendor_ie_microsoft(self):
        oui = b"\x00\x50\xF2"
        vendor_tag = bytes([221, len(oui) + 2]) + oui + b"\x01\x00"
        frame = self._build_probe_frame(tagged_params=vendor_tag)
        probe = self.parser.parse(frame)
        assert len(probe.vendor_ies) == 1
        assert probe.vendor_ies[0]["vendor"] == "Microsoft"

    def test_vendor_ie_wfa(self):
        oui = b"\x50\x6F\x9A"
        vendor_tag = bytes([221, len(oui) + 1]) + oui + b"\x09"
        frame = self._build_probe_frame(tagged_params=vendor_tag)
        probe = self.parser.parse(frame)
        assert probe.vendor_ies[0]["vendor"] == "Wi-Fi Alliance"

    # -- MAC properties -------------------------------------------------------

    def test_oui_extraction(self):
        probe = self.parser.from_fields(source_mac="AA:BB:CC:DD:EE:FF")
        assert probe.oui == "AA:BB:CC"

    def test_randomized_mac_detection(self):
        # Bit 1 of first octet set = locally administered
        probe = self.parser.from_fields(source_mac="02:00:00:00:00:00")
        assert probe.is_randomized_mac is True

    def test_non_randomized_mac(self):
        probe = self.parser.from_fields(source_mac="00:11:22:33:44:55")
        assert probe.is_randomized_mac is False

    def test_randomized_mac_common_patterns(self):
        # DA:xx and FE:xx are common randomized patterns
        probe = self.parser.from_fields(source_mac="DA:AB:CD:EF:12:34")
        assert probe.is_randomized_mac is True

    # -- from_fields convenience ----------------------------------------------

    def test_from_fields_basic(self):
        probe = self.parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            ssid="TestNet",
        )
        assert probe.source_mac == "aa:bb:cc:dd:ee:ff"
        assert probe.ssid == "TestNet"
        assert probe.is_broadcast_probe is False

    def test_from_fields_broadcast(self):
        probe = self.parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            ssid="",
        )
        assert probe.is_broadcast_probe is True

    def test_from_fields_with_tags(self):
        probe = self.parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            tags={
                0: b"TestNet",
                1: b"\x82\x84",
            },
        )
        assert probe.ssid == "TestNet"
        assert len(probe.supported_rates) == 2

    # -- Hex input ------------------------------------------------------------

    def test_parse_hex_string(self):
        frame = self._build_probe_frame()
        hex_str = frame.hex()
        probe = self.parser.parse(hex_str)
        assert probe.source_mac == "aa:bb:cc:dd:ee:ff"

    # -- Error handling -------------------------------------------------------

    def test_too_short_raises(self):
        with pytest.raises(ParseError, match="too short"):
            self.parser.parse(b"\x40\x00\x00\x00")

    def test_not_probe_request_raises(self):
        # Frame control for beacon (type=0, subtype=8)
        frame = b"\x80\x00" + b"\x00" * 22
        with pytest.raises(ParseError, match="Not a probe request"):
            self.parser.parse(frame)

    def test_invalid_hex_raises(self):
        with pytest.raises(ParseError, match="Invalid hex"):
            self.parser.parse("ZZZZZZZZZZZZZZZZZZ")

    def test_empty_source_mac(self):
        probe = self.parser.from_fields()
        assert probe.oui == ""
        assert probe.is_randomized_mac is False

    # -- Raw tags -------------------------------------------------------------

    def test_raw_tags_populated(self):
        ssid_tag = self._build_ssid_tag("Test")
        rates_tag = self._build_rates_tag([0x82])
        frame = self._build_probe_frame(tagged_params=ssid_tag + rates_tag)
        probe = self.parser.parse(frame)
        assert len(probe.raw_tags) == 2

    # -- Interworking ---------------------------------------------------------

    def test_interworking_tag(self):
        tag = bytes([107, 1, 0x00])  # Interworking IE
        frame = self._build_probe_frame(tagged_params=tag)
        probe = self.parser.parse(frame)
        assert probe.has_interworking is True

    def test_no_interworking(self):
        frame = self._build_probe_frame()
        probe = self.parser.parse(frame)
        assert probe.has_interworking is False


# =============================================================================
# AIS Parser (25+ tests)
# =============================================================================


class TestAISParser:
    """Tests for AISParser."""

    def setup_method(self):
        self.parser = AISParser()

    # -- Sentence decoding ----------------------------------------------------

    def test_decode_sentence_basic(self):
        sent = self.parser.decode_sentence(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert sent.sentence_type == "AIVDM"
        assert sent.fragment_count == 1
        assert sent.channel == "B"
        assert sent.payload == "13u@Dt002s000000000000000000"
        assert sent.pad_bits == 0

    def test_checksum_validation_pass(self):
        sent = self.parser.decode_sentence(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert sent.checksum_valid is True

    def test_checksum_validation_fail(self):
        sent = self.parser.decode_sentence(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*FF"
        )
        assert sent.checksum_valid is False

    def test_aivdo_sentence(self):
        # AIVDO is own-ship — should be accepted
        sent = self.parser.decode_sentence(
            "!AIVDO,1,1,,A,13u@Dt002s000000000000000000,0*62"
        )
        assert sent.sentence_type == "AIVDO"

    # -- Message type detection -----------------------------------------------

    def test_get_message_type(self):
        msg_type = self.parser.get_message_type(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert msg_type == 1

    def test_type_3_position(self):
        # Type 3 payload (starts with '3' = 0b000011)
        msg_type = self.parser.get_message_type(
            "!AIVDM,1,1,,A,35MsUV0Oh;H@@4WD>g3P0000000P,0*04"
        )
        assert msg_type == 3

    # -- Type 1/2/3 Position Reports ------------------------------------------

    def test_type1_position_report(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert isinstance(result, AISPositionReport)
        assert result.message_type == 1
        assert result.mmsi == 265557232

    def test_position_report_mmsi(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert result.mmsi > 0

    def test_position_report_speed(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert isinstance(result.speed_over_ground, float)

    def test_position_report_nav_status(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        assert isinstance(result.navigation_status_text, str)

    def test_type3_position_report(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,A,35MsUV0Oh;H@@4WD>g3P0000000P,0*04"
        )
        assert isinstance(result, AISPositionReport)
        assert result.message_type == 3

    def test_position_has_valid_check(self):
        result = self.parser.parse(
            "!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*63"
        )
        # Position is 0,0 for this test payload
        assert isinstance(result.has_valid_position, bool)

    # -- Type 5 Static Data ---------------------------------------------------

    def test_type5_static_data(self):
        # Type 5 requires 2 fragments; testing with a single complete payload
        # This is a synthetic type 5 message
        # 424 bits needed, 71 6-bit chars
        payload = "5" + "0" * 70
        sentence = f"!AIVDM,1,1,,A,{payload},0*"
        # Compute valid checksum
        chk = 0
        body = sentence[1:sentence.index("*")]
        for c in body:
            chk ^= ord(c)
        sentence += f"{chk:02X}"
        result = self.parser.parse(sentence)
        assert isinstance(result, AISStaticData)
        assert result.message_type == 5

    def test_static_data_length_property(self):
        sd = AISStaticData(dimension_bow=100, dimension_stern=50)
        assert sd.length == 150

    def test_static_data_beam_property(self):
        sd = AISStaticData(dimension_port=10, dimension_starboard=15)
        assert sd.beam == 25

    def test_static_data_eta_string(self):
        sd = AISStaticData(eta_month=3, eta_day=25, eta_hour=14, eta_minute=30)
        assert sd.eta_string == "03-25 14:30"

    def test_static_data_eta_empty(self):
        sd = AISStaticData(eta_month=0, eta_day=0)
        assert sd.eta_string == ""

    # -- Type 18 Class B Position ---------------------------------------------

    def test_type18_class_b(self):
        # Type 18 payload (starts with 'B' = 0b001011 = 11, but we need
        # message type 18 = 0b010010, armored char = 'J')
        # Build synthetic type 18 sentence
        # Msg type 18 = 010010 -> char = chr(48+18) = chr(66) = 'B'...
        # Wait: 18 in 6-bit = 010010. 48+18=66='B'. But 'B'=66-48=18.
        # So a payload starting with 'B' is msg type 18!
        payload = "B" + "0" * 27  # 168 bits = 28 chars
        sentence = f"!AIVDM,1,1,,A,{payload},0*"
        chk = 0
        body = sentence[1:sentence.index("*")]
        for c in body:
            chk ^= ord(c)
        sentence += f"{chk:02X}"
        result = self.parser.parse(sentence)
        assert isinstance(result, AISPositionReport)
        assert result.message_type == 18

    # -- Error handling -------------------------------------------------------

    def test_empty_sentence_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse("")

    def test_too_few_fields_raises(self):
        with pytest.raises(ParseError, match="Too few"):
            self.parser.decode_sentence("!AIVDM,1,1")

    def test_unknown_type_raises(self):
        with pytest.raises(ParseError, match="Unknown sentence type"):
            self.parser.decode_sentence("!GPGGA,1,1,,B,test,0*00")

    def test_unsupported_msg_type_raises(self):
        # Message type 6 = 0b000110 -> char = chr(48+6) = '6'
        payload = "6" + "0" * 15
        sentence = f"!AIVDM,1,1,,A,{payload},0*"
        chk = 0
        body = sentence[1:sentence.index("*")]
        for c in body:
            chk ^= ord(c)
        sentence += f"{chk:02X}"
        with pytest.raises(ParseError, match="Unsupported message type"):
            self.parser.parse(sentence)

    def test_short_payload_raises(self):
        sentence = f"!AIVDM,1,1,,A,,0*"
        chk = 0
        body = sentence[1:sentence.index("*")]
        for c in body:
            chk ^= ord(c)
        sentence += f"{chk:02X}"
        with pytest.raises(ParseError, match="too short"):
            self.parser.parse(sentence)

    # -- Bit manipulation -----------------------------------------------------

    def test_bits_to_uint(self):
        bits = [1, 0, 1, 0]  # = 10
        val = self.parser._bits_to_uint(bits, 0, 4)
        assert val == 10

    def test_bits_to_int_positive(self):
        bits = [0, 1, 0, 1]  # = 5
        val = self.parser._bits_to_int(bits, 0, 4)
        assert val == 5

    def test_bits_to_int_negative(self):
        bits = [1, 1, 1, 0]  # = -2 (two's complement)
        val = self.parser._bits_to_int(bits, 0, 4)
        assert val == -2

    def test_payload_to_bits_length(self):
        bits = self.parser._payload_to_bits("13u@Dt")
        assert len(bits) == 36  # 6 chars * 6 bits

    def test_char_to_payload_basic(self):
        # '0' -> 0, '1' -> 1, 'w' -> ord('w')-48-8 = 119-48-8 = 63
        assert self.parser._char_to_payload("0") == 0
        assert self.parser._char_to_payload("1") == 1

    # -- Type 24 Class B Static -----------------------------------------------

    def test_type24_recognized(self):
        # Type 24 = 0b011000 = 24 -> chr(48+24)=chr(72)='H'
        payload = "H" + "0" * 27
        sentence = f"!AIVDM,1,1,,A,{payload},0*"
        chk = 0
        body = sentence[1:sentence.index("*")]
        for c in body:
            chk ^= ord(c)
        sentence += f"{chk:02X}"
        result = self.parser.parse(sentence)
        assert isinstance(result, AISStaticData)
        assert result.message_type == 24


# =============================================================================
# ADS-B Parser (25+ tests)
# =============================================================================


class TestADSBParser:
    """Tests for ADSBParser."""

    def setup_method(self):
        self.parser = ADSBParser()

    # -- Aircraft Identification (TC 1-4) -------------------------------------

    def test_identification_message(self):
        # DF17, TC=4 (category D), callsign "KLM1023 "
        # This is a well-known test vector
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert isinstance(msg, ADSBIdentification)
        assert msg.icao_hex == "4840D6"

    def test_identification_callsign(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert isinstance(msg, ADSBIdentification)
        assert len(msg.callsign) > 0

    def test_identification_type_code(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert 1 <= msg.type_code <= 4

    def test_df17_extraction(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert msg.downlink_format == 17

    # -- Airborne Position (TC 9-18) ------------------------------------------

    def test_airborne_position(self):
        # DF17, TC=11, airborne position
        msg = self.parser.parse("8D40621D58C382D690C8AC2863A7")
        assert isinstance(msg, ADSBAirbornePosition)
        assert msg.type_code >= 9

    def test_airborne_position_altitude(self):
        msg = self.parser.parse("8D40621D58C382D690C8AC2863A7")
        if isinstance(msg, ADSBAirbornePosition):
            assert isinstance(msg.altitude_ft, float)

    def test_airborne_position_cpr(self):
        msg = self.parser.parse("8D40621D58C382D690C8AC2863A7")
        if isinstance(msg, ADSBAirbornePosition):
            assert isinstance(msg.cpr_lat, int)
            assert isinstance(msg.cpr_lon, int)
            assert isinstance(msg.cpr_odd, bool)

    def test_airborne_position_barometric(self):
        msg = self.parser.parse("8D40621D58C382D690C8AC2863A7")
        if isinstance(msg, ADSBAirbornePosition):
            assert msg.is_barometric is True

    # -- Velocity (TC 19) -----------------------------------------------------

    def test_velocity_message(self):
        # DF17, TC=19, velocity
        msg = self.parser.parse("8D485020994409940838175B284F")
        assert isinstance(msg, ADSBVelocity)
        assert msg.type_code == 19

    def test_velocity_speed(self):
        msg = self.parser.parse("8D485020994409940838175B284F")
        if isinstance(msg, ADSBVelocity):
            assert isinstance(msg.speed_kt, float)
            assert msg.speed_kt >= 0

    def test_velocity_heading(self):
        msg = self.parser.parse("8D485020994409940838175B284F")
        if isinstance(msg, ADSBVelocity):
            assert isinstance(msg.heading, float)

    def test_velocity_vertical_rate(self):
        msg = self.parser.parse("8D485020994409940838175B284F")
        if isinstance(msg, ADSBVelocity):
            assert isinstance(msg.vertical_rate_fpm, float)

    # -- DF11 All-Call --------------------------------------------------------

    def test_df11_all_call(self):
        # DF11 = 01011 -> first byte = 0101 1xxx = 0x5A for CA=2
        # ICAO = AABBCC, parity = 3 bytes
        msg = self.parser.parse("5AAABBCC112233")
        assert isinstance(msg, ADSBMessage)
        assert msg.downlink_format == 11
        assert msg.icao_hex == "AABBCC"

    # -- DF5/DF21 Squawk Reply ------------------------------------------------

    def test_df5_squawk(self):
        # DF5 = 00101 -> first byte = 0010 1xxx = 0x28
        # Build a synthetic 56-bit DF5 message
        msg_hex = "2800000000000000"[:14]  # 56 bits = 14 hex chars
        msg = self.parser.parse(msg_hex)
        assert isinstance(msg, ADSBSquawk)
        assert msg.downlink_format == 5

    def test_squawk_emergency_property(self):
        sq = ADSBSquawk(squawk="7700")
        assert sq.is_emergency is True

    def test_squawk_hijack_property(self):
        sq = ADSBSquawk(squawk="7500")
        assert sq.is_hijack is True

    def test_squawk_normal(self):
        sq = ADSBSquawk(squawk="1200")
        assert sq.is_emergency is False
        assert sq.is_hijack is False

    # -- DF4/DF20 Altitude Reply ----------------------------------------------

    def test_df4_altitude_reply(self):
        # DF4 = 00100 -> first byte = 0010 0xxx = 0x20
        msg_hex = "2000000000000000"[:14]
        msg = self.parser.parse(msg_hex)
        assert isinstance(msg, ADSBAltitudeReply)
        assert msg.downlink_format == 4

    # -- SBS parsing ----------------------------------------------------------

    def test_sbs_parse_basic(self):
        line = "MSG,3,1,1,A1B2C3,1,2026/03/25,10:00:00.000,2026/03/25,10:00:00.000,UAL123,35000,450,270,33.45,-112.07,500,1200,0,0,0,0"
        result = self.parser.parse_sbs(line)
        assert result["icao_hex"] == "A1B2C3"
        assert result["callsign"] == "UAL123"
        assert result["altitude_ft"] == 35000
        assert result["ground_speed_kt"] == 450
        assert result["heading"] == 270
        assert result["latitude"] == 33.45
        assert result["longitude"] == -112.07

    def test_sbs_empty_fields(self):
        line = "MSG,3,1,1,A1B2C3,1,2026/03/25,10:00:00.000,2026/03/25,10:00:00.000,,,,,,,,,,,,0"
        result = self.parser.parse_sbs(line)
        assert result["icao_hex"] == "A1B2C3"
        assert "callsign" not in result

    def test_sbs_squawk_field(self):
        # 22 fields: MSG,type,session,aircraft,icao,flight,gen_date,gen_time,log_date,log_time,
        #            callsign,altitude,speed,heading,lat,lon,vertical_rate,squawk,alert,emergency,spi,on_ground
        line = "MSG,3,1,1,A1B2C3,1,2026/03/25,10:00:00.000,2026/03/25,10:00:00.000,,,,,,,,7700,,,,-1"
        result = self.parser.parse_sbs(line)
        assert result.get("squawk") == "7700"

    def test_sbs_empty_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse_sbs("")

    def test_sbs_too_short_raises(self):
        with pytest.raises(ParseError, match="too short"):
            self.parser.parse_sbs("MSG,3,1,1")

    def test_sbs_not_msg_raises(self):
        with pytest.raises(ParseError, match="Not an SBS"):
            self.parser.parse_sbs("CLK,,,,,,,,,,,,,,,,,,,,,")

    # -- Error handling -------------------------------------------------------

    def test_empty_message_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse("")

    def test_invalid_hex_raises(self):
        with pytest.raises(ParseError, match="Invalid hex"):
            self.parser.parse("ZZZZZZZZZZZZZZ")

    def test_too_short_raises(self):
        with pytest.raises(ParseError, match="too short"):
            self.parser.parse("8D")

    def test_strip_decorations(self):
        """Should strip leading * and trailing ; (Beast format)."""
        msg = self.parser.parse("*8D4840D6202CC371C32CE0576098;")
        assert isinstance(msg, ADSBIdentification)

    # -- Message properties ---------------------------------------------------

    def test_df_name(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert msg.df_name == "Extended Squitter"

    def test_raw_hex_stored(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert len(msg.raw_hex) == 28

    def test_raw_bits_stored(self):
        msg = self.parser.parse("8D4840D6202CC371C32CE0576098")
        assert msg.raw_bits == 112


# =============================================================================
# Meshtastic Parser (25+ tests)
# =============================================================================


class TestMeshtasticParser:
    """Tests for MeshtasticParser."""

    def setup_method(self):
        self.parser = MeshtasticParser()

    # -- from_dict: text messages ---------------------------------------------

    def test_text_message(self):
        pkt = self.parser.from_dict({
            "from": 0xAABBCCDD,
            "to": 0xFFFFFFFF,
            "id": 12345,
            "decoded": {
                "portnum": 1,
                "payload": b"Hello mesh!",
                "text": "Hello mesh!",
            },
        })
        assert pkt.text == "Hello mesh!"
        assert pkt.portnum == 1
        assert pkt.portnum_name == "TEXT_MESSAGE_APP"

    def test_text_from_payload(self):
        pkt = self.parser.from_dict({
            "from": 0xAABBCCDD,
            "to": 0xFFFFFFFF,
            "id": 1,
            "decoded": {
                "portnum": 1,
                "payload": b"Fallback text",
            },
        })
        assert pkt.text == "Fallback text"

    def test_broadcast_detection(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "to": 0xFFFFFFFF,
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.is_broadcast is True

    def test_unicast_detection(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "to": 2,
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.is_broadcast is False

    # -- from_dict: position --------------------------------------------------

    def test_position_from_dict(self):
        pkt = self.parser.from_dict({
            "from": 0x11223344,
            "to": 0xFFFFFFFF,
            "decoded": {
                "portnum": 3,
                "position": {
                    "latitudeI": 334500000,
                    "longitudeI": -1120700000,
                    "altitude": 500,
                    "satsInView": 8,
                },
            },
        })
        assert pkt.position is not None
        assert abs(pkt.position.latitude - 33.45) < 0.001
        assert abs(pkt.position.longitude - (-112.07)) < 0.001
        assert pkt.position.altitude == 500
        assert pkt.position.has_valid_position is True

    def test_position_has_valid_check(self):
        pos = MeshtasticPosition(latitude_i=0, longitude_i=0)
        assert pos.has_valid_position is False

    # -- from_dict: node info -------------------------------------------------

    def test_nodeinfo_from_dict(self):
        pkt = self.parser.from_dict({
            "from": 0x11223344,
            "decoded": {
                "portnum": 4,
                "user": {
                    "id": "!11223344",
                    "longName": "My Node",
                    "shortName": "MN",
                    "hwModel": "HELTEC_V3",
                },
            },
        })
        assert pkt.node_info is not None
        assert pkt.node_info.long_name == "My Node"
        assert pkt.node_info.short_name == "MN"
        assert pkt.node_info.hw_model_name == "HELTEC_V3"

    def test_nodeinfo_hw_model_lookup(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 4,
                "user": {"id": "!00000001", "hwModel": "TBEAM"},
            },
        })
        assert pkt.node_info.hw_model == 4
        assert pkt.node_info.hw_model_name == "TBEAM"

    # -- from_dict: telemetry -------------------------------------------------

    def test_telemetry_from_dict(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 67,
                "telemetry": {
                    "time": 1711000000,
                    "deviceMetrics": {
                        "batteryLevel": 85,
                        "voltage": 3.95,
                        "channelUtilization": 12.5,
                        "airUtilTx": 3.2,
                        "uptimeSeconds": 86400,
                    },
                },
            },
        })
        assert pkt.telemetry is not None
        assert pkt.telemetry.battery_level == 85
        assert pkt.telemetry.voltage == 3.95
        assert pkt.telemetry.uptime_seconds == 86400

    def test_telemetry_environment(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 67,
                "telemetry": {
                    "deviceMetrics": {},
                    "environmentMetrics": {
                        "temperature": 22.5,
                        "relativeHumidity": 65.0,
                        "barometricPressure": 1013.25,
                    },
                },
            },
        })
        assert pkt.telemetry.temperature == 22.5
        assert pkt.telemetry.relative_humidity == 65.0
        assert pkt.telemetry.barometric_pressure == 1013.25

    # -- from_dict: routing ---------------------------------------------------

    def test_routing_from_dict(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 5,
                "routing": {"errorReason": "NONE"},
            },
        })
        assert pkt.routing is not None
        assert pkt.routing.error_reason == 0
        assert pkt.routing.error_text == "NONE"

    def test_routing_error(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 5,
                "routing": {"errorReason": 1},
            },
        })
        assert pkt.routing.error_reason == 1
        assert pkt.routing.error_text == "NO_ROUTE"

    # -- from_dict: neighbor info ---------------------------------------------

    def test_neighborinfo_from_dict(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": 72,
                "neighborinfo": {
                    "nodeId": 1,
                    "neighbors": [
                        {"nodeId": 2, "snr": 10.5},
                        {"nodeId": 3, "snr": 5.0},
                    ],
                },
            },
        })
        assert pkt.neighbor_info is not None
        assert len(pkt.neighbor_info.neighbors) == 2
        assert pkt.neighbor_info.neighbors[0]["snr"] == 10.5

    # -- from_dict: metadata --------------------------------------------------

    def test_rx_metadata(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "to": 0xFFFFFFFF,
            "rxTime": 1711000000,
            "rxSnr": 10.5,
            "rxRssi": -80,
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.rx_time == 1711000000
        assert pkt.rx_snr == 10.5
        assert pkt.rx_rssi == -80

    def test_source_hex_format(self):
        pkt = self.parser.from_dict({
            "from": 0xAABBCCDD,
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.source_hex == "!aabbccdd"

    def test_destination_hex_format(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "to": 0xFFFFFFFF,
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.destination_hex == "!ffffffff"

    # -- from_dict: string portnum --------------------------------------------

    def test_string_portnum(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "decoded": {
                "portnum": "TEXT_MESSAGE_APP",
                "text": "hi",
            },
        })
        assert pkt.portnum == 1
        assert pkt.portnum_name == "TEXT_MESSAGE_APP"

    # -- from_dict: string node IDs -------------------------------------------

    def test_string_source_id(self):
        pkt = self.parser.from_dict({
            "from": "!aabbccdd",
            "decoded": {"portnum": 1, "text": "hi"},
        })
        assert pkt.source == 0xAABBCCDD

    # -- Raw packet parsing ---------------------------------------------------

    def test_raw_packet_parse(self):
        # Build minimal packet: dest(4) + src(4) + id(4) + flags(1) + channel(1) = 14
        data = struct.pack("<III", 0xFFFFFFFF, 0x11223344, 99) + b"\x03\x00"
        pkt = self.parser.parse(data)
        assert pkt.destination == 0xFFFFFFFF
        assert pkt.source == 0x11223344
        assert pkt.packet_id == 99

    def test_raw_packet_flags(self):
        data = struct.pack("<III", 0xFFFFFFFF, 1, 1) + b"\x0B\x00"
        pkt = self.parser.parse(data)
        assert pkt.hop_limit == 3
        assert pkt.want_ack is True

    def test_raw_packet_hex_input(self):
        data = struct.pack("<III", 0xFFFFFFFF, 1, 1) + b"\x03\x00"
        pkt = self.parser.parse(data.hex())
        assert pkt.source == 1

    def test_raw_packet_too_short(self):
        with pytest.raises(ParseError, match="too short"):
            self.parser.parse(b"\x00\x01\x02")

    # -- Error handling -------------------------------------------------------

    def test_invalid_dict_type(self):
        with pytest.raises(ParseError, match="Expected dict"):
            self.parser.from_dict("not a dict")

    def test_invalid_hex_raises(self):
        with pytest.raises(ParseError, match="Invalid hex"):
            self.parser.parse("ZZZZ")

    # -- PORTNUM and HW_MODEL maps -------------------------------------------

    def test_portnum_map_coverage(self):
        assert PORTNUM_MAP[1] == "TEXT_MESSAGE_APP"
        assert PORTNUM_MAP[3] == "POSITION_APP"
        assert PORTNUM_MAP[4] == "NODEINFO_APP"
        assert PORTNUM_MAP[67] == "TELEMETRY_APP"

    def test_hw_model_map_coverage(self):
        assert HW_MODEL_MAP[4] == "TBEAM"
        assert HW_MODEL_MAP[43] == "HELTEC_V3"
        assert HW_MODEL_MAP[9] == "RAK4631"

    # -- Encrypted packet -----------------------------------------------------

    def test_encrypted_flag(self):
        pkt = self.parser.from_dict({
            "from": 1,
            "encrypted": True,
            "decoded": {},
        })
        assert pkt.encrypted is True


# =============================================================================
# NMEA Parser (25+ tests)
# =============================================================================


class TestNMEAParser:
    """Tests for NMEAParser."""

    def setup_method(self):
        self.parser = NMEAParser()

    # -- GGA ------------------------------------------------------------------

    def test_gga_basic(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert isinstance(result, NMEAGGA)
        assert result.talker_id == "GP"

    def test_gga_position(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert abs(result.position.latitude - 48.1173) < 0.001
        assert abs(result.position.longitude - 11.5167) < 0.001

    def test_gga_altitude(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert result.position.altitude_m == 545.4
        assert result.position.geoid_separation_m == 47.0

    def test_gga_fix_quality(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert result.fix_quality == 1
        assert result.fix_quality_text == "gps"

    def test_gga_satellites(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert result.satellites_used == 8

    def test_gga_hdop(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert result.hdop == 0.9

    def test_gga_time(self):
        result = self.parser.parse(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        )
        assert result.time.hours == 12
        assert result.time.minutes == 35
        assert result.time.seconds == 19.0

    def test_gga_south_west(self):
        result = self.parser.parse(
            "$GPGGA,000000,3345.000,S,11207.000,W,1,04,1.0,100.0,M,0.0,M,,*7D"
        )
        assert result.position.latitude < 0
        assert result.position.longitude < 0

    # -- RMC ------------------------------------------------------------------

    def test_rmc_basic(self):
        result = self.parser.parse(
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63"
        )
        assert isinstance(result, NMEARMC)
        assert result.status == "active"

    def test_rmc_speed(self):
        result = self.parser.parse(
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63"
        )
        assert result.speed_knots == 22.4

    def test_rmc_course(self):
        result = self.parser.parse(
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63"
        )
        assert result.course_degrees == 84.4

    def test_rmc_date(self):
        result = self.parser.parse(
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63"
        )
        assert result.date.day == 23
        assert result.date.month == 3
        assert result.date.year == 2026

    def test_rmc_void_status(self):
        result = self.parser.parse(
            "$GPRMC,000000,V,0000.000,N,00000.000,E,0.0,0.0,010100,0.0,E*61"
        )
        assert result.status == "void"

    def test_rmc_magnetic_variation_west(self):
        result = self.parser.parse(
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63"
        )
        assert result.magnetic_variation == -3.1

    # -- GSA ------------------------------------------------------------------

    def test_gsa_basic(self):
        result = self.parser.parse(
            "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
        )
        assert isinstance(result, NMEAGSA)
        assert result.fix_type == 3
        assert result.fix_type_text == "3d"

    def test_gsa_satellite_prns(self):
        result = self.parser.parse(
            "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
        )
        assert 4 in result.satellite_prns
        assert 5 in result.satellite_prns
        assert 24 in result.satellite_prns

    def test_gsa_dop_values(self):
        result = self.parser.parse(
            "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
        )
        assert result.pdop == 2.5
        assert result.hdop == 1.3
        assert result.vdop == 2.1

    def test_gsa_no_fix(self):
        result = self.parser.parse(
            "$GPGSA,A,1,,,,,,,,,,,,,,99.99,99.99,99.99*1C"
        )
        assert result.fix_type == 1
        assert result.fix_type_text == "no_fix"

    # -- VTG ------------------------------------------------------------------

    def test_vtg_basic(self):
        result = self.parser.parse(
            "$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"
        )
        assert isinstance(result, NMEAVTG)
        assert result.course_true == 54.7
        assert result.course_magnetic == 34.4
        assert result.speed_knots == 5.5
        assert result.speed_kmh == 10.2

    # -- GLL ------------------------------------------------------------------

    def test_gll_basic(self):
        result = self.parser.parse(
            "$GPGLL,4807.038,N,01131.000,E,123519,A*25"
        )
        assert isinstance(result, NMEAGLL)
        assert result.status == "active"
        assert abs(result.position.latitude - 48.1173) < 0.001

    def test_gll_void(self):
        result = self.parser.parse(
            "$GPGLL,0000.000,N,00000.000,E,000000,V*3D"
        )
        assert result.status == "void"

    # -- GSV ------------------------------------------------------------------

    def test_gsv_basic(self):
        result = self.parser.parse(
            "$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74"
        )
        assert isinstance(result, NMEAGSV)
        assert result.total_satellites == 11
        assert len(result.satellites) == 4

    def test_gsv_satellite_fields(self):
        result = self.parser.parse(
            "$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74"
        )
        sat = result.satellites[0]
        assert sat.prn == 3
        assert sat.elevation == 3
        assert sat.azimuth == 111
        assert sat.snr == 0

    # -- Multi-GNSS talker IDs -----------------------------------------------

    def test_gn_talker_id(self):
        result = self.parser.parse(
            "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*51"
        )
        assert isinstance(result, NMEAGGA)
        assert result.talker_id == "GN"

    # -- Checksum validation ---------------------------------------------------

    def test_valid_checksum(self):
        assert self.parser.validate_checksum(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        ) is True

    def test_invalid_checksum_raises(self):
        with pytest.raises(ParseError, match="Checksum mismatch"):
            self.parser.parse(
                "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*FF"
            )

    def test_no_checksum_ok(self):
        # No * means no checksum to validate
        assert self.parser.validate_checksum("$GPGGA,1,2,3") is True

    # -- Error handling -------------------------------------------------------

    def test_empty_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse("")

    def test_no_dollar_raises(self):
        with pytest.raises(ParseError, match="must start with"):
            self.parser.parse("GPGGA,1,2,3")

    def test_unsupported_type_raises(self):
        with pytest.raises(ParseError, match="Unsupported"):
            self.parser.parse("$GPXXX,1,2,3,4*4B")

    def test_too_few_fields_raises(self):
        # Sentence ID "GP" is too short (< 4 chars)
        with pytest.raises(ParseError):
            self.parser.parse("$GP,data")

    # -- parse_multi -----------------------------------------------------------

    def test_parse_multi(self):
        data = (
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F\n"
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63\n"
        )
        results = self.parser.parse_multi(data)
        assert len(results) == 2
        assert isinstance(results[0], NMEAGGA)
        assert isinstance(results[1], NMEARMC)

    def test_parse_multi_skips_invalid(self):
        data = (
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F\n"
            "GARBAGE LINE\n"
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230326,003.1,W*63\n"
        )
        results = self.parser.parse_multi(data)
        assert len(results) == 2

    # -- Dataclass properties -------------------------------------------------

    def test_nmea_time_str(self):
        t = NMEATime(hours=12, minutes=35, seconds=19.5)
        assert str(t) == "12:35:19.500"

    def test_nmea_date_str(self):
        d = NMEADate(day=25, month=3, year=2026)
        assert str(d) == "2026-03-25"

    def test_position_is_valid(self):
        pos = NMEAPosition(latitude=48.117, longitude=11.517)
        assert pos.is_valid is True

    def test_position_invalid(self):
        pos = NMEAPosition(latitude=0.0, longitude=0.0)
        assert pos.is_valid is False


# =============================================================================
# Import tests
# =============================================================================


class TestProtocolImports:
    """Verify top-level imports work."""

    def test_import_all_parsers(self):
        from tritium_lib.protocols import (
            BLEAdvertParser,
            WiFiProbeParser,
            AISParser,
            ADSBParser,
            MeshtasticParser,
            NMEAParser,
            ParseError,
        )
        assert BLEAdvertParser is not None
        assert WiFiProbeParser is not None
        assert AISParser is not None
        assert ADSBParser is not None
        assert MeshtasticParser is not None
        assert NMEAParser is not None
        assert ParseError is not None

    def test_parsers_instantiate(self):
        BLEAdvertParser()
        WiFiProbeParser()
        AISParser()
        ADSBParser()
        MeshtasticParser()
        NMEAParser()
