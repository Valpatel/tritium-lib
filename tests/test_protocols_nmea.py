# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.protocols.nmea — NMEA 0183 GPS parser."""

import pytest

from tritium_lib.protocols.errors import ParseError
from tritium_lib.protocols.nmea import (
    NMEAGGA,
    NMEAGLL,
    NMEAGSA,
    NMEAGSV,
    NMEAParser,
    NMEAPosition,
    NMEARMC,
    NMEATime,
    NMEADate,
    NMEASatellite,
    NMEAVTG,
)


@pytest.fixture
def parser():
    return NMEAParser()


class TestNMEAPosition:
    def test_valid(self):
        pos = NMEAPosition(latitude=48.117, longitude=11.517)
        assert pos.is_valid is True

    def test_invalid_zeroes(self):
        pos = NMEAPosition()
        assert pos.is_valid is False


class TestNMEATime:
    def test_str(self):
        t = NMEATime(hours=12, minutes=35, seconds=19.0)
        assert str(t) == "12:35:19.000"


class TestNMEADate:
    def test_str(self):
        d = NMEADate(day=23, month=3, year=2026)
        assert str(d) == "2026-03-23"


class TestNMEAParserGGA:
    def test_parse_gga(self, parser):
        sentence = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAGGA)
        assert result.talker_id == "GP"
        assert result.fix_quality == 1
        assert result.fix_quality_text == "gps"
        assert result.satellites_used == 8
        assert abs(result.position.latitude - 48.1173) < 0.001
        assert abs(result.position.longitude - 11.5167) < 0.001
        assert result.position.altitude_m == 545.4
        assert result.hdop == 0.9

    def test_parse_gga_time(self, parser):
        sentence = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        result = parser.parse(sentence)
        assert result.time is not None
        assert result.time.hours == 12
        assert result.time.minutes == 35
        assert result.time.seconds == 19.0


class TestNMEAParserRMC:
    def test_parse_rmc(self, parser):
        sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
        result = parser.parse(sentence)
        assert isinstance(result, NMEARMC)
        assert result.status == "active"
        assert result.speed_knots == 22.4
        assert result.course_degrees == 84.4
        assert abs(result.position.latitude - 48.1173) < 0.001

    def test_parse_rmc_date(self, parser):
        sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
        result = parser.parse(sentence)
        assert result.date is not None
        assert result.date.day == 23
        assert result.date.month == 3
        assert result.date.year == 1994

    def test_parse_rmc_void_status(self, parser):
        sentence = "$GPRMC,123519,V,,,,,,,230394,,,*1F"
        result = parser.parse(sentence)
        assert isinstance(result, NMEARMC)
        assert result.status == "void"


class TestNMEAParserGSA:
    def test_parse_gsa(self, parser):
        sentence = "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAGSA)
        assert result.mode == "A"
        assert result.fix_type == 3
        assert result.fix_type_text == "3d"
        assert result.pdop == 2.5
        assert result.hdop == 1.3
        assert result.vdop == 2.1
        assert 4 in result.satellite_prns
        assert 5 in result.satellite_prns


class TestNMEAParserVTG:
    def test_parse_vtg(self, parser):
        sentence = "$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAVTG)
        assert result.course_true == 54.7
        assert result.course_magnetic == 34.4
        assert result.speed_knots == 5.5
        assert result.speed_kmh == 10.2


class TestNMEAParserGLL:
    def test_parse_gll(self, parser):
        sentence = "$GPGLL,4916.45,N,12311.12,W,225444,A,*1D"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAGLL)
        assert result.status == "active"
        assert abs(result.position.latitude - 49.274) < 0.01
        assert result.position.longitude < 0  # West


class TestNMEAParserGSV:
    def test_parse_gsv(self, parser):
        sentence = "$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAGSV)
        assert result.total_messages == 3
        assert result.message_number == 1
        assert result.total_satellites == 11
        assert len(result.satellites) == 4
        assert result.satellites[0].prn == 3


class TestNMEAParserErrors:
    def test_empty(self, parser):
        with pytest.raises(ParseError):
            parser.parse("")

    def test_no_dollar(self, parser):
        with pytest.raises(ParseError):
            parser.parse("GPGGA,123519,...")

    def test_bad_checksum(self, parser):
        with pytest.raises(ParseError):
            parser.parse("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*00")

    def test_unsupported_sentence(self, parser):
        with pytest.raises(ParseError):
            parser.parse("$GPXYZ,1,2,3*00")

    def test_too_few_fields(self, parser):
        with pytest.raises(ParseError):
            parser.parse("$GP*00")

    def test_short_sentence_id(self, parser):
        with pytest.raises(ParseError):
            parser.parse("$GP,data*00")


class TestNMEAParserMulti:
    def test_parse_multi(self, parser):
        data = """$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F
$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
invalid line
$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"""
        results = parser.parse_multi(data)
        assert len(results) == 3

    def test_parse_multi_empty(self, parser):
        results = parser.parse_multi("")
        assert results == []

    def test_parse_multi_skips_invalid(self, parser):
        data = "$GPXYZ,bad*00\n$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"
        results = parser.parse_multi(data)
        assert len(results) == 1


class TestNMEAParserChecksum:
    def test_valid_checksum(self, parser):
        sentence = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*4F"
        assert parser.validate_checksum(sentence) is True

    def test_no_checksum(self, parser):
        sentence = "$GPGGA,123519,4807.038,N,01131.000,E"
        assert parser.validate_checksum(sentence) is True

    def test_gn_talker(self, parser):
        sentence = "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*51"
        result = parser.parse(sentence)
        assert isinstance(result, NMEAGGA)
        assert result.talker_id == "GN"


class TestNMEAParserHelpers:
    def test_parse_lat_south(self):
        lat = NMEAParser._parse_lat("3347.567", "S")
        assert lat < 0
        assert abs(lat - (-33.79278)) < 0.01

    def test_parse_lon_west(self):
        lon = NMEAParser._parse_lon("11831.456", "W")
        assert lon < 0

    def test_parse_lat_empty(self):
        assert NMEAParser._parse_lat("", "N") == 0.0

    def test_parse_lon_empty(self):
        assert NMEAParser._parse_lon("", "E") == 0.0

    def test_safe_float_empty(self):
        assert NMEAParser._safe_float("") == 0.0

    def test_safe_float_invalid(self):
        assert NMEAParser._safe_float("abc") == 0.0

    def test_safe_int_empty(self):
        assert NMEAParser._safe_int("") == 0

    def test_safe_int_invalid(self):
        assert NMEAParser._safe_int("xyz") == 0

    def test_parse_date_year_2000s(self):
        d = NMEAParser._parse_date("230326")
        assert d.year == 2026

    def test_parse_date_year_1900s(self):
        d = NMEAParser._parse_date("230394")
        assert d.year == 1994

    def test_parse_date_empty(self):
        assert NMEAParser._parse_date("") is None

    def test_parse_time_empty(self):
        assert NMEAParser._parse_time("") is None

    def test_parse_time_short(self):
        assert NMEAParser._parse_time("12") is None
