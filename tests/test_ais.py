# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AIS and ADS-B models."""

import pytest

from tritium_lib.models.ais import (
    AISPosition,
    AISVessel,
    VesselType,
    NavigationStatus,
    ADSBPosition,
    ADSBFlight,
    FlightCategory,
    SquawkCode,
)


# -- AIS Vessel tests -----------------------------------------------------

class TestAISVessel:
    def test_create_vessel(self):
        vessel = AISVessel(mmsi=211234567, name="HAMBURG EXPRESS")
        assert vessel.mmsi == 211234567
        assert vessel.name == "HAMBURG EXPRESS"
        assert vessel.vessel_type == VesselType.UNKNOWN

    def test_vessel_with_position(self):
        pos = AISPosition(
            latitude=51.5074,
            longitude=-0.1278,
            heading=90.0,
            speed_over_ground=12.5,
        )
        vessel = AISVessel(
            mmsi=211234567,
            name="TEST SHIP",
            position=pos,
            vessel_type=VesselType.CARGO,
        )
        assert vessel.position.latitude == 51.5074
        assert vessel.position.speed_over_ground == 12.5
        assert vessel.vessel_type == VesselType.CARGO

    def test_compute_target_id(self):
        vessel = AISVessel(mmsi=123456789)
        assert vessel.compute_target_id() == "ais_123456789"

    def test_to_target_dict(self):
        vessel = AISVessel(
            mmsi=211234567,
            name="CARGO ONE",
            call_sign="DLFG",
            vessel_type=VesselType.TANKER,
            length=200.0,
            beam=30.0,
            destination="ROTTERDAM",
            position=AISPosition(
                latitude=52.0, longitude=4.0, heading=180.0, speed_over_ground=15.0,
            ),
        )
        d = vessel.to_target_dict()
        assert d["target_id"] == "ais_211234567"
        assert d["name"] == "CARGO ONE"
        assert d["source"] == "ais"
        assert d["asset_type"] == "tanker"
        assert d["position"]["lat"] == 52.0
        assert d["heading"] == 180.0
        assert d["speed"] == 15.0
        assert d["metadata"]["destination"] == "ROTTERDAM"
        assert d["metadata"]["call_sign"] == "DLFG"

    def test_vessel_types(self):
        for vtype in VesselType:
            vessel = AISVessel(mmsi=100000000, vessel_type=vtype)
            assert vessel.vessel_type == vtype

    def test_navigation_status(self):
        vessel = AISVessel(
            mmsi=100000000,
            navigation_status=NavigationStatus.MOORED,
        )
        assert vessel.navigation_status == NavigationStatus.MOORED

    def test_vessel_default_name_in_target(self):
        """Unnamed vessel should use MMSI as target name."""
        vessel = AISVessel(mmsi=999999999)
        d = vessel.to_target_dict()
        assert "999999999" in d["name"]

    def test_vessel_dimensions(self):
        vessel = AISVessel(
            mmsi=100000000, length=150.0, beam=25.0, draught=8.5,
        )
        assert vessel.length == 150.0
        assert vessel.beam == 25.0
        assert vessel.draught == 8.5


# -- AIS Position tests ----------------------------------------------------

class TestAISPosition:
    def test_default_position(self):
        pos = AISPosition()
        assert pos.latitude == 0.0
        assert pos.longitude == 0.0
        assert pos.heading == 0.0

    def test_position_fields(self):
        pos = AISPosition(
            latitude=33.45,
            longitude=-112.07,
            heading=270.0,
            course_over_ground=265.0,
            speed_over_ground=8.0,
            rate_of_turn=-5.0,
        )
        assert pos.latitude == 33.45
        assert pos.course_over_ground == 265.0
        assert pos.rate_of_turn == -5.0


# -- ADS-B Flight tests ---------------------------------------------------

class TestADSBFlight:
    def test_create_flight(self):
        flight = ADSBFlight(icao_hex="A1B2C3", callsign="UAL123")
        assert flight.icao_hex == "A1B2C3"
        assert flight.callsign == "UAL123"
        assert flight.category == FlightCategory.UNKNOWN

    def test_compute_target_id(self):
        flight = ADSBFlight(icao_hex="A1B2C3")
        assert flight.compute_target_id() == "adsb_a1b2c3"

    def test_flight_with_position(self):
        pos = ADSBPosition(
            latitude=33.45,
            longitude=-112.07,
            altitude_ft=35000,
            heading=90.0,
            ground_speed=450.0,
            vertical_rate=500.0,
        )
        flight = ADSBFlight(
            icao_hex="A1B2C3",
            callsign="UAL123",
            position=pos,
            category=FlightCategory.LARGE,
        )
        assert flight.position.altitude_ft == 35000
        assert flight.position.ground_speed == 450.0
        assert flight.category == FlightCategory.LARGE

    def test_to_target_dict(self):
        flight = ADSBFlight(
            icao_hex="ABCDEF",
            callsign="AAL456",
            aircraft_type="B738",
            operator="American Airlines",
            category=FlightCategory.LARGE,
            position=ADSBPosition(
                latitude=40.0,
                longitude=-74.0,
                altitude_ft=30000,
                heading=270.0,
                ground_speed=400.0,
            ),
        )
        d = flight.to_target_dict()
        assert d["target_id"] == "adsb_abcdef"
        assert d["name"] == "AAL456"
        assert d["source"] == "adsb"
        assert d["position"]["lat"] == 40.0
        assert d["metadata"]["aircraft_type"] == "B738"
        assert d["metadata"]["altitude_ft"] == 30000

    def test_default_name_in_target(self):
        """Unnamed flight should use ICAO as target name."""
        flight = ADSBFlight(icao_hex="ABCDEF")
        d = flight.to_target_dict()
        assert "ABCDEF" in d["name"]

    def test_flight_categories(self):
        for cat in FlightCategory:
            flight = ADSBFlight(icao_hex="000001", category=cat)
            assert flight.category == cat


# -- ADS-B Position tests --------------------------------------------------

class TestADSBPosition:
    def test_compute_altitude_m(self):
        pos = ADSBPosition(altitude_ft=10000)
        assert abs(pos.compute_altitude_m() - 3048.0) < 1.0

    def test_zero_altitude(self):
        pos = ADSBPosition(altitude_ft=0)
        assert pos.compute_altitude_m() == 0.0

    def test_vertical_rate(self):
        pos = ADSBPosition(vertical_rate=-1000.0)
        assert pos.vertical_rate == -1000.0


# -- Squawk code tests -----------------------------------------------------

class TestSquawkCode:
    def test_normal_squawk(self):
        sq = SquawkCode(code="1200")
        assert not sq.is_emergency
        assert not sq.is_hijack
        assert not sq.is_radio_failure

    def test_hijack_squawk(self):
        sq = SquawkCode(code="7500")
        assert sq.is_hijack
        assert not sq.is_radio_failure

    def test_radio_failure(self):
        sq = SquawkCode(code="7600")
        assert sq.is_radio_failure
        assert not sq.is_hijack

    def test_emergency_squawk(self):
        sq = SquawkCode(code="7700")
        assert sq.is_emergency

    def test_emergency_flag(self):
        sq = SquawkCode(code="1234", emergency=True)
        assert sq.is_emergency

    def test_hijack_flight_alliance(self):
        """Hijack squawk should result in hostile alliance."""
        flight = ADSBFlight(
            icao_hex="ABCDEF",
            squawk=SquawkCode(code="7500"),
        )
        d = flight.to_target_dict()
        assert d["alliance"] == "hostile"

    def test_emergency_flight_alliance(self):
        """Emergency squawk should result in neutral alliance."""
        flight = ADSBFlight(
            icao_hex="ABCDEF",
            squawk=SquawkCode(code="7700"),
        )
        assert flight.is_emergency()
        d = flight.to_target_dict()
        assert d["alliance"] == "neutral"


# -- Import from top-level ------------------------------------------------

class TestImports:
    def test_import_from_models(self):
        from tritium_lib.models import AISVessel, ADSBFlight, SquawkCode
        assert AISVessel is not None
        assert ADSBFlight is not None
        assert SquawkCode is not None
