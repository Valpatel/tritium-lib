# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""RF signature simulation for ambient entities.

Every simulated person, vehicle, and building emits realistic radio signals
— BLE advertisements, WiFi probe requests, TPMS transmissions — in the
EXACT same format that real sensors produce.  This means the simulation
exercises the full sensor-fusion pipeline (edge_tracker, wifi_fingerprint,
TargetTracker) end-to-end.

Usage::

    from tritium_lib.sim_engine.ai.rf_signatures import RFSignatureGenerator

    person = RFSignatureGenerator.random_person()
    ads = person.emit_ble_advertisements(position=(100.0, 200.0))
    probes = person.emit_wifi_probes(position=(100.0, 200.0))

    vehicle = RFSignatureGenerator.random_vehicle()
    tpms = vehicle.emit_tpms(position=(300.0, 400.0))

    building = RFSignatureGenerator.random_building("commercial")
    beacons = building.emit_wifi_beacons(position=(50.0, 50.0))
"""

from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — realistic device distributions
# ---------------------------------------------------------------------------

# BLE company IDs (Bluetooth SIG assigned numbers)
APPLE_COMPANY_ID = 0x004C
SAMSUNG_COMPANY_ID = 0x0075
GOOGLE_COMPANY_ID = 0x00E0
MICROSOFT_COMPANY_ID = 0x0006
HUAWEI_COMPANY_ID = 0x027D
XIAOMI_COMPANY_ID = 0x038F
FITBIT_COMPANY_ID = 0x0224
GARMIN_COMPANY_ID = 0x0087
SONY_COMPANY_ID = 0x012D
LG_COMPANY_ID = 0x00C7

# Phone models by ecosystem
_APPLE_PHONES = [
    "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15", "iPhone 15 Plus",
    "iPhone 14 Pro Max", "iPhone 14 Pro", "iPhone 14", "iPhone 13",
    "iPhone SE (3rd gen)",
]
_ANDROID_PHONES = [
    "Galaxy S24 Ultra", "Galaxy S24+", "Galaxy S24", "Galaxy S23",
    "Galaxy A54", "Galaxy A34", "Pixel 8 Pro", "Pixel 8", "Pixel 7a",
    "OnePlus 12", "OnePlus 11", "Xiaomi 14", "Xiaomi Redmi Note 13",
    "Huawei P60 Pro", "Sony Xperia 1 V", "LG V60",
]

# Watch models
_APPLE_WATCHES = [
    "Apple Watch Ultra 2", "Apple Watch Series 9", "Apple Watch SE (2nd gen)",
]
_ANDROID_WATCHES = [
    "Galaxy Watch6 Classic", "Galaxy Watch6", "Google Pixel Watch 2",
    "Garmin Venu 3", "Garmin Fenix 7", "Fitbit Sense 2", "Fitbit Versa 4",
]

# Earbuds
_APPLE_EARBUDS = ["AirPods Pro (2nd gen)", "AirPods (3rd gen)", "AirPods Max"]
_ANDROID_EARBUDS = [
    "Galaxy Buds2 Pro", "Galaxy Buds FE", "Pixel Buds Pro",
    "Sony WF-1000XM5", "Jabra Elite 85t", "JBL Tour Pro 2",
]

# Common WiFi SSID patterns that phones probe for
_COMMON_SSID_PATTERNS = [
    "HOME-{four_hex}", "NETGEAR{two_digit}", "linksys{two_digit}",
    "xfinitywifi", "ATT-WIFI-{four_hex}", "DIRECT-{two_char}",
    "MySpectrumWiFi{two_char}-{freq}", "Verizon_{four_hex}",
    "TP-Link_{four_hex}", "ASUS_{four_hex}",
    "CenturyLink{four_digit}", "Google_Home_{four_hex}",
    "AndroidAP_{four_hex}", "iPhone ({name})",
]

# Common open / public SSIDs
_PUBLIC_SSIDS = [
    "xfinitywifi", "attwifi", "Starbucks WiFi", "Google Starbucks",
    "McDonald's Free WiFi", "XFINITY", "optimumwifi",
    "CableWiFi", "TWCWiFi", "BrightHouseWiFi",
]

# Vehicle makes and models
_VEHICLE_MAKES = [
    ("Toyota", ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"]),
    ("Honda", ["Civic", "Accord", "CR-V", "Pilot"]),
    ("Ford", ["F-150", "Escape", "Explorer", "Mustang"]),
    ("Chevrolet", ["Silverado", "Equinox", "Malibu", "Tahoe"]),
    ("Tesla", ["Model 3", "Model Y", "Model S", "Model X"]),
    ("Hyundai", ["Elantra", "Tucson", "Santa Fe", "Ioniq 5"]),
    ("Nissan", ["Altima", "Rogue", "Sentra", "Frontier"]),
    ("BMW", ["3 Series", "5 Series", "X3", "X5"]),
    ("Mercedes-Benz", ["C-Class", "E-Class", "GLC", "GLE"]),
    ("Subaru", ["Outback", "Forester", "Crosstrek", "Impreza"]),
]

_VEHICLE_COLORS = [
    "White", "Black", "Silver", "Gray", "Red", "Blue",
    "Brown", "Green", "Beige", "Orange",
]

# Building WiFi and IoT
_RESIDENTIAL_SSID_PATTERNS = [
    "HOME-{four_hex}", "NETGEAR{two_digit}", "linksys{two_digit}",
    "TP-Link_{four_hex}", "ASUS_{four_hex}", "{family}_WiFi",
]

_COMMERCIAL_SSID_PATTERNS = [
    "{business}_Guest", "{business}_Staff", "{business}_IoT",
    "FREE_{business}_WIFI", "Corp-{four_hex}",
]

_IOT_DEVICE_TYPES = [
    ("doorbell", "ble"), ("security_camera", "wifi"), ("thermostat", "wifi"),
    ("smart_lock", "ble"), ("smart_plug", "wifi"), ("smart_bulb", "ble"),
    ("smoke_detector", "ble"), ("garage_opener", "wifi"),
    ("sprinkler_controller", "wifi"), ("robot_vacuum", "wifi"),
]

_FAMILY_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Hernandez", "Moore", "Jackson", "Lee", "Perez", "White", "Harris",
]

_BUSINESS_NAMES = [
    "Office", "Suite", "Corp", "HQ", "Lobby", "Workshop", "Studio",
    "Clinic", "Shop", "Cafe",
]

# US state license plate formats (simplified)
_PLATE_FORMATS: dict[str, str] = {
    "CA": "{digit}{LLL}{digit}{digit}{digit}",
    "TX": "{LLL}-{digit}{digit}{digit}{digit}",
    "NY": "{LLL}-{digit}{digit}{digit}{digit}",
    "FL": "{LLL}{LLL}{digit}{digit}",
    "IL": "{LL} {digit}{digit}{digit}{digit}{digit}",
}

# MAC rotation interval in seconds (simulating iOS/Android MAC randomization)
MAC_ROTATION_INTERVAL_S = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _random_mac(rng: random.Random | None = None) -> str:
    """Generate a random locally-administered unicast MAC address."""
    r = rng or random
    # Set locally administered bit (bit 1 of first octet) and clear
    # multicast bit (bit 0 of first octet).
    first_octet = (r.randint(0, 255) | 0x02) & 0xFE
    octets = [first_octet] + [r.randint(0, 255) for _ in range(5)]
    return ":".join(f"{b:02X}" for b in octets)


def _random_oui_mac(company_id: int, rng: random.Random | None = None) -> str:
    """Generate a MAC that encodes a company hint in the last 3 bytes.

    Real BLE MACs from Apple/Samsung are randomized but the company ID
    shows up in the advertisement manufacturer data, not in the MAC itself.
    We store the company ID in metadata; the MAC is still random.
    """
    return _random_mac(rng)


def _random_tpms_id(rng: random.Random | None = None) -> str:
    """Generate a random 32-bit TPMS sensor ID as hex string."""
    r = rng or random
    return f"{r.randint(0, 0xFFFFFFFF):08X}"


def _random_plate(state: str = "CA", rng: random.Random | None = None) -> str:
    """Generate a random US license plate for the given state."""
    r = rng or random
    fmt = _PLATE_FORMATS.get(state, _PLATE_FORMATS["CA"])
    result = []
    i = 0
    while i < len(fmt):
        if fmt[i] == '{':
            end = fmt.index('}', i)
            token = fmt[i + 1:end]
            if token == "digit":
                result.append(str(r.randint(0, 9)))
            elif token == "LLL":
                result.append("".join(r.choices(string.ascii_uppercase, k=3)))
            elif token == "LL":
                result.append("".join(r.choices(string.ascii_uppercase, k=2)))
            elif token == "L":
                result.append(r.choice(string.ascii_uppercase))
            i = end + 1
        else:
            result.append(fmt[i])
            i += 1
    return "".join(result)


def _generate_ssid(pattern: str, rng: random.Random | None = None) -> str:
    """Fill in an SSID pattern with random values."""
    r = rng or random
    result = pattern
    result = result.replace("{four_hex}", f"{r.randint(0, 0xFFFF):04X}")
    result = result.replace("{two_digit}", f"{r.randint(10, 99)}")
    result = result.replace("{four_digit}", f"{r.randint(1000, 9999)}")
    result = result.replace("{two_char}", "".join(r.choices(string.ascii_uppercase, k=2)))
    result = result.replace("{freq}", r.choice(["2G", "5G"]))
    result = result.replace("{name}", r.choice(_FAMILY_NAMES))
    result = result.replace("{family}", r.choice(_FAMILY_NAMES))
    result = result.replace("{business}", r.choice(_BUSINESS_NAMES))
    return result


# ---------------------------------------------------------------------------
# PersonRFProfile
# ---------------------------------------------------------------------------

@dataclass
class PersonRFProfile:
    """RF signature of a simulated person.

    Generates BLE advertisements and WiFi probe requests that match the
    exact format consumed by edge_tracker and wifi_fingerprint plugins.
    """

    # Phone
    has_phone: bool = True
    phone_mac: str = ""
    phone_model: str = ""
    phone_ble_company_id: int = APPLE_COMPANY_ID
    phone_wifi_probes: list[str] = field(default_factory=list)
    phone_ecosystem: str = "apple"  # "apple", "android", "other"

    # Smartwatch
    has_smartwatch: bool = False
    watch_mac: str = ""
    watch_model: str = ""
    watch_ble_company_id: int = APPLE_COMPANY_ID

    # Earbuds
    has_earbuds: bool = False
    earbuds_mac: str = ""
    earbuds_model: str = ""
    earbuds_ble_company_id: int = APPLE_COMPANY_ID

    # MAC rotation tracking
    _mac_last_rotated: float = field(default_factory=time.time)
    _original_company_id: int = 0

    def emit_ble_advertisements(
        self, position: tuple[float, float], rssi_at_1m: int = -59
    ) -> list[dict]:
        """Emit BLE advertisements from all devices this person carries.

        Returns dicts matching the format expected by edge_tracker:
        ``{mac, rssi, name, company_id, position_x, position_y, source, ...}``
        """
        now = time.time()
        ads: list[dict] = []

        if self.has_phone:
            ads.append({
                "mac": self.phone_mac,
                "rssi": rssi_at_1m + random.randint(-6, 6),
                "name": self.phone_model,
                "company_id": self.phone_ble_company_id,
                "source": "ble",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "device_type": "phone",
                "simulated": True,
                "metadata": {
                    "ecosystem": self.phone_ecosystem,
                    "model": self.phone_model,
                    "company_id_hex": f"0x{self.phone_ble_company_id:04X}",
                },
            })

        if self.has_smartwatch:
            ads.append({
                "mac": self.watch_mac,
                "rssi": rssi_at_1m + random.randint(-10, 2),
                "name": self.watch_model,
                "company_id": self.watch_ble_company_id,
                "source": "ble",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "device_type": "smartwatch",
                "simulated": True,
                "metadata": {
                    "model": self.watch_model,
                    "company_id_hex": f"0x{self.watch_ble_company_id:04X}",
                },
            })

        if self.has_earbuds:
            ads.append({
                "mac": self.earbuds_mac,
                "rssi": rssi_at_1m + random.randint(-12, 0),
                "name": self.earbuds_model,
                "company_id": self.earbuds_ble_company_id,
                "source": "ble",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "device_type": "earbuds",
                "simulated": True,
                "metadata": {
                    "model": self.earbuds_model,
                    "company_id_hex": f"0x{self.earbuds_ble_company_id:04X}",
                },
            })

        return ads

    def emit_wifi_probes(
        self, position: tuple[float, float]
    ) -> list[dict]:
        """Emit WiFi probe requests from the phone.

        Returns dicts matching the format expected by wifi_fingerprint:
        ``{mac, ssid, rssi, position_x, position_y, source, ...}``
        """
        if not self.has_phone or not self.phone_wifi_probes:
            return []

        now = time.time()
        # A phone typically probes for 1-3 SSIDs at a time (not all at once)
        count = min(len(self.phone_wifi_probes), random.randint(1, 3))
        ssids = random.sample(self.phone_wifi_probes, count)

        probes: list[dict] = []
        for ssid in ssids:
            probes.append({
                "mac": self.phone_mac,
                "ssid": ssid,
                "rssi": random.randint(-75, -45),
                "source": "wifi_probe",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "device_type": "phone",
                "simulated": True,
            })

        return probes

    def rotate_mac(self) -> None:
        """Simulate MAC address randomization (iOS/Android rotate every ~15 min).

        The company ID and probe list stay the same — only the MAC changes.
        This is how real phones behave: the MAC rotates but the BLE
        manufacturer data (company_id) and preferred network list persist.
        """
        if self.has_phone:
            self.phone_mac = _random_mac()
        if self.has_smartwatch:
            self.watch_mac = _random_mac()
        if self.has_earbuds:
            self.earbuds_mac = _random_mac()
        self._mac_last_rotated = time.time()

    def should_rotate_mac(self) -> bool:
        """Check if it's time to rotate MAC addresses."""
        return (time.time() - self._mac_last_rotated) >= MAC_ROTATION_INTERVAL_S


# ---------------------------------------------------------------------------
# VehicleRFProfile
# ---------------------------------------------------------------------------

@dataclass
class VehicleRFProfile:
    """RF signature of a simulated vehicle.

    Vehicles emit TPMS sensor data (ISM band), keyfob BLE, and optionally
    dashcam WiFi hotspot signals.
    """

    # TPMS (tire pressure monitoring system)
    tpms_ids: list[str] = field(default_factory=list)  # 4 tire sensor IDs
    tpms_frequency: float = 315.0  # MHz (315.0 US / 433.92 EU)

    # Keyfob BLE
    has_keyfob: bool = True
    keyfob_mac: str = ""
    keyfob_ble_company_id: int = 0

    # Dashcam WiFi hotspot
    has_dashcam_wifi: bool = False
    dashcam_ssid: str = ""
    dashcam_mac: str = ""

    # Vehicle identity
    license_plate: str = ""
    make_model: str = ""
    year: int = 2020
    color: str = ""

    def emit_tpms(self, position: tuple[float, float]) -> list[dict]:
        """Emit TPMS sensor readings for all 4 tires.

        Returns dicts compatible with ISMDevice.to_target_dict() format.
        TPMS transmits every 30-60 seconds in real life.
        """
        now = time.time()
        readings: list[dict] = []
        tire_labels = ["FL", "FR", "RL", "RR"]

        for i, (sensor_id, label) in enumerate(zip(self.tpms_ids, tire_labels)):
            # Realistic tire pressure: 30-36 PSI, temperature: 20-45 C
            pressure_psi = round(random.uniform(30.0, 36.0), 1)
            temp_c = round(random.uniform(20.0, 45.0), 1)

            readings.append({
                "target_id": f"ism_tpms_{sensor_id}",
                "source": "sdr_ism",
                "classification": "ism_device",
                "alliance": "unknown",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "simulated": True,
                "metadata": {
                    "device_type": "TPMS",
                    "protocol": "tpms",
                    "frequency_mhz": self.tpms_frequency,
                    "device_id": sensor_id,
                    "tire_position": label,
                    "pressure_psi": pressure_psi,
                    "temperature_c": temp_c,
                    "vehicle_plate": self.license_plate,
                    "vehicle_make_model": self.make_model,
                },
            })

        return readings

    def emit_keyfob_ble(self, position: tuple[float, float]) -> list[dict]:
        """Emit BLE advertisement from the keyfob (when vehicle is running).

        Returns dicts matching edge_tracker BLE sighting format.
        """
        if not self.has_keyfob:
            return []

        now = time.time()
        return [{
            "mac": self.keyfob_mac,
            "rssi": random.randint(-70, -50),
            "name": f"Keyfob {self.make_model}",
            "company_id": self.keyfob_ble_company_id,
            "source": "ble",
            "position_x": position[0],
            "position_y": position[1],
            "timestamp": now,
            "device_type": "keyfob",
            "simulated": True,
            "metadata": {
                "vehicle_plate": self.license_plate,
                "vehicle_make_model": self.make_model,
            },
        }]

    def emit_dashcam_wifi(self, position: tuple[float, float]) -> list[dict]:
        """Emit WiFi beacon from the dashcam hotspot."""
        if not self.has_dashcam_wifi:
            return []

        now = time.time()
        return [{
            "bssid": self.dashcam_mac,
            "ssid": self.dashcam_ssid,
            "rssi": random.randint(-65, -40),
            "channel": random.choice([1, 6, 11]),
            "source": "wifi_beacon",
            "position_x": position[0],
            "position_y": position[1],
            "timestamp": now,
            "device_type": "dashcam",
            "simulated": True,
        }]


# ---------------------------------------------------------------------------
# BuildingRFProfile
# ---------------------------------------------------------------------------

@dataclass
class BuildingRFProfile:
    """RF signature of a building (WiFi access points, IoT devices).

    Buildings are stationary RF emitters — they provide the WiFi and IoT
    background that real sensors would see.
    """

    building_type: str = "residential"  # residential, commercial, industrial
    wifi_aps: list[dict] = field(default_factory=list)
    iot_devices: list[dict] = field(default_factory=list)

    def emit_wifi_beacons(self, position: tuple[float, float]) -> list[dict]:
        """Emit WiFi beacon frames from all access points.

        Returns dicts compatible with wifi_fingerprint plugin format.
        """
        now = time.time()
        beacons: list[dict] = []

        for ap in self.wifi_aps:
            beacons.append({
                "bssid": ap["bssid"],
                "ssid": ap["ssid"],
                "rssi": ap.get("signal_strength", -50) + random.randint(-3, 3),
                "channel": ap.get("channel", 6),
                "source": "wifi_beacon",
                "position_x": position[0],
                "position_y": position[1],
                "timestamp": now,
                "device_type": "access_point",
                "simulated": True,
                "metadata": {
                    "building_type": self.building_type,
                    "encryption": ap.get("encryption", "WPA2"),
                },
            })

        return beacons

    def emit_iot_signals(self, position: tuple[float, float]) -> list[dict]:
        """Emit BLE/WiFi signals from IoT devices in the building."""
        now = time.time()
        signals: list[dict] = []

        for device in self.iot_devices:
            protocol = device.get("protocol", "ble")
            signals.append({
                "mac": device["mac"],
                "rssi": random.randint(-80, -55),
                "name": device.get("name", device["type"]),
                "source": protocol,
                "position_x": position[0] + random.uniform(-5, 5),
                "position_y": position[1] + random.uniform(-5, 5),
                "timestamp": now,
                "device_type": device["type"],
                "simulated": True,
                "metadata": {
                    "building_type": self.building_type,
                    "iot_type": device["type"],
                    "protocol": protocol,
                },
            })

        return signals


# ---------------------------------------------------------------------------
# RFSignatureGenerator — factory for realistic RF profiles
# ---------------------------------------------------------------------------

class RFSignatureGenerator:
    """Generates realistic RF profiles for simulation entities.

    Uses realistic distributions:
    - 85% of people have phones, 30% have smartwatches, 60% have earbuds
    - Apple 45%, Android 45%, other 10%
    - Phones probe for 2-6 known SSIDs
    - Cars emit TPMS every 30-60s, 20% have dashcam WiFi
    - Buildings have 1-4 WiFi APs and 0-8 IoT devices
    """

    @staticmethod
    def random_mac(rng: random.Random | None = None) -> str:
        """Generate a random locally-administered unicast MAC."""
        return _random_mac(rng)

    @staticmethod
    def random_tpms_id(rng: random.Random | None = None) -> str:
        """Generate a random 32-bit TPMS sensor ID."""
        return _random_tpms_id(rng)

    @staticmethod
    def random_plate(state: str = "CA", rng: random.Random | None = None) -> str:
        """Generate a random US license plate."""
        return _random_plate(state, rng)

    @staticmethod
    def random_person(
        age_range: tuple[int, int] = (18, 65),
        rng: random.Random | None = None,
    ) -> PersonRFProfile:
        """Generate a realistic PersonRFProfile.

        Device ownership probabilities:
        - Phone: 85% (higher for younger age ranges)
        - Smartwatch: 30%
        - Earbuds: 60%
        - Apple ecosystem: 45%, Android: 45%, Other: 10%
        """
        r = rng or random

        # Adjust phone ownership by age (younger = more likely)
        age = r.randint(*age_range)
        phone_chance = 0.95 if age < 40 else 0.85 if age < 60 else 0.70
        watch_chance = 0.35 if age < 45 else 0.25
        earbuds_chance = 0.70 if age < 35 else 0.55

        has_phone = r.random() < phone_chance
        has_watch = r.random() < watch_chance
        has_earbuds = r.random() < earbuds_chance and has_phone

        # Choose ecosystem
        eco_roll = r.random()
        if eco_roll < 0.45:
            ecosystem = "apple"
            company_id = APPLE_COMPANY_ID
            phone_model = r.choice(_APPLE_PHONES) if has_phone else ""
            watch_model = r.choice(_APPLE_WATCHES) if has_watch else ""
            earbuds_model = r.choice(_APPLE_EARBUDS) if has_earbuds else ""
        elif eco_roll < 0.90:
            ecosystem = "android"
            phone_model = r.choice(_ANDROID_PHONES) if has_phone else ""
            watch_model = r.choice(_ANDROID_WATCHES) if has_watch else ""
            earbuds_model = r.choice(_ANDROID_EARBUDS) if has_earbuds else ""
            # Pick company ID based on specific model
            if has_phone and "Galaxy" in phone_model:
                company_id = SAMSUNG_COMPANY_ID
            elif has_phone and "Pixel" in phone_model:
                company_id = GOOGLE_COMPANY_ID
            elif has_phone and "Xiaomi" in phone_model:
                company_id = XIAOMI_COMPANY_ID
            elif has_phone and "Huawei" in phone_model:
                company_id = HUAWEI_COMPANY_ID
            elif has_phone and "Sony" in phone_model:
                company_id = SONY_COMPANY_ID
            else:
                company_id = r.choice([
                    SAMSUNG_COMPANY_ID, GOOGLE_COMPANY_ID, XIAOMI_COMPANY_ID,
                ])
        else:
            ecosystem = "other"
            company_id = r.choice([
                HUAWEI_COMPANY_ID, XIAOMI_COMPANY_ID, LG_COMPANY_ID,
            ])
            phone_model = r.choice(_ANDROID_PHONES) if has_phone else ""
            watch_model = r.choice(_ANDROID_WATCHES) if has_watch else ""
            earbuds_model = r.choice(_ANDROID_EARBUDS) if has_earbuds else ""

        # Generate WiFi probe list (SSIDs this phone has connected to before)
        probe_count = r.randint(2, 6)
        probes: list[str] = []
        # Always include 0-2 public SSIDs
        pub_count = r.randint(0, min(2, probe_count))
        probes.extend(r.sample(_PUBLIC_SSIDS, min(pub_count, len(_PUBLIC_SSIDS))))
        # Fill rest with home/personal SSIDs
        while len(probes) < probe_count:
            pattern = r.choice(_COMMON_SSID_PATTERNS)
            probes.append(_generate_ssid(pattern, r))

        return PersonRFProfile(
            has_phone=has_phone,
            phone_mac=_random_mac(r) if has_phone else "",
            phone_model=phone_model,
            phone_ble_company_id=company_id,
            phone_wifi_probes=probes if has_phone else [],
            phone_ecosystem=ecosystem,
            has_smartwatch=has_watch,
            watch_mac=_random_mac(r) if has_watch else "",
            watch_model=watch_model,
            watch_ble_company_id=company_id,
            has_earbuds=has_earbuds,
            earbuds_mac=_random_mac(r) if has_earbuds else "",
            earbuds_model=earbuds_model,
            earbuds_ble_company_id=company_id,
            _original_company_id=company_id,
        )

    @staticmethod
    def random_vehicle(
        rng: random.Random | None = None,
    ) -> VehicleRFProfile:
        """Generate a realistic VehicleRFProfile.

        All vehicles have TPMS (mandatory in US since 2007).
        80% have keyfob BLE, 20% have dashcam WiFi.
        """
        r = rng or random

        make, models = r.choice(_VEHICLE_MAKES)
        model = r.choice(models)
        year = r.randint(2010, 2026)
        color = r.choice(_VEHICLE_COLORS)

        # TPMS frequency: 315 MHz in US, 433.92 in EU
        tpms_freq = r.choice([315.0, 315.0, 315.0, 433.92])  # 75% US

        # Keyfob manufacturer company ID
        keyfob_companies = {
            "Tesla": 0x0000,  # Tesla uses their own BLE
            "BMW": 0x0006,
            "Mercedes-Benz": 0x0006,
        }
        keyfob_cid = keyfob_companies.get(make, r.randint(0x0100, 0x0FFF))

        # Dashcam SSID naming patterns
        dashcam_ssids = [
            f"VIOFO-A229-{r.randint(1000, 9999)}",
            f"BlackVue-{r.randint(100000, 999999)}",
            f"Garmin-Dash-{r.randint(1000, 9999)}",
            f"Nextbase-{r.randint(10000, 99999)}",
            f"REXING-{r.randint(1000, 9999)}",
        ]

        return VehicleRFProfile(
            tpms_ids=[_random_tpms_id(r) for _ in range(4)],
            tpms_frequency=tpms_freq,
            has_keyfob=r.random() < 0.80,
            keyfob_mac=_random_mac(r),
            keyfob_ble_company_id=keyfob_cid,
            has_dashcam_wifi=r.random() < 0.20,
            dashcam_ssid=r.choice(dashcam_ssids),
            dashcam_mac=_random_mac(r),
            license_plate=_random_plate("CA", r),
            make_model=f"{year} {make} {model}",
            year=year,
            color=color,
        )

    @staticmethod
    def random_building(
        building_type: str = "residential",
        rng: random.Random | None = None,
    ) -> BuildingRFProfile:
        """Generate a realistic BuildingRFProfile.

        Residential: 1-2 APs, 1-5 IoT devices
        Commercial: 2-4 APs, 3-8 IoT devices
        Industrial: 1-3 APs, 1-3 IoT devices
        """
        r = rng or random

        # WiFi AP count by building type
        ap_counts = {
            "residential": (1, 2),
            "commercial": (2, 4),
            "industrial": (1, 3),
        }
        min_ap, max_ap = ap_counts.get(building_type, (1, 2))
        ap_count = r.randint(min_ap, max_ap)

        # IoT device count
        iot_counts = {
            "residential": (1, 5),
            "commercial": (3, 8),
            "industrial": (1, 3),
        }
        min_iot, max_iot = iot_counts.get(building_type, (1, 3))
        iot_count = r.randint(min_iot, max_iot)

        # Generate APs
        ssid_patterns = (
            _COMMERCIAL_SSID_PATTERNS
            if building_type == "commercial"
            else _RESIDENTIAL_SSID_PATTERNS
        )
        wifi_aps: list[dict] = []
        channels_2g = [1, 6, 11]
        channels_5g = [36, 40, 44, 48, 149, 153, 157, 161]
        for i in range(ap_count):
            is_5g = i > 0 and r.random() < 0.6  # first AP always 2.4 GHz
            channel = r.choice(channels_5g if is_5g else channels_2g)
            ssid = _generate_ssid(r.choice(ssid_patterns), r)
            if is_5g and not ssid.endswith("_5G"):
                ssid += "_5G"
            wifi_aps.append({
                "ssid": ssid,
                "bssid": _random_mac(r),
                "channel": channel,
                "signal_strength": r.randint(-65, -35),
                "encryption": r.choice(["WPA2", "WPA3", "WPA2"]),
            })

        # Generate IoT devices
        iot_devices: list[dict] = []
        available_types = list(_IOT_DEVICE_TYPES)
        r.shuffle(available_types)
        for i in range(min(iot_count, len(available_types))):
            dev_type, protocol = available_types[i]
            iot_devices.append({
                "type": dev_type,
                "mac": _random_mac(r),
                "protocol": protocol,
                "name": f"{dev_type.replace('_', ' ').title()}",
            })

        return BuildingRFProfile(
            building_type=building_type,
            wifi_aps=wifi_aps,
            iot_devices=iot_devices,
        )
