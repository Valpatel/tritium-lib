# Fingerprint Data

**Where you are:** `tritium-lib/src/tritium_lib/data/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

JSON lookup tables used by the DeviceClassifier for BLE and WiFi device identification. These databases map hardware identifiers (MAC OUI prefixes, BLE service UUIDs, GAP appearance codes, company IDs) and software signatures (device name patterns, SSID patterns, vendor fingerprints) to device types and manufacturers.

## Key Files

| File | Purpose |
|------|---------|
| `ble_fingerprints.json` | Known BLE device fingerprints — MAC + name + type combinations |
| `ble_appearance_values.json` | BLE GAP appearance code to device type (phone, watch, keyboard, etc.) |
| `ble_service_uuids.json` | BLE service UUID to device type mapping |
| `ble_company_ids.json` | Bluetooth SIG company ID to manufacturer name |
| `ble_name_patterns.json` | Regex patterns for matching BLE advertised names |
| `apple_continuity_types.json` | Apple continuity protocol message type IDs |
| `oui_device_types.json` | IEEE OUI (first 3 bytes of MAC) to device type heuristics |
| `wifi_ssid_patterns.json` | WiFi SSID regex patterns for network classification (corporate, hotspot, IoT) |
| `wifi_vendor_fingerprints.json` | WiFi vendor-specific fingerprints |
| `device_classification_rules.json` | Composite multi-signal classification rules |

## Related

- [../classifier/](../classifier/) — DeviceClassifier that loads and queries these tables
- [../models/ble.py](../models/ble.py) — BLE data models (BLESighting, BLEDevice)
