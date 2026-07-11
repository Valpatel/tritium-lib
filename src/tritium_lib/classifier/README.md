# Device Classifier

**Where you are:** `tritium-lib/src/tritium_lib/classifier/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

Multi-signal BLE and WiFi device type classifier. Combines all available signals (MAC OUI, device name, GAP appearance, service UUIDs, company ID, Apple continuity data, Google Fast Pair model ID, WiFi SSID patterns) to produce the best possible device type classification with confidence scores.

Each signal contributes a (device_type, confidence) vote. The final classification picks the highest-confidence vote, with ties broken by signal priority: appearance > service_uuid > company_id > name_pattern > oui.

## Key Files

| File | Purpose |
|------|---------|
| `device_classifier.py` | DeviceClassifier — multi-signal classification engine for BLE and WiFi devices |

## Usage

```python
from tritium_lib.classifier import DeviceClassifier

dc = DeviceClassifier()

# BLE classification
result = dc.classify_ble(mac="AC:BC:32:AA:BB:CC", name="iPhone 15")
print(result.device_type)   # "phone"
print(result.confidence)    # 0.9
print(result.manufacturer)  # "Apple"

# WiFi classification
wifi = dc.classify_wifi(ssid="DIRECT-HP-Printer", bssid="00:17:88:AA:BB:CC")
print(wifi.device_type)     # "printer"
```

## Fingerprint Databases

The classifier loads JSON lookup tables from the sibling `data/` directory:

| Database | Contents |
|----------|----------|
| `ble_fingerprints.json` | Known BLE device fingerprints |
| `ble_appearance_values.json` | BLE GAP appearance code to device type mapping |
| `ble_service_uuids.json` | BLE service UUID to device type mapping |
| `ble_company_ids.json` | BLE company ID to manufacturer mapping |
| `ble_name_patterns.json` | Regex patterns for BLE device name matching |
| `apple_continuity_types.json` | Apple continuity protocol type IDs |
| `oui_device_types.json` | OUI prefix to device type mapping |
| `wifi_ssid_patterns.json` | WiFi SSID patterns for network classification |
| `wifi_vendor_fingerprints.json` | WiFi vendor fingerprints |
| `device_classification_rules.json` | Composite classification rules |

## Consumed by (dated 2026-07-11, grep `from tritium_lib.classifier`)

- **tritium-sc (the app): 2 sites** — `src/engine/tactical/enrichment.py`
  (enriches tracked targets with a device type) and
  `plugins/edge_tracker/plugin.py` (classifies edge-reported sightings).
- **lib-internal: 1 site** — a sibling package re-uses it.
- **tests: 2 sites**.

## Related

- [../data/](../data/) — JSON lookup tables loaded by the classifier
- [../ontology/](../ontology/) — ontology types that classification results map to
- [../tracking/ble_classifier.py](../tracking/ble_classifier.py) — the tracking-side
  `BLEClassifier` (known/unknown/new/suspicious levels) — a coarser, threat-level
  classifier, distinct from this fine-grained device-type engine
