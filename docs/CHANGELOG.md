# Tritium-Lib Changelog

Changes tracked with verification status. All changes on `dev` branch.

## Verification Levels

| Level | Meaning |
|-------|---------|
| **Unit Tested** | Passes `pytest tests/` |
| **Consumer Tested** | Verified working in tritium-edge or tritium-sc imports |
| **Human Verified** | Manually reviewed by a human |

---

## 2026-03-13 — Wave 9: Graph Database & Ontology Schema

### KuzuDB Graph Store
| Change | Verification |
|--------|-------------|
| `graph/kuzu_store.py` — KuzuDB embedded graph database for ontology layer | Unit Tested |
| Node CRUD: create, read, update, delete typed entities | Unit Tested |
| Edge CRUD: typed relationships between entities (CARRIES, DETECTED_WITH, etc.) | Unit Tested |
| Cypher query interface for traversal and pattern matching | Unit Tested |

### Ontology Schema & Registry
| Change | Verification |
|--------|-------------|
| `ontology/schema.py` — formal ontology: 10 entity types, 12 relationships, 3 interfaces | Unit Tested |
| Entity types: Person, Device, Vehicle, Location, Network, Organization, Event, Alert, Asset, Zone | Unit Tested |
| Relationship types: OWNS, CARRIES, DETECTED_AT, CONNECTED_TO, MEMBER_OF, etc. | Unit Tested |
| Schema validation and type-safe entity/relationship construction | Unit Tested |

### DossierStore Enhancements
| Change | Verification |
|--------|-------------|
| `_update_json_field` helper for atomic tag/note updates in DossierStore | Unit Tested |

---

## 2026-03-13 — Wave 7: Dossiers & Target Intelligence

### Models — Dossier
| Change | Verification |
|--------|-------------|
| `models/dossier.py` — Target Dossier model for persistent entity intelligence | Unit Tested |

### Stores — DossierStore
| Change | Verification |
|--------|-------------|
| `store/dossier.py` — SQLite-backed DossierStore for persistent target intelligence | Unit Tested |

---

## 2026-03-13

### Models — New
| Change | Verification |
|--------|-------------|
| `models/meshtastic.py` — MeshtasticNode, MeshtasticMessage, MeshtasticWaypoint, MeshtasticStatus | Unit Tested |
| `models/camera.py` — CameraSource, CameraFrame, CameraDetection, BoundingBox | Unit Tested |
| All models exported from `models/__init__.py` | Unit Tested |

### MQTT Topics — New
| Change | Verification |
|--------|-------------|
| `meshtastic_nodes()`, `meshtastic_message()`, `meshtastic_command()` | Unit Tested |
| `camera_feed()`, `camera_snapshot()` | Unit Tested |
| `all_meshtastic()` wildcard subscription | Unit Tested |

### Infrastructure
| Change | Verification |
|--------|-------------|
| `testing/__init__.py` — lazy imports for cv2/numpy/requests deps | Unit Tested |
| `pyproject.toml` — `[testing]` optional dep group | Unit Tested |

### Documentation
| Change | Verification |
|--------|-------------|
| `CLAUDE.md` — submodule context with polyglot vision | N/A (docs) |
| `src/tritium_lib/README.md` — package reference with model categories | N/A (docs) |
| `README.md` — updated model table | N/A (docs) |
| `LICENSE` — AGPL-3.0 added | N/A (legal) |

---

## Test Baseline

| Suite | Count | Status | Date |
|-------|-------|--------|------|
| pytest tests/ | 833 | All passing | 2026-03-13 |
| Meshtastic model tests | 27 | All passing | 2026-03-13 |
| Camera model tests | 27 | All passing | 2026-03-13 |
| MQTT topic tests | 29 | All passing | 2026-03-13 |
