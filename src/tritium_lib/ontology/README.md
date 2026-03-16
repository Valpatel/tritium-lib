# Ontology

**Where you are:** `tritium-lib/src/tritium_lib/ontology/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

The ontology defines the semantic type system for all entities and relationships in Tritium. It provides typed definitions for entity types (device, person, vehicle, location, network), relationship types (carries, detected_with, traveled_with), interfaces, and properties. The OntologyRegistry provides runtime lookup, validation, and schema enforcement.

This is the semantic backbone — every object that flows through MQTT, gets stored in SQLite, or appears on a dashboard is an instance of an ontology entity type.

## Key Files

| File | Purpose |
|------|---------|
| `schema.py` | Type definitions — EntityType, RelationshipType, InterfaceDef, PropertyDef, DataType, Cardinality, and the built-in TRITIUM_ONTOLOGY |
| `registry.py` | OntologyRegistry — runtime type lookup, validation, schema loading from OntologySchema |

## Usage

```python
from tritium_lib.ontology.schema import TRITIUM_ONTOLOGY
from tritium_lib.ontology.registry import OntologyRegistry

registry = OntologyRegistry()
registry.load_schema(TRITIUM_ONTOLOGY)

device_type = registry.get_entity_type("device")
carries_rel = registry.get_relationship_type("carries")
```

## Related

- [../graph/](../graph/) — KuzuDB graph store that persists ontology instances
- [../models/](../models/) — Pydantic models that implement ontology entity types
- [../classifier/](../classifier/) — DeviceClassifier that assigns ontology types to detected devices
