# Ontology

**Where you are:** `tritium-lib/src/tritium_lib/ontology/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

The ontology package defines a typed VIEW of Tritium's semantic layer:
10 entity types (`device`, `mesh_node`, `edge_unit`, `sighting`,
`track`, `chat_message`, `command`, `classification`, `investigation`,
`zone`), 12 relationship types (`detected_by`, `sighting_of`,
`part_of_track`, `connected_to`, `mesh_peer`, `managed_by`,
`located_at`, `within_zone`, `classified_as`, `related_to`,
`subject_of`, `sent_by`), and 3 interfaces (`trackable`,
`identifiable`, `classifiable`). The OntologyRegistry provides runtime
lookup and validation.

**Honesty note (2026-06-11, ontology study):** the CANONICAL semantic
layer is `tritium_lib/models/` (the Pydantic contracts every consumer
imports) — this package is a typed view that is not yet derived from
it, and `OntologyRegistry.validate_entity` has no production callers
today. The live `/api/v1/ontology` router carries its own third
vocabulary. Reconciliation (derive-vs-retire) is charter-proposed in
`docs/QUESTIONS.md`; see
`docs/research/ontology-principles-palantir.md` §2.1. This README
previously advertised entity types (`person`, `vehicle`) and
relationships (`carries`) that never existed in `schema.py`.

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
related_rel = registry.get_relationship_type("related_to")
```

## Related

- [../graph/](../graph/) — KuzuDB graph store that persists ontology instances
- [../models/](../models/) — Pydantic models that implement ontology entity types
- [../classifier/](../classifier/) — DeviceClassifier that assigns ontology types to detected devices
