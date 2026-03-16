# Graph Store

**Where you are:** `tritium-lib/src/tritium_lib/graph/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

Embedded graph database layer using KuzuDB for modeling entities and relationships discovered by the Tritium sensor network. Stores the ontology as a property graph: nodes are targets (Person, Device, Vehicle, Location, Network, Camera, MeshNode, Zone) and edges are relationships (CARRIES, DETECTED_WITH, TRAVELED_WITH, CONNECTED_TO, etc.).

This is the persistence layer for the graph ontology vision — every entity gets a node, every observed relationship gets an edge, building a knowledge graph over time.

## Key Files

| File | Purpose |
|------|---------|
| `store.py` | TritiumGraph — KuzuDB wrapper with schema creation, node/edge CRUD, and Cypher queries |

## Dependencies

Requires `kuzu` Python package. Install with:
```bash
pip install 'tritium-lib[graph]'
```

## Usage

```python
from tritium_lib.graph.store import TritiumGraph

graph = TritiumGraph("/path/to/graph.db")
graph.add_node("Device", {"id": "ble_aa:bb:cc", "type": "phone", "name": "iPhone"})
graph.add_edge("CARRIES", "person_001", "ble_aa:bb:cc")
results = graph.query("MATCH (p:Person)-[:CARRIES]->(d:Device) RETURN p, d")
```

## Related

- [../ontology/](../ontology/) — Schema and registry that define valid entity/relationship types
- [../models/](../models/) — Pydantic models for the entities stored in the graph
- [../store/](../store/) — Other persistence stores (BLE sightings, targets)
- [../../../../tritium-sc/src/engine/tactical/](../../../../tritium-sc/src/engine/tactical/) — Tactical engine that queries the graph for intelligence
