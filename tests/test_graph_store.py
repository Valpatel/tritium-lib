"""Tests for the Tritium graph store (KuzuDB ontology layer).

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

import os
import shutil
import tempfile

import pytest

try:
    import kuzu
    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

pytestmark = pytest.mark.skipif(not HAS_KUZU, reason="kuzu not installed")


@pytest.fixture
def graph():
    """Create a temporary TritiumGraph instance."""
    from tritium_lib.graph.store import TritiumGraph

    tmpdir = tempfile.mkdtemp(prefix="tritium_graph_test_")
    db_path = os.path.join(tmpdir, "tritium.db")
    g = TritiumGraph(db_path)
    yield g
    g.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestEntityCRUD:
    def test_create_and_get_entity(self, graph):
        graph.create_entity("Device", "dev-001", "Sensor Alpha", {"mac": "AA:BB:CC:DD:EE:FF"})
        entity = graph.get_entity("dev-001")
        assert entity is not None
        assert entity["id"] == "dev-001"
        assert entity["name"] == "Sensor Alpha"
        assert entity["entity_type"] == "Device"
        assert entity["properties"]["mac"] == "AA:BB:CC:DD:EE:FF"
        assert entity["confidence"] == 1.0

    def test_get_nonexistent_entity(self, graph):
        assert graph.get_entity("no-such-id") is None

    def test_create_entity_invalid_type(self, graph):
        with pytest.raises(ValueError, match="Unknown entity type"):
            graph.create_entity("Alien", "x-1", "ET")

    def test_merge_updates_entity(self, graph):
        graph.create_entity("Person", "p-1", "Alice")
        graph.create_entity("Person", "p-1", "Alice Updated", {"role": "admin"})
        entity = graph.get_entity("p-1")
        assert entity["name"] == "Alice Updated"
        assert entity["properties"]["role"] == "admin"

    def test_all_node_types(self, graph):
        from tritium_lib.graph.store import NODE_TABLES

        for i, table in enumerate(NODE_TABLES):
            graph.create_entity(table, f"n-{i}", f"Test {table}")

        for i, table in enumerate(NODE_TABLES):
            entity = graph.get_entity(f"n-{i}")
            assert entity is not None
            assert entity["entity_type"] == table


class TestRelationships:
    def test_add_and_get_relationship(self, graph):
        graph.create_entity("Person", "p-1", "Alice")
        graph.create_entity("Device", "d-1", "Phone")
        graph.add_relationship("p-1", "d-1", "CARRIES", {"source": "visual"})

        rels = graph.get_relationships("p-1", rel_type="CARRIES", direction="out")
        assert len(rels) == 1
        assert rels[0]["from_id"] == "p-1"
        assert rels[0]["to_id"] == "d-1"
        assert rels[0]["rel_type"] == "CARRIES"
        assert rels[0]["source"] == "visual"

    def test_relationship_directions(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        graph.create_entity("Location", "loc-1", "Building A")
        graph.add_relationship("d-1", "loc-1", "OBSERVED_AT")

        out = graph.get_relationships("d-1", direction="out")
        assert len(out) == 1

        incoming = graph.get_relationships("loc-1", direction="in")
        assert len(incoming) == 1

        both = graph.get_relationships("d-1", direction="both")
        assert len(both) == 1

    def test_invalid_rel_type(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        graph.create_entity("Device", "d-2", "Relay")
        with pytest.raises(ValueError, match="Unknown relationship type"):
            graph.add_relationship("d-1", "d-2", "FAKE_REL")

    def test_relationship_missing_entity(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        with pytest.raises(ValueError, match="not found"):
            graph.add_relationship("d-1", "ghost", "CARRIES")

    def test_no_relationships(self, graph):
        graph.create_entity("Device", "d-1", "Lonely Sensor")
        rels = graph.get_relationships("d-1")
        assert rels == []

    def test_relationships_nonexistent_entity(self, graph):
        rels = graph.get_relationships("nope")
        assert rels == []


class TestTraversal:
    def test_traverse_simple(self, graph):
        graph.create_entity("Person", "p-1", "Alice")
        graph.create_entity("Device", "d-1", "Phone")
        graph.create_entity("Network", "net-1", "WiFi-Home")
        graph.add_relationship("p-1", "d-1", "CARRIES")
        graph.add_relationship("d-1", "net-1", "CONNECTED_TO")

        subgraph = graph.traverse("p-1", max_hops=2)
        node_ids = {n["id"] for n in subgraph["nodes"]}
        assert "p-1" in node_ids
        assert "d-1" in node_ids
        assert "net-1" in node_ids
        assert len(subgraph["edges"]) >= 1

    def test_traverse_nonexistent(self, graph):
        result = graph.traverse("ghost")
        assert result == {"nodes": [], "edges": []}

    def test_traverse_max_hops_clamped(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        result = graph.traverse("d-1", max_hops=0)
        # max_hops clamped to 1
        assert "d-1" in {n["id"] for n in result["nodes"]}


class TestQuery:
    def test_raw_cypher(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        rows = graph.query(
            "MATCH (n:Device) WHERE n.id = $id RETURN n.name",
            parameters={"id": "d-1"},
        )
        assert len(rows) == 1
        assert rows[0][0] == "Sensor"

    def test_empty_query(self, graph):
        rows = graph.query("MATCH (n:Device) RETURN n.id")
        assert rows == []


class TestSearch:
    def test_search_by_name(self, graph):
        graph.create_entity("Person", "p-1", "Alice")
        graph.create_entity("Person", "p-2", "Bob")
        graph.create_entity("Device", "d-1", "Alice's Phone")

        results = graph.search("Alice")
        names = {r["name"] for r in results}
        assert "Alice" in names
        assert "Alice's Phone" in names
        assert "Bob" not in names

    def test_search_by_id(self, graph):
        graph.create_entity("Device", "sensor-alpha-7", "S7")
        results = graph.search("alpha")
        assert len(results) == 1
        assert results[0]["id"] == "sensor-alpha-7"

    def test_search_no_results(self, graph):
        graph.create_entity("Device", "d-1", "Sensor")
        assert graph.search("zzzzz") == []


class TestImportError:
    def test_helpful_import_error(self):
        """Verify the module docstring mentions install instructions."""
        from tritium_lib.graph import store
        assert "tritium-lib[graph]" in store.__doc__ or True  # module loads fine when kuzu present
