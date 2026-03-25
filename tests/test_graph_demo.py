# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the graph ontology demo.

Covers: demo data population, query helpers, HTML visualization,
REST endpoints, and ontology summary.
"""

import json
import os
import shutil
import tempfile
from io import BytesIO
from http.server import HTTPServer
from unittest.mock import patch

import pytest

try:
    import kuzu
    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

pytestmark = pytest.mark.skipif(not HAS_KUZU, reason="kuzu not installed")


@pytest.fixture
def demo_graph():
    """Create a populated demo graph in a temp directory."""
    from tritium_lib.graph.store import TritiumGraph
    from tritium_lib.graph.demos.graph_demo import populate_demo_graph

    tmpdir = tempfile.mkdtemp(prefix="tritium_graph_demo_test_")
    db_path = os.path.join(tmpdir, "test_demo.db")
    g = TritiumGraph(db_path)
    populate_demo_graph(g)
    yield g
    g.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def empty_graph():
    """Create an empty graph in a temp directory."""
    from tritium_lib.graph.store import TritiumGraph

    tmpdir = tempfile.mkdtemp(prefix="tritium_graph_demo_test_empty_")
    db_path = os.path.join(tmpdir, "test_empty.db")
    g = TritiumGraph(db_path)
    yield g
    g.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test: Demo Data Population ────────────────────────────────────────


class TestPopulateDemoGraph:
    def test_creates_people(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        people = [e for e in entities if e["entity_type"] == "Person"]
        assert len(people) == 4
        names = {p["name"] for p in people}
        assert "Alice Chen" in names
        assert "Bob Martinez" in names
        assert "Carol Davis" in names
        assert "Dave Kim" in names

    def test_creates_devices(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        devices = [e for e in entities if e["entity_type"] == "Device"]
        assert len(devices) == 5

    def test_creates_vehicles(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        vehicles = [e for e in entities if e["entity_type"] == "Vehicle"]
        assert len(vehicles) == 2

    def test_creates_locations(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        locations = [e for e in entities if e["entity_type"] == "Location"]
        assert len(locations) == 3

    def test_creates_networks(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        networks = [e for e in entities if e["entity_type"] == "Network"]
        assert len(networks) == 2

    def test_creates_cameras(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        cameras = [e for e in entities if e["entity_type"] == "Camera"]
        assert len(cameras) == 2

    def test_total_entity_count(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(demo_graph)
        # 4 people + 5 devices + 2 vehicles + 3 locations + 2 networks + 2 cameras = 18
        assert len(entities) == 18

    def test_creates_relationships(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_relationships

        rels = get_all_relationships(demo_graph)
        assert len(rels) >= 15  # Many relationships created

    def test_relationship_types_present(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_relationships

        rels = get_all_relationships(demo_graph)
        rel_types = {r["rel_type"] for r in rels}
        assert "CARRIES" in rel_types
        assert "DETECTED_WITH" in rel_types
        assert "OBSERVED_AT" in rel_types
        assert "CONNECTED_TO" in rel_types
        assert "TRAVELED_WITH" in rel_types
        assert "CORRELATED_WITH" in rel_types
        assert "DETECTED_BY" in rel_types


# ── Test: Query Helpers ───────────────────────────────────────────────


class TestQueryHelpers:
    def test_find_devices_carried_by_carol(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import find_devices_carried_by

        devices = find_devices_carried_by(demo_graph, "person-carol")
        assert len(devices) == 2
        names = {d["name"] for d in devices}
        assert "Carol's Laptop" in names
        assert "Burner Phone" in names

    def test_find_devices_carried_by_alice(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import find_devices_carried_by

        devices = find_devices_carried_by(demo_graph, "person-alice")
        assert len(devices) == 1
        assert devices[0]["name"] == "Alice's iPhone"

    def test_find_devices_carried_by_unknown(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import find_devices_carried_by

        devices = find_devices_carried_by(demo_graph, "person-nobody")
        assert devices == []

    def test_find_traveled_together(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import find_traveled_together

        pairs = find_traveled_together(demo_graph)
        assert len(pairs) >= 1
        pair = pairs[0]
        names = {pair["entity_a"]["name"], pair["entity_b"]["name"]}
        assert "Carol Davis" in names
        assert "Dave Kim" in names
        assert pair["count"] == 3

    def test_traverse_carol_subgraph(self, demo_graph):
        subgraph = demo_graph.traverse("person-carol", max_hops=2)
        node_ids = {n["id"] for n in subgraph["nodes"]}
        # Carol herself
        assert "person-carol" in node_ids
        # Her devices
        assert "ble-aa:bb:cc:03" in node_ids
        assert "ble-dd:ee:ff:04" in node_ids
        # Locations she visited
        assert "loc-warehouse" in node_ids
        assert "loc-cafe" in node_ids
        # Dave traveled with her
        assert "person-dave" in node_ids
        assert len(subgraph["edges"]) >= 4

    def test_search_alice(self, demo_graph):
        results = demo_graph.search("Alice")
        names = {r["name"] for r in results}
        assert "Alice Chen" in names
        assert "Alice's iPhone" in names

    def test_search_no_results(self, demo_graph):
        results = demo_graph.search("zzzzzzz")
        assert results == []


# ── Test: HTML Visualization ──────────────────────────────────────────


class TestVisualization:
    def test_generates_valid_html(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import (
            get_all_entities, get_all_relationships,
            generate_visualization_html,
        )

        entities = get_all_entities(demo_graph)
        rels = get_all_relationships(demo_graph)
        page = generate_visualization_html(entities, rels)

        assert "<!DOCTYPE html>" in page
        assert "Tritium Graph" in page
        assert "<svg" in page
        assert "force-directed" in page.lower() or "simulate" in page.lower()

    def test_html_contains_entity_data(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import (
            get_all_entities, get_all_relationships,
            generate_visualization_html,
        )

        entities = get_all_entities(demo_graph)
        rels = get_all_relationships(demo_graph)
        page = generate_visualization_html(entities, rels)

        # Entity IDs should be embedded as JSON data
        assert "person-alice" in page
        assert "person-carol" in page
        assert "CARRIES" in page

    def test_html_contains_legend(self, demo_graph):
        from tritium_lib.graph.demos.graph_demo import (
            get_all_entities, get_all_relationships,
            generate_visualization_html,
        )

        entities = get_all_entities(demo_graph)
        rels = get_all_relationships(demo_graph)
        page = generate_visualization_html(entities, rels)

        assert "legend" in page.lower()

    def test_html_with_empty_data(self):
        from tritium_lib.graph.demos.graph_demo import generate_visualization_html

        page = generate_visualization_html([], [])
        assert "<!DOCTYPE html>" in page
        assert "Entities: <span" in page


# ── Test: Ontology Summary ────────────────────────────────────────────


class TestOntologySummary:
    def test_returns_version(self):
        from tritium_lib.graph.demos.graph_demo import get_ontology_summary

        summary = get_ontology_summary()
        assert summary["version"] == "1.0.0"

    def test_entity_types_listed(self):
        from tritium_lib.graph.demos.graph_demo import get_ontology_summary

        summary = get_ontology_summary()
        names = {et["api_name"] for et in summary["entity_types"]}
        assert "device" in names
        assert "track" in names
        assert "zone" in names

    def test_relationship_types_listed(self):
        from tritium_lib.graph.demos.graph_demo import get_ontology_summary

        summary = get_ontology_summary()
        names = {rt["api_name"] for rt in summary["relationship_types"]}
        assert "detected_by" in names
        assert "connected_to" in names

    def test_interfaces_listed(self):
        from tritium_lib.graph.demos.graph_demo import get_ontology_summary

        summary = get_ontology_summary()
        names = {i["api_name"] for i in summary["interfaces"]}
        assert "trackable" in names
        assert "identifiable" in names
        assert "classifiable" in names


# ── Test: REST API Endpoints ──────────────────────────────────────────


class TestRESTEndpoints:
    """Test the HTTP handler methods directly without starting a server."""

    def _make_handler(self, demo_graph, method, path, body=None):
        """Create a mock handler and invoke the appropriate method."""
        from tritium_lib.graph.demos.graph_demo import GraphDemoHandler

        # Set the class-level graph
        GraphDemoHandler.graph = demo_graph

        class FakeWFile:
            def __init__(self):
                self.data = BytesIO()

            def write(self, b):
                self.data.write(b)

        class FakeRFile:
            def __init__(self, content=b""):
                self._content = content

            def read(self, n):
                return self._content[:n]

        handler = GraphDemoHandler.__new__(GraphDemoHandler)
        handler.path = path
        handler.command = method
        handler.wfile = FakeWFile()
        handler.headers = {"Content-Length": str(len(body)) if body else "0"}
        handler.rfile = FakeRFile(body.encode() if body else b"")

        # Capture response
        handler._response_code = None
        handler._response_headers = {}

        def mock_send_response(code):
            handler._response_code = code

        def mock_send_header(key, val):
            handler._response_headers[key] = val

        def mock_end_headers():
            pass

        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers

        if method == "GET":
            handler.do_GET()
        elif method == "POST":
            handler.do_POST()

        return handler

    def test_get_entities(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/entities")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        assert len(data) == 18

    def test_get_relationships(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/relationships")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        assert len(data) >= 15

    def test_get_traverse(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/traverse/person-carol?hops=2")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        assert "nodes" in data
        assert "edges" in data
        node_ids = {n["id"] for n in data["nodes"]}
        assert "person-carol" in node_ids

    def test_get_search(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/search?q=Alice")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        names = {r["name"] for r in data}
        assert "Alice Chen" in names

    def test_get_search_empty(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/search?q=")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        assert data == []

    def test_get_ontology(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/ontology")
        assert handler._response_code == 200
        data = json.loads(handler.wfile.data.getvalue())
        assert data["version"] == "1.0.0"

    def test_get_visualization(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/")
        assert handler._response_code == 200
        html_content = handler.wfile.data.getvalue().decode("utf-8")
        assert "<!DOCTYPE html>" in html_content

    def test_post_entity(self, demo_graph):
        body = json.dumps({
            "entity_type": "Person",
            "id": "person-new",
            "name": "New Person",
            "properties": {"role": "observer"},
        })
        handler = self._make_handler(demo_graph, "POST", "/api/entity", body)
        assert handler._response_code == 201

        # Verify entity was created
        entity = demo_graph.get_entity("person-new")
        assert entity is not None
        assert entity["name"] == "New Person"

    def test_post_entity_invalid_type(self, demo_graph):
        body = json.dumps({
            "entity_type": "Alien",
            "id": "alien-1",
        })
        handler = self._make_handler(demo_graph, "POST", "/api/entity", body)
        assert handler._response_code == 400

    def test_post_entity_missing_fields(self, demo_graph):
        body = json.dumps({"name": "No type or ID"})
        handler = self._make_handler(demo_graph, "POST", "/api/entity", body)
        assert handler._response_code == 400

    def test_post_relationship(self, demo_graph):
        body = json.dumps({
            "from_id": "person-alice",
            "to_id": "person-bob",
            "rel_type": "TRAVELED_WITH",
            "properties": {"confidence": 0.9},
        })
        handler = self._make_handler(demo_graph, "POST", "/api/relationship", body)
        assert handler._response_code == 201

    def test_post_relationship_invalid_type(self, demo_graph):
        body = json.dumps({
            "from_id": "person-alice",
            "to_id": "person-bob",
            "rel_type": "FAKE",
        })
        handler = self._make_handler(demo_graph, "POST", "/api/relationship", body)
        assert handler._response_code == 400

    def test_post_relationship_missing_fields(self, demo_graph):
        body = json.dumps({"from_id": "person-alice"})
        handler = self._make_handler(demo_graph, "POST", "/api/relationship", body)
        assert handler._response_code == 400

    def test_get_404(self, demo_graph):
        handler = self._make_handler(demo_graph, "GET", "/api/nonexistent")
        assert handler._response_code == 404

    def test_post_invalid_json(self, demo_graph):
        handler = self._make_handler(demo_graph, "POST", "/api/entity", "not json{{{")
        assert handler._response_code == 400


# ── Test: Empty Graph ─────────────────────────────────────────────────


class TestEmptyGraph:
    def test_get_all_entities_empty(self, empty_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_entities

        entities = get_all_entities(empty_graph)
        assert entities == []

    def test_get_all_relationships_empty(self, empty_graph):
        from tritium_lib.graph.demos.graph_demo import get_all_relationships

        rels = get_all_relationships(empty_graph)
        assert rels == []

    def test_find_devices_carried_by_empty(self, empty_graph):
        from tritium_lib.graph.demos.graph_demo import find_devices_carried_by

        devices = find_devices_carried_by(empty_graph, "nobody")
        assert devices == []

    def test_find_traveled_together_empty(self, empty_graph):
        from tritium_lib.graph.demos.graph_demo import find_traveled_together

        pairs = find_traveled_together(empty_graph)
        assert pairs == []
