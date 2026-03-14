"""
TritiumGraph — Embedded graph database for the Tritium ontology layer.

Uses KuzuDB (embedded, single-file, Cypher queries) to model entities
and relationships discovered by the Tritium sensor network.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import kuzu
except ImportError:
    raise ImportError(
        "KuzuDB is required for the graph store. "
        "Install it with: pip install 'tritium-lib[graph]'"
    )


# ── Schema constants ────────────────────────────────────────────────

NODE_TABLES: list[str] = [
    "Person",
    "Device",
    "Vehicle",
    "Location",
    "Network",
    "Camera",
    "MeshNode",
    "Zone",
]

REL_TABLES: list[str] = [
    "CARRIES",
    "DETECTED_WITH",
    "OBSERVED_AT",
    "ENTERED",
    "EXITED",
    "DETECTED_BY",
    "PROBED_FOR",
    "CONNECTED_TO",
    "CORRELATED_WITH",
    "TRAVELED_WITH",
]


class TritiumGraph:
    """Embedded graph database wrapping KuzuDB for the Tritium ontology."""

    def __init__(self, db_path: str | Path) -> None:
        """Open or create a Tritium graph database.

        Args:
            db_path: Path for the KuzuDB database (file or directory).
                     Parent directories are created if needed.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    # ── Schema ───────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create node and relationship tables if they don't exist."""
        for table in NODE_TABLES:
            self._conn.execute(
                f"CREATE NODE TABLE IF NOT EXISTS {table}("
                f"id STRING, "
                f"name STRING, "
                f"entity_type STRING, "
                f"first_seen STRING, "
                f"last_seen STRING, "
                f"confidence DOUBLE, "
                f"properties STRING, "
                f"PRIMARY KEY(id))"
            )

        # Each rel table connects any node type to any node type.
        # Use REL TABLE GROUP so a single rel name spans all node combos.
        pairs = ", ".join(
            f"FROM {a} TO {b}" for a in NODE_TABLES for b in NODE_TABLES
        )
        for rel in REL_TABLES:
            self._conn.execute(
                f"CREATE REL TABLE GROUP IF NOT EXISTS {rel}("
                f"{pairs}, "
                f"timestamp STRING, "
                f"confidence DOUBLE, "
                f"source STRING, "
                f"count INT64)"
            )

    # ── Entity CRUD ──────────────────────────────────────────────────

    def create_entity(
        self,
        entity_type: str,
        id: str,
        name: str = "",
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Create or merge a node in the graph.

        Args:
            entity_type: One of the NODE_TABLES (Person, Device, etc.).
            id: Unique identifier for the entity.
            name: Human-readable name.
            properties: Arbitrary JSON-serializable properties.
            confidence: Confidence score 0.0–1.0.

        Raises:
            ValueError: If entity_type is not a valid node table.
        """
        if entity_type not in NODE_TABLES:
            raise ValueError(
                f"Unknown entity type '{entity_type}'. "
                f"Must be one of: {NODE_TABLES}"
            )

        now = _now_iso()
        props_json = json.dumps(properties or {})

        self._conn.execute(
            f"MERGE (n:{entity_type} {{id: $id}}) "
            f"ON CREATE SET n.name = $name, n.entity_type = $entity_type, "
            f"n.first_seen = $now, n.last_seen = $now, "
            f"n.confidence = $confidence, n.properties = $props "
            f"ON MATCH SET n.last_seen = $now, n.name = $name, "
            f"n.confidence = $confidence, n.properties = $props",
            parameters={
                "id": id,
                "name": name,
                "entity_type": entity_type,
                "now": now,
                "confidence": confidence,
                "props": props_json,
            },
        )

    def get_entity(self, id: str) -> dict[str, Any] | None:
        """Retrieve an entity by ID across all node tables.

        Returns:
            Dict with all properties, or None if not found.
        """
        for table in NODE_TABLES:
            result = self._conn.execute(
                f"MATCH (n:{table}) WHERE n.id = $id "
                f"RETURN n.id, n.name, n.entity_type, n.first_seen, "
                f"n.last_seen, n.confidence, n.properties",
                parameters={"id": id},
            )
            if result.has_next():
                row = result.get_next()
                return _row_to_entity(result.get_column_names(), row)
        return None

    # ── Relationships ────────────────────────────────────────────────

    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two existing entities.

        Args:
            from_id: Source entity ID.
            to_id: Target entity ID.
            rel_type: One of the REL_TABLES (CARRIES, DETECTED_WITH, etc.).
            properties: Optional dict with timestamp, confidence, source, count.

        Raises:
            ValueError: If rel_type is not valid or entities not found.
        """
        if rel_type not in REL_TABLES:
            raise ValueError(
                f"Unknown relationship type '{rel_type}'. "
                f"Must be one of: {REL_TABLES}"
            )

        props = properties or {}
        timestamp = props.get("timestamp", _now_iso())
        confidence = float(props.get("confidence", 1.0))
        source = props.get("source", "")
        count = int(props.get("count", 1))

        from_table = self._find_entity_table(from_id)
        to_table = self._find_entity_table(to_id)
        if from_table is None or to_table is None:
            missing = from_id if from_table is None else to_id
            raise ValueError(f"Entity '{missing}' not found in graph")

        self._conn.execute(
            f"MATCH (a:{from_table}), (b:{to_table}) "
            f"WHERE a.id = $from_id AND b.id = $to_id "
            f"CREATE (a)-[:{rel_type} {{"
            f"timestamp: $ts, confidence: $conf, "
            f"source: $src, count: $cnt}}]->(b)",
            parameters={
                "from_id": from_id,
                "to_id": to_id,
                "ts": timestamp,
                "conf": confidence,
                "src": source,
                "cnt": count,
            },
        )

    def get_relationships(
        self,
        id: str,
        rel_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get relationships for an entity.

        Args:
            id: Entity ID to query relationships for.
            rel_type: Optional filter by relationship type.
            direction: "out", "in", or "both".

        Returns:
            List of relationship dicts with from_id, to_id, rel_type, and properties.
        """
        table = self._find_entity_table(id)
        if table is None:
            return []

        rel_filter = f":{rel_type}" if rel_type else ""
        results: list[dict[str, Any]] = []

        if direction in ("out", "both"):
            r = self._conn.execute(
                f"MATCH (a:{table})-[r{rel_filter}]->(b) "
                f"WHERE a.id = $id "
                f"RETURN a.id, b.id, label(r), r.timestamp, "
                f"r.confidence, r.source, r.count",
                parameters={"id": id},
            )
            while r.has_next():
                row = r.get_next()
                results.append(_row_to_rel(row, direction="out"))

        if direction in ("in", "both"):
            r = self._conn.execute(
                f"MATCH (a)-[r{rel_filter}]->(b:{table}) "
                f"WHERE b.id = $id "
                f"RETURN a.id, b.id, label(r), r.timestamp, "
                f"r.confidence, r.source, r.count",
                parameters={"id": id},
            )
            while r.has_next():
                row = r.get_next()
                results.append(_row_to_rel(row, direction="in"))

        return results

    # ── Traversal ────────────────────────────────────────────────────

    def traverse(
        self, start_id: str, max_hops: int = 2
    ) -> dict[str, Any]:
        """Traverse the graph from a starting entity.

        Args:
            start_id: Entity ID to start traversal from.
            max_hops: Maximum number of hops (1–10).

        Returns:
            Subgraph dict with 'nodes' and 'edges' lists.
        """
        table = self._find_entity_table(start_id)
        if table is None:
            return {"nodes": [], "edges": []}

        max_hops = max(1, min(max_hops, 10))

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        # Get the start node
        start = self.get_entity(start_id)
        if start:
            nodes[start_id] = start

        # Variable-length path query
        r = self._conn.execute(
            f"MATCH (a:{table})-[r*1..{max_hops}]-(b) "
            f"WHERE a.id = $id "
            f"RETURN DISTINCT b.id, b.name, b.entity_type, "
            f"b.first_seen, b.last_seen, b.confidence, b.properties",
            parameters={"id": start_id},
        )
        cols = r.get_column_names()
        while r.has_next():
            row = r.get_next()
            entity = _row_to_entity(cols, row)
            if entity and entity["id"]:
                nodes[entity["id"]] = entity

        # Get all edges between discovered nodes
        node_ids = list(nodes.keys())
        for nid in node_ids:
            rels = self.get_relationships(nid, direction="out")
            for rel in rels:
                if rel["to_id"] in nodes:
                    edges.append(rel)

        return {"nodes": list(nodes.values()), "edges": edges}

    # ── Query ────────────────────────────────────────────────────────

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[list[Any]]:
        """Execute a raw Cypher query.

        Args:
            cypher: Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result rows (each row is a list of values).
        """
        result = self._conn.execute(cypher, parameters=parameters or {})
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    # ── Search ───────────────────────────────────────────────────────

    def search(self, text: str) -> list[dict[str, Any]]:
        """Search entities by name or ID substring (case-insensitive).

        Args:
            text: Search text to match against entity name or ID.

        Returns:
            List of matching entity dicts.
        """
        results: list[dict[str, Any]] = []
        pattern = f"%{text}%"

        for table in NODE_TABLES:
            r = self._conn.execute(
                f"MATCH (n:{table}) "
                f"WHERE n.id CONTAINS $text OR n.name CONTAINS $text "
                f"RETURN n.id, n.name, n.entity_type, n.first_seen, "
                f"n.last_seen, n.confidence, n.properties",
                parameters={"text": text},
            )
            cols = r.get_column_names()
            while r.has_next():
                row = r.get_next()
                entity = _row_to_entity(cols, row)
                if entity:
                    results.append(entity)

        return results

    # ── Helpers ───────────────────────────────────────────────────────

    def _find_entity_table(self, id: str) -> str | None:
        """Find which node table contains an entity by ID."""
        for table in NODE_TABLES:
            r = self._conn.execute(
                f"MATCH (n:{table}) WHERE n.id = $id RETURN n.id",
                parameters={"id": id},
            )
            if r.has_next():
                return table
        return None

    def close(self) -> None:
        """Close the database connection."""
        # KuzuDB cleans up via Python garbage collection
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]


# ── Module-level helpers ─────────────────────────────────────────────


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_entity(
    columns: list[str], row: list[Any]
) -> dict[str, Any]:
    """Convert a query result row to an entity dict."""
    data: dict[str, Any] = {}
    for col, val in zip(columns, row):
        key = col.split(".")[-1]  # strip table prefix (e.g., "n.name" → "name")
        if key == "properties" and isinstance(val, str):
            try:
                data[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                data[key] = val
        else:
            data[key] = val
    return data


def _row_to_rel(row: list[Any], direction: str = "out") -> dict[str, Any]:
    """Convert a relationship query row to a dict."""
    return {
        "from_id": row[0],
        "to_id": row[1],
        "rel_type": row[2],
        "direction": direction,
        "timestamp": row[3],
        "confidence": row[4],
        "source": row[5],
        "count": row[6],
    }
