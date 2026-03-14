# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the ReID embedding store and models."""

from __future__ import annotations

import math

import pytest

from tritium_lib.models.reid import ReIDEmbedding, ReIDMatch
from tritium_lib.store.reid import (
    ReIDStore,
    cosine_similarity,
    _vector_to_blob,
    _blob_to_vector,
    _cosine_similarity_pure,
)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestReIDModels:
    def test_embedding_defaults(self) -> None:
        e = ReIDEmbedding()
        assert e.embedding_id == ""
        assert e.target_id == ""
        assert e.source_camera == ""
        assert e.timestamp is None
        assert e.dimensions == 0

    def test_embedding_with_values(self) -> None:
        e = ReIDEmbedding(
            embedding_id="abc123",
            target_id="person_42",
            source_camera="cam_front",
            dimensions=128,
        )
        assert e.embedding_id == "abc123"
        assert e.target_id == "person_42"
        assert e.dimensions == 128

    def test_match_defaults(self) -> None:
        m = ReIDMatch()
        assert m.query_id == ""
        assert m.matched_id == ""
        assert m.similarity == 0.0
        assert m.source_cameras == []

    def test_match_is_strong(self) -> None:
        weak = ReIDMatch(similarity=0.7)
        strong = ReIDMatch(similarity=0.9)
        assert not weak.is_strong_match
        assert strong.is_strong_match

    def test_match_source_cameras(self) -> None:
        m = ReIDMatch(source_cameras=["cam_a", "cam_b"])
        assert len(m.source_cameras) == 2


# ---------------------------------------------------------------------------
# Vector serialization tests
# ---------------------------------------------------------------------------

class TestVectorSerialization:
    def test_roundtrip(self) -> None:
        vec = [1.0, 2.5, -3.0, 0.0, 0.123456]
        blob = _vector_to_blob(vec)
        restored = _blob_to_vector(blob)
        assert len(restored) == len(vec)
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-5

    def test_empty_vector(self) -> None:
        blob = _vector_to_blob([])
        assert _blob_to_vector(blob) == []


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_pure_python_fallback(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        sim = _cosine_similarity_pure(a, b)
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        assert abs(sim - expected) < 1e-6


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> ReIDStore:
    s = ReIDStore(":memory:")
    yield s
    s.close()


class TestReIDStore:
    def test_store_and_retrieve(self, store: ReIDStore) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        eid = store.store_embedding("target_1", vec, "cam_a", embedding_id="e1")
        assert eid == "e1"

        result = store.get_embedding("e1")
        assert result is not None
        assert result["target_id"] == "target_1"
        assert result["source_camera"] == "cam_a"
        assert len(result["embedding"]) == 4
        for a, b in zip(result["embedding"], vec):
            assert abs(a - b) < 1e-5

    def test_get_missing_embedding(self, store: ReIDStore) -> None:
        assert store.get_embedding("nonexistent") is None

    def test_auto_generated_id(self, store: ReIDStore) -> None:
        eid = store.store_embedding("t1", [1.0, 2.0], "cam_x")
        assert len(eid) == 16  # sha256 truncated to 16 hex chars

    def test_get_embeddings_for_target(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0], "cam_a", embedding_id="e1")
        store.store_embedding("t1", [0.0, 1.0], "cam_b", embedding_id="e2")
        store.store_embedding("t2", [1.0, 1.0], "cam_a", embedding_id="e3")

        results = store.get_embeddings_for_target("t1")
        assert len(results) == 2
        assert all(r["target_id"] == "t1" for r in results)

    def test_find_similar_exact_match(self, store: ReIDStore) -> None:
        vec = [1.0, 0.0, 0.0, 0.0]
        store.store_embedding("t1", vec, "cam_a", embedding_id="e1")

        results = store.find_similar(vec, threshold=0.9)
        assert len(results) == 1
        assert results[0]["embedding_id"] == "e1"
        assert results[0]["similarity"] > 0.99

    def test_find_similar_threshold(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0, 0.0], "cam_a", embedding_id="e1")
        store.store_embedding("t2", [0.0, 1.0, 0.0], "cam_b", embedding_id="e2")
        store.store_embedding("t3", [0.9, 0.1, 0.0], "cam_c", embedding_id="e3")

        query = [1.0, 0.0, 0.0]
        results = store.find_similar(query, threshold=0.8)
        # e1 should be exact match (sim=1.0), e3 should be high sim
        # e2 should be filtered out (orthogonal)
        assert len(results) >= 1
        assert results[0]["embedding_id"] == "e1"
        ids = [r["embedding_id"] for r in results]
        assert "e2" not in ids

    def test_find_similar_limit(self, store: ReIDStore) -> None:
        for i in range(20):
            # All similar vectors
            store.store_embedding(
                f"t{i}", [1.0, float(i) * 0.01, 0.0], "cam_a",
                embedding_id=f"e{i}",
            )
        results = store.find_similar([1.0, 0.0, 0.0], threshold=0.5, limit=5)
        assert len(results) == 5

    def test_find_similar_dimension_mismatch(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0], "cam_a", embedding_id="e1")
        # Query with different dimension — should be skipped
        results = store.find_similar([1.0, 0.0, 0.0], threshold=0.5)
        assert len(results) == 0

    def test_record_and_get_matches(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0], "cam_a", embedding_id="e1")
        store.store_embedding("t2", [0.9, 0.1], "cam_b", embedding_id="e2")

        mid = store.record_match("e1", "e2", 0.95)
        assert mid is not None

        matches = store.get_matches("t1")
        assert len(matches) == 1
        assert matches[0]["similarity"] == 0.95
        assert matches[0]["query_camera"] == "cam_a"
        assert matches[0]["matched_camera"] == "cam_b"

    def test_get_matches_bidirectional(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0], "cam_a", embedding_id="e1")
        store.store_embedding("t2", [0.9, 0.1], "cam_b", embedding_id="e2")
        store.record_match("e1", "e2", 0.95)

        # Should find the match from either target's perspective
        assert len(store.get_matches("t1")) == 1
        assert len(store.get_matches("t2")) == 1

    def test_count_embeddings(self, store: ReIDStore) -> None:
        assert store.count_embeddings() == 0
        store.store_embedding("t1", [1.0], "cam_a", embedding_id="e1")
        store.store_embedding("t2", [2.0], "cam_b", embedding_id="e2")
        assert store.count_embeddings() == 2

    def test_count_matches(self, store: ReIDStore) -> None:
        assert store.count_matches() == 0
        store.store_embedding("t1", [1.0], "cam_a", embedding_id="e1")
        store.store_embedding("t2", [2.0], "cam_b", embedding_id="e2")
        store.record_match("e1", "e2", 0.9)
        assert store.count_matches() == 1

    def test_delete_target(self, store: ReIDStore) -> None:
        store.store_embedding("t1", [1.0, 0.0], "cam_a", embedding_id="e1")
        store.store_embedding("t1", [0.9, 0.1], "cam_b", embedding_id="e2")
        store.store_embedding("t2", [0.0, 1.0], "cam_c", embedding_id="e3")
        store.record_match("e1", "e3", 0.5)

        deleted = store.delete_target("t1")
        assert deleted == 2
        assert store.count_embeddings() == 1
        assert store.count_matches() == 0  # match was cleaned up

    def test_delete_nonexistent_target(self, store: ReIDStore) -> None:
        assert store.delete_target("ghost") == 0

    def test_confidence_stored(self, store: ReIDStore) -> None:
        store.store_embedding(
            "t1", [1.0, 0.0], "cam_a",
            confidence=0.85, embedding_id="e1",
        )
        result = store.get_embedding("e1")
        assert result is not None
        assert abs(result["confidence"] - 0.85) < 1e-6


class TestReIDStoreImports:
    """Verify the store is accessible from the package __init__."""

    def test_store_import(self) -> None:
        from tritium_lib.store import ReIDStore as RS
        assert RS is ReIDStore

    def test_model_import(self) -> None:
        from tritium_lib.models import ReIDEmbedding, ReIDMatch
        assert ReIDEmbedding is not None
        assert ReIDMatch is not None
