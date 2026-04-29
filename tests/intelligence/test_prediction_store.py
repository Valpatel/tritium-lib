# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for PredictionStore — persistent RL prediction history (B-6)."""

from __future__ import annotations

import time

import pytest

from tritium_lib.intelligence.prediction_store import PredictionStore
from tritium_lib.intelligence.rl_metrics import PredictionRecord, RLMetrics


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "predictions.db"


class TestPredictionStoreCore:
    """Core deque-compatible API."""

    def test_init_creates_db(self, store_path):
        store = PredictionStore(store_path)
        assert store_path.exists()
        assert len(store) == 0
        assert list(store) == []

    def test_append_and_iter(self, store_path):
        store = PredictionStore(store_path)
        store.append(PredictionRecord(
            timestamp=1000.0,
            predicted_class=1,
            probability=0.82,
            correct=True,
        ))
        store.append(PredictionRecord(
            timestamp=1001.0,
            predicted_class=0,
            probability=0.21,
            correct=False,
        ))
        store.append(PredictionRecord(
            timestamp=1002.0,
            predicted_class=1,
            probability=0.55,
            correct=None,
        ))

        assert len(store) == 3
        records = list(store)
        assert len(records) == 3
        assert records[0].timestamp == 1000.0
        assert records[0].correct is True
        assert records[1].correct is False
        assert records[2].correct is None

    def test_clear(self, store_path):
        store = PredictionStore(store_path)
        store.append(PredictionRecord(
            timestamp=1.0, predicted_class=1, probability=0.6,
        ))
        assert len(store) == 1
        store.clear()
        assert len(store) == 0
        assert list(store) == []

    def test_extend_bulk(self, store_path):
        store = PredictionStore(store_path)
        store.extend([
            PredictionRecord(
                timestamp=t, predicted_class=t % 2, probability=0.5 + (t / 100),
                correct=(t % 2 == 0),
            )
            for t in range(20)
        ])
        assert len(store) == 20

    def test_max_records_prunes_oldest(self, store_path):
        store = PredictionStore(store_path, max_records=5)
        for t in range(10):
            store.append(PredictionRecord(
                timestamp=float(t), predicted_class=1, probability=0.5,
            ))
        assert len(store) == 5
        records = store.fetch()
        # Oldest five should have been pruned -> only timestamps 5..9 remain
        timestamps = [r.timestamp for r in records]
        assert timestamps == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_fetch_filters(self, store_path):
        store = PredictionStore(store_path)
        for t in range(5):
            store.append(PredictionRecord(
                timestamp=1000.0 + t, predicted_class=1, probability=0.7,
            ))
        recent = store.fetch(since=1003.0)
        assert len(recent) == 2
        for r in recent:
            assert r.timestamp >= 1003.0

    def test_stats(self, store_path):
        store = PredictionStore(store_path)
        store.extend([
            PredictionRecord(
                timestamp=10.0, predicted_class=1, probability=0.7, correct=True,
            ),
            PredictionRecord(
                timestamp=20.0, predicted_class=0, probability=0.4, correct=False,
            ),
            PredictionRecord(
                timestamp=30.0, predicted_class=1, probability=0.9, correct=None,
            ),
        ])
        stats = store.stats()
        assert stats["count"] == 3
        assert stats["correct"] == 1
        assert stats["incorrect"] == 1
        assert stats["first_timestamp"] == 10.0
        assert stats["last_timestamp"] == 30.0


class TestPredictionStoreRestartSurvival:
    """B-6 acceptance: restart-survival of historical predictions."""

    def test_restart_survives(self, store_path):
        # First "process": write
        store_a = PredictionStore(store_path)
        for t in range(3):
            store_a.append(PredictionRecord(
                timestamp=100.0 + t,
                predicted_class=1,
                probability=0.6 + (t / 10.0),
                correct=True,
            ))
        del store_a

        # Second "process": new store, same path
        store_b = PredictionStore(store_path)
        assert len(store_b) == 3
        assert list(store_b)[0].timestamp == 100.0

    def test_rl_metrics_hydrates_from_store(self, store_path):
        # Phase 1: write history through RLMetrics
        store = PredictionStore(store_path)
        m1 = RLMetrics(prediction_store=store)
        m1.record_prediction(predicted_class=1, probability=0.8, correct=True)
        m1.record_prediction(predicted_class=0, probability=0.2, correct=False)
        del m1

        # Phase 2: a brand-new RLMetrics should pre-populate from the store
        store2 = PredictionStore(store_path)
        m2 = RLMetrics(prediction_store=store2)
        # The deque was hydrated from the store
        assert len(m2._predictions) == 2

    def test_reset_does_not_clear_store_by_default(self, store_path):
        store = PredictionStore(store_path)
        m = RLMetrics(prediction_store=store)
        m.record_prediction(predicted_class=1, probability=0.8, correct=True)
        assert len(store) == 1

        m.reset()  # default: clear_store=False
        assert len(store) == 1  # persistent record survives

    def test_reset_can_clear_store(self, store_path):
        store = PredictionStore(store_path)
        m = RLMetrics(prediction_store=store)
        m.record_prediction(predicted_class=1, probability=0.8, correct=True)
        assert len(store) == 1
        m.reset(clear_store=True)
        assert len(store) == 0
