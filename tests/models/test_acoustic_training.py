# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AcousticTrainingExample and AcousticTrainingSet models."""

import pytest

from tritium_lib.models.acoustic_training import (
    AcousticTrainingExample,
    AcousticTrainingSet,
    TrainingSource,
)


class TestTrainingSource:
    def test_enum_values(self):
        assert TrainingSource.SYNTHETIC == "synthetic"
        assert TrainingSource.RECORDED == "recorded"
        assert TrainingSource.DATASET == "dataset"
        assert TrainingSource.AUGMENTED == "augmented"
        assert TrainingSource.MANUAL == "manual"


class TestAcousticTrainingExample:
    def _make_example(self, label="gunshot"):
        features = list(range(18))  # 13 MFCC + centroid + zcr + rms + bw + dur
        return AcousticTrainingExample(
            audio_features=[float(x) for x in features],
            label=label,
            source=TrainingSource.DATASET,
            confidence=0.95,
            dataset_name="ESC-50",
            filename="test.wav",
        )

    def test_create(self):
        ex = self._make_example()
        assert ex.label == "gunshot"
        assert ex.source == TrainingSource.DATASET
        assert ex.confidence == 0.95

    def test_mfcc_property(self):
        ex = self._make_example()
        assert len(ex.mfcc) == 13
        assert ex.mfcc == [float(i) for i in range(13)]

    def test_spectral_properties(self):
        ex = self._make_example()
        assert ex.spectral_centroid == 13.0
        assert ex.zcr == 14.0
        assert ex.rms_energy == 15.0
        assert ex.spectral_bandwidth == 16.0
        assert ex.duration_ms == 17

    def test_to_training_tuple(self):
        ex = self._make_example()
        tup = ex.to_training_tuple()
        assert tup[0] == "gunshot"
        assert len(tup[1]) == 13  # MFCC
        assert tup[2] == 13.0  # centroid
        assert tup[3] == 14.0  # zcr
        assert tup[4] == 15.0  # rms
        assert tup[5] == 16.0  # bw
        assert tup[6] == 17    # dur

    def test_to_dict_from_dict(self):
        ex = self._make_example()
        d = ex.to_dict()
        assert d["label"] == "gunshot"
        assert d["source"] == "dataset"

        restored = AcousticTrainingExample.from_dict(d)
        assert restored.label == ex.label
        assert restored.source == ex.source
        assert restored.confidence == ex.confidence
        assert restored.audio_features == ex.audio_features

    def test_short_features(self):
        ex = AcousticTrainingExample(audio_features=[1.0, 2.0], label="test")
        assert len(ex.mfcc) == 2
        assert ex.spectral_centroid == 0.0
        assert ex.duration_ms == 0


class TestAcousticTrainingSet:
    def _make_set(self, n_per_class=5):
        ts = AcousticTrainingSet(name="test")
        for label in ["gunshot", "voice", "vehicle"]:
            for i in range(n_per_class):
                ts.add(AcousticTrainingExample(
                    audio_features=[float(i + j) for j in range(18)],
                    label=label,
                    source=TrainingSource.DATASET,
                    confidence=0.9,
                ))
        return ts

    def test_size_and_labels(self):
        ts = self._make_set()
        assert ts.size == 15
        assert ts.labels == {"gunshot", "voice", "vehicle"}

    def test_label_counts(self):
        ts = self._make_set()
        counts = ts.label_counts
        assert counts["gunshot"] == 5
        assert counts["voice"] == 5
        assert counts["vehicle"] == 5

    def test_add_many(self):
        ts = AcousticTrainingSet()
        examples = [
            AcousticTrainingExample(audio_features=[1.0] * 18, label="test")
            for _ in range(10)
        ]
        added = ts.add_many(examples)
        assert added == 10
        assert ts.size == 10

    def test_remove_by_label(self):
        ts = self._make_set()
        removed = ts.remove(label="gunshot")
        assert removed == 5
        assert ts.size == 10
        assert "gunshot" not in ts.labels

    def test_remove_by_source(self):
        ts = self._make_set()
        ts.add(AcousticTrainingExample(
            audio_features=[0.0] * 18,
            label="test",
            source=TrainingSource.SYNTHETIC,
        ))
        assert ts.size == 16
        removed = ts.remove(source=TrainingSource.SYNTHETIC)
        assert removed == 1
        assert ts.size == 15

    def test_get_examples_filtered(self):
        ts = self._make_set()
        gunshots = ts.get_examples(label="gunshot")
        assert len(gunshots) == 5
        assert all(ex.label == "gunshot" for ex in gunshots)

    def test_split_stratified(self):
        ts = self._make_set(n_per_class=10)
        train, test = ts.split(test_fraction=0.3, stratify=True)
        assert train.size + test.size == ts.size
        # Each class should be represented in both
        assert "gunshot" in train.labels
        assert "gunshot" in test.labels
        assert "voice" in train.labels
        assert "voice" in test.labels

    def test_split_unstratified(self):
        ts = self._make_set(n_per_class=10)
        train, test = ts.split(test_fraction=0.2, stratify=False)
        assert train.size + test.size == ts.size

    def test_balance_oversample(self):
        ts = AcousticTrainingSet()
        # Add imbalanced data
        for i in range(10):
            ts.add(AcousticTrainingExample(
                audio_features=[float(i)] * 18, label="gunshot",
                source=TrainingSource.DATASET,
            ))
        for i in range(3):
            ts.add(AcousticTrainingExample(
                audio_features=[float(i)] * 18, label="voice",
                source=TrainingSource.DATASET,
            ))
        assert ts.label_counts["gunshot"] == 10
        assert ts.label_counts["voice"] == 3

        total = ts.balance(method="oversample")
        assert ts.label_counts["voice"] == 10
        assert total == 20

    def test_balance_undersample(self):
        ts = AcousticTrainingSet()
        for i in range(10):
            ts.add(AcousticTrainingExample(
                audio_features=[float(i)] * 18, label="gunshot",
                source=TrainingSource.DATASET,
            ))
        for i in range(3):
            ts.add(AcousticTrainingExample(
                audio_features=[float(i)] * 18, label="voice",
                source=TrainingSource.DATASET,
            ))

        total = ts.balance(method="undersample")
        assert ts.label_counts["gunshot"] == 3
        assert ts.label_counts["voice"] == 3
        assert total == 6

    def test_to_training_data(self):
        ts = self._make_set(n_per_class=2)
        data = ts.to_training_data()
        assert len(data) == 6
        assert all(isinstance(t, tuple) for t in data)
        assert data[0][0] in {"gunshot", "voice", "vehicle"}

    def test_to_dicts_from_dicts(self):
        ts = self._make_set(n_per_class=2)
        dicts = ts.to_dicts()
        assert len(dicts) == 6

        restored = AcousticTrainingSet.from_dicts("restored", dicts)
        assert restored.size == 6
        assert restored.labels == ts.labels

    def test_summary(self):
        ts = self._make_set()
        s = ts.summary()
        assert s["name"] == "test"
        assert s["total_examples"] == 15
        assert s["num_classes"] == 3
        assert s["avg_confidence"] == pytest.approx(0.9)
        assert "dataset" in s["source_counts"]

    def test_empty_set(self):
        ts = AcousticTrainingSet()
        assert ts.size == 0
        assert ts.labels == set()
        assert ts.label_counts == {}
        s = ts.summary()
        assert s["avg_confidence"] == 0.0

    def test_balance_empty(self):
        ts = AcousticTrainingSet()
        assert ts.balance() == 0
