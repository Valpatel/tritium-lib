# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Structured audio ML training data models.

Provides AcousticTrainingExample and AcousticTrainingSet for managing
labeled audio feature datasets used to train acoustic classifiers.

Supports:
- Adding/removing examples with labels and provenance
- Train/test splitting with stratification
- Class balancing via oversampling or undersampling
- Export to classifier-compatible format
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrainingSource(str, Enum):
    """Where the training example came from."""

    SYNTHETIC = "synthetic"     # Generated from known profiles
    RECORDED = "recorded"       # Captured from real microphone
    DATASET = "dataset"         # From labeled dataset (ESC-50, UrbanSound8K, etc.)
    AUGMENTED = "augmented"     # Derived from augmentation of existing sample
    MANUAL = "manual"           # Manually labeled by operator


@dataclass
class AcousticTrainingExample:
    """A single labeled audio feature vector for ML training.

    Attributes:
        audio_features: Feature vector (13 MFCCs + spectral_centroid + zcr +
            rms_energy + spectral_bandwidth + duration_ms).
        label: Classification label (e.g. "gunshot", "voice", "vehicle").
        source: Provenance of this example.
        confidence: Label confidence (1.0 = certain ground truth).
        dataset_name: Origin dataset name (e.g. "ESC-50").
        filename: Source audio filename if applicable.
        timestamp: When this example was created.
        metadata: Optional extra metadata.
    """

    audio_features: list[float]
    label: str
    source: TrainingSource = TrainingSource.SYNTHETIC
    confidence: float = 1.0
    dataset_name: str = ""
    filename: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def mfcc(self) -> list[float]:
        """Extract the 13 MFCC coefficients from the feature vector."""
        return self.audio_features[:13] if len(self.audio_features) >= 13 else self.audio_features

    @property
    def spectral_centroid(self) -> float:
        return self.audio_features[13] if len(self.audio_features) > 13 else 0.0

    @property
    def zcr(self) -> float:
        return self.audio_features[14] if len(self.audio_features) > 14 else 0.0

    @property
    def rms_energy(self) -> float:
        return self.audio_features[15] if len(self.audio_features) > 15 else 0.0

    @property
    def spectral_bandwidth(self) -> float:
        return self.audio_features[16] if len(self.audio_features) > 16 else 0.0

    @property
    def duration_ms(self) -> int:
        return int(self.audio_features[17]) if len(self.audio_features) > 17 else 0

    def to_training_tuple(self) -> tuple:
        """Convert to the (label, mfcc, centroid, zcr, rms, bw, dur) format
        expected by MFCCClassifier.train().
        """
        return (
            self.label,
            self.mfcc,
            self.spectral_centroid,
            self.zcr,
            self.rms_energy,
            self.spectral_bandwidth,
            self.duration_ms,
        )

    def to_dict(self) -> dict:
        return {
            "audio_features": self.audio_features,
            "label": self.label,
            "source": self.source.value,
            "confidence": self.confidence,
            "dataset_name": self.dataset_name,
            "filename": self.filename,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AcousticTrainingExample:
        return cls(
            audio_features=d.get("audio_features", []),
            label=d.get("label", ""),
            source=TrainingSource(d.get("source", "synthetic")),
            confidence=d.get("confidence", 1.0),
            dataset_name=d.get("dataset_name", ""),
            filename=d.get("filename", ""),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
        )


class AcousticTrainingSet:
    """A managed collection of AcousticTrainingExample instances.

    Provides add/remove, train/test split, and class balancing
    for acoustic classifier training.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._examples: list[AcousticTrainingExample] = []

    @property
    def size(self) -> int:
        return len(self._examples)

    @property
    def labels(self) -> set[str]:
        """All unique labels in the set."""
        return {ex.label for ex in self._examples}

    @property
    def label_counts(self) -> dict[str, int]:
        """Count of examples per label."""
        counts: dict[str, int] = {}
        for ex in self._examples:
            counts[ex.label] = counts.get(ex.label, 0) + 1
        return counts

    def add(self, example: AcousticTrainingExample) -> None:
        """Add a training example."""
        self._examples.append(example)

    def add_many(self, examples: list[AcousticTrainingExample]) -> int:
        """Add multiple examples. Returns count added."""
        self._examples.extend(examples)
        return len(examples)

    def remove(self, label: Optional[str] = None, source: Optional[TrainingSource] = None) -> int:
        """Remove examples matching criteria. Returns count removed."""
        before = len(self._examples)
        self._examples = [
            ex for ex in self._examples
            if not ((label is None or ex.label == label) and
                    (source is None or ex.source == source))
        ]
        return before - len(self._examples)

    def get_examples(
        self,
        label: Optional[str] = None,
        source: Optional[TrainingSource] = None,
        min_confidence: float = 0.0,
    ) -> list[AcousticTrainingExample]:
        """Get examples matching optional filters."""
        result = self._examples
        if label is not None:
            result = [ex for ex in result if ex.label == label]
        if source is not None:
            result = [ex for ex in result if ex.source == source]
        if min_confidence > 0.0:
            result = [ex for ex in result if ex.confidence >= min_confidence]
        return result

    def split(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
        stratify: bool = True,
    ) -> tuple[AcousticTrainingSet, AcousticTrainingSet]:
        """Split into train and test sets.

        Args:
            test_fraction: Fraction of data for test set (0.0-1.0).
            seed: Random seed for reproducibility.
            stratify: If True, maintain label proportions in both sets.

        Returns:
            (train_set, test_set) tuple.
        """
        rng = random.Random(seed)
        train_set = AcousticTrainingSet(name=f"{self.name}_train")
        test_set = AcousticTrainingSet(name=f"{self.name}_test")

        if stratify:
            # Group by label
            by_label: dict[str, list[AcousticTrainingExample]] = {}
            for ex in self._examples:
                by_label.setdefault(ex.label, []).append(ex)

            for label, examples in by_label.items():
                shuffled = list(examples)
                rng.shuffle(shuffled)
                n_test = max(1, int(len(shuffled) * test_fraction))
                test_set.add_many(shuffled[:n_test])
                train_set.add_many(shuffled[n_test:])
        else:
            shuffled = list(self._examples)
            rng.shuffle(shuffled)
            n_test = int(len(shuffled) * test_fraction)
            test_set.add_many(shuffled[:n_test])
            train_set.add_many(shuffled[n_test:])

        return train_set, test_set

    def balance(self, method: str = "oversample", target_count: Optional[int] = None) -> int:
        """Balance class distribution.

        Args:
            method: "oversample" (duplicate minority) or "undersample" (trim majority).
            target_count: Target count per class. Defaults to max (oversample) or min (undersample).

        Returns:
            Total example count after balancing.
        """
        counts = self.label_counts
        if not counts:
            return 0

        if method == "oversample":
            target = target_count or max(counts.values())
            by_label: dict[str, list[AcousticTrainingExample]] = {}
            for ex in self._examples:
                by_label.setdefault(ex.label, []).append(ex)

            new_examples: list[AcousticTrainingExample] = []
            rng = random.Random(42)
            for label, examples in by_label.items():
                new_examples.extend(examples)
                deficit = target - len(examples)
                if deficit > 0:
                    # Oversample by duplicating with augmented source tag
                    for i in range(deficit):
                        orig = examples[i % len(examples)]
                        dup = AcousticTrainingExample(
                            audio_features=list(orig.audio_features),
                            label=orig.label,
                            source=TrainingSource.AUGMENTED,
                            confidence=orig.confidence * 0.95,
                            dataset_name=orig.dataset_name,
                            filename=orig.filename,
                            metadata={**orig.metadata, "augmented_from": i % len(examples)},
                        )
                        new_examples.append(dup)
            self._examples = new_examples

        elif method == "undersample":
            target = target_count or min(counts.values())
            by_label = {}
            for ex in self._examples:
                by_label.setdefault(ex.label, []).append(ex)

            new_examples = []
            rng = random.Random(42)
            for label, examples in by_label.items():
                if len(examples) > target:
                    rng.shuffle(examples)
                    new_examples.extend(examples[:target])
                else:
                    new_examples.extend(examples)
            self._examples = new_examples

        return len(self._examples)

    def to_training_data(self) -> list[tuple]:
        """Export as list of tuples for MFCCClassifier.train()."""
        return [ex.to_training_tuple() for ex in self._examples]

    def to_dicts(self) -> list[dict]:
        """Export all examples as dicts."""
        return [ex.to_dict() for ex in self._examples]

    @classmethod
    def from_dicts(cls, name: str, dicts: list[dict]) -> AcousticTrainingSet:
        """Create from list of dicts."""
        ts = cls(name=name)
        for d in dicts:
            ts.add(AcousticTrainingExample.from_dict(d))
        return ts

    def summary(self) -> dict:
        """Get summary statistics."""
        counts = self.label_counts
        sources: dict[str, int] = {}
        for ex in self._examples:
            sources[ex.source.value] = sources.get(ex.source.value, 0) + 1

        return {
            "name": self.name,
            "total_examples": self.size,
            "num_classes": len(counts),
            "label_counts": counts,
            "source_counts": sources,
            "avg_confidence": (
                sum(ex.confidence for ex in self._examples) / self.size
                if self.size > 0 else 0.0
            ),
        }
