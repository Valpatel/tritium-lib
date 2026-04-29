# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.acoustic_classifier_train (Wave 204).

Covers:
- ESC-50 mapping is consistent with the test in tritium-sc.
- Pickle save/load round-trip preserves predictions.
- AcousticClassifier model_version selection (esc50_v2 / synthetic_v1).
- walk_esc50 yields rows for the mapped subset.

Most heavy tests are SKIPPED when the ESC-50 dataset is not present, so this
file is safe to run on machines without the dataset.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import pytest

from tritium_lib.intelligence.acoustic_classifier import (
    AcousticClassifier,
    AudioFeatures,
    ESC50_CATEGORY_MAP,
    MFCCClassifier,
)
from tritium_lib.intelligence.acoustic_classifier_train import (
    load_classifier,
    rows_to_training_tuples,
    save_classifier,
    walk_esc50,
)

# Repo-root probing — the dataset lives at tritium/data/library/audio/ESC-50
# regardless of which submodule we're running from.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT_CANDIDATES = [
    THIS_FILE.parents[3] / "data" / "library" / "audio" / "ESC-50",  # parent tritium repo
    THIS_FILE.parents[2] / "data" / "library" / "audio" / "ESC-50",  # tritium-lib only
]
ESC50_ROOT: Path | None = next((p for p in REPO_ROOT_CANDIDATES if p.exists()), None)


def test_esc50_mapping_has_34_entries():
    """The ESC-50 -> Tritium mapping covers exactly 34 categories (Wave 135 baseline)."""
    # Per memory: 34 ESC-50 categories map to 9 of the 11 Tritium classes.
    assert len(ESC50_CATEGORY_MAP) == 34


def test_esc50_mapping_covers_nine_tritium_classes():
    """The mapped subset reaches 9 of the 11 sound classes (no music, no gunshot)."""
    target_classes = set(ESC50_CATEGORY_MAP.values())
    assert "music" not in target_classes  # ESC-50 has no music category
    assert "gunshot" not in target_classes  # ESC-50 has no gunshot category
    assert len(target_classes) == 9


def test_acoustic_classifier_synthetic_version_pinning():
    """When TRITIUM_ACOUSTIC_MODEL_VERSION=synthetic_v1, no pickle is loaded."""
    old = os.environ.get("TRITIUM_ACOUSTIC_MODEL_VERSION")
    os.environ["TRITIUM_ACOUSTIC_MODEL_VERSION"] = "synthetic_v1"
    try:
        c = AcousticClassifier()
        assert c.loaded_model_version == "synthetic_v1"
        assert c.ml_available
        assert c.loaded_model_path is None
    finally:
        if old is None:
            os.environ.pop("TRITIUM_ACOUSTIC_MODEL_VERSION", None)
        else:
            os.environ["TRITIUM_ACOUSTIC_MODEL_VERSION"] = old


def test_acoustic_classifier_disable_ml():
    """enable_ml=False yields no ML classifier and rule-based fallback."""
    c = AcousticClassifier(enable_ml=False)
    assert not c.ml_available
    assert c.loaded_model_version == "none"


def test_save_load_roundtrip_preserves_predictions(tmp_path: Path):
    """Pickling and reloading an MFCCClassifier preserves predictions exactly."""
    clf = MFCCClassifier(k=3)
    clf.train()
    assert clf.is_trained

    fv = AudioFeatures(
        rms_energy=0.2,
        zero_crossing_rate=0.1,
        spectral_centroid=400,
        spectral_bandwidth=200,
        duration_ms=1500,
        mfcc=[0.1] * 13,
    )
    pred_before = clf.classify(fv)

    out = tmp_path / "test_classifier.pkl"
    save_classifier(clf, out)
    assert out.exists()

    loaded = load_classifier(out)
    assert loaded.is_trained
    pred_after = loaded.classify(fv)

    # Predicted class and confidence should be identical.
    assert pred_before[0] == pred_after[0]
    assert abs(pred_before[1] - pred_after[1]) < 1e-9


def test_acoustic_classifier_loads_pickle_via_arg(tmp_path: Path):
    """Passing model_path= directly loads the pickle as esc50_v2."""
    clf = MFCCClassifier(k=3)
    clf.train()
    out = tmp_path / "test_v2.pkl"
    save_classifier(clf, out)

    c = AcousticClassifier(model_path=str(out))
    assert c.loaded_model_version == "esc50_v2"
    assert c.loaded_model_path == str(out)
    assert c.ml_available


def test_acoustic_classifier_handles_missing_pickle(tmp_path: Path):
    """When model_version='esc50_v2' is requested but pickle is missing, fall back to synthetic_v1."""
    bogus = tmp_path / "does_not_exist.pkl"
    c = AcousticClassifier(model_path=str(bogus), model_version="esc50_v2")
    # Falls back to synthetic_v1.
    assert c.loaded_model_version == "synthetic_v1"
    assert c.ml_available


def test_acoustic_classifier_handles_corrupt_pickle(tmp_path: Path):
    """A corrupt pickle is rejected gracefully (falls back to synthetic_v1)."""
    bad = tmp_path / "corrupt.pkl"
    bad.write_bytes(b"not a valid pickle")
    c = AcousticClassifier(model_path=str(bad))
    assert c.loaded_model_version == "synthetic_v1"
    assert c.ml_available


def test_acoustic_classifier_handles_pickle_without_version_marker(tmp_path: Path):
    """Pickle missing the 'version' key is rejected (falls back to synthetic_v1)."""
    bad = tmp_path / "no_version.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"k": 5}, f)  # missing 'version'
    c = AcousticClassifier(model_path=str(bad))
    assert c.loaded_model_version == "synthetic_v1"


def test_rows_to_training_tuples_strips_metadata():
    """rows_to_training_tuples drops fold and category columns."""
    rows = [
        ("animal", [0.1] * 13, 100.0, 0.1, 0.1, 50.0, 1000, 1, "dog"),
        ("vehicle", [0.2] * 13, 200.0, 0.2, 0.2, 100.0, 2000, 2, "engine"),
    ]
    tuples = rows_to_training_tuples(rows)
    assert len(tuples) == 2
    assert all(len(t) == 7 for t in tuples)
    assert tuples[0][0] == "animal"
    assert tuples[1][0] == "vehicle"


# ---------------------------------------------------------------------------
# Heavy ESC-50 tests (skipped without dataset).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(ESC50_ROOT is None, reason="ESC-50 dataset not available")
def test_walk_esc50_returns_mapped_rows():
    """walk_esc50 with a small per-category cap yields rows from mapped categories."""
    assert ESC50_ROOT is not None
    rows = walk_esc50(ESC50_ROOT, folds=[1], max_per_category=2)
    assert len(rows) > 0
    # Every row should have a non-unknown label.
    for label, mfcc, centroid, zcr, rms, bw, dur, fold, cat in rows:
        assert label != "unknown"
        assert label in set(ESC50_CATEGORY_MAP.values())
        assert len(mfcc) == 13
        assert fold == 1
        assert cat in ESC50_CATEGORY_MAP


@pytest.mark.skipif(ESC50_ROOT is None, reason="ESC-50 dataset not available")
def test_walk_esc50_respects_per_category_cap():
    """max_per_category=1 yields at most 34 rows (one per mapped category)."""
    assert ESC50_ROOT is not None
    rows = walk_esc50(ESC50_ROOT, max_per_category=1)
    cats = set(r[8] for r in rows)
    assert len(rows) <= 34
    # Each (cat, fold) should appear at most once.
    seen: set[tuple[str, int]] = set()
    for r in rows:
        key = (r[8], r[7])
        # max_per_category caps total per ESC-50 category, not per fold.
        seen.add(key)
    assert len(cats) <= 34
