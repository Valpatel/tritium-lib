# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Acoustic classifier ESC-50 training and evaluation harness.

Walks the ESC-50 dataset, extracts MFCC + spectral features, builds (X, y)
pairs for the 34 mapped categories, then trains an MFCCClassifier with
held-out evaluation. Reports per-class precision/recall and persists the
trained model to a pickle.

Usage:
    python -m tritium_lib.intelligence.acoustic_classifier_train \
        --dataset esc50 \
        --esc50-root data/library/audio/ESC-50\
        --out tritium-lib/data/models/acoustic_classifier_v2.pkl

Strategies:
    --mode holdout      Train on fold 1-4, test on fold 5 (default)
    --mode all          Train on all folds (no eval), used for final shipping
    --mode synthetic    Train on synthetic profiles only (baseline)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tritium_lib.intelligence.acoustic_classifier import (
    ESC50_CATEGORY_MAP,
    MFCCClassifier,
    TRAINING_DATA,
    _extract_wav_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset walking
# ---------------------------------------------------------------------------


def walk_esc50(
    esc50_root: Path,
    category_map: Optional[dict[str, str]] = None,
    folds: Optional[list[int]] = None,
    max_per_category: Optional[int] = None,
    progress: bool = False,
) -> list[tuple[str, list[float], float, float, float, float, int, int, str]]:
    """Iterate ESC-50 metadata, extract features per WAV, return rows.

    Returns list of tuples:
        (label, mfcc, centroid, zcr, rms, bandwidth, duration_ms, fold, category)

    The fold and category fields are only used by the training harness; they
    are stripped before training.
    """
    cat_map = category_map or ESC50_CATEGORY_MAP
    csv_path = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"

    if not csv_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata CSV not found at {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found at {audio_dir}")

    fold_set = set(folds) if folds else None
    counts: dict[str, int] = defaultdict(int)
    rows: list[tuple[str, list[float], float, float, float, float, int, int, str]] = []
    skipped = 0
    processed = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category", "")
            label = cat_map.get(cat)
            if not label or label == "unknown":
                continue
            try:
                fold_int = int(row.get("fold", "0"))
            except ValueError:
                continue
            if fold_set is not None and fold_int not in fold_set:
                continue
            if max_per_category is not None and counts[cat] >= max_per_category:
                continue

            filename = row.get("filename", "")
            if not filename or ".." in filename or filename.startswith("/"):
                skipped += 1
                continue
            wav_path = audio_dir / filename
            if not wav_path.exists():
                skipped += 1
                continue

            result = _extract_wav_features(str(wav_path))
            if result is None:
                skipped += 1
                continue

            _, mfcc, centroid, zcr, rms, bw, dur = result
            rows.append((label, mfcc, centroid, zcr, rms, bw, dur, fold_int, cat))
            counts[cat] += 1
            processed += 1

            if progress and processed % 100 == 0:
                logger.info("ESC-50: extracted %d files (%d skipped)", processed, skipped)

    if progress:
        logger.info(
            "ESC-50 walk done: %d rows from %d categories (%d skipped)",
            len(rows),
            len(counts),
            skipped,
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    classifier: MFCCClassifier,
    test_rows: list[tuple[str, list[float], float, float, float, float, int, int, str]],
) -> dict:
    """Evaluate classifier on held-out test rows.

    Returns dict with overall_accuracy, per-class precision/recall/f1,
    confusion matrix, and confidence histogram.
    """
    # Per-class TP/FP/FN
    classes = sorted(set(r[0] for r in test_rows))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    total_correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    confidences: list[float] = []
    per_category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    from tritium_lib.intelligence.acoustic_classifier import AudioFeatures

    for row in test_rows:
        label, mfcc, centroid, zcr, rms, bw, dur, _fold, cat = row
        feats = AudioFeatures(
            rms_energy=rms,
            zero_crossing_rate=zcr,
            spectral_centroid=centroid,
            spectral_bandwidth=bw,
            duration_ms=dur,
            mfcc=mfcc,
        )
        predicted, confidence, _preds = classifier.classify(feats)
        confidences.append(confidence)
        confusion[label][predicted] += 1
        per_category_stats[cat]["total"] += 1
        if predicted == label:
            tp[label] += 1
            total_correct += 1
            per_category_stats[cat]["correct"] += 1
        else:
            fn[label] += 1
            fp[predicted] += 1
        total += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp[cls] + fn[cls],
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
        }

    macro_precision = (
        sum(per_class[c]["precision"] for c in classes) / len(classes) if classes else 0.0
    )
    macro_recall = (
        sum(per_class[c]["recall"] for c in classes) / len(classes) if classes else 0.0
    )
    macro_f1 = sum(per_class[c]["f1"] for c in classes) / len(classes) if classes else 0.0

    return {
        "overall_accuracy": round(total_correct / total, 4) if total > 0 else 0.0,
        "total_samples": total,
        "total_correct": total_correct,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_category": dict(per_category_stats),
        "mean_confidence": round(sum(confidences) / len(confidences), 4)
        if confidences
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def rows_to_training_tuples(
    rows: list[tuple[str, list[float], float, float, float, float, int, int, str]],
) -> list[tuple[str, list[float], float, float, float, float, int]]:
    """Strip fold/category from rows to produce TRAINING_DATA-compatible tuples."""
    return [(label, mfcc, centroid, zcr, rms, bw, dur) for label, mfcc, centroid, zcr, rms, bw, dur, _f, _c in rows]


def train_holdout(
    esc50_root: Path,
    test_fold: int = 5,
    k: int = 5,
    augment_synthetic: bool = False,
    max_per_category: Optional[int] = None,
    progress: bool = False,
) -> tuple[MFCCClassifier, dict, dict]:
    """Train on folds != test_fold, evaluate on test_fold.

    Returns (classifier, train_metrics, test_metrics).
    """
    train_folds = [f for f in range(1, 6) if f != test_fold]

    if progress:
        logger.info("Walking ESC-50 train folds %s ...", train_folds)
    train_rows = walk_esc50(
        esc50_root,
        folds=train_folds,
        max_per_category=max_per_category,
        progress=progress,
    )
    if progress:
        logger.info("Walking ESC-50 test fold %d ...", test_fold)
    test_rows = walk_esc50(
        esc50_root,
        folds=[test_fold],
        max_per_category=max_per_category,
        progress=progress,
    )

    train_tuples = rows_to_training_tuples(train_rows)
    if augment_synthetic:
        train_tuples = list(train_tuples) + list(TRAINING_DATA)

    classifier = MFCCClassifier(k=k)
    classifier.train(train_tuples)

    train_metrics = {
        "n_train": len(train_tuples),
        "n_classes": classifier._training_class_count,
        "uses_sklearn": classifier.uses_sklearn,
        "augmented_with_synthetic": augment_synthetic,
        "k": k,
        "test_fold": test_fold,
    }
    test_metrics = evaluate_classifier(classifier, test_rows)
    return classifier, train_metrics, test_metrics


def train_all(
    esc50_root: Path,
    k: int = 5,
    augment_synthetic: bool = False,
    max_per_category: Optional[int] = None,
    progress: bool = False,
) -> tuple[MFCCClassifier, dict]:
    """Train on every mapped ESC-50 sample (no held-out evaluation)."""
    if progress:
        logger.info("Walking ALL ESC-50 folds ...")
    rows = walk_esc50(
        esc50_root,
        max_per_category=max_per_category,
        progress=progress,
    )
    train_tuples = rows_to_training_tuples(rows)
    if augment_synthetic:
        train_tuples = list(train_tuples) + list(TRAINING_DATA)

    classifier = MFCCClassifier(k=k)
    classifier.train(train_tuples)

    train_metrics = {
        "n_train": len(train_tuples),
        "n_classes": classifier._training_class_count,
        "uses_sklearn": classifier.uses_sklearn,
        "augmented_with_synthetic": augment_synthetic,
        "k": k,
    }
    return classifier, train_metrics


def evaluate_synthetic_baseline(
    esc50_root: Path,
    k: int = 5,
    progress: bool = False,
) -> dict:
    """Train MFCCClassifier on TRAINING_DATA only, eval on all ESC-50 mapped."""
    classifier = MFCCClassifier(k=k)
    classifier.train()  # uses TRAINING_DATA by default
    if progress:
        logger.info("Walking ALL ESC-50 folds for baseline evaluation ...")
    test_rows = walk_esc50(esc50_root, progress=progress)
    return evaluate_classifier(classifier, test_rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_classifier(classifier: MFCCClassifier, out_path: Path) -> None:
    """Persist trained classifier state to pickle.

    We save only the data needed to reconstruct the classifier — sklearn
    estimators are pickleable on the same major version. Pure-python state
    is always saved as a fallback.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MFCCClassifier.MODEL_VERSION,
        "k": classifier.k,
        "training_vectors": classifier._training_vectors,
        "feature_means": classifier._feature_means,
        "feature_stds": classifier._feature_stds,
        "training_sample_count": classifier._training_sample_count,
        "training_class_count": classifier._training_class_count,
        "use_sklearn": classifier._use_sklearn,
        "label_list": classifier._label_list,
        "sklearn_knn": classifier._sklearn_knn,
        "sklearn_scaler": classifier._sklearn_scaler,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


def load_classifier(in_path: Path) -> MFCCClassifier:
    """Reconstruct an MFCCClassifier from a saved pickle."""
    with open(in_path, "rb") as f:
        payload = pickle.load(f)
    classifier = MFCCClassifier(k=payload["k"])
    classifier._training_vectors = payload["training_vectors"]
    classifier._feature_means = payload["feature_means"]
    classifier._feature_stds = payload["feature_stds"]
    classifier._training_sample_count = payload["training_sample_count"]
    classifier._training_class_count = payload["training_class_count"]
    classifier._use_sklearn = payload["use_sklearn"]
    classifier._label_list = payload["label_list"]
    classifier._sklearn_knn = payload["sklearn_knn"]
    classifier._sklearn_scaler = payload["sklearn_scaler"]
    classifier._trained = True
    return classifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  overall_accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  macro_f1:         {metrics['macro_f1']:.4f}")
    print(f"  macro_precision:  {metrics['macro_precision']:.4f}")
    print(f"  macro_recall:     {metrics['macro_recall']:.4f}")
    print(f"  total_samples:    {metrics['total_samples']}")
    print(f"  mean_confidence:  {metrics['mean_confidence']:.4f}")
    print("\n  Per Tritium class:")
    for cls in sorted(metrics["per_class"].keys()):
        s = metrics["per_class"][cls]
        print(
            f"    {cls:12s} P={s['precision']:.3f} R={s['recall']:.3f} "
            f"F1={s['f1']:.3f} (TP={s['tp']:3d} FP={s['fp']:3d} "
            f"FN={s['fn']:3d} N={s['support']:3d})"
        )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Train acoustic classifier on ESC-50 with held-out evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=["esc50"],
        default="esc50",
        help="Training dataset (only ESC-50 supported for now)",
    )
    parser.add_argument(
        "--esc50-root",
        type=Path,
        default=Path("data/library/audio/ESC-50"),
        help="Path to ESC-50 dataset root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tritium-lib/data/models/acoustic_classifier_v2.pkl"),
        help="Output pickle path",
    )
    parser.add_argument(
        "--mode",
        choices=["holdout", "all", "baseline", "compare"],
        default="compare",
        help="holdout: fold5 eval. all: train on everything. baseline: synthetic-only eval. compare: run all three",
    )
    parser.add_argument("--test-fold", type=int, default=5)
    parser.add_argument("--k", type=int, default=5, help="KNN k")
    parser.add_argument(
        "--augment-synthetic",
        action="store_true",
        help="Mix in TRAINING_DATA synthetic profiles alongside real WAVs",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Cap samples per ESC-50 category (None=use all 40)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON metrics report path",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    progress = not args.quiet

    if not args.esc50_root.exists():
        print(f"ERROR: ESC-50 root not found at {args.esc50_root}", file=sys.stderr)
        return 2

    t0 = time.time()
    report: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "timestamp": time.time(),
    }

    if args.mode in ("baseline", "compare"):
        print("\n[1/3] Synthetic-only baseline (TRAINING_DATA -> all ESC-50 mapped)...")
        baseline = evaluate_synthetic_baseline(args.esc50_root, k=args.k, progress=progress)
        report["baseline_synthetic"] = baseline
        _print_report("BASELINE: synthetic-trained on real ESC-50", baseline)

    if args.mode in ("holdout", "compare"):
        print(f"\n[2/3] Holdout: train on folds!={args.test_fold}, test on fold {args.test_fold}...")
        classifier, train_metrics, test_metrics = train_holdout(
            args.esc50_root,
            test_fold=args.test_fold,
            k=args.k,
            augment_synthetic=args.augment_synthetic,
            max_per_category=args.max_per_category,
            progress=progress,
        )
        report["holdout_train"] = train_metrics
        report["holdout_test"] = test_metrics
        _print_report(
            f"HOLDOUT: trained on real ESC-50 (folds!={args.test_fold}), eval on fold {args.test_fold}",
            test_metrics,
        )
        # Save the holdout-trained model by default unless mode == 'all'
        if args.mode == "holdout":
            save_classifier(classifier, args.out)
            print(f"\nSaved holdout-trained classifier to {args.out}")

    if args.mode in ("all", "compare"):
        print("\n[3/3] All-fold training (production model)...")
        classifier_all, train_metrics_all = train_all(
            args.esc50_root,
            k=args.k,
            augment_synthetic=args.augment_synthetic,
            max_per_category=args.max_per_category,
            progress=progress,
        )
        report["all_train"] = train_metrics_all
        if args.mode in ("all", "compare"):
            save_classifier(classifier_all, args.out)
            print(f"Saved all-fold classifier to {args.out}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    report["elapsed_s"] = round(elapsed, 1)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Wrote report to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
