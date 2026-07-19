"""A detection must carry WHO produced its label, not just the label.

The defect this file pins down: :class:`BackgroundMotionDetector` is a
background-subtraction blob finder with no classifier in it at all, yet it
emitted COCO class names -- and ``person`` was the *catch-all fallback*, so
every blob that was neither tall nor wide reached the operator's wall badged
PERSON beside a fill-ratio dressed up as a confidence.

The contract here: a motion backend reports ``motion`` (honest: something
moved), records its geometry guess in ``shape_hint`` (honest: a hint, not a
class), and stamps ``class_source="heuristic"`` so anything downstream can
refuse to treat it as an identity.  A real classifier stamps ``classifier``.
"""

import numpy as np
import pytest

from tritium_lib.models.camera import CameraDetection
from tritium_lib.perception.detector import BackgroundMotionDetector


def _blank(w: int = 320, h: int = 240) -> np.ndarray:
    return np.full((h, w, 3), 90, dtype=np.uint8)


def _stamp(frame: np.ndarray, cx: int, cy: int, pw: int, ph: int) -> np.ndarray:
    out = frame.copy()
    out[max(0, cy - ph // 2):cy + ph // 2, max(0, cx - pw // 2):cx + pw // 2] = 240
    return out


def _learn(det: BackgroundMotionDetector, frame: np.ndarray, n: int = 20) -> None:
    for _ in range(n):
        det.detect(frame, "cam1")


def _detect_blob(pw: int, ph: int) -> CameraDetection:
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn(det, bg)
    dets = det.detect(_stamp(bg, 160, 120, pw, ph), "cam1")
    assert dets, f"no detection for {pw}x{ph} blob"
    return max(dets, key=lambda d: d.bbox.area)


# --- the core refusal -------------------------------------------------------

@pytest.mark.parametrize("pw,ph,label", [
    (24, 70, "tall blob"),
    (70, 24, "wide blob"),
    (40, 40, "square blob (the old catch-all -> PERSON)"),
])
def test_motion_backend_never_emits_a_classifier_class_name(pw, ph, label):
    """No motion blob, of any shape, may be labelled with a COCO class."""
    d = _detect_blob(pw, ph)
    assert d.class_name == "motion", (
        f"{label}: motion backend emitted {d.class_name!r} -- a class no "
        f"classifier produced"
    )
    assert d.class_name not in ("person", "car", "vehicle", "truck")


def test_motion_detection_is_marked_unclassified():
    d = _detect_blob(40, 40)
    assert d.class_source == "heuristic"
    assert d.is_classified is False


def test_shape_hint_still_carries_the_geometry_guess():
    """The aspect-ratio signal is not thrown away -- it is just demoted."""
    assert _detect_blob(24, 70).shape_hint == "tall"
    assert _detect_blob(70, 24).shape_hint == "wide"
    # Neither tall nor wide: no hint at all, rather than a fabricated one.
    assert _detect_blob(40, 40).shape_hint is None


# --- the model-level contract ----------------------------------------------

def test_camera_detection_defaults_to_unclassified():
    """An unstamped detection must not be mistaken for a classified one."""
    d = CameraDetection(source_id="cam", class_name="person")
    assert d.class_source == ""
    assert d.is_classified is False


def test_a_classifier_stamped_detection_is_classified():
    d = CameraDetection(source_id="cam", class_name="person",
                        confidence=0.92, class_source="classifier")
    assert d.is_classified is True


def test_display_label_does_not_promote_a_hint_to_a_class():
    """What the operator's badge shows for an unclassified blob."""
    motion = CameraDetection(source_id="cam", class_name="motion",
                             class_source="heuristic", shape_hint="tall")
    assert motion.display_label == "MOTION"
    assert "PERSON" not in motion.display_label.upper()

    real = CameraDetection(source_id="cam", class_name="person",
                           class_source="classifier", confidence=0.9)
    assert real.display_label == "PERSON"


# --- the end-to-end claim ---------------------------------------------------

def test_a_motion_blob_does_not_become_a_person_on_the_tactical_map():
    """The headline: an unclassified blip must not reach the map as PERSON.

    This is the whole point of the change -- everything above is machinery.
    """
    from tritium_lib.tracking.target_tracker import TargetTracker

    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn(det, bg)
    blobs = det.detect(_stamp(bg, 160, 120, 40, 40), "cam1")
    assert blobs

    tracker = TargetTracker()
    tid = tracker.update_from_detection({
        "class_name": blobs[0].class_name,
        "class_source": blobs[0].class_source,
        "shape_hint": blobs[0].shape_hint,
        "confidence": blobs[0].confidence,
        "center_x": 5.0, "center_y": 5.0,
        "source_camera": "cam1",
    })
    assert tid is not None
    target = tracker.get_target(tid)
    assert target.asset_type != "person", "unclassified blob landed as a person"
    assert "person" not in tid.lower()
    assert "person" not in (target.name or "").lower()
    assert target.alliance == "unknown"


def test_solidity_score_is_not_laundered_into_classification_confidence():
    """A fill-ratio is not a classifier's certainty and must not pose as one."""
    from tritium_lib.tracking.target_tracker import TargetTracker

    tracker = TargetTracker()
    tid = tracker.update_from_detection({
        "class_name": "motion", "class_source": "heuristic",
        "shape_hint": "tall", "confidence": 0.93,
        "center_x": 1.0, "center_y": 1.0, "source_camera": "cam1",
    })
    target = tracker.get_target(tid)
    assert target.classification_confidence == 0.0, (
        "a non-classifying backend's solidity score was reported as "
        "classification confidence"
    )


def test_classifier_confidence_is_preserved():
    from tritium_lib.tracking.target_tracker import TargetTracker

    tracker = TargetTracker()
    tid = tracker.update_from_detection({
        "class_name": "person", "class_source": "classifier",
        "confidence": 0.91,
        "center_x": 1.0, "center_y": 1.0, "source_camera": "cam1",
    })
    target = tracker.get_target(tid)
    assert target.classification_confidence == pytest.approx(0.91)
    assert target.asset_type == "person"


def test_track_exposes_class_provenance_to_api_consumers():
    """to_dict() must let the operator's UI tell a guess from a verdict."""
    from tritium_lib.tracking.target_tracker import TargetTracker

    tracker = TargetTracker()
    tid = tracker.update_from_detection({
        "class_name": "motion", "class_source": "heuristic",
        "shape_hint": "tall", "confidence": 0.8,
        "center_x": 2.0, "center_y": 2.0, "source_camera": "cam1",
    })
    d = tracker.get_target(tid).to_dict()
    assert d["class_source"] == "heuristic"
