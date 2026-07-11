# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Frame object detection — pixels in, bounding-box detections out.

This is the reusable perception primitive that turns a raw camera frame
into a list of :class:`~tritium_lib.models.camera.CameraDetection` boxes.
It is the missing rung between the L0-L2 :class:`FrameAnalyzer` (which only
reports a single motion centroid) and the multi-sensor
:class:`TargetTracker` (which wants class + position per object).

Two backends fill the SAME :class:`FrameObjectDetector` interface so the
runtime can pick the best available without the caller caring:

  * :class:`BackgroundMotionDetector` — pure OpenCV background-subtraction
    (MOG2) + contour extraction.  No neural net, no GPU, no weights, no
    torch.  Deterministic for a fixed frame sequence.  This is the honest
    always-available detector: a classic static-CCTV foreground detector,
    exactly what runs when a real camera has no accelerator.
  * :class:`YoloObjectDetector` — a graceful wrapper over ultralytics YOLO.
    If ultralytics/torch are not importable (the common case on a headless
    box), it is simply unavailable and the factory falls back to motion.

Production ↔ fun: the SAME detector runs on synthetic demo frames (a person
walking through a simulated camera lights up the tactical map) and on real
RTSP frames from a security camera (the production track).  Swapping a real
camera in changes the frame source, not this code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import cv2
import numpy as np

from tritium_lib.models.camera import BoundingBox, CameraDetection

logger = logging.getLogger("tritium.perception.detector")


# COCO-ish classes a security camera cares about (mirrors the SC YOLO
# pipeline's RELEVANT_CLASSES so a real-YOLO backend and the motion backend
# agree on the vocabulary the tracker consumes).
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
}


class FrameObjectDetector(ABC):
    """Abstract detector: a BGR frame -> a list of CameraDetection boxes."""

    #: Human-readable backend name for telemetry ("motion", "yolo:yolov8n").
    backend_name: str = "abstract"

    @abstractmethod
    def detect(self, frame: np.ndarray, source_id: str = "") -> list[CameraDetection]:
        """Detect objects in a single BGR frame.

        Args:
            frame: HxWx3 BGR uint8 image.
            source_id: Camera id, stamped onto each detection for attribution.

        Returns:
            Detections with *pixel-space* bounding boxes, highest-confidence
            (or largest) first.
        """

    def reset(self) -> None:
        """Reset any temporal state (e.g. a learned background). No-op by default."""


class BackgroundMotionDetector(FrameObjectDetector):
    """Classic static-camera foreground detector (MOG2 + contours).

    A security camera is usually static, so anything that moves against the
    learned background is a candidate object.  This is deterministic given a
    fixed frame sequence (MOG2 uses no RNG at inference), needs no model
    weights, and runs in a few milliseconds on CPU — the correct honest
    detector when there is no accelerator (and a legitimate production
    fallback the moment a GPU is unavailable).

    Classification is a coarse geometric heuristic on the blob's aspect
    ratio: tall blobs read as ``person``, wide blobs as ``car``; the tracker
    downstream fuses these with richer sensors for a final identity.
    """

    backend_name = "motion"

    def __init__(
        self,
        min_area: int = 250,
        max_detections: int = 20,
        history: int = 120,
        var_threshold: float = 24.0,
        learning_rate: float = -1.0,
        person_aspect: float = 1.25,
        vehicle_aspect: float = 1.4,
    ) -> None:
        """
        Args:
            min_area: Minimum blob area in pixels to report (rejects noise).
            max_detections: Cap on detections per frame (largest kept).
            history: MOG2 background history length (frames).
            var_threshold: MOG2 Mahalanobis variance threshold.
            learning_rate: MOG2 apply() learning rate (-1 = auto).
            person_aspect: h/w above which a blob reads as "person".
            vehicle_aspect: w/h above which a blob reads as "car".
        """
        self.min_area = int(min_area)
        self.max_detections = int(max_detections)
        self.learning_rate = float(learning_rate)
        self.person_aspect = float(person_aspect)
        self.vehicle_aspect = float(vehicle_aspect)
        self._history = int(history)
        self._var_threshold = float(var_threshold)
        self._bg = self._new_bg()
        # 3x3 kernel to close speckle holes and merge adjacent foreground.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _new_bg(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )

    def reset(self) -> None:
        """Forget the learned background (e.g. after a scene cut)."""
        self._bg = self._new_bg()

    def detect(self, frame: np.ndarray, source_id: str = "") -> list[CameraDetection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        if frame.ndim == 2:  # grayscale in -> fake a 3rd axis for MOG2
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        fgmask = self._bg.apply(frame, learningRate=self.learning_rate)
        # Binarize (MOG2 with detectShadows=False is already 0/255, but be safe)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # Denoise: open removes speckle, close fills a body into one blob.
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        ts = datetime.now(timezone.utc)
        dets: list[CameraDetection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            class_name = self._classify(w, h)
            # Fill ratio: how solid the blob is inside its box (a real object
            # fills more of its box than scattered noise).  Maps to confidence.
            fill = float(area) / float(w * h)
            confidence = max(0.4, min(0.98, 0.45 + 0.5 * fill))
            dets.append(
                CameraDetection(
                    source_id=source_id or "motion",
                    class_name=class_name,
                    confidence=round(confidence, 3),
                    bbox=BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                    timestamp=ts,
                )
            )

        # Largest boxes first, capped.
        dets.sort(key=lambda d: d.bbox.area, reverse=True)
        return dets[: self.max_detections]

    def _classify(self, w: int, h: int) -> str:
        if h >= w * self.person_aspect:
            return "person"
        if w >= h * self.vehicle_aspect:
            return "car"
        return "person"


class YoloObjectDetector(FrameObjectDetector):
    """Graceful ultralytics-YOLO backend.

    Constructing this raises ``RuntimeError`` when ultralytics/torch are not
    importable, so :func:`build_frame_detector` can catch it and fall back to
    :class:`BackgroundMotionDetector`.  When available it is a drop-in
    replacement — same interface, richer classes — so a box with a GPU gets
    real YOLO while everything downstream (projection, tracker, map) is
    unchanged.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as exc:  # ImportError, or torch load failure
            raise RuntimeError(
                f"ultralytics YOLO not available ({exc}); "
                "use BackgroundMotionDetector"
            ) from exc

        from ultralytics import YOLO

        self.confidence_threshold = float(confidence_threshold)
        self._model = YOLO(model_name)
        if device:
            self._model.to(device)
        self._names = self._model.names
        self.backend_name = f"yolo:{model_name}"

    def detect(self, frame: np.ndarray, source_id: str = "") -> list[CameraDetection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        ts = datetime.now(timezone.utc)
        results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        dets: list[CameraDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in RELEVANT_CLASSES:
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(float)
                dets.append(
                    CameraDetection(
                        source_id=source_id or "yolo",
                        class_name=self._names.get(cls_id, RELEVANT_CLASSES[cls_id]),
                        confidence=round(conf, 3),
                        bbox=BoundingBox(
                            x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1),
                        ),
                        timestamp=ts,
                    )
                )
        return dets


def yolo_available() -> bool:
    """True if the ultralytics/torch YOLO backend can be imported."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> dict[str, bool]:
    """Report which detector backends are usable in this environment."""
    return {"motion": True, "yolo": yolo_available()}


def build_frame_detector(
    prefer: str = "auto",
    *,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    device: str | None = None,
    **motion_kwargs,
) -> FrameObjectDetector:
    """Build the best available frame detector.

    Args:
        prefer: "auto" (YOLO if available else motion), "yolo" (require/try
            YOLO, fall back to motion on failure), or "motion" (force the
            classical detector).
        model_name/confidence_threshold/device: forwarded to the YOLO backend.
        **motion_kwargs: forwarded to :class:`BackgroundMotionDetector`.

    Returns:
        A ready :class:`FrameObjectDetector`.  Its ``backend_name`` reports
        which backend was selected.
    """
    prefer = (prefer or "auto").lower()
    if prefer in ("yolo", "auto"):
        try:
            det = YoloObjectDetector(
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            logger.info("Frame detector: %s", det.backend_name)
            return det
        except Exception as exc:
            if prefer == "yolo":
                logger.warning("YOLO requested but unavailable (%s); using motion", exc)
            else:
                logger.info("YOLO unavailable (%s); using motion detector", exc)
    det = BackgroundMotionDetector(**motion_kwargs)
    logger.info("Frame detector: %s", det.backend_name)
    return det
