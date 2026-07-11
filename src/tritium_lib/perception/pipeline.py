# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Frame -> detection -> world -> sink pipeline.

Ties the reusable perception pieces into one provider-driven loop:

    frame_provider() -> BGR frame
        -> detector.detect(frame)          (bbox detections)
        -> camera_model.project(det)       (world x/y/lat/lng)
        -> detection_sink(payload)         (e.g. TargetTracker.update_from_detection)

It owns NO framework objects — the frame source, camera pose, and detection
sink are all injected callables — so the SAME pipeline drives a synthetic
demo feed, an RTSP security camera, or a unit-test fixture.  The sink payload
matches exactly what :meth:`TargetTracker.update_from_detection` expects
(``class_name``, ``confidence``, ``center_x``, ``center_y`` in local metres),
so camera detections become ``det_*`` tracks that fuse with BLE/mesh/sim into
one unique target id.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

from tritium_lib.models.camera import CameraDetection
from tritium_lib.perception.detector import FrameObjectDetector
from tritium_lib.perception.projection import GroundCameraModel

logger = logging.getLogger("tritium.perception.pipeline")

FrameProvider = Callable[[], Optional[np.ndarray]]
ModelProvider = Callable[[], Optional[GroundCameraModel]]
DetectionSink = Callable[[dict], None]


class FrameDetectionPipeline:
    """Run detection on frames and push world-positioned detections to a sink."""

    def __init__(
        self,
        detector: FrameObjectDetector,
        frame_provider: FrameProvider,
        detection_sink: DetectionSink,
        model_provider: Optional[ModelProvider] = None,
        source_id: str = "",
        interval: float = 1.0,
        min_confidence: float = 0.4,
    ) -> None:
        self.detector = detector
        self._frame_provider = frame_provider
        self._sink = detection_sink
        self._model_provider = model_provider
        self.source_id = source_id
        self.interval = max(0.1, float(interval))
        self.min_confidence = float(min_confidence)

        self.detections_total = 0
        self.detections_last_tick = 0
        self.frames_processed = 0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- one-shot ----------------------------------------------------------

    def tick(self) -> int:
        """Grab one frame, detect, project, and emit. Returns detections made."""
        frame = self._safe_frame()
        if frame is None:
            self.detections_last_tick = 0
            return 0
        self.frames_processed += 1
        model = self._safe_model()

        try:
            detections = self.detector.detect(frame, self.source_id)
        except Exception as exc:  # never let a bad frame kill the loop
            logger.debug("detector error on %s: %s", self.source_id, exc)
            self.detections_last_tick = 0
            return 0

        emitted = 0
        for det in detections:
            if det.confidence < self.min_confidence:
                continue
            payload = self._build_payload(det, model, frame.shape)
            try:
                self._sink(payload)
                emitted += 1
            except Exception as exc:
                logger.debug("sink error on %s: %s", self.source_id, exc)

        self.detections_last_tick = emitted
        self.detections_total += emitted
        return emitted

    def _build_payload(self, det: CameraDetection, model, frame_shape) -> dict:
        payload = {
            "class_name": det.class_name,
            "confidence": det.confidence,
            "source_camera": self.source_id,
            "bbox": {
                "x": det.bbox.x, "y": det.bbox.y,
                "w": det.bbox.w, "h": det.bbox.h,
            },
        }
        if model is not None:
            # Keep the model's image size honest against the real frame.
            if frame_shape is not None and len(frame_shape) >= 2:
                model.image_h = int(frame_shape[0])
                model.image_w = int(frame_shape[1])
            world = model.project(det)
            payload["center_x"] = world["x"]
            payload["center_y"] = world["y"]
            if world.get("lat") is not None:
                payload["lat"] = world["lat"]
                payload["lng"] = world["lng"]
            payload["bearing_deg"] = world["bearing_deg"]
            payload["distance_m"] = world["distance_m"]
        else:
            # No pose: fall back to normalized image-centre coords (the
            # tracker treats |c|<2 as normalized and clusters by IoU-ish
            # proximity — still yields a det_* track, just not geo-placed).
            w = frame_shape[1] if frame_shape is not None and len(frame_shape) >= 2 else 1
            h = frame_shape[0] if frame_shape is not None and len(frame_shape) >= 2 else 1
            payload["center_x"] = (det.bbox.x + det.bbox.w / 2.0) / max(1, w)
            payload["center_y"] = (det.bbox.y + det.bbox.h / 2.0) / max(1, h)
        return payload

    # -- background loop ---------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"frame-detect-{self.source_id or 'cam'}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Frame detection pipeline started for %s (%s @ %.1fs)",
                    self.source_id, self.detector.backend_name, self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                logger.debug("pipeline tick error on %s: %s", self.source_id, exc)
            self._stop.wait(self.interval)

    # -- safe accessors ----------------------------------------------------

    def _safe_frame(self) -> Optional[np.ndarray]:
        try:
            return self._frame_provider()
        except Exception as exc:
            logger.debug("frame provider error on %s: %s", self.source_id, exc)
            return None

    def _safe_model(self) -> Optional[GroundCameraModel]:
        if self._model_provider is None:
            return None
        try:
            return self._model_provider()
        except Exception as exc:
            logger.debug("model provider error on %s: %s", self.source_id, exc)
            return None
