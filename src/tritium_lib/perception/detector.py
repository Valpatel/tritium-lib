# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Frame object detection — pixels in, bounding-box detections out.

This is the reusable perception primitive that turns a raw camera frame
into a list of :class:`~tritium_lib.models.camera.CameraDetection` boxes.
It is the missing rung between the L0-L2 :class:`FrameAnalyzer` (which only
reports a single motion centroid) and the multi-sensor
:class:`TargetTracker` (which wants class + position per object).

Three backends fill the SAME :class:`FrameObjectDetector` interface so the
runtime can pick the best available without the caller caring:

  * :class:`BackgroundMotionDetector` — pure OpenCV background-subtraction
    (MOG2) + contour extraction.  No neural net, no GPU, no weights, no
    torch.  Deterministic for a fixed frame sequence.  This is the honest
    always-available detector: a classic static-CCTV foreground detector,
    exactly what runs when a real camera has no accelerator.
  * :class:`YoloObjectDetector` — a graceful wrapper over ultralytics YOLO.
    If ultralytics/torch are not importable (the common case on a headless
    box), it is simply unavailable and the factory falls back.
  * :class:`OnnxYoloDetector` — a YOLO ``.onnx`` model run through
    onnxruntime on CPU.  Real class labels without torch/ultralytics:
    letterbox preprocess + v8/v5 output decode + NMS live here in plain
    numpy/cv2 (see :func:`letterbox_frame` / :func:`decode_yolo_predictions`,
    which are pure and unit-testable without a model).  The model path comes
    from the caller or the ``TRITIUM_YOLO_ONNX`` env var
    (:func:`resolve_onnx_model`).

Production ↔ fun: the SAME detector runs on synthetic demo frames (a person
walking through a simulated camera lights up the tactical map) and on real
RTSP frames from a security camera (the production track).  Swapping a real
camera in changes the frame source, not this code.
"""

from __future__ import annotations

import logging
import os
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

# Full COCO 80-class vocabulary, index == class id.  A raw ONNX YOLO export
# carries no names metadata, so the decode step maps class ids through this
# tuple (RELEVANT_CLASSES is the id-filter; this is the id->name map).
COCO80_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


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


# ------------------------------------------------------------------ ONNX
# Pure preprocessing/decode math for the ONNX backend.  Kept module-level so
# they are unit-testable without a model file or an onnxruntime install.

def letterbox_frame(
    frame: np.ndarray, size: int = 640,
) -> tuple[np.ndarray, float, int, int]:
    """Letterbox a BGR frame into a square YOLO input blob.

    Scales the frame to fit ``size`` x ``size`` preserving aspect ratio, pads
    the remainder with the YOLO-standard grey (114), converts BGR->RGB, and
    normalizes to [0, 1] float32 in NCHW layout.

    Args:
        frame: HxWx3 BGR uint8 image.
        size: Square model input edge (e.g. 640).

    Returns:
        ``(blob, ratio, pad_x, pad_y)`` where ``blob`` is float32
        ``[1, 3, size, size]``, ``ratio`` is the original->letterbox scale,
        and ``pad_x``/``pad_y`` are the left/top pad offsets in pixels.
    """
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    return blob, float(ratio), int(pad_x), int(pad_y)


def decode_yolo_predictions(
    pred: np.ndarray,
    ratio: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    input_size: int = 640,
) -> list[tuple[int, float, float, float, float, float]]:
    """Decode a raw YOLO output tensor into NMS-filtered pixel boxes.

    Handles BOTH standard export layouts (with or without a leading batch
    dim):

      * YOLOv8: ``(4+nc, anchors)`` e.g. (84, 8400) — [cx, cy, w, h] then
        ``nc`` class scores, no objectness.
      * YOLOv5: ``(anchors, 5+nc)`` e.g. (25200, 85) — [cx, cy, w, h, obj]
        then ``nc`` class scores; confidence is ``obj * cls``.

    The layouts are told apart by orientation: the anchor axis is always the
    (much) longer one.  Some exports emit box coordinates NORMALIZED to
    [0, 1] instead of letterbox pixels; that is auto-detected (every kept
    coordinate <= 2) and rescaled by ``input_size``.  Boxes are then
    un-letterboxed back to ORIGINAL image pixel coordinates using
    ``ratio``/``pad_x``/``pad_y`` from :func:`letterbox_frame`, and
    per-class NMS'd via ``cv2.dnn.NMSBoxes``.

    Returns:
        ``[(cls_id, conf, x, y, w, h), ...]`` with x/y the top-left corner in
        original-image pixels, sorted by confidence descending.  Boxes are
        NOT clipped to any frame bounds here (the caller knows the frame).
    """
    a = np.asarray(pred, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] < 5:
        return []

    if a.shape[0] <= a.shape[1]:
        # (channels, anchors) — v8 layout: transpose, no objectness column.
        a = a.T
        boxes = a[:, :4]
        scores = a[:, 4:]
    else:
        # (anchors, channels) — v5 layout: objectness * class scores.
        boxes = a[:, :4]
        scores = a[:, 5:] * a[:, 4:5]
    if scores.shape[1] == 0:
        return []

    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]
    keep = confs >= float(conf_threshold)
    if not np.any(keep):
        return []
    boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]

    # Normalized-coordinate exports: every surviving cx/cy/w/h sits in
    # [0, 1] (letterbox-pixel exports have coords in the hundreds).
    if float(boxes.max()) <= 2.0:
        boxes = boxes * float(input_size)

    inv = 1.0 / (float(ratio) if ratio else 1.0)
    x = (boxes[:, 0] - boxes[:, 2] / 2.0 - float(pad_x)) * inv
    y = (boxes[:, 1] - boxes[:, 3] / 2.0 - float(pad_y)) * inv
    w = boxes[:, 2] * inv
    h = boxes[:, 3] * inv

    out: list[tuple[int, float, float, float, float, float]] = []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        cand = np.stack([x[m], y[m], w[m], h[m]], axis=1)
        cand_conf = confs[m]
        idxs = cv2.dnn.NMSBoxes(
            cand.tolist(), cand_conf.tolist(),
            float(conf_threshold), float(iou_threshold),
        )
        if idxs is None or len(idxs) == 0:
            continue
        for i in np.asarray(idxs).flatten():
            out.append((
                int(cid), float(cand_conf[i]),
                float(cand[i, 0]), float(cand[i, 1]),
                float(cand[i, 2]), float(cand[i, 3]),
            ))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def resolve_onnx_model(explicit: str | None = None) -> str | None:
    """Resolve an ONNX YOLO model path, or ``None`` if none is available.

    Order: ``explicit`` argument if that file exists, else the
    ``TRITIUM_YOLO_ONNX`` environment variable if set and the file exists,
    else ``None``.
    """
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        logger.debug("Explicit ONNX model path does not exist: %s", explicit)
    env_path = os.environ.get("TRITIUM_YOLO_ONNX", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    return None


class OnnxYoloDetector(FrameObjectDetector):
    """YOLO ``.onnx`` model through onnxruntime — real classes without torch.

    Constructing this raises ``RuntimeError`` (mirroring
    :class:`YoloObjectDetector`) when onnxruntime is not importable, the
    model file is missing, or the session fails to load, so
    :func:`build_frame_detector` can catch it and fall back gracefully.
    CPU-only by design (``CPUExecutionProvider``): this is the accelerator-
    free path that still yields honest class labels on a headless box.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # ImportError or a broken install
            raise RuntimeError(
                f"onnxruntime not available ({exc}); "
                "use BackgroundMotionDetector"
            ) from exc
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError(f"ONNX model not found: {model_path!r}")
        try:
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load ONNX model {model_path!r}: {exc}"
            ) from exc

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._output_name = self._session.get_outputs()[0].name
        # Static export: [1, 3, H, W].  Dynamic dims arrive as strings/None.
        shape = list(model_input.shape or [])
        size = 640
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            size = int(shape[2])
        self.input_size = size
        self.backend_name = f"onnx:{os.path.basename(model_path)}"

    def detect(self, frame: np.ndarray, source_id: str = "") -> list[CameraDetection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        if frame.ndim == 2:  # grayscale in -> 3 channels, like the motion backend
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_h, frame_w = frame.shape[:2]
        blob, ratio, pad_x, pad_y = letterbox_frame(frame, self.input_size)
        pred = self._session.run([self._output_name], {self._input_name: blob})[0]
        raw = decode_yolo_predictions(
            pred, ratio, pad_x, pad_y,
            self.confidence_threshold, self.iou_threshold,
            input_size=self.input_size,
        )

        ts = datetime.now(timezone.utc)
        dets: list[CameraDetection] = []
        for cls_id, conf, x, y, w, h in raw:
            if cls_id not in RELEVANT_CLASSES:
                continue
            # Clip to the original frame bounds; drop degenerate remnants.
            x1 = min(max(x, 0.0), float(frame_w))
            y1 = min(max(y, 0.0), float(frame_h))
            x2 = min(max(x + w, 0.0), float(frame_w))
            y2 = min(max(y + h, 0.0), float(frame_h))
            if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
                continue
            dets.append(
                CameraDetection(
                    source_id=source_id or "onnx",
                    class_name=COCO80_NAMES[cls_id],
                    confidence=round(min(1.0, conf), 3),
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    timestamp=ts,
                )
            )
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


def yolo_available() -> bool:
    """True if the ultralytics/torch YOLO backend can be imported."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def onnx_available() -> bool:
    """True if onnxruntime is importable (model availability is separate)."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> dict[str, bool]:
    """Report which detector backends are usable in this environment."""
    return {"motion": True, "yolo": yolo_available(), "onnx": onnx_available()}


def build_frame_detector(
    prefer: str = "auto",
    *,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    device: str | None = None,
    onnx_model_path: str | None = None,
    iou_threshold: float = 0.45,
    **motion_kwargs,
) -> FrameObjectDetector:
    """Build the best available frame detector.

    Args:
        prefer: Backend preference —
            * ``"auto"``: ultralytics YOLO if importable, else the ONNX
              backend if a model resolves (``onnx_model_path`` arg or the
              ``TRITIUM_YOLO_ONNX`` env var), else motion.
            * ``"yolo"``: a real-YOLO ask — same chain as "auto"
              (ultralytics, then ONNX, then motion) since ONNX still gives
              real YOLO classes; failures are logged as warnings.
            * ``"onnx"``: the ONNX backend, warning + motion fallback when
              no model resolves or the session fails.
            * ``"motion"``: force the classical detector.
        model_name/confidence_threshold/device: forwarded to the ultralytics
            backend (confidence_threshold also feeds the ONNX backend).
        onnx_model_path: explicit ONNX model path for the ONNX backend (see
            :func:`resolve_onnx_model` for the env-var fallback).
        iou_threshold: NMS IoU threshold, forwarded to the ONNX backend.
        **motion_kwargs: forwarded to :class:`BackgroundMotionDetector`.

    Returns:
        A ready :class:`FrameObjectDetector`.  Its ``backend_name`` reports
        which backend was selected.
    """
    prefer = (prefer or "auto").lower()
    explicit_ask = prefer in ("yolo", "onnx")

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
            logger.log(
                logging.WARNING if prefer == "yolo" else logging.INFO,
                "ultralytics YOLO unavailable (%s); trying ONNX", exc,
            )

    if prefer in ("yolo", "auto", "onnx"):
        resolved = resolve_onnx_model(onnx_model_path)
        if resolved is not None:
            try:
                det = OnnxYoloDetector(
                    resolved,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
                logger.info("Frame detector: %s", det.backend_name)
                return det
            except Exception as exc:
                logger.log(
                    logging.WARNING if explicit_ask else logging.INFO,
                    "ONNX backend failed (%s); using motion detector", exc,
                )
        elif explicit_ask:
            logger.warning(
                "No ONNX model resolves (path=%r, env TRITIUM_YOLO_ONNX "
                "unset or missing); using motion detector", onnx_model_path,
            )

    det = BackgroundMotionDetector(**motion_kwargs)
    logger.info("Frame detector: %s", det.backend_name)
    return det
