# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Perception pipeline — frame analysis, object detection, fact extraction.

L0: Quality gate (sharpness, brightness)
L1: Complexity (edge density)
L2: Motion (frame diff)
L3: Object detection (frame -> bounding-box detections -> world position)
L3+: Depth enrichment (aligned depth frame -> range_m + camera-frame xyz)
One-call: process_depth_frame / DepthCameraPipeline (detect -> enrich ->
place -> fuse, unique det_* map targets in a single call)
Plus: LLM chat API client, regex fact extraction from conversation.

Framework-free: pure OpenCV/numpy/stdlib. The LLM host is injected via
set_ollama_host(); the PTZ pose estimator accepts any PTZPosition; the
object detector has graceful YOLO backends (ultralytics, or ONNX via
onnxruntime CPU) and an always-available classical (motion) backend.
"""

from tritium_lib.perception.perception import (
    CameraPose,
    FrameAnalyzer,
    FrameMetrics,
    PoseEstimator,
    PTZPosition,
)
from tritium_lib.perception.detector import (
    COCO80_NAMES,
    RELEVANT_CLASSES,
    BackgroundMotionDetector,
    FrameObjectDetector,
    OnnxYoloDetector,
    YoloObjectDetector,
    available_backends,
    build_frame_detector,
    decode_yolo_predictions,
    letterbox_frame,
    onnx_available,
    resolve_onnx_model,
    yolo_available,
)
from tritium_lib.perception.depth import (
    CameraIntrinsics,
    deproject_pixel,
    enrich_detections_with_depth,
    range_for_bbox,
)
from tritium_lib.perception.depth_codec import (
    DEPTH_SCALE_MM,
    decode_depth16_png,
    encode_depth16_png,
)
from tritium_lib.perception.multipart import (
    MultipartPart,
    boundary_from_content_type,
    iter_multipart,
)
from tritium_lib.perception.projection import (
    CameraWorldPose,
    GroundCameraModel,
    place_detections_on_map,
    world_from_camera_xyz,
)
from tritium_lib.perception.pipeline import FrameDetectionPipeline
from tritium_lib.perception.depth_pipeline import (
    DepthCameraPipeline,
    process_depth_frame,
)
from tritium_lib.perception.extraction import extract_facts, extract_person_name
from tritium_lib.perception.vision import (
    check_radio_detection,
    ollama_chat,
    set_ollama_host,
)

__all__ = [
    "CameraPose",
    "FrameAnalyzer",
    "FrameMetrics",
    "PoseEstimator",
    "PTZPosition",
    "COCO80_NAMES",
    "RELEVANT_CLASSES",
    "BackgroundMotionDetector",
    "FrameObjectDetector",
    "OnnxYoloDetector",
    "YoloObjectDetector",
    "CameraIntrinsics",
    "deproject_pixel",
    "enrich_detections_with_depth",
    "range_for_bbox",
    "DEPTH_SCALE_MM",
    "decode_depth16_png",
    "encode_depth16_png",
    "MultipartPart",
    "boundary_from_content_type",
    "iter_multipart",
    "CameraWorldPose",
    "GroundCameraModel",
    "place_detections_on_map",
    "world_from_camera_xyz",
    "FrameDetectionPipeline",
    "DepthCameraPipeline",
    "process_depth_frame",
    "available_backends",
    "build_frame_detector",
    "decode_yolo_predictions",
    "letterbox_frame",
    "onnx_available",
    "resolve_onnx_model",
    "yolo_available",
    "check_radio_detection",
    "extract_facts",
    "extract_person_name",
    "ollama_chat",
    "set_ollama_host",
]
